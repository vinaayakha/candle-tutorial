use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;

/// Helper: compute summary stats (mean, min, max) from a tensor for logging.
fn tensor_stats(t: &Tensor) -> (f32, f32, f32) {
    let flat = t.flatten_all().and_then(|f| f.to_dtype(DType::F32));
    let Ok(flat) = flat else { return (0.0, 0.0, 0.0) };
    let vals: Vec<f32> = flat.to_vec1().unwrap_or_default();
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let sum: f32 = vals.iter().sum();
    let mean = sum / vals.len() as f32;
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, min, max)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RmsNorm (manual, to match weight path exactly)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_dtype = xs.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let xs = xs.to_dtype(internal_dtype)?;
        let variance = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

struct Qwen3RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl Qwen3RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 * 2.0 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        let freqs = positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.i(offset..offset + seq_len)?.to_dtype(q.dtype())?;
        let q_rot = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

// ---------------------------------------------------------------------------
// MLP: gate_proj + up_proj -> SiLU-gated -> down_proj
// ---------------------------------------------------------------------------

struct Qwen3MLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    layer_idx: usize,
}

impl Qwen3MLP {
    fn load(vb: VarBuilder, config: &Qwen3Config, layer_idx: usize) -> Result<Self> {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            layer_idx,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        println!(
            "    [Layer {:>2}] MLP input: {:?}",
            self.layer_idx,
            xs.dims()
        );

        let gate = self.gate_proj.forward(xs)?;
        let gate_act = gate.silu()?;
        let up = self.up_proj.forward(xs)?;
        let gated = (&gate_act * &up)?;
        let out = self.down_proj.forward(&gated)?;

        let (g_mean, g_min, g_max) = tensor_stats(&gate_act);
        let (o_mean, o_min, o_max) = tensor_stats(&out);
        println!(
            "    [Layer {:>2}] MLP gate_silu stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx, g_mean, g_min, g_max
        );
        println!(
            "    [Layer {:>2}] MLP output: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx,
            out.dims(),
            o_mean,
            o_min,
            o_max
        );
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Attention (GQA with QK-norm, fused SDPA, preallocated KV-cache)
// ---------------------------------------------------------------------------

struct Qwen3Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_cache: candle_nn::kv_cache::KvCache,
    layer_idx: usize,
}

impl Qwen3Attention {
    fn load(vb: VarBuilder, config: &Qwen3Config, layer_idx: usize) -> Result<Self> {
        let h = config.hidden_size;
        let hd = config.head_dim;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;

        let q_proj = candle_nn::linear_no_bias(h, nh * hd, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(nh * hd, h, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::load(hd, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(hd, config.rms_norm_eps, vb.pp("k_norm"))?;

        // dim=2 is the seq_len dimension in [b, heads, seq, head_dim]
        let kv_cache = candle_nn::kv_cache::KvCache::new(2, config.max_position_embeddings);

        let scale = 1.0 / (hd as f32).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            scale,
            kv_cache,
            layer_idx,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rope: &Qwen3RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        println!(
            "    [Layer {:>2}] Attention input: [{}, {}, {}], offset={}",
            self.layer_idx, b, seq_len, hidden, offset
        );

        // Project Q/K/V
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to [b, heads, seq, head_dim]
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        println!(
            "    [Layer {:>2}] Q: {:?}, K: {:?}, V: {:?}",
            self.layer_idx,
            q.dims(),
            k.dims(),
            v.dims()
        );

        // QK-norm (per-head RmsNorm)
        let q = self.apply_head_norm(&self.q_norm, &q)?;
        let k = self.apply_head_norm(&self.k_norm, &k)?;

        let (qn_mean, qn_min, qn_max) = tensor_stats(&q);
        println!(
            "    [Layer {:>2}] After QK-norm: Q stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx, qn_mean, qn_min, qn_max
        );

        // RoPE
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Preallocated KV-cache: append new K/V and get full accumulated tensors
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let kv_seq_len = k.dim(2)?;
        println!(
            "    [Layer {:>2}] KV-cache total seq_len: {}",
            self.layer_idx, kv_seq_len
        );

        // Fused SDPA — handles GQA natively (no repeat_kv needed),
        // uses Metal GPU kernel with causal masking built-in.
        // For prefill (seq>1): do_causal=true generates the mask internally.
        // For decode (seq=1): no mask needed, uses optimized vector kernel.
        let do_causal = seq_len > 1;
        let attn_output = candle_nn::ops::sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            None,      // no explicit mask — do_causal handles it
            do_causal,
            self.scale,
            1.0,       // softcapping=1.0 means disabled
        )?;

        println!(
            "    [Layer {:>2}] SDPA output: {:?}",
            self.layer_idx,
            attn_output.dims()
        );

        // Reshape back to [b, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        let out = self.o_proj.forward(&attn_output)?;
        let (om, omin, omax) = tensor_stats(&out);
        println!(
            "    [Layer {:>2}] Attention output: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx,
            out.dims(),
            om,
            omin,
            omax
        );

        Ok(out)
    }

    fn apply_head_norm(&self, norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
        let (b, heads, seq, hd) = xs.dims4()?;
        let xs = xs.reshape((b * heads * seq, hd))?;
        let xs = norm.forward(&xs.unsqueeze(0)?)?;
        xs.squeeze(0)?.reshape((b, heads, seq, hd))
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    layer_idx: usize,
}

impl Qwen3DecoderLayer {
    fn load(vb: VarBuilder, config: &Qwen3Config, layer_idx: usize) -> Result<Self> {
        let self_attn = Qwen3Attention::load(vb.pp("self_attn"), config, layer_idx)?;
        let mlp = Qwen3MLP::load(vb.pp("mlp"), config, layer_idx)?;
        let input_layernorm =
            RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            layer_idx,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rope: &Qwen3RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let (in_mean, in_min, in_max) = tensor_stats(xs);
        println!(
            "  [Layer {:>2}] === ENTER === input: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx,
            xs.dims(),
            in_mean,
            in_min,
            in_max
        );

        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;

        let (ln_mean, ln_min, ln_max) = tensor_stats(&xs);
        println!(
            "    [Layer {:>2}] After input_layernorm: stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx, ln_mean, ln_min, ln_max
        );

        let xs = self.self_attn.forward(&xs, rope, offset)?;
        let xs = (residual + xs)?;

        let (r1_mean, _, _) = tensor_stats(&xs);
        println!(
            "    [Layer {:>2}] After attn + residual: mean={:.4}",
            self.layer_idx, r1_mean
        );

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let out = (residual + xs)?;

        let (out_mean, out_min, out_max) = tensor_stats(&out);
        println!(
            "  [Layer {:>2}] === EXIT ===  output: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            self.layer_idx,
            out.dims(),
            out_mean,
            out_min,
            out_max
        );

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Qwen3Model (embed + layers + final norm)
// ---------------------------------------------------------------------------

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    pub device: Device,
}

impl Qwen3Model {
    fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let device = vb.device().clone();
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::load(
                vb.pp(format!("layers.{i}")),
                config,
                i,
            )?);
        }
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = Qwen3RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            &device,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            device,
        })
    }

    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        println!(
            "[Model] Forward: input_ids=[{}, {}], offset={}",
            batch, seq_len, offset
        );

        let mut xs = self.embed_tokens.forward(input_ids)?;
        let (em, emin, emax) = tensor_stats(&xs);
        println!(
            "[Model] Embedding output: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            xs.dims(),
            em,
            emin,
            emax
        );

        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            println!("--- Layer {}/{} ---", i, num_layers - 1);
            xs = layer.forward(&xs, &self.rotary_emb, offset)?;
        }

        let xs = self.norm.forward(&xs)?;
        let (nm, nmin, nmax) = tensor_stats(&xs);
        println!(
            "[Model] Final RmsNorm output: {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            xs.dims(),
            nm,
            nmin,
            nmax
        );

        Ok(xs)
    }

    pub fn reset_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.self_attn.reset_kv_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3ForCausalLM (model + lm_head with weight tying)
// ---------------------------------------------------------------------------

pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head: candle_nn::Linear,
}

impl Qwen3ForCausalLM {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let model = Qwen3Model::load(vb.pp("model"), config)?;

        let lm_head = if config.tie_word_embeddings {
            let embed_weight = model.embed_tokens.embeddings().clone();
            candle_nn::Linear::new(embed_weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, offset)?;
        let logits = self.lm_head.forward(&hidden)?;

        let (lm, lmin, lmax) = tensor_stats(&logits);
        println!(
            "[CausalLM] lm_head output (logits): {:?}, stats: mean={:.4}, min={:.4}, max={:.4}",
            logits.dims(),
            lm,
            lmin,
            lmax
        );

        Ok(logits)
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }
}
