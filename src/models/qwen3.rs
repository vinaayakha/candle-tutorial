use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;

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
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [half]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?; // [max_seq_len]

        // [max_seq_len, half]
        let freqs = positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE to query/key tensors.
    /// Input shape: [batch, num_heads, seq_len, head_dim]
    /// offset: position offset for KV-cache continuation
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?; // [seq_len, half]
        let sin = self.sin.i(offset..offset + seq_len)?;
        let q_rot = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

// ---------------------------------------------------------------------------
// MLP: gate_proj + up_proj → SiLU-gated → down_proj
// ---------------------------------------------------------------------------

struct Qwen3MLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Qwen3MLP {
    fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Repeat KV heads for GQA
// ---------------------------------------------------------------------------

fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))
}

// ---------------------------------------------------------------------------
// Attention (GQA with QK-norm, KV-cache)
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
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3Attention {
    fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
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
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        rope: &Qwen3RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Project
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

        // QK-norm (per-head RmsNorm applied on the last dimension)
        let q = self.apply_head_norm(&self.q_norm, &q)?;
        let k = self.apply_head_norm(&self.k_norm, &k)?;

        // RoPE
        let (q, k) = rope.apply(&q, &k, offset)?;

        // KV-cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expansion
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [b, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn apply_head_norm(&self, norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
        // xs: [b, heads, seq, head_dim] — apply RmsNorm on last dim per-head
        let (b, heads, seq, hd) = xs.dims4()?;
        let xs = xs.reshape((b * heads * seq, hd))?;
        let xs = norm.forward(&xs.unsqueeze(0)?)?;
        xs.squeeze(0)?.reshape((b, heads, seq, hd))
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
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
}

impl Qwen3DecoderLayer {
    fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let self_attn = Qwen3Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Qwen3MLP::load(vb.pp("mlp"), config)?;
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
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        rope: &Qwen3RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, rope, offset)?;
        let xs = (residual + xs)?;

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
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
        let (_, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        // Causal mask
        let mask = if seq_len > 1 {
            Some(Self::build_causal_mask(seq_len, offset, &self.device)?)
        } else {
            None
        };

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), &self.rotary_emb, offset)?;
        }

        self.norm.forward(&xs)
    }

    fn build_causal_mask(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
        let total_len = offset + seq_len;
        // Create a [seq_len, total_len] mask where future positions are -inf
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?;
        Ok(mask)
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
            // Weight tying: reuse embed_tokens weight as lm_head
            let embed_weight = model.embed_tokens.embeddings().clone();
            candle_nn::Linear::new(embed_weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self { model, lm_head })
    }

    /// Returns logits for the full sequence: [batch, seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids, offset)?;
        self.lm_head.forward(&hidden)
    }

    pub fn reset_kv_cache(&mut self) {
        self.model.reset_kv_cache();
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }
}
