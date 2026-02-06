use candle_core::{IndexOp, Result, Tensor};
use rand::distributions::{Distribution, WeightedIndex};

use crate::models::qwen3::Qwen3ForCausalLM;

pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
    pub eos_token_id: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            eos_token_id: 151645, // <|im_end|> for Qwen3
        }
    }
}

fn sample_token(logits: &Tensor, params: &SamplingParams) -> Result<u32> {
    let logits = logits.to_dtype(candle_core::DType::F32)?;
    let logits_vec: Vec<f32> = logits.to_vec1()?;

    if params.temperature <= 0.0 {
        // Greedy
        let (idx, val) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        println!(
            "[Sampling] Greedy: token_id={}, logit={:.4}",
            idx, val
        );
        return Ok(idx as u32);
    }

    // Temperature scaling
    let scaled: Vec<f64> = logits_vec
        .iter()
        .map(|&v| (v as f64) / params.temperature)
        .collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut probs: Vec<(usize, f64)> = exps.iter().enumerate().map(|(i, &e)| (i, e / sum)).collect();

    // Top-p (nucleus) sampling
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cumulative = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= params.top_p {
            cutoff_idx = i + 1;
            break;
        }
    }
    let filtered = &probs[..cutoff_idx];

    // Log top-5 candidates
    let show_n = filtered.len().min(5);
    let top_candidates: Vec<String> = filtered[..show_n]
        .iter()
        .map(|(id, p)| format!("{}({:.4})", id, p))
        .collect();
    println!(
        "[Sampling] temp={:.2}, top_p={:.2}, nucleus_size={}, top-5: [{}]",
        params.temperature,
        params.top_p,
        filtered.len(),
        top_candidates.join(", ")
    );

    let weights: Vec<f64> = filtered.iter().map(|&(_, p)| p).collect();
    let indices: Vec<usize> = filtered.iter().map(|&(i, _)| i).collect();

    let dist = WeightedIndex::new(&weights).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut rng = rand::thread_rng();
    let sampled = dist.sample(&mut rng);

    let chosen_id = indices[sampled] as u32;
    let chosen_prob = weights[sampled];
    println!(
        "[Sampling] Sampled: token_id={}, prob={:.4}",
        chosen_id, chosen_prob
    );

    Ok(chosen_id)
}

/// Generate tokens autoregressively.
///
/// `input_ids`: [1, seq_len] -- the prompt tokens
/// Returns the generated token IDs (excluding the prompt).
pub fn generate(
    model: &mut Qwen3ForCausalLM,
    input_ids: &Tensor,
    params: &SamplingParams,
) -> Result<Vec<u32>> {
    model.reset_kv_cache();

    let (_, prompt_len) = input_ids.dims2()?;
    let device = model.device().clone();

    println!("========================================");
    println!("[Generate] PREFILL: prompt_len={}", prompt_len);
    println!("========================================");

    // Prefill: forward the entire prompt at once
    let logits = model.forward(input_ids, 0)?;
    let last_logits = logits.i((0, prompt_len - 1))?;

    // Log logit stats for the last position
    let logits_vec: Vec<f32> = last_logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let l_min = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let l_max = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let l_mean: f32 = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
    println!(
        "[Generate] Prefill logits (last pos): vocab_size={}, mean={:.4}, min={:.4}, max={:.4}",
        logits_vec.len(),
        l_mean,
        l_min,
        l_max
    );

    let mut next_token = sample_token(&last_logits, params)?;
    let mut generated = vec![next_token];

    println!(
        "[Generate] Prefill done. First generated token: {}",
        next_token
    );

    if next_token == params.eos_token_id {
        println!("[Generate] EOS after prefill, stopping.");
        return Ok(generated);
    }

    println!("========================================");
    println!(
        "[Generate] DECODE: max_tokens={}",
        params.max_tokens
    );
    println!("========================================");

    // Decode one token at a time
    for step in 0..params.max_tokens - 1 {
        let offset = prompt_len + step;
        println!(
            "---- Decode step {}/{}, offset={}, input_token={} ----",
            step + 1,
            params.max_tokens - 1,
            offset,
            next_token
        );

        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, offset)?;
        let last_logits = logits.i((0, 0))?;

        // Log logit stats
        let logits_vec: Vec<f32> = last_logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;
        let l_min = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let l_max = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "[Generate] Step {} logits: min={:.4}, max={:.4}",
            step + 1,
            l_min,
            l_max
        );

        next_token = sample_token(&last_logits, params)?;
        generated.push(next_token);

        println!(
            "[Generate] Step {} => token_id={}, total_generated={}",
            step + 1,
            next_token,
            generated.len()
        );

        if next_token == params.eos_token_id {
            println!("[Generate] EOS token hit at step {}, stopping.", step + 1);
            break;
        }
    }

    println!("========================================");
    println!(
        "[Generate] DONE: generated {} tokens",
        generated.len()
    );
    println!("========================================");

    Ok(generated)
}
