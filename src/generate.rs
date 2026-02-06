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
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
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

    let weights: Vec<f64> = filtered.iter().map(|&(_, p)| p).collect();
    let indices: Vec<usize> = filtered.iter().map(|&(i, _)| i).collect();

    let dist = WeightedIndex::new(&weights).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut rng = rand::thread_rng();
    let sampled = dist.sample(&mut rng);

    Ok(indices[sampled] as u32)
}

/// Generate tokens autoregressively.
///
/// `input_ids`: [1, seq_len] — the prompt tokens
/// Returns the generated token IDs (excluding the prompt).
pub fn generate(
    model: &mut Qwen3ForCausalLM,
    input_ids: &Tensor,
    params: &SamplingParams,
) -> Result<Vec<u32>> {
    model.reset_kv_cache();

    let (_, prompt_len) = input_ids.dims2()?;
    let device = model.device().clone();

    // Prefill: forward the entire prompt at once
    let logits = model.forward(input_ids, 0)?;
    // Take logits for the last position
    let last_logits = logits.i((0, prompt_len - 1))?;
    let mut next_token = sample_token(&last_logits, params)?;

    let mut generated = vec![next_token];

    if next_token == params.eos_token_id {
        return Ok(generated);
    }

    // Decode one token at a time
    for step in 0..params.max_tokens - 1 {
        let offset = prompt_len + step;
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?; // [1, 1]
        let logits = model.forward(&input, offset)?;
        let last_logits = logits.i((0, 0))?;
        next_token = sample_token(&last_logits, params)?;
        generated.push(next_token);
        if next_token == params.eos_token_id {
            break;
        }
    }

    Ok(generated)
}
