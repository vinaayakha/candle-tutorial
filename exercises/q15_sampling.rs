/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 15: Token Sampling & Generation
/// ============================================================================
/// Difficulty: ADVANCED (NUANCES)
///
/// Sampling is how we convert logits into actual tokens. It deeply affects
/// output quality and is a common interview topic for LLM positions.
///
/// KEY CONCEPTS:
///   - Greedy: argmax(logits) — deterministic, often repetitive
///   - Temperature: logits / T — higher T = more random, lower T = more peaked
///   - Top-p (nucleus): keep smallest set of tokens whose cumulative prob >= p
///   - Top-k: keep only top-k highest probability tokens
///   - Softmax: exp(x_i) / sum(exp(x_j)) — convert logits to probabilities
///   - Log-softmax trick: subtract max before exp to prevent overflow
///
/// INTERVIEW NUANCE:
///   - Temperature 0 ≈ greedy (implementation uses <= 0 check)
///   - Temperature > 1 makes distribution flatter (more creative/random)
///   - Temperature < 1 makes distribution sharper (more deterministic)
///   - top_p=0.9 means: keep tokens until their cumulative probability hits 90%
///   - top_p=1.0 disables nucleus sampling (keep all tokens)
///   - The sampling order matters: temp -> softmax -> top-p -> sample
///   - EOS token must be filtered from output AFTER generation, not during
///   - Candle does NOT have built-in sampling; you must implement it yourself
///   - WeightedIndex from rand crate is used for multinomial sampling
///
/// HINTS:
///   - Sort by probability descending, accumulate until >= top_p
///   - Use rand::distributions::WeightedIndex for weighted random choice
///   - Always handle the temp<=0 case specially (greedy)
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q15a: Implement softmax with the numerical stability trick.
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// Input: 1D tensor [vocab_size]. Output: 1D tensor [vocab_size] (probabilities).
pub fn stable_softmax(logits: &Tensor) -> Result<Vec<f64>> {
    // TODO: Implement stable softmax manually
    // Step 1: Get logits as Vec<f64>
    // Step 2: Find max value
    // Step 3: Subtract max, exp each
    // Step 4: Divide by sum
    //
    // HINT:
    //   let logits_vec: Vec<f32> = logits.to_vec1()?;
    //   let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    //   let exps: Vec<f64> = logits_vec.iter().map(|&v| ((v - max_val) as f64).exp()).collect();
    //   let sum: f64 = exps.iter().sum();
    //   Ok(exps.into_iter().map(|e| e / sum).collect())
    //
    // INTERVIEW Q: Why subtract max before exp?
    //   Answer: exp(large_number) overflows to inf. Subtracting max ensures
    //   the largest exponent is exp(0) = 1. This doesn't change the result
    //   because softmax(x) == softmax(x - c) for any constant c.
    todo!("Stable softmax")
}

/// Q15b: Implement temperature scaling.
/// logits_scaled = logits / temperature
/// Then apply softmax.
pub fn temperature_softmax(logits: &Tensor, temperature: f64) -> Result<Vec<f64>> {
    // TODO: Scale logits by temperature, then softmax
    // HINT:
    //   let scaled: Vec<f32> = logits.to_vec1::<f32>()?
    //       .iter().map(|&v| v as f32 / temperature as f32).collect();
    //   let scaled_tensor = Tensor::new(scaled.as_slice(), logits.device())?;
    //   stable_softmax(&scaled_tensor)
    //
    // INTERVIEW Q: What happens as temperature -> 0?
    //   Answer: The distribution becomes a one-hot vector (all probability
    //   on the argmax). This is equivalent to greedy decoding.
    // INTERVIEW Q: What happens as temperature -> infinity?
    //   Answer: The distribution becomes uniform (all tokens equally likely).
    todo!("Temperature scaling")
}

/// Q15c: Implement top-p (nucleus) sampling.
/// Given probabilities, keep the smallest set whose cumulative sum >= top_p.
/// Return the filtered (token_id, probability) pairs.
pub fn top_p_filter(probs: &[(usize, f64)], top_p: f64) -> Vec<(usize, f64)> {
    // TODO: Implement nucleus sampling filter
    // Step 1: Sort by probability descending
    // Step 2: Accumulate probabilities
    // Step 3: Keep tokens until cumulative >= top_p
    //
    // HINT:
    //   let mut sorted = probs.to_vec();
    //   sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    //   let mut cumulative = 0.0;
    //   let mut filtered = Vec::new();
    //   for (id, p) in sorted {
    //       cumulative += p;
    //       filtered.push((id, p));
    //       if cumulative >= top_p {
    //           break;
    //       }
    //   }
    //   filtered
    //
    // INTERVIEW Q: Why top-p over top-k?
    //   Answer: top-k uses a fixed number of tokens regardless of distribution shape.
    //   If the model is very confident, top-k=50 includes many near-zero tokens.
    //   top-p adapts: confident = fewer tokens, uncertain = more tokens.
    todo!("Top-p filter")
}

/// Q15d: Implement greedy decoding (argmax).
/// Return the token_id with highest logit.
pub fn greedy_decode(logits: &Tensor) -> Result<u32> {
    // TODO: Find argmax
    // HINT:
    //   let logits_vec: Vec<f32> = logits.to_vec1()?;
    //   let (idx, _) = logits_vec.iter().enumerate()
    //       .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //       .unwrap();
    //   Ok(idx as u32)
    todo!("Greedy decode")
}

/// Q15e: FULL SAMPLING PIPELINE
/// Combine temperature, softmax, top-p, and weighted random sampling.
/// logits: [vocab_size]
/// temperature: f64
/// top_p: f64
/// Return: sampled token_id
///
/// Use a fixed seed for reproducibility in tests.
pub fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    // TODO: Full sampling pipeline
    // If temperature <= 0: return greedy
    // Otherwise:
    //   1. Temperature scale
    //   2. Softmax
    //   3. Top-p filter
    //   4. Weighted random sample
    //
    // HINT for weighted sampling:
    //   use rand::distributions::{Distribution, WeightedIndex};
    //   use rand::SeedableRng;
    //   let weights: Vec<f64> = filtered.iter().map(|&(_, p)| p).collect();
    //   let indices: Vec<usize> = filtered.iter().map(|&(i, _)| i).collect();
    //   let dist = WeightedIndex::new(&weights).unwrap();
    //   let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    //   let sampled = dist.sample(&mut rng);
    //   Ok(indices[sampled] as u32)
    todo!("Full sampling pipeline")
}

/// Q15f: INTERVIEW QUESTION — The generation loop.
/// Explain the prefill vs decode phases and how the offset changes.
/// Fill in the blanks:
pub fn generation_loop_quiz() -> (usize, usize, usize) {
    // Given: prompt_len = 10, max_tokens = 5
    // Q: What is the offset for:
    //   - Prefill phase? (forward entire prompt)
    //   - Decode step 1? (first generated token)
    //   - Decode step 3? (third generated token)
    //
    // TODO: Return (prefill_offset, decode1_offset, decode3_offset)
    // HINT:
    //   Prefill: offset = 0 (start from beginning)
    //   Decode step i: offset = prompt_len + i  (0-indexed)
    //   So: decode step 1: offset = 10 + 0 = 10
    //       decode step 3: offset = 10 + 2 = 12
    todo!("Generation loop quiz")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_softmax() -> Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let probs = stable_softmax(&logits)?;
        assert_eq!(probs.len(), 3);
        // Sum should be 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Probabilities should be increasing
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
        Ok(())
    }

    #[test]
    fn test_softmax_with_large_values() -> Result<()> {
        // This should NOT overflow thanks to the stability trick
        let logits = Tensor::new(&[1000.0f32, 1001.0, 1002.0], &Device::Cpu)?;
        let probs = stable_softmax(&logits)?;
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1 even with large values");
        Ok(())
    }

    #[test]
    fn test_temperature() -> Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 10.0], &Device::Cpu)?;

        // Low temperature = sharper distribution (more confident)
        let sharp = temperature_softmax(&logits, 0.1)?;
        // High temperature = flatter distribution
        let flat = temperature_softmax(&logits, 10.0)?;

        // With sharp temp, the max prob should be much higher
        let sharp_max = sharp.iter().cloned().fold(0.0f64, f64::max);
        let flat_max = flat.iter().cloned().fold(0.0f64, f64::max);
        assert!(sharp_max > flat_max, "Low temp should give sharper distribution");
        Ok(())
    }

    #[test]
    fn test_top_p() {
        let probs = vec![(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)];
        let filtered = top_p_filter(&probs, 0.8);
        // Should keep tokens 0 and 1 (0.5 + 0.3 = 0.8 >= 0.8)
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 0);
        assert_eq!(filtered[1].0, 1);
    }

    #[test]
    fn test_top_p_all() {
        let probs = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let filtered = top_p_filter(&probs, 1.0);
        // top_p=1.0 should keep all tokens
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_greedy() -> Result<()> {
        let logits = Tensor::new(&[0.1f32, 0.5, 0.3, 0.9, 0.2], &Device::Cpu)?;
        let token = greedy_decode(&logits)?;
        assert_eq!(token, 3); // index of 0.9
        Ok(())
    }

    #[test]
    fn test_generation_loop_quiz() {
        let (prefill, decode1, decode3) = generation_loop_quiz();
        assert_eq!(prefill, 0);
        assert_eq!(decode1, 10);
        assert_eq!(decode3, 12);
    }
}

fn main() -> Result<()> {
    println!("=== Q15: Token Sampling & Generation ===\n");
    println!("Run `cargo test --example q15_sampling` to verify.");
    Ok(())
}
