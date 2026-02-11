/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 12: KV-Cache Implementation
/// ============================================================================
/// Difficulty: ADVANCED
///
/// KV-cache is what makes autoregressive generation efficient.
/// Without it, every new token requires re-computing attention for all past tokens.
///
/// KEY CONCEPTS:
///   - candle_nn::kv_cache::KvCache::new(dim, max_seq_len)
///   - .append(&k, &v) — adds new K/V and returns full accumulated tensors
///   - .reset() — clears cache between different prompts
///   - dim=2 for attention cache: [batch, heads, SEQ, head_dim]
///   - Prefill: forward entire prompt, cache all K/V
///   - Decode: forward 1 token, append to cache, attend to all cached K/V
///
/// INTERVIEW NUANCE:
///   - Without KV-cache: decode step i takes O(i * d^2) compute
///   - With KV-cache: decode step i takes O(d^2) compute (just the new token!)
///   - Total: O(n^2 * d^2) without vs O(n * d^2) with KV-cache
///   - The cache is on dimension 2 because attention shape is [b, h, s, d]
///     and we're appending along the sequence dimension
///   - K and V must be .contiguous() before appending to cache!
///   - After prefill: cache has seq_len entries
///   - After decode step i: cache has seq_len + i entries
///   - The "offset" parameter in forward tells the model the absolute position
///   - Memory grows linearly with generated tokens (bounded by max_seq_len)
///
/// HINTS:
///   - KvCache::new(2, max_positions) — dim=2 for seq dimension
///   - cache.append(&k.contiguous()?, &v.contiguous()?)? returns (full_k, full_v)
///   - cache.reset() between generations
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

/// Q12a: Create a KV-cache for attention layers.
/// dim: which dimension to concatenate along (should be 2 for attention)
/// max_len: maximum sequence length
pub fn create_kv_cache(dim: usize, max_len: usize) -> KvCache {
    // TODO: Create a KvCache
    // HINT: KvCache::new(dim, max_len)
    todo!("Create KV-cache")
}

/// Q12b: Simulate a PREFILL + DECODE cycle with KV-cache.
/// This tests your understanding of the generation loop.
///
/// 1. Create cache
/// 2. Prefill: append K/V of shape [1, 2, 5, 4] (batch=1, heads=2, seq=5, dim=4)
/// 3. Verify cache now has seq_len=5
/// 4. Decode step 1: append K/V of shape [1, 2, 1, 4] (single new token)
/// 5. Verify total cached seq_len=6
/// 6. Decode step 2: append another [1, 2, 1, 4]
/// 7. Verify total cached seq_len=7
/// 8. Reset cache
/// 9. Verify cache is empty
///
/// Return: Vec of seq_lens observed at steps 3, 5, 7, and after reset.
pub fn simulate_generation() -> Result<Vec<usize>> {
    // TODO: Implement the simulation
    // HINT:
    //   let mut cache = KvCache::new(2, 100);
    //   let k_prefill = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu)?;
    //   let v_prefill = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu)?;
    //   let (k_full, _v_full) = cache.append(&k_prefill, &v_prefill)?;
    //   let seq1 = k_full.dim(2)?;  // Should be 5
    //   ... continue for decode steps ...
    //   cache.reset();
    //   // After reset, current_seq_len returns 0
    todo!("Simulate generation with KV-cache")
}

/// Q12c: INTERVIEW QUESTION — Calculate the memory usage of KV-cache.
/// Given: num_layers, num_kv_heads, head_dim, dtype_bytes, max_seq_len
/// Return the total bytes for KV-cache across all layers.
///
/// Formula: 2 * num_layers * batch * num_kv_heads * max_seq * head_dim * dtype_bytes
/// (2 for K and V)
pub fn kv_cache_memory_bytes(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    dtype_bytes: usize, // 2 for BF16, 4 for F32
) -> usize {
    // TODO: Calculate total KV-cache memory (batch=1)
    // HINT: 2 * num_layers * 1 * num_kv_heads * max_seq_len * head_dim * dtype_bytes
    // INTERVIEW Q: For Qwen3-0.6B (28 layers, 8 kv_heads, 128 head_dim, BF16):
    //   What's the KV-cache for 4096 tokens?
    //   = 2 * 28 * 8 * 4096 * 128 * 2 = ~469 MB
    //   This is why KV-cache is the main memory bottleneck for long contexts!
    todo!("Calculate KV-cache memory")
}

/// Q12d: Explain why contiguous() is needed before cache.append().
pub fn explain_contiguous_for_cache() -> &'static str {
    // TODO: Return your explanation
    // HINT: After transpose(1,2), K and V are non-contiguous views.
    // The KV-cache concatenates along dim 2. If the tensor isn't contiguous,
    // the concatenation would produce incorrect results because the memory
    // layout doesn't match the logical layout.
    todo!("Explain contiguous for cache")
}

/// Q12e: BONUS — Explain the offset parameter in model.forward().
/// During prefill: offset = 0
/// During decode step i: offset = prompt_len + i
///
/// What does offset affect?
pub fn explain_offset() -> &'static str {
    // TODO: Return your explanation
    // HINT: The offset is used for:
    // 1. RoPE: slice cos/sin table at the correct position
    // 2. The model needs to know absolute position for position embeddings
    // 3. It does NOT affect KV-cache (cache handles its own length)
    todo!("Explain offset parameter")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cache() {
        let cache = create_kv_cache(2, 100);
        assert_eq!(cache.current_seq_len(), 0);
    }

    #[test]
    fn test_simulate_generation() -> Result<()> {
        let seq_lens = simulate_generation()?;
        assert_eq!(seq_lens, vec![5, 6, 7, 0]);
        Ok(())
    }

    #[test]
    fn test_kv_cache_memory() {
        // Qwen3-0.6B: 28 layers, 8 kv_heads, 128 head_dim, BF16 (2 bytes)
        let bytes = kv_cache_memory_bytes(28, 8, 128, 4096, 2);
        // 2 * 28 * 8 * 4096 * 128 * 2 = 469,762,048
        assert_eq!(bytes, 2 * 28 * 8 * 4096 * 128 * 2);
    }
}

fn main() -> Result<()> {
    println!("=== Q12: KV-Cache Implementation ===\n");
    println!("Run `cargo test --example q12_kv_cache` to verify.");
    Ok(())
}
