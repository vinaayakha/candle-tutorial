/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 10: Rotary Position Embeddings (RoPE)
/// ============================================================================
/// Difficulty: ADVANCED
///
/// RoPE encodes position info by rotating Q/K vectors in pairs.
/// It's used in LLaMA, Qwen, Mistral, and most modern LLMs.
///
/// KEY CONCEPTS:
///   - inv_freq[i] = 1 / (theta ^ (2i / head_dim))  for i in 0..head_dim/2
///   - freqs = positions outer_product inv_freq  -> [seq_len, head_dim/2]
///   - cos = freqs.cos(), sin = freqs.sin()
///   - Applied to Q and K (NOT V!)
///   - candle_nn::rotary_emb::rope() applies the rotation
///
/// INTERVIEW NUANCE:
///   - theta is typically 10000 for original, 1_000_000 for extended context
///   - RoPE is applied AFTER QK-norm (if used) in Qwen3
///   - The cos/sin table is precomputed for ALL positions up to max_seq_len
///   - During decode, we slice the table at the current offset
///   - RoPE works on pairs of dimensions, so head_dim must be even
///   - The cos/sin are shape [seq_len, head_dim/2], NOT [seq_len, head_dim]
///     The rope() function internally handles the pairing
///   - Must convert cos/sin to the same dtype as q/k before applying
///
/// HINTS:
///   - Precompute: inv_freq -> positions -> outer product -> cos/sin
///   - Apply: slice cos/sin for current offset, cast dtype, rope()
///   - candle_nn::rotary_emb::rope(q, &cos, &sin) does the actual rotation
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q10a: Compute the inverse frequency vector for RoPE.
/// inv_freq[i] = 1.0 / (theta ^ (2*i / head_dim)) for i in 0..head_dim/2
/// Return as a 1D tensor [head_dim/2].
pub fn compute_inv_freq(head_dim: usize, theta: f64, device: &Device) -> Result<Tensor> {
    // TODO: Compute inverse frequencies
    // HINT:
    //   let half = head_dim / 2;
    //   let inv_freq: Vec<f32> = (0..half)
    //       .map(|i| 1.0 / (theta as f32).powf(i as f32 * 2.0 / head_dim as f32))
    //       .collect();
    //   Tensor::new(inv_freq.as_slice(), device)
    todo!("Compute inverse frequencies")
}

/// Q10b: Compute the frequency table (outer product of positions and inv_freq).
/// positions: [0, 1, 2, ..., max_seq_len-1]
/// inv_freq: [head_dim/2]
/// Result: freqs [max_seq_len, head_dim/2]
pub fn compute_freq_table(
    max_seq_len: usize,
    inv_freq: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    // TODO: Outer product: positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    // HINT:
    //   let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    //   let positions = Tensor::new(positions.as_slice(), device)?;
    //   positions.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)
    todo!("Compute frequency table")
}

/// Q10c: Compute cos and sin tables from frequencies.
/// Return (cos_table, sin_table), each [max_seq_len, head_dim/2].
pub fn compute_cos_sin(freqs: &Tensor) -> Result<(Tensor, Tensor)> {
    // TODO: Apply cos and sin
    // HINT: (freqs.cos()?, freqs.sin()?)
    todo!("Compute cos/sin tables")
}

/// Q10d: Slice the cos/sin tables for the current generation offset.
/// Given tables of shape [max_seq_len, half_dim], extract rows
/// from offset to offset+seq_len and cast to target dtype.
pub fn slice_and_cast(
    cos: &Tensor,
    sin: &Tensor,
    offset: usize,
    seq_len: usize,
    target_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // TODO: Slice and cast to target dtype
    // HINT:
    //   use candle_core::IndexOp;
    //   let cos = cos.i(offset..offset + seq_len)?.to_dtype(target_dtype)?;
    //   let sin = sin.i(offset..offset + seq_len)?.to_dtype(target_dtype)?;
    //   Ok((cos, sin))
    //
    // INTERVIEW Q: Why is dtype casting needed here?
    //   Answer: cos/sin are precomputed in F32 for precision,
    //   but Q/K may be BF16 on GPU. The rope function requires matching dtypes.
    todo!("Slice and cast RoPE tables")
}

/// Q10e: FULL ROPE APPLICATION
/// Given Q and K tensors, cos and sin slices, apply rotary embeddings.
/// Q shape: [batch, num_heads, seq_len, head_dim]
/// K shape: [batch, num_kv_heads, seq_len, head_dim]
/// cos/sin: [seq_len, head_dim/2]
pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // TODO: Apply RoPE to Q and K
    // HINT: candle_nn::rotary_emb::rope(q, cos, sin)?
    //       candle_nn::rotary_emb::rope(k, cos, sin)?
    // INTERVIEW Q: Why is RoPE applied to Q and K but NOT V?
    //   Answer: RoPE encodes relative position via the angle between
    //   Q and K dot products. V carries the content, not position.
    //   Q_i . K_j encodes the relative distance (i-j) via rotation.
    todo!("Apply RoPE to Q and K")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_freq() -> Result<()> {
        let inv = compute_inv_freq(8, 10000.0, &Device::Cpu)?;
        assert_eq!(inv.dims(), &[4]); // head_dim/2 = 4
        let vals = inv.to_vec1::<f32>()?;
        // First element: 1.0 / 10000^(0/8) = 1.0
        assert!((vals[0] - 1.0).abs() < 1e-4);
        // Second: 1.0 / 10000^(2/8) = 1/10 = 0.1
        assert!((vals[1] - 0.1).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_freq_table() -> Result<()> {
        let inv = compute_inv_freq(8, 10000.0, &Device::Cpu)?;
        let freqs = compute_freq_table(16, &inv, &Device::Cpu)?;
        assert_eq!(freqs.dims(), &[16, 4]);
        // Position 0 should be all zeros
        let row0 = freqs.i(0)?.to_vec1::<f32>()?;
        assert!(row0.iter().all(|v| v.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn test_cos_sin() -> Result<()> {
        let freqs = Tensor::zeros((10, 4), DType::F32, &Device::Cpu)?;
        let (cos, sin) = compute_cos_sin(&freqs)?;
        assert_eq!(cos.dims(), &[10, 4]);
        assert_eq!(sin.dims(), &[10, 4]);
        // cos(0) = 1, sin(0) = 0
        let cos_vals = cos.i(0)?.to_vec1::<f32>()?;
        assert!(cos_vals.iter().all(|v| (v - 1.0).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn test_slice_and_cast() -> Result<()> {
        let cos = Tensor::ones((100, 4), DType::F32, &Device::Cpu)?;
        let sin = Tensor::zeros((100, 4), DType::F32, &Device::Cpu)?;
        let (c, s) = slice_and_cast(&cos, &sin, 10, 5, DType::F32)?;
        assert_eq!(c.dims(), &[5, 4]);
        assert_eq!(s.dims(), &[5, 4]);
        Ok(())
    }

    #[test]
    fn test_apply_rope() -> Result<()> {
        // Simple test: RoPE at position 0 with cos=1, sin=0 should be identity
        let q = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu)?;
        let cos = Tensor::ones((1, 4), DType::F32, &Device::Cpu)?;
        let sin = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        let (q_rot, k_rot) = apply_rope(&q, &k, &cos, &sin)?;
        assert_eq!(q_rot.dims(), &[1, 2, 1, 8]);
        assert_eq!(k_rot.dims(), &[1, 2, 1, 8]);
        // With sin=0, cos=1, rope should be identity
        let q_vals = q_rot.flatten_all()?.to_vec1::<f32>()?;
        assert!(q_vals.iter().all(|v| (v - 1.0).abs() < 1e-4));
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q10: Rotary Position Embeddings (RoPE) ===\n");
    println!("Run `cargo test --example q10_rotary_embeddings` to verify.");
    Ok(())
}
