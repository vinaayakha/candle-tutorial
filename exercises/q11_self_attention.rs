/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 11: Self-Attention from Scratch
/// ============================================================================
/// Difficulty: ADVANCED
///
/// Self-attention is the core of transformers. This exercise builds it
/// step by step, from basic scaled dot-product attention to GQA with SDPA.
///
/// KEY CONCEPTS:
///   - Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
///   - Multi-Head: split Q/K/V into heads, attend independently, concat
///   - Grouped Query Attention (GQA): fewer K/V heads than Q heads
///   - candle_nn::ops::sdpa() — fused, optimized attention (Metal/CUDA)
///   - Causal masking for autoregressive (decoder) models
///
/// INTERVIEW NUANCE:
///   - GQA ratio: num_heads / num_kv_heads (e.g., 16/8=2 in Qwen3)
///   - With sdpa(), GQA is handled NATIVELY — no need for repeat_kv!
///   - For manual attention, you'd need to repeat K/V to match Q heads
///   - do_causal=true in sdpa generates causal mask internally
///   - For decode (seq=1), no mask needed — use do_causal=false
///   - softcapping=1.0 means disabled in sdpa
///   - The attention scale is 1/sqrt(head_dim), NOT 1/sqrt(hidden_size)
///   - After attention: transpose(1,2) -> contiguous() -> reshape is MANDATORY
///
/// HINTS:
///   - Q/K/V projections: linear_no_bias(hidden, heads*head_dim, vb)
///   - Reshape: [b, s, nh*hd] -> [b, s, nh, hd] -> transpose(1,2) -> [b, nh, s, hd]
///   - Scale: 1.0 / (head_dim as f32).sqrt()
///   - sdpa(q, k, v, mask, do_causal, scale, softcapping)
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q11a: Compute scaled dot-product attention MANUALLY (no sdpa).
/// Q: [batch, heads, seq_q, head_dim]
/// K: [batch, heads, seq_k, head_dim]
/// V: [batch, heads, seq_k, head_dim]
/// scale: f32
/// Return: [batch, heads, seq_q, head_dim]
///
/// Formula: softmax(Q @ K^T * scale) @ V
pub fn manual_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    // TODO: Implement scaled dot-product attention
    // Step 1: scores = Q @ K^T  (K^T means transpose last two dims)
    // Step 2: scores = scores * scale
    // Step 3: attn_weights = softmax(scores, dim=-1)
    // Step 4: output = attn_weights @ V
    //
    // HINT:
    //   let scores = q.matmul(&k.transpose(2, 3)?)?;  // K^T over last 2 dims
    //   let scores = (scores * scale as f64)?;
    //   let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    //   attn_weights.matmul(v)
    todo!("Manual scaled dot-product attention")
}

/// Q11b: Implement the full attention reshape pattern.
/// Input hidden: [batch, seq, hidden_size]
/// Project Q/K/V, reshape to multi-head format, compute attention, reshape back.
///
/// This is the COMPLETE attention forward pass (minus RoPE and KV-cache).
pub fn multi_head_attention(
    hidden: &Tensor,
    wq: &Tensor,  // [num_heads * head_dim, hidden_size]
    wk: &Tensor,  // [num_kv_heads * head_dim, hidden_size]
    wv: &Tensor,  // [num_kv_heads * head_dim, hidden_size]
    wo: &Tensor,  // [hidden_size, num_heads * head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (b, seq_len, _hidden) = hidden.dims3()?;

    // TODO: Implement multi-head attention
    // Step 1: Project Q, K, V using matmul with transposed weights
    //   let q = hidden.matmul(&wq.t()?)?;
    //   let k = hidden.matmul(&wk.t()?)?;
    //   let v = hidden.matmul(&wv.t()?)?;
    //
    // Step 2: Reshape to multi-head format [b, s, nh, hd] -> [b, nh, s, hd]
    //   let q = q.reshape((b, seq_len, num_heads, head_dim))?.transpose(1, 2)?;
    //   let k = k.reshape((b, seq_len, num_kv_heads, head_dim))?.transpose(1, 2)?;
    //   let v = v.reshape((b, seq_len, num_kv_heads, head_dim))?.transpose(1, 2)?;
    //
    // Step 3: Use sdpa for fused attention (handles GQA natively!)
    //   let scale = 1.0 / (head_dim as f32).sqrt();
    //   let attn = candle_nn::ops::sdpa(
    //       &q.contiguous()?, &k.contiguous()?, &v.contiguous()?,
    //       None, true, scale, 1.0,
    //   )?;
    //
    // Step 4: Reshape back: [b, nh, s, hd] -> [b, s, nh*hd]
    //   let attn = attn.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, num_heads * head_dim))?;
    //
    // Step 5: Output projection
    //   attn.matmul(&wo.t()?)
    todo!("Multi-head attention")
}

/// Q11c: INTERVIEW QUESTION — Explain why contiguous() is needed in attention.
/// Write your answer as comments and return a key insight.
pub fn explain_contiguous_in_attention() -> &'static str {
    // TODO: Fill in your understanding
    // After transpose(1,2), the tensor is a VIEW with non-contiguous memory.
    // matmul/sdpa may give wrong results or panic on non-contiguous tensors.
    // contiguous() copies data to be packed in memory order matching the shape.
    //
    // PLACES WHERE contiguous() IS CRITICAL:
    // 1. After transpose(1,2) for Q/K/V before sdpa
    // 2. After transpose(1,2) on attention output before reshape
    // 3. Before appending to KV-cache (cache expects contiguous tensors)
    todo!("Explain contiguous in attention")
}

/// Q11d: Compute attention with causal masking MANUALLY.
/// Add a lower-triangular mask of -inf to scores before softmax.
/// This prevents attending to future positions.
pub fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // TODO: Same as manual_attention but with causal mask
    // After computing scores, add causal mask:
    //   let seq_len = q.dim(2)?;
    //   Create a mask where mask[i][j] = 0 if j <= i, else -inf
    //   Add mask to scores before softmax
    //
    // HINT for creating the mask:
    //   let mask = Tensor::zeros((seq_len, seq_len), DType::F32, q.device())?;
    //   // You can use a loop or create from Vec
    //   let mut mask_data = vec![vec![0.0f32; seq_len]; seq_len];
    //   for i in 0..seq_len {
    //       for j in i+1..seq_len {
    //           mask_data[i][j] = f32::NEG_INFINITY;
    //       }
    //   }
    //   let mask = Tensor::new(mask_data, q.device())?;
    //   let scores = scores.broadcast_add(&mask)?;
    todo!("Causal attention")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_attention() -> Result<()> {
        // Simple test: Q=K=V=identity-like, should output V
        let q = Tensor::ones((1, 1, 2, 4), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 2, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::new(&[[[[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]], &Device::Cpu)?;
        let result = manual_attention(&q, &k, &v, 0.5)?;
        assert_eq!(result.dims(), &[1, 1, 2, 4]);
        Ok(())
    }

    #[test]
    fn test_multi_head_attention() -> Result<()> {
        let hidden = Tensor::zeros((1, 4, 32), DType::F32, &Device::Cpu)?;
        // 4 heads, 2 kv_heads, head_dim=8, hidden=32
        let wq = Tensor::zeros((32, 32), DType::F32, &Device::Cpu)?; // 4*8=32
        let wk = Tensor::zeros((16, 32), DType::F32, &Device::Cpu)?; // 2*8=16
        let wv = Tensor::zeros((16, 32), DType::F32, &Device::Cpu)?;
        let wo = Tensor::zeros((32, 32), DType::F32, &Device::Cpu)?;
        let result = multi_head_attention(&hidden, &wq, &wk, &wv, &wo, 4, 2, 8)?;
        assert_eq!(result.dims(), &[1, 4, 32]);
        Ok(())
    }

    #[test]
    fn test_causal_attention() -> Result<()> {
        // With causal masking, each position should only attend to itself and earlier
        let q = Tensor::ones((1, 1, 3, 4), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 3, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::new(
            &[[[[1.0f32, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]]]],
            &Device::Cpu,
        )?;
        let result = causal_attention(&q, &k, &v, 0.5)?;
        assert_eq!(result.dims(), &[1, 1, 3, 4]);
        // Position 0 should only see V[0], so output[0] = [1, 0, 0, 0]
        let pos0 = result.i((0, 0, 0))?.to_vec1::<f32>()?;
        assert!((pos0[0] - 1.0).abs() < 1e-4, "pos0 should be V[0]");
        assert!((pos0[1] - 0.0).abs() < 1e-4);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q11: Self-Attention from Scratch ===\n");
    println!("Run `cargo test --example q11_self_attention` to verify.");
    Ok(())
}
