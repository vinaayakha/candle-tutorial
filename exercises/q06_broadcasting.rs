/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 6: Broadcasting Rules
/// ============================================================================
/// Difficulty: INTERMEDIATE
///
/// Broadcasting in Candle follows NumPy rules. It allows operations between
/// tensors of different shapes by virtually expanding dimensions.
///
/// KEY CONCEPTS:
///   - .broadcast_add(), .broadcast_sub(), .broadcast_mul(), .broadcast_div()
///   - Explicit broadcast_* methods (unlike PyTorch's implicit broadcasting)
///   - .broadcast_left(n) — prepend a dimension of size n
///   - Rules: dimensions are compared right-to-left; each must be equal or 1
///
/// INTERVIEW NUANCE:
///   - Candle does NOT auto-broadcast with +, -, *, /. You MUST use broadcast_*
///     methods when shapes differ. This is a common source of bugs when porting
///     from PyTorch!
///   - broadcast_add of [2,1] + [1,3] gives [2,3] (both dims expand)
///   - LayerNorm normalizes then does: x * weight + bias where weight is [hidden]
///     but x is [batch, seq, hidden]. Need broadcast_mul and broadcast_add.
///   - Residual connections use regular + because shapes already match
///
/// HINTS:
///   - x.broadcast_add(&bias)? — bias [hidden] broadcasts to [batch, seq, hidden]
///   - .broadcast_left(batch_size)? adds a leading dim
///   - Think of broadcasting as "stretching" size-1 dims to match
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q6a: Add a bias vector [hidden] to a 2D tensor [seq, hidden].
/// This simulates adding bias in a Linear layer.
pub fn add_bias_2d(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    // TODO: Broadcast-add bias to x
    // HINT: x.broadcast_add(bias)?
    x.broadcast_add(bias)
}

/// Q6b: Add a bias vector [hidden] to a 3D tensor [batch, seq, hidden].
/// Same as above but with batch dimension.
pub fn add_bias_3d(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    // TODO: Broadcast-add bias to x
    // HINT: Same method — broadcasting auto-handles extra dims
    x.broadcast_add(bias)
}

/// Q6c: Subtract the mean from a tensor along the last dimension.
/// Input: [batch, seq, hidden]. Mean: [batch, seq, 1].
/// This is the first step of LayerNorm!
pub fn subtract_mean(x: &Tensor) -> Result<Tensor> {
    // TODO: Compute mean along last dim with keepdim, then broadcast-subtract
    // HINT: let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    //       x.broadcast_sub(&mean)?
    // INTERVIEW Q: Why use D::Minus1 instead of 2?
    //   Answer: D::Minus1 works regardless of tensor rank, making code generic.
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    x.broadcast_sub(&mean)
}

/// Q6d: Multiply weight [hidden] with normalized tensor [batch, seq, hidden].
/// Then add bias [hidden]. This is the final step of LayerNorm.
pub fn apply_weight_and_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    // TODO: broadcast_mul with weight, then broadcast_add with bias
    // HINT: x.broadcast_mul(weight)?.broadcast_add(bias)?
    x.broadcast_mul(weight)?.broadcast_add(bias)
}

/// Q6e: Scale attention scores by dividing by sqrt(head_dim).
/// Scores shape: [batch, heads, seq_q, seq_k]. Scalar: f64.
/// INTERVIEW Q: Why do we scale by 1/sqrt(d_k)?
///   Answer: Dot products grow with dimension, pushing softmax into
///   saturation. Scaling keeps gradients healthy.
pub fn scale_attention(scores: &Tensor, head_dim: usize) -> Result<Tensor> {
    // TODO: Divide scores by sqrt(head_dim)
    // HINT: You can use (scores / (head_dim as f64).sqrt())?
    // OR: scores.affine(1.0 / (head_dim as f64).sqrt(), 0.0)?
    scores.affine(1.0 / (head_dim as f64).sqrt(), 0.0)
}

/// Q6f: Broadcast a linear weight [out, in] to [batch, out, in].
/// This is needed in the custom Linear implementation for batched matmul.
pub fn broadcast_weight(weight: &Tensor, batch_size: usize) -> Result<Tensor> {
    // TODO: Add a batch dimension at the front
    // HINT: weight.broadcast_left(batch_size)?
    // INTERVIEW Q: Why does the custom Linear use broadcast_left + transpose?
    //   Answer: For batched matmul, we need weight to be [b, in, out]
    //   so: broadcast_left(b) gives [b, out, in], then .t()? gives [b, in, out]
    weight.broadcast_left(batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn test_add_bias_2d() -> Result<()> {
        let x = Tensor::ones((3, 4), DType::F32, &Device::Cpu)?;
        let bias = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let result = add_bias_2d(&x, &bias)?;
        assert_eq!(result.dims(), &[3, 4]);
        // First row should be [2, 3, 4, 5]
        let row0: Vec<f32> = result.i(0)?.to_vec1()?;
        assert_eq!(row0, vec![2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_add_bias_3d() -> Result<()> {
        let x = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
        let bias = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &Device::Cpu)?;
        let result = add_bias_3d(&x, &bias)?;
        assert_eq!(result.dims(), &[2, 3, 4]);
        let val: f32 = result.i((0, 0, 0))?.to_scalar()?;
        assert_eq!(val, 11.0);
        Ok(())
    }

    #[test]
    fn test_subtract_mean() -> Result<()> {
        // [1, 1, 4] tensor with values [1, 2, 3, 4], mean=2.5
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?;
        let result = subtract_mean(&x)?;
        let vals = result.i((0, 0))?.to_vec1::<f32>()?;
        assert!((vals[0] - (-1.5)).abs() < 1e-6);
        assert!((vals[3] - 1.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_apply_weight_and_bias() -> Result<()> {
        let x = Tensor::ones((1, 1, 3), DType::F32, &Device::Cpu)?;
        let weight = Tensor::new(&[2.0f32, 3.0, 4.0], &Device::Cpu)?;
        let bias = Tensor::new(&[0.1f32, 0.2, 0.3], &Device::Cpu)?;
        let result = apply_weight_and_bias(&x, &weight, &bias)?;
        let vals = result.i((0, 0))?.to_vec1::<f32>()?;
        assert!((vals[0] - 2.1).abs() < 1e-6);
        assert!((vals[1] - 3.2).abs() < 1e-6);
        assert!((vals[2] - 4.3).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_scale_attention() -> Result<()> {
        let scores = Tensor::new(&[[[[64.0f32]]]], &Device::Cpu)?;
        let result = scale_attention(&scores, 64)?;
        let val: f32 = result.i((0, 0, 0, 0))?.to_scalar()?;
        assert!((val - 8.0).abs() < 1e-6); // 64 / sqrt(64) = 64/8 = 8
        Ok(())
    }

    #[test]
    fn test_broadcast_weight() -> Result<()> {
        let weight = Tensor::zeros((4, 3), DType::F32, &Device::Cpu)?;
        let result = broadcast_weight(&weight, 2)?;
        assert_eq!(result.dims(), &[2, 4, 3]);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q6: Broadcasting Rules ===\n");
    println!("Run `cargo test --example q06_broadcasting` to verify.");
    Ok(())
}
