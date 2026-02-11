/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 8: Implementing LayerNorm from Scratch
/// ============================================================================
/// Difficulty: INTERMEDIATE-ADVANCED
///
/// LayerNorm normalizes across the feature dimension (last dim), NOT batch.
/// Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
///
/// KEY CONCEPTS:
///   - Normalize across last dimension (hidden_size)
///   - mean_keepdim / sum_keepdim for reduction with dim preservation
///   - eps (e.g., 1e-5 or 1e-12) prevents division by zero
///   - Affine parameters: weight (gamma) and bias (beta)
///   - MUST upcast to F32 for BF16/F16 inputs for numerical stability
///
/// INTERVIEW NUANCE:
///   - Post-norm (BERT/RoBERTa): LayerNorm AFTER attention/FFN + residual
///   - Pre-norm (GPT/LLaMA/Qwen): LayerNorm BEFORE attention/FFN
///   - Candle's built-in candle_nn::LayerNorm exists, but many projects
///     implement their own for exact weight path matching
///   - The weight path in HuggingFace models can be "LayerNorm" (capital L!)
///     or "layer_norm" or "norm" — this matters for VarBuilder::pp()
///   - Some models use "gamma"/"beta" instead of "weight"/"bias"
///     The roberta.rs code handles both!
///
/// HINTS:
///   - Variance = mean((x - mean)^2) = mean(x^2) - mean(x)^2
///   - Use .broadcast_sub(), .broadcast_div(), .broadcast_mul(), .broadcast_add()
///   - Cast to F32 first, normalize, cast back, THEN apply weight/bias
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q8a: Compute the mean of tensor along the last dimension, keeping the dim.
/// Input: [batch, seq, hidden]. Output: [batch, seq, 1].
pub fn compute_mean(x: &Tensor) -> Result<Tensor> {
    // TODO: Sum along last dim then divide by dim size
    // HINT: let hidden = x.dim(candle_core::D::Minus1)?;
    //       (x.sum_keepdim(candle_core::D::Minus1)? / hidden as f64)?
    // OR: x.mean_keepdim(candle_core::D::Minus1)?
    todo!("Compute mean along last dim")
}

/// Q8b: Compute variance of tensor along last dimension, keeping dim.
/// variance = mean((x - mean)^2)
pub fn compute_variance(x: &Tensor) -> Result<Tensor> {
    // TODO: Compute variance
    // Method 1: centered = x - mean; variance = mean(centered^2)
    // Method 2: variance = mean(x^2) - mean(x)^2
    // HINT (method 1):
    //   let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    //   let centered = x.broadcast_sub(&mean)?;
    //   centered.sqr()?.mean_keepdim(candle_core::D::Minus1)
    todo!("Compute variance along last dim")
}

/// Q8c: FULL LAYERNORM IMPLEMENTATION
/// Implement the complete LayerNorm forward pass.
/// x: [batch, seq, hidden]
/// weight: [hidden]
/// bias: [hidden]
/// eps: f64
///
/// Steps:
/// 1. Upcast x to F32 if BF16/F16
/// 2. Compute mean along last dim (keepdim)
/// 3. Center: x - mean
/// 4. Compute variance of centered values (keepdim)
/// 5. Normalize: centered / sqrt(variance + eps)
/// 6. Downcast back to original dtype
/// 7. Apply affine: normalized * weight + bias
pub fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    // TODO: Implement the full LayerNorm
    // HINT: Follow the steps above. Key operations:
    //   let x_dtype = x.dtype();
    //   let internal_dtype = match x_dtype {
    //       DType::F16 | DType::BF16 => DType::F32,
    //       d => d,
    //   };
    //   let x = x.to_dtype(internal_dtype)?;
    //   ... normalize ...
    //   let x = x_normed.to_dtype(x_dtype)?;
    //   x.broadcast_mul(weight)?.broadcast_add(bias)
    //
    // INTERVIEW Q: Why upcast BEFORE normalizing but apply weight/bias
    //   AFTER downcasting?
    //   Answer: The division in normalization is numerically unstable in
    //   low precision. But weight/bias multiplication is fine in BF16/F16.
    //   Applying them after downcast saves memory bandwidth.
    todo!("Full LayerNorm implementation")
}

/// Q8d: BONUS — What's the difference between LayerNorm and RmsNorm?
/// Implement a simplified comparison.
/// LayerNorm: (x - mean) / sqrt(var + eps) * w + b
/// RmsNorm:   x / sqrt(mean(x^2) + eps) * w      (no bias, no mean subtraction!)
///
/// Return: (layernorm_output, rmsnorm_output)
pub fn compare_norms(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    // TODO: Implement both and return them
    // For RmsNorm:
    //   let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    //   let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    //   let rms_out = x_normed.broadcast_mul(weight)?;
    //
    // INTERVIEW Q: Why do modern LLMs (LLaMA, Qwen) prefer RmsNorm?
    //   Answer: 1) Simpler (no mean subtraction, no bias)
    //           2) Slightly faster
    //           3) Empirically works just as well for large models
    //           4) LLaMA paper showed no degradation vs LayerNorm
    todo!("Compare LayerNorm vs RmsNorm")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean() -> Result<()> {
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?;
        let mean = compute_mean(&x)?;
        assert_eq!(mean.dims(), &[1, 1, 1]);
        let val: f32 = mean.i((0, 0, 0))?.to_scalar()?;
        assert!((val - 2.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_compute_variance() -> Result<()> {
        let x = Tensor::new(&[[[1.0f32, 3.0]]], &Device::Cpu)?;
        let var = compute_variance(&x)?;
        assert_eq!(var.dims(), &[1, 1, 1]);
        let val: f32 = var.i((0, 0, 0))?.to_scalar()?;
        // mean=2, centered=[-1, 1], var=mean([1,1])=1
        assert!((val - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?;
        let weight = Tensor::ones((4,), DType::F32, &Device::Cpu)?;
        let bias = Tensor::zeros((4,), DType::F32, &Device::Cpu)?;
        let result = layer_norm(&x, &weight, &bias, 1e-5)?;
        assert_eq!(result.dims(), &[1, 1, 4]);

        // With unit weight and zero bias, output should have mean~0, var~1
        let vals = result.i((0, 0))?.to_vec1::<f32>()?;
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
        Ok(())
    }

    #[test]
    fn test_compare_norms() -> Result<()> {
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?;
        let weight = Tensor::ones((4,), DType::F32, &Device::Cpu)?;
        let bias = Tensor::zeros((4,), DType::F32, &Device::Cpu)?;
        let (ln, rms) = compare_norms(&x, &weight, &bias, 1e-5)?;
        assert_eq!(ln.dims(), rms.dims());
        // They should produce different values since RmsNorm doesn't subtract mean
        let ln_vals = ln.i((0, 0))?.to_vec1::<f32>()?;
        let rms_vals = rms.i((0, 0))?.to_vec1::<f32>()?;
        assert!(
            (ln_vals[0] - rms_vals[0]).abs() > 1e-4,
            "LayerNorm and RmsNorm should differ"
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q8: Implementing LayerNorm ===\n");
    println!("Run `cargo test --example q08_layer_norm` to verify.");
    Ok(())
}
