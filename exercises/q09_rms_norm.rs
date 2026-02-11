/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 9: RmsNorm Implementation
/// ============================================================================
/// Difficulty: INTERMEDIATE-ADVANCED
///
/// RmsNorm (Root Mean Square Normalization) is used in LLaMA, Qwen, Mistral.
/// It's simpler than LayerNorm: no mean subtraction, no bias.
/// Formula: y = x / sqrt(mean(x^2) + eps) * weight
///
/// KEY CONCEPTS:
///   - RmsNorm is the de facto norm for modern LLMs
///   - It normalizes by the root-mean-square, not mean and variance
///   - Must upcast to F32 for BF16/F16 (critical for numerical stability!)
///   - Weight path in Qwen3: "model.layers.{i}.input_layernorm.weight"
///   - VarBuilder::pp() must match the exact safetensors key path
///
/// INTERVIEW NUANCE:
///   - The dtype handling is the #1 source of bugs when porting models:
///     1. Save original dtype
///     2. Upcast to F32
///     3. Normalize in F32
///     4. Downcast back
///     5. Multiply by weight (in original dtype)
///   - The Qwen3 code applies RmsNorm per-HEAD for QK-norm, not per-token!
///     This requires reshaping: [b, heads, seq, hd] -> [b*heads*seq, hd]
///     Apply RmsNorm, then reshape back.
///   - .sqr()?.mean_keepdim(D::Minus1)? computes RMS denominator
///
/// HINTS:
///   - Key formula: x / sqrt(mean(x^2) + eps) * weight
///   - .broadcast_div() for dividing by [b, s, 1] tensor
///   - .broadcast_mul() for applying [hidden] weight
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q9a: Implement RmsNorm forward pass.
/// x: [batch, seq, hidden] or [batch_combined, hidden]
/// weight: [hidden]
/// eps: f64
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // TODO: Implement RmsNorm
    // Step 1: Save original dtype
    // Step 2: Upcast to F32 if needed
    // Step 3: Compute variance = mean(x^2) along last dim with keepdim
    // Step 4: Normalize: x / sqrt(variance + eps)
    // Step 5: Downcast back to original dtype
    // Step 6: broadcast_mul with weight
    //
    // HINT:
    //   let x_dtype = x.dtype();
    //   let internal = match x_dtype { DType::F16 | DType::BF16 => DType::F32, d => d };
    //   let x = x.to_dtype(internal)?;
    //   let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    //   let x = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    //   x.to_dtype(x_dtype)?.broadcast_mul(weight)
    todo!("Implement RmsNorm")
}

/// Q9b: Apply RmsNorm per-head (QK-norm pattern from Qwen3).
/// Input: [batch, heads, seq, head_dim]
/// Weight: [head_dim]
/// This requires flattening b*heads*seq into one dim, normalizing, reshaping back.
pub fn per_head_rms_norm(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    // TODO: Apply RmsNorm per-head
    // Step 1: Get dims: let (b, heads, seq, hd) = x.dims4()?;
    // Step 2: Reshape to [b*heads*seq, hd]
    // Step 3: Unsqueeze to [1, b*heads*seq, hd] (RmsNorm expects this)
    //         OR just apply directly on 2D
    // Step 4: Apply rms_norm
    // Step 5: Reshape back to [b, heads, seq, hd]
    //
    // HINT: This is exactly what Qwen3Attention::apply_head_norm does:
    //   let (b, heads, seq, hd) = x.dims4()?;
    //   let xs = x.reshape((b * heads * seq, hd))?;
    //   let xs = rms_norm(&xs.unsqueeze(0)?, weight, eps)?;
    //   xs.squeeze(0)?.reshape((b, heads, seq, hd))
    todo!("Per-head RmsNorm")
}

/// Q9c: INTERVIEW QUESTION — Why is the unsqueeze(0) needed in per-head norm?
/// Fill in the answer as a string.
pub fn explain_unsqueeze_in_per_head_norm() -> &'static str {
    // TODO: Replace with your answer
    // HINT: Think about what rms_norm expects for its input dimensions
    // and how mean_keepdim(D::Minus1) behaves on 2D vs 3D tensors.
    todo!("Explain why unsqueeze is needed")
    // Expected answer (something like):
    // "RmsNorm uses mean_keepdim on the last dim. For a 2D [N, hd] tensor,
    //  D::Minus1 reduces along hd which is correct. The unsqueeze(0) adds a
    //  batch dim to make it [1, N, hd] so that the norm function works
    //  uniformly with its expected 3D input. Without it, if the norm function
    //  uses .dims3()?, it would fail on 2D input."
}

/// Q9d: Load RmsNorm from a VarBuilder (mimicking model loading).
/// Given a VarBuilder scoped to the right path, load weight and create
/// a struct that can do forward passes.
pub struct MyRmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl MyRmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // TODO: Load the weight tensor from VarBuilder
        // HINT: let weight = vb.get(size, "weight")?;
        //       Ok(Self { weight, eps })
        todo!("Load RmsNorm from VarBuilder")
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO: Call your rms_norm function
        rms_norm(x, &self.weight, self.eps)
    }
}

use candle_nn::VarBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        // For x = [1, 1, 1], RMS = 1, so output = x * weight / 1 = weight
        let x = Tensor::ones((1, 1, 3), DType::F32, &Device::Cpu)?;
        let weight = Tensor::new(&[2.0f32, 3.0, 4.0], &Device::Cpu)?;
        let result = rms_norm(&x, &weight, 1e-6)?;
        let vals = result.i((0, 0))?.to_vec1::<f32>()?;
        // RMS of [1,1,1] = sqrt(mean([1,1,1])) = 1
        // So output = [1,1,1] / 1 * [2,3,4] = [2,3,4]
        assert!((vals[0] - 2.0).abs() < 1e-4);
        assert!((vals[1] - 3.0).abs() < 1e-4);
        assert!((vals[2] - 4.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_rms_norm_scaling() -> Result<()> {
        // For x = [3, 4], RMS = sqrt(mean([9, 16])) = sqrt(12.5) ≈ 3.536
        let x = Tensor::new(&[[[3.0f32, 4.0]]], &Device::Cpu)?;
        let weight = Tensor::ones((2,), DType::F32, &Device::Cpu)?;
        let result = rms_norm(&x, &weight, 1e-6)?;
        let vals = result.i((0, 0))?.to_vec1::<f32>()?;
        let rms = (12.5f32).sqrt();
        assert!((vals[0] - 3.0 / rms).abs() < 1e-4);
        assert!((vals[1] - 4.0 / rms).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_per_head_rms_norm() -> Result<()> {
        let x = Tensor::ones((1, 2, 3, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::ones((4,), DType::F32, &Device::Cpu)?;
        let result = per_head_rms_norm(&x, &weight, 1e-6)?;
        assert_eq!(result.dims(), &[1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_load_rms_norm() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let norm = MyRmsNorm::load(64, 1e-6, vb)?;
        assert_eq!(norm.weight.dims(), &[64]);
        assert_eq!(norm.eps, 1e-6);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q9: RmsNorm Implementation ===\n");
    println!("Run `cargo test --example q09_rms_norm` to verify.");
    Ok(())
}
