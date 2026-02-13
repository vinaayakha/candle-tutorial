/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 7: Implementing a Linear Layer
/// ============================================================================
/// Difficulty: INTERMEDIATE
///
/// The Linear layer is y = xW^T + b. It's the most fundamental NN building block.
/// In Candle, there are multiple ways to create one.
///
/// KEY CONCEPTS:
///   - candle_nn::Linear::new(weight, bias) — from raw tensors
///   - candle_nn::linear(in, out, vb) — with bias, from VarBuilder
///   - candle_nn::linear_no_bias(in, out, vb) — without bias
///   - Manual: x.matmul(&weight.t()?)? + broadcast bias
///   - Weight shape convention: [out_features, in_features] (TRANSPOSED!)
///
/// INTERVIEW NUANCE:
///   - Weight is [out, in] not [in, out]! This catches people from PyTorch.
///   - candle_nn::Linear stores weight as [out, in] and transposes during forward
///   - For batched inputs [b, seq, in], the forward does:
///     weight.broadcast_left(b)?.t()? to get [b, in, out], then matmul
///   - The custom Linear in model_utils.rs handles batched 3D input specially
///   - linear_no_bias is used in modern transformers (LLaMA, Qwen, Mistral)
///
/// HINTS:
///   - For manual linear: x.matmul(&weight.t()?)? then broadcast_add bias
///   - VarBuilder::zeros(dtype, device) creates zero-filled weights for testing
///   - candle_nn::Module trait has fn forward(&self, x: &Tensor) -> Result<Tensor>
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// Q7a: Implement a manual linear layer forward pass.
/// Compute y = x @ W^T + b
/// x: [batch, in_features], weight: [out_features, in_features], bias: [out_features]
pub fn manual_linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    // TODO: Implement y = x @ W^T + b
    // Step 1: Transpose weight from [out, in] to [in, out]
    // Step 2: Matrix multiply x @ W^T
    // Step 3: Add bias via broadcast
    // HINT: x.matmul(&weight.t()?)?.broadcast_add(bias)?
   x.matmul(&weight.t()?)?.broadcast_add(bias)
}

/// Q7b: Implement manual linear for 3D input (batched).
/// x: [batch, seq_len, in_features]
/// weight: [out_features, in_features]
/// bias: Option<Tensor> (may be None for no-bias variant)
pub fn manual_linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    // TODO: Handle 3D input
    // HINT: For 3D, you need weight.broadcast_left(batch)?.t()? then matmul
    // OR: candle_nn::Linear handles this automatically
    // INTERVIEW Q: Why does the custom Linear check x.dims() for 3D?
    //   Answer: matmul with [b, s, in] @ [in, out] works in Candle for 3D
    //   but weight needs to be [b, in, out] or use broadcast_left.
    //   Actually, matmul in Candle handles batched matmul if the last 2 dims match.
    let (bsize, _seq, _) = x.dims3()?;
    let w = weight.broadcast_left(bsize)?.t()?;
    let out = x.matmul(&w)?;
    match bias {
        Some(b) => out.broadcast_add(b),
        None => Ok(out),
    }
}

/// Q7c: Create a candle_nn::Linear layer using VarBuilder with zeros.
/// Then run a forward pass.
/// Return the output tensor.
pub fn linear_with_varbuilder() -> Result<Tensor> {
    // TODO:
    // Step 1: Create a VarBuilder with zeros
    // Step 2: Create a candle_nn::linear(in=3, out=5, vb)
    // Step 3: Forward pass with a random input [2, 3]
    // HINT:
    //   let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    //   let layer = candle_nn::linear(3, 5, vb)?;
    //   let input = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    //   layer.forward(&input)?
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer = candle_nn::linear(3,5, vb)?;
    let input = Tensor::zeros((2,3), DType::F32, &Device::Cpu)?;
    layer.forward(&input)
}

/// Q7d: Create a no-bias linear layer (used in LLaMA/Qwen attention projections).
/// Return the output tensor.
pub fn linear_no_bias_example() -> Result<Tensor> {
    // TODO: Use candle_nn::linear_no_bias
    // HINT:
    //   let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    //   let layer = candle_nn::linear_no_bias(4, 8, vb)?;
    //   let input = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu)?;
    //   layer.forward(&input)?
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let layer = candle_nn::linear_no_bias(4,8, vb)?;
    let input = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu)?;
    layer.forward(&input)
}

/// Q7e: INTERVIEW QUESTION
/// Explain weight tying and implement it.
/// In weight tying, the lm_head shares weights with the embedding layer.
/// Given an embedding weight [vocab_size, hidden_size], create a Linear
/// layer that uses it (transposed) as the lm_head.
pub fn create_tied_lm_head(embed_weight: &Tensor) -> Result<candle_nn::Linear> {
    // TODO: Create a Linear with the embedding weight and no bias
    // The embedding weight is [vocab_size, hidden_size]
    // Linear expects [out_features, in_features] = [vocab_size, hidden_size]
    // So we can use it directly!
    // HINT: candle_nn::Linear::new(embed_weight.clone(), None)
    // INTERVIEW Q: Why does weight tying work?
    //   Answer: The embedding maps token_id -> vector. The lm_head maps
    //   vector -> logits over vocab. They are conceptual inverses, so
    //   sharing weights acts as a regularizer and saves parameters.
    Ok(candle_nn::Linear::new(embed_weight.clone(),None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_linear() -> Result<()> {
        // Identity-like test: weight = I, bias = 0
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu)?;
        let weight = Tensor::new(
            &[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &Device::Cpu,
        )?;
        let bias = Tensor::zeros((3,), DType::F32, &Device::Cpu)?;
        let y = manual_linear(&x, &weight, &bias)?;
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![1.0, 2.0, 3.0]]);
        Ok(())
    }

    #[test]
    fn test_manual_linear_3d() -> Result<()> {
        let x = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
        let weight = Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?;
        let result = manual_linear_3d(&x, &weight, None)?;
        assert_eq!(result.dims(), &[2, 3, 8]);
        Ok(())
    }

    #[test]
    fn test_linear_varbuilder() -> Result<()> {
        let output = linear_with_varbuilder()?;
        assert_eq!(output.dims(), &[2, 5]);
        // With zero weights, output should be all zeros (0*x + 0 = 0)
        let sum: f32 = output.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);
        Ok(())
    }

    #[test]
    fn test_linear_no_bias() -> Result<()> {
        let output = linear_no_bias_example()?;
        assert_eq!(output.dims(), &[1, 3, 8]);
        Ok(())
    }

    #[test]
    fn test_tied_lm_head() -> Result<()> {
        let embed_weight = Tensor::zeros((1000, 64), DType::F32, &Device::Cpu)?;
        let lm_head = create_tied_lm_head(&embed_weight)?;
        let hidden = Tensor::zeros((1, 5, 64), DType::F32, &Device::Cpu)?;
        let logits = lm_head.forward(&hidden)?;
        assert_eq!(logits.dims(), &[1, 5, 1000]);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q7: Implementing a Linear Layer ===\n");
    println!("Run `cargo test --example q07_linear_layer` to verify.");
    Ok(())
}
