/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 4: Reshaping, Squeeze, Unsqueeze, Transpose
/// ============================================================================
/// Difficulty: BASIC-INTERMEDIATE
///
/// Shape manipulation is the bread and butter of ML code. In Candle, getting
/// this wrong causes cryptic runtime errors, not compile-time errors.
///
/// KEY CONCEPTS:
///   - .reshape(shape) — returns view if contiguous, copies otherwise
///   - .squeeze(dim) — removes dim of size 1
///   - .unsqueeze(dim) — adds dim of size 1
///   - .transpose(d1, d2) — swaps two dimensions (returns lazy view!)
///   - .contiguous() — ensures data is packed in memory
///   - .t() — shorthand for .transpose(0, 1) on 2D tensors
///   - .flatten_all() — flatten to 1D
///   - .dims2(), .dims3(), .dims4() — unpack dimensions with compile-time arity
///
/// INTERVIEW NUANCE:
///   - transpose() creates a VIEW, not a copy! The underlying data is NOT moved.
///   - After transpose, tensor is NOT contiguous. matmul may fail silently or
///     give wrong results. Always call .contiguous()? after transpose if needed.
///   - reshape() with -1 is NOT supported in Candle (unlike PyTorch).
///     You must compute all dimensions explicitly.
///   - The attention pattern: reshape -> transpose -> contiguous is ubiquitous
///
/// HINTS:
///   - .reshape((b, seq, heads, hd))? then .transpose(1, 2)?
///   - For squeeze: .squeeze(0)? removes the batch dim if size 1
///   - For dims unpacking: let (b, s, h) = tensor.dims3()?;
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q4a: Reshape a [12] tensor into [3, 4].
pub fn reshape_1d_to_2d(t: &Tensor) -> Result<Tensor> {
    // TODO: Reshape from [12] to [3, 4]
    // HINT: t.reshape((3, 4))
    t.reshape((3, 4))
}

/// Q4b: Reshape a [2, 6] tensor into [2, 3, 2].
pub fn reshape_2d_to_3d(t: &Tensor) -> Result<Tensor> {
    // TODO: Reshape from [2, 6] to [2, 3, 2]
    t.reshape((2,3,2))
}

/// Q4c: Add a batch dimension. Input: [seq_len]. Output: [1, seq_len].
/// This is needed when feeding a single sequence to a model.
pub fn add_batch_dim(t: &Tensor) -> Result<Tensor> {
    // TODO: Add a dimension at position 0
    // HINT: t.unsqueeze(0)?
   t.unsqueeze(0)
}

/// Q4d: Remove the batch dimension. Input: [1, seq_len, hidden]. Output: [seq_len, hidden].
/// Only valid when batch=1.
pub fn remove_batch_dim(t: &Tensor) -> Result<Tensor> {
    // TODO: Remove dimension at position 0
    // HINT: t.squeeze(0)?
    t.squeeze(0)
}

/// Q4e: THIS IS THE CRITICAL ATTENTION RESHAPE PATTERN.
/// Input: [batch, seq_len, num_heads * head_dim]
/// Output: [batch, num_heads, seq_len, head_dim]
///
/// This is used in EVERY transformer attention implementation.
/// Steps: reshape to [b, s, nh, hd] -> transpose(1, 2) to get [b, nh, s, hd]
pub fn attention_reshape(
    t: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // TODO: Implement the attention reshape pattern
    // Step 1: reshape to [batch, seq_len, num_heads, head_dim]
    // Step 2: transpose dims 1 and 2
    // HINT: t.reshape((batch, seq_len, num_heads, head_dim))?.transpose(1, 2)?
    // INTERVIEW Q: Why transpose(1,2) and not transpose(2,3)?
    //   Answer: We want [b, heads, seq, dim] so heads become the "batch-like"
    //   dimension for parallel attention computation.
    t.reshape((batch, seq_len, num_heads, head_dim))?.transpose(1, 2)
}

/// Q4f: Reverse the attention reshape. Go from [b, nh, s, hd] back to [b, s, nh*hd].
/// This is the post-attention reshape before the output projection.
pub fn reverse_attention_reshape(t: &Tensor) -> Result<Tensor> {
    // TODO: Reverse: transpose(1,2) -> contiguous -> reshape
    // HINT: let (b, nh, s, hd) = t.dims4()?;
    //       t.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))
    // INTERVIEW Q: Why is .contiguous()? needed here?
    //   Answer: transpose() creates a non-contiguous view. reshape() requires
    //   contiguous data. Without it, you get a runtime error.
    t.transpose(1, 2)?.contiguous()?.reshape((t.dims()[0], t.dims()[2], t.dims()[1] * t.dims()[3]))
}

/// Q4g: Flatten a tensor completely to 1D.
pub fn flatten_all(t: &Tensor) -> Result<Tensor> {
    // TODO: Flatten to 1D
    // HINT: t.flatten_all()?
    t.flatten_all()
}

#[cfg(test)]
mod tests {
    use candle_core::IndexOp;

    use super::*;

    #[test]
    fn test_reshape_1d_to_2d() -> Result<()> {
        let t = Tensor::new(&[1.0f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], &Device::Cpu)?;
        let r = reshape_1d_to_2d(&t)?;
        assert_eq!(r.dims(), &[3, 4]);
        Ok(())
    }

    #[test]
    fn test_reshape_2d_to_3d() -> Result<()> {
        let t = Tensor::zeros((2, 6), DType::F32, &Device::Cpu)?;
        let r = reshape_2d_to_3d(&t)?;
        assert_eq!(r.dims(), &[2, 3, 2]);
        Ok(())
    }

    #[test]
    fn test_add_batch_dim() -> Result<()> {
        let t = Tensor::zeros((10,), DType::F32, &Device::Cpu)?;
        let r = add_batch_dim(&t)?;
        assert_eq!(r.dims(), &[1, 10]);
        Ok(())
    }

    #[test]
    fn test_remove_batch_dim() -> Result<()> {
        let t = Tensor::zeros((1, 10, 64), DType::F32, &Device::Cpu)?;
        let r = remove_batch_dim(&t)?;
        assert_eq!(r.dims(), &[10, 64]);
        Ok(())
    }

    #[test]
    fn test_attention_reshape() -> Result<()> {
        // batch=1, seq=4, 8 heads, head_dim=64, so hidden=512
        let t = Tensor::zeros((1, 4, 512), DType::F32, &Device::Cpu)?;
        let r = attention_reshape(&t, 1, 4, 8, 64)?;
        println!("Attention reshape output dims: {:?}", r.dims());
        assert_eq!(r.dims(), &[1, 8, 4, 64]);
        Ok(())
    }

    #[test]
    fn test_reverse_attention_reshape() -> Result<()> {
        let t = Tensor::zeros((1, 8, 4, 64), DType::F32, &Device::Cpu)?;
        let r = reverse_attention_reshape(&t)?;
        assert_eq!(r.dims(), &[1, 4, 512]);
        Ok(())
    }

    #[test]
    fn test_flatten() -> Result<()> {
        let t = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
        let r = flatten_all(&t)?;
        assert_eq!(r.dims(), &[24]);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q4: Reshaping, Squeeze, Unsqueeze, Transpose ===\n");
    println!("Run `cargo test --example q04_reshaping` to verify.");
    Ok(())
}
