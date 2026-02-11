/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 1: Tensor Creation & Inspection
/// ============================================================================
/// Difficulty: BASIC
///
/// In this exercise you will learn to create tensors from Rust data,
/// inspect their shape/dtype/device, and extract values back to Rust types.
///
/// KEY CONCEPTS:
///   - Tensor::new() creates tensors from slices, arrays, vecs
///   - Tensor::zeros() / Tensor::ones() for constant tensors
///   - Tensor::rand() for random tensors
///   - .dims(), .shape(), .dtype(), .device() for inspection
///   - .to_vec1(), .to_vec2(), .to_scalar() to extract values
///
/// HINTS:
///   - Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu) creates a 1D f32 tensor
///   - For 2D: Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)
///   - Tensor::zeros((rows, cols), DType::F32, &Device::Cpu) for zero tensor
///   - The type suffix on literals (f32, u32, etc.) matters for inference
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q1a: Create a 1D tensor containing [1.0, 2.0, 3.0, 4.0, 5.0] on CPU.
/// Return the tensor.
pub fn create_1d_tensor() -> Result<Tensor> {
    // TODO: Create a 1D f32 tensor with values [1.0, 2.0, 3.0, 4.0, 5.0]
    // HINT: Use Tensor::new() with an f32 slice and Device::Cpu
    todo!("Create a 1D tensor")
}

/// Q1b: Create a 2D tensor (matrix) with shape [2, 3] containing:
///   [[1, 2, 3],
///    [4, 5, 6]]
/// Use u32 dtype.
pub fn create_2d_tensor() -> Result<Tensor> {
    // TODO: Create a 2D u32 tensor
    // HINT: Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu)
    todo!("Create a 2D tensor")
}

/// Q1c: Create a zeros tensor with shape [3, 4] and DType::F32.
pub fn create_zeros() -> Result<Tensor> {
    // TODO: Use Tensor::zeros with explicit shape and dtype
    // HINT: Tensor::zeros((rows, cols), dtype, device)
    todo!("Create a zeros tensor")
}

/// Q1d: Given a tensor, extract ALL values into a Vec<f32>.
/// The input tensor is guaranteed to be 1D and f32.
pub fn extract_values(t: &Tensor) -> Result<Vec<f32>> {
    // TODO: Convert the tensor back to a Rust Vec<f32>
    // HINT: Use .to_vec1::<f32>()?
    todo!("Extract tensor values")
}

/// Q1e: Given a tensor, return a tuple of (num_dims, total_elements, dtype_name).
/// dtype_name should be "F32", "F64", "BF16", "U32", etc.
pub fn inspect_tensor(t: &Tensor) -> Result<(usize, usize, String)> {
    // TODO: Return (number of dimensions, total element count, dtype as string)
    // HINT: .dims().len() for ndims, .elem_count() for total elements
    // HINT: format!("{:?}", t.dtype()) gives the dtype name
    todo!("Inspect tensor properties")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_tensor() -> Result<()> {
        let t = create_1d_tensor()?;
        assert_eq!(t.dims(), &[5]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_2d_tensor() -> Result<()> {
        let t = create_2d_tensor()?;
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::U32);
        assert_eq!(t.to_vec2::<u32>()?, vec![vec![1, 2, 3], vec![4, 5, 6]]);
        Ok(())
    }

    #[test]
    fn test_zeros() -> Result<()> {
        let t = create_zeros()?;
        assert_eq!(t.dims(), &[3, 4]);
        assert_eq!(t.dtype(), DType::F32);
        let sum: f32 = t.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);
        Ok(())
    }

    #[test]
    fn test_extract_values() -> Result<()> {
        let t = Tensor::new(&[10.0f32, 20.0, 30.0], &Device::Cpu)?;
        let vals = extract_values(&t)?;
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
        Ok(())
    }

    #[test]
    fn test_inspect_tensor() -> Result<()> {
        let t = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu)?;
        let (ndims, nelems, dtype_name) = inspect_tensor(&t)?;
        assert_eq!(ndims, 3);
        assert_eq!(nelems, 24);
        assert_eq!(dtype_name, "F32");
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q1: Tensor Creation & Inspection ===\n");

    let t = create_1d_tensor()?;
    println!("1D tensor: {:?}", t);

    let t = create_2d_tensor()?;
    println!("2D tensor: {:?}", t);

    let t = create_zeros()?;
    println!("Zeros: {:?}", t);

    println!("\nAll exercises complete! Run `cargo test --example q01_tensor_creation` to verify.");
    Ok(())
}
