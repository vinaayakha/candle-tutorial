/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 3: Tensor Arithmetic & Operations
/// ============================================================================
/// Difficulty: BASIC
///
/// Candle overloads +, -, *, / for tensors but they return Result<Tensor>.
/// This means you need the ? operator or explicit unwrap.
///
/// KEY CONCEPTS:
///   - (a + b)? — element-wise add (note the parentheses around the op!)
///   - (&a + &b)? — same but borrows (avoids moving)
///   - .matmul() — matrix multiplication
///   - .sum_all(), .mean_all() — reductions
///   - .sum_keepdim(dim), .mean_keepdim(dim) — reduce along axis, keep dim
///   - .sqr(), .sqrt(), .neg(), .recip(), .exp(), .log() — unary ops
///   - .affine(mul, add) — fused multiply-add: x * mul + add
///
/// INTERVIEW NUANCE:
///   - Unlike PyTorch, Candle ops return Result, not Tensor directly
///   - The & reference pattern matters: (&a * &b)? vs (a * b)? — the
///     latter MOVES a and b, so you can't use them again!
///   - .contiguous()? may be needed before matmul after transpose
///
/// HINTS:
///   - For matrix multiply: a.matmul(&b)?
///   - For element-wise: (&a * &b)? or a.mul(&b)?
///   - For scalar ops: (tensor + 1.0)? or tensor.affine(1.0, 5.0)?
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q3a: Add two tensors element-wise. Both are shape [3].
pub fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: Return a + b
    // HINT: (a + b)? works because Tensor implements Add
    a + b 
}

/// Q3b: Multiply two tensors element-wise.
pub fn multiply_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: Return a * b element-wise
    // HINT: (a * b)? -- remember the parentheses!
    a * b
}

/// Q3c: Matrix multiplication. a is [2, 3], b is [3, 4]. Return [2, 4].
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // TODO: Return the matrix product a @ b
    // HINT: a.matmul(b)?
    a.matmul(b)
}

/// Q3d: Compute the sum of ALL elements in a tensor, returning a scalar f32.
pub fn sum_all(t: &Tensor) -> Result<f32> {
    // TODO: Sum all elements and return as f32
    // HINT: t.sum_all()?.to_scalar::<f32>()?
    t.sum_all()?.to_scalar::<f32>()
}

/// Q3e: Compute mean along dimension 1, keeping the dimension.
/// Input shape: [2, 4]. Output shape: [2, 1].
pub fn mean_along_dim1(t: &Tensor) -> Result<Tensor> {
    // TODO: Compute mean along dim 1 with keepdim
    // HINT: t.mean_keepdim(1)?
    // INTERVIEW Q: Why is keepdim important for broadcasting later?
    t.mean_keepdim(1)
}

/// Q3f: Implement the sigmoid function: 1 / (1 + exp(-x))
/// Do NOT use a built-in sigmoid. Compose it from primitive ops.
pub fn manual_sigmoid(x: &Tensor) -> Result<Tensor> {
    // TODO: Compute sigmoid from primitives
    // HINT: (x.neg()?.exp()? + 1.0)?.recip()
    // INTERVIEW Q: Why does Candle not have a built-in sigmoid on Tensor?
    //   (Answer: it does via candle_nn::ops::sigmoid, but it's good to
    //    understand the composition)
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// Q3g: Apply the affine transformation: y = x * 2.0 + 3.0
/// Use the .affine() method.
pub fn affine_transform(x: &Tensor) -> Result<Tensor> {
    // TODO: Use .affine(mul, add)
    // HINT: x.affine(2.0, 3.0)
    // INTERVIEW Q: When is .affine() better than separate mul+add?
    //   (Answer: fused ops reduce memory traffic and are faster)
    x.affine(2.0,3.0 )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() -> Result<()> {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let b = Tensor::new(&[4.0f32, 5.0, 6.0], &Device::Cpu)?;
        let c = add_tensors(&a, &b)?;
        assert_eq!(c.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_multiply() -> Result<()> {
        let a = Tensor::new(&[2.0f32, 3.0, 4.0], &Device::Cpu)?;
        let b = Tensor::new(&[5.0f32, 6.0, 7.0], &Device::Cpu)?;
        let c = multiply_tensors(&a, &b)?;
        assert_eq!(c.to_vec1::<f32>()?, vec![10.0, 18.0, 28.0]);
        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<()> {
        let a = Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], &Device::Cpu)?;
        let b = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            &Device::Cpu,
        )?;
        let c = matmul(&a, &b)?;
        assert_eq!(c.dims(), &[2, 4]);
        assert_eq!(
            c.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
        );
        Ok(())
    }

    #[test]
    fn test_sum_all() -> Result<()> {
        let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        let s = sum_all(&t)?;
        assert_eq!(s, 10.0);
        Ok(())
    }

    #[test]
    fn test_mean_dim1() -> Result<()> {
        let t = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &Device::Cpu)?;
        let m = mean_along_dim1(&t)?;
        assert_eq!(m.dims(), &[2, 1]);
        let vals = m.to_vec2::<f32>()?;
        assert!((vals[0][0] - 2.5).abs() < 1e-6);
        assert!((vals[1][0] - 6.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let x = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let s = manual_sigmoid(&x)?;
        let val: f32 = s.to_vec1::<f32>()?[0];
        assert!((val - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {}", val);

        let x = Tensor::new(&[100.0f32], &Device::Cpu)?;
        let s = manual_sigmoid(&x)?;
        let val: f32 = s.to_vec1::<f32>()?[0];
        assert!((val - 1.0).abs() < 1e-4, "sigmoid(100) should be ~1.0, got {}", val);
        Ok(())
    }

    #[test]
    fn test_affine() -> Result<()> {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let y = affine_transform(&x)?;
        assert_eq!(y.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q3: Tensor Arithmetic & Operations ===\n");
    println!("Run `cargo test --example q03_tensor_arithmetic` to verify.");
    Ok(())
}
