/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 2: DType & Device Management
/// ============================================================================
/// Difficulty: BASIC
///
/// Understanding DType and Device is critical for real-world inference.
/// Wrong dtype = silent precision loss. Wrong device = CPU/GPU mismatch panics.
///
/// KEY CONCEPTS:
///   - DType: F16, BF16, F32, F64, U8, U32, I64
///   - Device: Cpu, Cuda(ordinal), Metal(ordinal)
///   - .to_dtype() converts between dtypes
///   - .to_device() moves tensors between devices
///   - device.bf16_default_to_f32() selects BF16 on GPU, F32 on CPU
///
/// INTERVIEW NUANCE:
///   - BF16 has same exponent range as F32 but fewer mantissa bits
///   - F16 has smaller exponent range, can overflow more easily
///   - When doing LayerNorm/RmsNorm, you MUST upcast to F32 internally
///     to avoid precision issues, then downcast back
///   - Candle's to_dtype is lazy for same-dtype (returns self)
///
/// HINTS:
///   - tensor.to_dtype(DType::BF16)? converts dtype
///   - DType::F16 | DType::BF16 => DType::F32 is the standard upcast pattern
///   - Device::Cpu is an enum variant, not a function
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};

/// Q2a: Convert a F32 tensor to BF16 and back to F32.
/// Return the round-tripped tensor.
/// INTERVIEW Q: Why might values change after F32->BF16->F32?
pub fn roundtrip_bf16(t: &Tensor) -> Result<Tensor> {
    // TODO: Convert to BF16 then back to F32
    // HINT: chain two .to_dtype() calls
    todo!("Round-trip through BF16")
}

/// Q2b: Given a dtype, return the "internal compute dtype" that should be
/// used for numerical stability in normalization layers.
/// Rule: F16 and BF16 should upcast to F32. Everything else stays the same.
pub fn get_internal_dtype(dtype: DType) -> DType {
    // TODO: Implement the upcast logic
    // HINT: match on the dtype, return F32 for F16/BF16, pass through otherwise
    todo!("Get internal compute dtype")
}

/// Q2c: Create a tensor on CPU and verify it's on CPU.
/// Return (tensor, is_cpu_bool).
pub fn verify_device() -> Result<(Tensor, bool)> {
    // TODO: Create any tensor on Device::Cpu
    // Return (tensor, true if device is CPU)
    // HINT: Check tensor.device() - comparing with Device::Cpu
    // HINT: matches!(tensor.device(), Device::Cpu)
    todo!("Verify device placement")
}

/// Q2d: Given two tensors, check if they are on the same device AND same dtype.
/// Operations between tensors require both to match!
pub fn check_compatible(a: &Tensor, b: &Tensor) -> (bool, bool) {
    // TODO: Return (same_device, same_dtype)
    // HINT: Use .device() and .dtype() comparisons
    // NOTE: Device implements PartialEq, DType implements PartialEq
    todo!("Check tensor compatibility")
}

/// Q2e: INTERVIEW QUESTION - What does device.bf16_default_to_f32() do?
/// Implement the equivalent logic manually:
/// - If the device is CUDA or Metal, return DType::BF16
/// - If the device is CPU, return DType::F32
pub fn select_dtype_for_device(device: &Device) -> DType {
    // TODO: Implement the device-dependent dtype selection
    // HINT: match on Device variants
    // HINT: Device::Cpu => F32, Device::Cuda(_) | Device::Metal(_) => BF16
    todo!("Select dtype for device")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_bf16() -> Result<()> {
        let t = Tensor::new(&[1.0f32, 2.5, 3.14159], &Device::Cpu)?;
        let rt = roundtrip_bf16(&t)?;
        assert_eq!(rt.dtype(), DType::F32);
        // Values should be close but may not be exactly equal due to BF16 precision
        let orig: Vec<f32> = t.to_vec1()?;
        let result: Vec<f32> = rt.to_vec1()?;
        for (a, b) in orig.iter().zip(result.iter()) {
            assert!((a - b).abs() < 0.02, "BF16 roundtrip error too large: {} vs {}", a, b);
        }
        Ok(())
    }

    #[test]
    fn test_internal_dtype() {
        assert_eq!(get_internal_dtype(DType::F16), DType::F32);
        assert_eq!(get_internal_dtype(DType::BF16), DType::F32);
        assert_eq!(get_internal_dtype(DType::F32), DType::F32);
        assert_eq!(get_internal_dtype(DType::F64), DType::F64);
        assert_eq!(get_internal_dtype(DType::U32), DType::U32);
    }

    #[test]
    fn test_verify_device() -> Result<()> {
        let (t, is_cpu) = verify_device()?;
        assert!(is_cpu);
        assert!(matches!(t.device(), Device::Cpu));
        Ok(())
    }

    #[test]
    fn test_check_compatible() -> Result<()> {
        let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let b = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
        let (same_dev, same_dtype) = check_compatible(&a, &b);
        assert!(same_dev);
        assert!(same_dtype);

        let c = Tensor::ones((2, 3), DType::F64, &Device::Cpu)?;
        let (same_dev, same_dtype) = check_compatible(&a, &c);
        assert!(same_dev);
        assert!(!same_dtype);
        Ok(())
    }

    #[test]
    fn test_select_dtype() {
        assert_eq!(select_dtype_for_device(&Device::Cpu), DType::F32);
        // Note: Can't test CUDA/Metal without hardware, but the logic should be clear
    }
}

fn main() -> Result<()> {
    println!("=== Q2: DType & Device Management ===\n");
    println!("Run `cargo test --example q02_dtype_and_device` to verify.");
    Ok(())
}
