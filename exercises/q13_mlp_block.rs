/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 13: MLP / Feed-Forward Block
/// ============================================================================
/// Difficulty: ADVANCED
///
/// The MLP block is the "thinking" part of each transformer layer.
/// Modern LLMs use a gated variant: SiLU(gate_proj(x)) * up_proj(x), then down_proj.
///
/// KEY CONCEPTS:
///   - Classic FFN: Linear(hidden, intermediate) -> GELU -> Linear(intermediate, hidden)
///   - Gated FFN (LLaMA/Qwen): gate=Linear(h, i), up=Linear(h, i), down=Linear(i, h)
///     output = down(silu(gate(x)) * up(x))
///   - SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
///   - All projections are no-bias in modern LLMs
///   - intermediate_size is typically ~2.67x hidden_size (with SwiGLU factor)
///
/// INTERVIEW NUANCE:
///   - Why gated MLP? The gate controls information flow, similar to LSTM gates.
///     SwiGLU (SiLU-gated) empirically outperforms standard FFN (PaLM paper).
///   - The "SwiGLU factor": classic FFN has 8h^2 params (2 matrices of 4h*h).
///     Gated FFN has 3 matrices. To match params: intermediate = 8h/3 ≈ 2.67h.
///   - The full decoder layer pattern (pre-norm):
///     residual = x
///     x = RmsNorm(x)
///     x = Attention(x) + residual    // first residual
///     residual = x
///     x = RmsNorm(x)
///     x = MLP(x) + residual          // second residual
///
/// HINTS:
///   - .silu()? applies SiLU activation
///   - candle_nn::linear_no_bias for projections
///   - (&gate_act * &up)? for element-wise gating
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// Q13a: Implement the SiLU (Sigmoid Linear Unit) activation manually.
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn manual_silu(x: &Tensor) -> Result<Tensor> {
    // TODO: Implement SiLU from primitives
    // HINT: x / (x.neg()?.exp()? + 1.0)?
    // OR just: x.silu() (Candle has it built-in!)
    // INTERVIEW Q: How does SiLU compare to ReLU?
    //   Answer: SiLU is smooth and non-monotonic (slight negative values
    //   survive), which helps gradient flow in deep networks.
    todo!("Manual SiLU")
}

/// Q13b: Implement the gated MLP block.
/// gate_proj, up_proj: [hidden_size, intermediate_size]
/// down_proj: [intermediate_size, hidden_size]
///
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub fn gated_mlp(
    x: &Tensor,
    gate_proj: &candle_nn::Linear,
    up_proj: &candle_nn::Linear,
    down_proj: &candle_nn::Linear,
) -> Result<Tensor> {
    // TODO: Implement gated MLP
    // Step 1: gate = gate_proj.forward(x)?
    // Step 2: gate_act = gate.silu()?
    // Step 3: up = up_proj.forward(x)?
    // Step 4: gated = (&gate_act * &up)?
    // Step 5: down_proj.forward(&gated)
    //
    // INTERVIEW Q: Why multiply gate and up BEFORE down_proj?
    //   Answer: The gate selectively activates features from up_proj.
    //   This is the "gating mechanism" — similar to how LSTM gates
    //   control information flow.
    todo!("Gated MLP block")
}

/// Q13c: Implement a FULL decoder layer (pre-norm style).
/// This combines RmsNorm + Attention-stub + Residual + RmsNorm + MLP + Residual.
/// For simplicity, use a dummy "attention" that just returns its input.
///
/// Pattern:
///   residual = x
///   x = rms_norm_1(x)
///   x = attention(x)  [stub: identity]
///   x = x + residual
///   residual = x
///   x = rms_norm_2(x)
///   x = mlp(x)
///   x = x + residual
pub fn decoder_layer(
    x: &Tensor,
    norm1_weight: &Tensor,
    norm2_weight: &Tensor,
    gate_proj: &candle_nn::Linear,
    up_proj: &candle_nn::Linear,
    down_proj: &candle_nn::Linear,
    eps: f64,
) -> Result<Tensor> {
    // TODO: Implement the pre-norm decoder layer pattern
    //
    // HINT: Use rms_norm from q09 or implement inline:
    //   fn rms_norm_inline(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    //       let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    //       let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    //       x_norm.broadcast_mul(w)
    //   }
    //
    // INTERVIEW Q: Why pre-norm instead of post-norm?
    //   Answer: Pre-norm is more stable for deep networks. The residual
    //   connection carries the unnormalized signal, preventing gradient
    //   vanishing. GPT-2 discovered this; now all modern LLMs use it.
    todo!("Full decoder layer")
}

/// Q13d: INTERVIEW QUESTION — What's the parameter count of a Qwen3-0.6B MLP block?
/// hidden_size = 1024, intermediate_size = 3072
/// 3 matrices (gate, up, down), no bias.
pub fn mlp_param_count(hidden: usize, intermediate: usize) -> usize {
    // TODO: Calculate total parameters
    // gate_proj: hidden * intermediate
    // up_proj: hidden * intermediate
    // down_proj: intermediate * hidden
    // Total: 3 * hidden * intermediate
    todo!("MLP parameter count")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_silu() -> Result<()> {
        let x = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let result = manual_silu(&x)?;
        let val: f32 = result.to_vec1::<f32>()?[0];
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((val - 0.0).abs() < 1e-6);

        let x = Tensor::new(&[1.0f32], &Device::Cpu)?;
        let result = manual_silu(&x)?;
        let val: f32 = result.to_vec1::<f32>()?[0];
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((val - 0.7311).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_gated_mlp() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let gate = candle_nn::linear_no_bias(8, 16, vb.pp("gate"))?;
        let up = candle_nn::linear_no_bias(8, 16, vb.pp("up"))?;
        let down = candle_nn::linear_no_bias(16, 8, vb.pp("down"))?;
        let x = Tensor::ones((1, 4, 8), DType::F32, &Device::Cpu)?;
        let result = gated_mlp(&x, &gate, &up, &down)?;
        assert_eq!(result.dims(), &[1, 4, 8]);
        Ok(())
    }

    #[test]
    fn test_decoder_layer() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let gate = candle_nn::linear_no_bias(8, 16, vb.pp("gate"))?;
        let up = candle_nn::linear_no_bias(8, 16, vb.pp("up"))?;
        let down = candle_nn::linear_no_bias(16, 8, vb.pp("down"))?;
        let norm1 = Tensor::ones((8,), DType::F32, &Device::Cpu)?;
        let norm2 = Tensor::ones((8,), DType::F32, &Device::Cpu)?;
        let x = Tensor::ones((1, 4, 8), DType::F32, &Device::Cpu)?;
        let result = decoder_layer(&x, &norm1, &norm2, &gate, &up, &down, 1e-6)?;
        assert_eq!(result.dims(), &[1, 4, 8]);
        // With zero weights, MLP output is 0, so result should equal input (residual)
        let diff = (&result - &x)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "With zero MLP weights, output should match input via residual");
        Ok(())
    }

    #[test]
    fn test_mlp_param_count() {
        // Qwen3-0.6B: hidden=1024, intermediate=3072
        let count = mlp_param_count(1024, 3072);
        assert_eq!(count, 3 * 1024 * 3072);
    }
}

fn main() -> Result<()> {
    println!("=== Q13: MLP / Feed-Forward Block ===\n");
    println!("Run `cargo test --example q13_mlp_block` to verify.");
    Ok(())
}
