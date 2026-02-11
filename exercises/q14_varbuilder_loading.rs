/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 14: VarBuilder & Model Weight Loading
/// ============================================================================
/// Difficulty: ADVANCED
///
/// VarBuilder is Candle's mechanism for loading model weights from safetensors
/// or initializing them for training. Understanding it is critical for porting models.
///
/// KEY CONCEPTS:
///   - VarBuilder::from_mmaped_safetensors() — zero-copy memory-mapped loading
///   - VarBuilder::zeros() — zero-initialized (for testing)
///   - VarBuilder::from_varmap() — for training (mutable weights)
///   - .pp("name") — "path prefix" scopes to a submodule
///   - .get(shape, "name") — retrieve a specific tensor
///   - Weight paths must EXACTLY match the safetensors keys!
///
/// INTERVIEW NUANCE:
///   - The #1 source of bugs: wrong weight path!
///     PyTorch: model.layers.0.self_attn.q_proj.weight
///     Candle:  vb.pp("model").pp("layers.0").pp("self_attn").pp("q_proj").get(shape, "weight")
///     OR:      vb.pp("model.layers.0.self_attn.q_proj").get(shape, "weight")
///   - .pp() can be chained: vb.pp("a").pp("b") == vb.pp("a.b")
///   - Memory mapping: weights stay on disk, loaded lazily into memory
///     This is why `unsafe { VarBuilder::from_mmaped_safetensors(...) }` is used
///   - The dtype parameter in from_mmaped_safetensors CASTS all weights
///   - VarBuilder is NOT Clone, but is passed by value into load functions
///   - For nested modules: pass vb.pp("submodule") to the child's load()
///
/// HINTS:
///   - vb.pp("layers.0") or vb.pp(&format!("layers.{i}"))
///   - vb.get((out, in), "weight")? for a Linear weight
///   - vb.get(size, "weight")? for a 1D weight (LayerNorm, RmsNorm)
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

/// Q14a: Load a simple Linear layer from VarBuilder.
/// The path should be "my_linear", weight shape [out, in].
pub fn load_linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    // TODO: Load a linear layer using candle_nn::linear
    // HINT: candle_nn::linear(in_dim, out_dim, vb.pp("my_linear"))
    todo!("Load linear from VarBuilder")
}

/// Q14b: Load an Embedding layer from VarBuilder.
/// Path: "embed_tokens", shape [vocab_size, hidden_size].
pub fn load_embedding(
    vocab_size: usize,
    hidden_size: usize,
    vb: VarBuilder,
) -> Result<Embedding> {
    // TODO: Load embedding
    // HINT: candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))
    todo!("Load embedding from VarBuilder")
}

/// Q14c: Simulate loading a transformer layer with proper path scoping.
/// The weight hierarchy should be:
///   model.layers.{layer_idx}.self_attn.q_proj.weight
///   model.layers.{layer_idx}.self_attn.k_proj.weight
///   model.layers.{layer_idx}.input_layernorm.weight
///
/// Return the q_proj weight shape to verify correct loading.
pub fn load_layer_weights(vb: VarBuilder, layer_idx: usize, hidden: usize, head_dim: usize, num_heads: usize) -> Result<Vec<usize>> {
    // TODO: Navigate the VarBuilder path hierarchy
    // HINT:
    //   let layer_vb = vb.pp("model").pp(&format!("layers.{layer_idx}"));
    //   let attn_vb = layer_vb.pp("self_attn");
    //   let q_proj = candle_nn::linear_no_bias(hidden, num_heads * head_dim, attn_vb.pp("q_proj"))?;
    //   // Get the weight shape
    //   Ok(q_proj.weight().dims().to_vec())
    todo!("Load layer weights with path scoping")
}

/// Q14d: INTERVIEW QUESTION — Explain the difference between these loading methods:
///   1. VarBuilder::from_mmaped_safetensors
///   2. VarBuilder::zeros
///   3. VarBuilder::from_varmap
pub fn explain_varbuilder_methods() -> (&'static str, &'static str, &'static str) {
    // TODO: Return explanations for each
    // HINT:
    //   mmaped: Memory-mapped loading from safetensors files. Weights are lazily
    //           loaded from disk, reducing peak memory. Used for inference.
    //   zeros:  All weights initialized to zero. Used for testing/debugging.
    //   varmap: Weights stored in a mutable VarMap. Used for training, as
    //           weights need to be updated by the optimizer.
    todo!("Explain VarBuilder methods")
}

/// Q14e: Implement the HuggingFace model loading pattern.
/// Given file path, dtype, device — create a VarBuilder and load an embedding.
/// This simulates what main.rs does.
pub fn hf_loading_pattern(
    dtype: DType,
    device: &Device,
) -> Result<Embedding> {
    // TODO: Create a VarBuilder and load an embedding
    // In real code:
    //   let vb = unsafe {
    //       VarBuilder::from_mmaped_safetensors(&[path], dtype, device)?
    //   };
    // For this exercise, use VarBuilder::zeros:
    //   let vb = VarBuilder::zeros(dtype, device);
    //   candle_nn::embedding(100, 32, vb.pp("model").pp("embed_tokens"))
    //
    // INTERVIEW Q: Why is from_mmaped_safetensors unsafe?
    //   Answer: Memory mapping relies on the file not being modified while
    //   mapped. If the file changes, you get undefined behavior. The `unsafe`
    //   block acknowledges this invariant.
    todo!("HF loading pattern")
}

/// Q14f: BONUS — Weight tying pattern.
/// After loading the model, extract the embedding weight and reuse it as lm_head.
/// Show how to access embeddings() from a loaded Embedding.
pub fn weight_tying_pattern(vb: VarBuilder) -> Result<(Embedding, candle_nn::Linear)> {
    // TODO: Load embedding, then create lm_head from its weights
    // HINT:
    //   let embed = candle_nn::embedding(100, 32, vb.pp("embed_tokens"))?;
    //   let lm_head = candle_nn::Linear::new(embed.embeddings().clone(), None);
    //   Ok((embed, lm_head))
    //
    // INTERVIEW Q: What does embed.embeddings() return?
    //   Answer: A reference to the underlying weight Tensor [vocab_size, hidden_size].
    todo!("Weight tying pattern")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_linear() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let layer = load_linear(4, 8, vb)?;
        let input = Tensor::ones((1, 4), DType::F32, &Device::Cpu)?;
        let output = layer.forward(&input)?;
        assert_eq!(output.dims(), &[1, 8]);
        Ok(())
    }

    #[test]
    fn test_load_embedding() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let embed = load_embedding(100, 32, vb)?;
        let ids = Tensor::new(&[0u32, 5, 10], &Device::Cpu)?;
        let output = embed.forward(&ids)?;
        assert_eq!(output.dims(), &[3, 32]);
        Ok(())
    }

    #[test]
    fn test_load_layer_weights() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let shape = load_layer_weights(vb, 0, 64, 8, 4)?;
        // q_proj weight: [num_heads * head_dim, hidden] = [32, 64]
        assert_eq!(shape, vec![32, 64]);
        Ok(())
    }

    #[test]
    fn test_hf_loading() -> Result<()> {
        let embed = hf_loading_pattern(DType::F32, &Device::Cpu)?;
        let ids = Tensor::new(&[0u32, 1, 2], &Device::Cpu)?;
        let output = embed.forward(&ids)?;
        assert_eq!(output.dims(), &[3, 32]);
        Ok(())
    }

    #[test]
    fn test_weight_tying() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let (embed, lm_head) = weight_tying_pattern(vb)?;
        let ids = Tensor::new(&[0u32, 1], &Device::Cpu)?;
        let hidden = embed.forward(&ids)?; // [2, 32]
        let logits = lm_head.forward(&hidden)?; // [2, 100]
        assert_eq!(logits.dims(), &[2, 100]);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q14: VarBuilder & Model Weight Loading ===\n");
    println!("Run `cargo test --example q14_varbuilder_loading` to verify.");
    Ok(())
}
