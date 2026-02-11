/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 5: Indexing, Slicing & Narrowing
/// ============================================================================
/// Difficulty: INTERMEDIATE
///
/// Candle has several indexing mechanisms. Unlike PyTorch's flexible indexing,
/// Candle is more explicit.
///
/// KEY CONCEPTS:
///   - .i(index) — IndexOp trait, supports ranges and tuples
///   - .narrow(dim, start, len) — slice along a dimension
///   - .gather(dim, index_tensor) — advanced indexing
///   - .index_select(dim, index_tensor) — select rows/cols by index
///   - IndexOp supports: single int, range, tuple of the above
///
/// INTERVIEW NUANCE:
///   - .i((0, seq_len - 1)) extracts a specific element from 2D
///   - .narrow(2, offset, len) is how KV-cache slicing works
///   - In generation: logits.i((0, prompt_len - 1))? gets last position
///   - .i(start..end) is a range slice (Python's tensor[start:end])
///   - Candle does NOT support negative indexing like Python's tensor[-1]
///     Instead use: tensor.i(tensor.dim(0)? - 1)?
///
/// HINTS:
///   - use candle_core::IndexOp; is needed for .i()
///   - .narrow(dim, start, length) — NOT start..end!
///   - .i((0,)) gives first row of 2D, .i((.., 0)) gives first column
/// ============================================================================

use candle_core::{DType, Device, IndexOp, Result, Tensor};

/// Q5a: Extract the element at position [0, 2] from a 2D tensor.
/// Return as a scalar f32.
pub fn get_element(t: &Tensor) -> Result<f32> {
    // TODO: Get element at row 0, col 2
    // HINT: t.i((0, 2))?.to_scalar::<f32>()?
    todo!("Get single element")
}

/// Q5b: Extract the last token's logits.
/// Input: logits tensor of shape [batch, seq_len, vocab_size]
/// Return: tensor of shape [vocab_size] (last position of first batch)
/// This is THE critical operation in autoregressive generation.
pub fn get_last_token_logits(logits: &Tensor) -> Result<Tensor> {
    // TODO: Get logits at position [0, seq_len-1]
    // HINT: let seq_len = logits.dim(1)?;
    //       logits.i((0, seq_len - 1))?
    // INTERVIEW Q: Why do we take the LAST position's logits?
    //   Answer: In causal LM, each position predicts the NEXT token.
    //   The last position predicts the first generated token.
    todo!("Get last token logits")
}

/// Q5c: Slice a tensor along dimension 0 using narrow.
/// Input: [10, hidden_size]. Get rows 3..7 (4 rows starting at 3).
pub fn narrow_slice(t: &Tensor) -> Result<Tensor> {
    // TODO: Use .narrow(dim, start, length)
    // HINT: t.narrow(0, 3, 4)?  — 4 rows starting at position 3
    todo!("Narrow slice")
}

/// Q5d: Extract RoPE cos/sin for the current offset.
/// Given cos table of shape [max_seq_len, half_head_dim],
/// extract rows from `offset` to `offset + seq_len`.
/// This is exactly what Qwen3RotaryEmbedding.apply() does.
pub fn slice_rope_table(cos: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
    // TODO: Slice the cos table for the current window
    // HINT: cos.i(offset..offset + seq_len)?
    // INTERVIEW Q: Why do we need offset for RoPE during decode?
    //   Answer: During decode, we generate one token at a time but need
    //   position-specific embeddings. offset tracks the absolute position.
    todo!("Slice RoPE table")
}

/// Q5e: Implement index_select to gather specific rows.
/// Given a tensor [vocab_size, hidden_size] and indices [seq_len],
/// select the rows corresponding to the indices.
/// This is what Embedding.forward() does internally!
pub fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    // TODO: Use index_select to do embedding lookup
    // HINT: weight.index_select(indices, 0)?
    // INTERVIEW Q: What does Embedding::forward() do under the hood?
    //   Answer: It's literally index_select on the embedding weight matrix.
    todo!("Embedding lookup")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_element() -> Result<()> {
        let t = Tensor::new(&[[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]], &Device::Cpu)?;
        let val = get_element(&t)?;
        assert_eq!(val, 30.0);
        Ok(())
    }

    #[test]
    fn test_last_token_logits() -> Result<()> {
        // [1, 3, 5] — batch=1, seq=3, vocab=5
        let logits = Tensor::new(
            &[[[0.1f32, 0.2, 0.3, 0.4, 0.5],
               [1.1, 1.2, 1.3, 1.4, 1.5],
               [2.1, 2.2, 2.3, 2.4, 2.5]]],
            &Device::Cpu,
        )?;
        let last = get_last_token_logits(&logits)?;
        assert_eq!(last.dims(), &[5]);
        assert_eq!(last.to_vec1::<f32>()?, vec![2.1, 2.2, 2.3, 2.4, 2.5]);
        Ok(())
    }

    #[test]
    fn test_narrow_slice() -> Result<()> {
        let t = Tensor::zeros((10, 4), DType::F32, &Device::Cpu)?;
        let s = narrow_slice(&t)?;
        assert_eq!(s.dims(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn test_slice_rope() -> Result<()> {
        let cos = Tensor::zeros((100, 64), DType::F32, &Device::Cpu)?;
        let s = slice_rope_table(&cos, 10, 5)?;
        assert_eq!(s.dims(), &[5, 64]);
        Ok(())
    }

    #[test]
    fn test_embedding_lookup() -> Result<()> {
        // Simple embedding: 4 words, 3-dim embeddings
        let weight = Tensor::new(
            &[[1.0f32, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]],
            &Device::Cpu,
        )?;
        let indices = Tensor::new(&[3u32, 0, 2], &Device::Cpu)?;
        let result = embedding_lookup(&weight, &indices)?;
        assert_eq!(result.dims(), &[3, 3]);
        assert_eq!(
            result.to_vec2::<f32>()?,
            vec![
                vec![1.0, 1.0, 1.0], // word 3
                vec![1.0, 0.0, 0.0], // word 0
                vec![0.0, 0.0, 1.0], // word 2
            ]
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Q5: Indexing, Slicing & Narrowing ===\n");
    println!("Run `cargo test --example q05_indexing_slicing` to verify.");
    Ok(())
}
