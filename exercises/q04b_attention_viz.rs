/// ============================================================================
/// CANDLE INTERVIEW PREP - Question 4B: Attention Shape Visualization
/// ============================================================================
/// Difficulty: INTERMEDIATE
///
/// Visualize the complete attention reshape lifecycle in the terminal.
/// Fill in the todo!()s to print each transformation step.
///
/// The full attention data flow is:
///
///   hidden [b, s, h]
///     │
///     ├── Q proj ──→ [b, s, nh*hd]
///     │                   │
///     │              reshape ──→ [b, s, nh, hd]
///     │                   │
///     │              transpose(1,2) ──→ [b, nh, s, hd]  ← Q
///     │
///     ├── K proj ──→ [b, s, nkv*hd]
///     │                   │
///     │              reshape ──→ [b, s, nkv, hd]
///     │                   │
///     │              transpose(1,2) ──→ [b, nkv, s, hd] ← K
///     │
///     └── V proj ──→ [b, s, nkv*hd]  (same as K)
///
///   Attention:
///     scores = Q @ K^T ──→ [b, nh, s, s]    (GQA: sdpa handles nh≠nkv)
///     weights = softmax(scores / sqrt(hd))
///     output = weights @ V ──→ [b, nh, s, hd]
///
///   Reshape back:
///     transpose(1,2) ──→ [b, s, nh, hd]
///     contiguous()
///     reshape ──→ [b, s, nh*hd]
///     │
///     └── O proj ──→ [b, s, h]
///
/// ============================================================================

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// Trace and print every shape transformation in a single attention layer.
/// Fill in each step. The function prints the visualization to stdout.
pub fn visualize_attention_shapes(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        ATTENTION SHAPE VISUALIZATION                       ║");
    println!("║  batch={}, seq={}, hidden={}, heads={}, kv_heads={}, hd={} ║",
        batch, seq_len, hidden_size, num_heads, num_kv_heads, head_dim);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── 1. Input hidden states ──────────────────────────────────────────
    let hidden = Tensor::zeros((batch, seq_len, hidden_size), DType::F32, &device)?;
    println!("① Input hidden states:        {:?}", hidden.dims());

    // ── 2. Q/K/V Projections ────────────────────────────────────────────
    // TODO: Create Q, K, V projection layers and run forward
    // Q projects: hidden_size → num_heads * head_dim
    // K projects: hidden_size → num_kv_heads * head_dim
    // V projects: hidden_size → num_kv_heads * head_dim
    //
    // HINT:
    //   let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
    //   let k_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
    //   let v_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
    //   let q = q_proj.forward(&hidden)?;
    //   let k = k_proj.forward(&hidden)?;
    //   let v = v_proj.forward(&hidden)?;
    let q_proj = todo!("Create Q projection");
    let k_proj = todo!("Create K projection");
    let v_proj = todo!("Create V projection");
    let q: Tensor = todo!("Forward Q");
    let k: Tensor = todo!("Forward K");
    let v: Tensor = todo!("Forward V");

    println!("\n② After Q/K/V projection:");
    println!("   Q projected:               {:?}", q.dims());
    println!("   K projected:               {:?}", k.dims());
    println!("   V projected:               {:?}", v.dims());

    // ── 3. Reshape to multi-head format ─────────────────────────────────
    // TODO: Reshape Q from [b, s, nh*hd] → [b, s, nh, hd]
    //       Reshape K from [b, s, nkv*hd] → [b, s, nkv, hd]
    //       Reshape V from [b, s, nkv*hd] → [b, s, nkv, hd]
    //
    // HINT:
    //   let q = q.reshape((batch, seq_len, num_heads, head_dim))?;
    //   let k = k.reshape((batch, seq_len, num_kv_heads, head_dim))?;
    //   let v = v.reshape((batch, seq_len, num_kv_heads, head_dim))?;
    let q: Tensor = todo!("Reshape Q to multi-head");
    let k: Tensor = todo!("Reshape K to multi-head");
    let v: Tensor = todo!("Reshape V to multi-head");

    println!("\n③ After reshape to multi-head:");
    println!("   Q reshaped:                {:?}", q.dims());
    println!("   K reshaped:                {:?}", k.dims());
    println!("   V reshaped:                {:?}", v.dims());

    // ── 4. Transpose to [b, heads, s, hd] ──────────────────────────────
    // TODO: Transpose dim 1 and dim 2 for Q, K, V
    //
    // HINT:
    //   let q = q.transpose(1, 2)?;
    //   let k = k.transpose(1, 2)?;
    //   let v = v.transpose(1, 2)?;
    let q: Tensor = todo!("Transpose Q");
    let k: Tensor = todo!("Transpose K");
    let v: Tensor = todo!("Transpose V");

    println!("\n④ After transpose(1,2) — ATTENTION-READY FORMAT:");
    println!("   Q [b, nh, s, hd]:          {:?}  ← {} query heads", q.dims(), num_heads);
    println!("   K [b, nkv, s, hd]:         {:?}  ← {} kv heads", k.dims(), num_kv_heads);
    println!("   V [b, nkv, s, hd]:         {:?}  ← {} kv heads", v.dims(), num_kv_heads);
    if num_heads != num_kv_heads {
        println!("   ⚡ GQA ratio:               {}:1 (Q heads per KV head)", num_heads / num_kv_heads);
    }

    // ── 5. Attention scores (Q @ K^T) ───────────────────────────────────
    // TODO: Compute attention scores manually: Q @ K^T
    // For GQA, we need to handle mismatched head counts.
    // For visualization, just show what the shapes WOULD be with sdpa.
    //
    // HINT (non-GQA manual):
    //   let scores = q.contiguous()?.matmul(&k.contiguous()?.transpose(2, 3)?)?;
    // For GQA, sdpa handles it natively. Just print the expected shape.
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("\n⑤ Attention computation:");
    println!("   Scale factor:              1/√{} = {:.4}", head_dim, scale);
    println!("   scores = Q @ K^T:          [{}, {}, {}, {}]",
        batch, num_heads, seq_len, seq_len);
    println!("   weights = softmax(scores):  [{}, {}, {}, {}]",
        batch, num_heads, seq_len, seq_len);

    // Use sdpa for the actual computation
    let attn_output = candle_nn::ops::sdpa(
        &q.contiguous()?,
        &k.contiguous()?,
        &v.contiguous()?,
        None,
        seq_len > 1, // causal for prefill
        scale,
        1.0,
    )?;

    println!("   output = weights @ V:      {:?}", attn_output.dims());

    // ── 6. Reshape back ─────────────────────────────────────────────────
    // TODO: Reverse the attention reshape:
    //   transpose(1, 2) → contiguous → reshape to [b, s, nh*hd]
    //
    // HINT:
    //   let (b, nh, s, hd) = attn_output.dims4()?;
    //   let output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
    let output: Tensor = todo!("Reverse attention reshape");

    println!("\n⑥ Reshape back to hidden:");
    println!("   transpose(1,2):            [{}, {}, {}, {}]", batch, seq_len, num_heads, head_dim);
    println!("   contiguous + reshape:      {:?}", output.dims());

    // ── 7. Output projection ────────────────────────────────────────────
    // TODO: Apply output projection: [b, s, nh*hd] → [b, s, hidden_size]
    //
    // HINT:
    //   let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
    //   let final_output = o_proj.forward(&output)?;
    let final_output: Tensor = todo!("Output projection");

    println!("   O projection:              {:?}  ← back to hidden_size!", final_output.dims());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  COMPLETE ATTENTION SHAPE FLOW SUMMARY                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  [{b},{s},{h}] ─Q─→ [{b},{s},{qd}] ─reshape─→ [{b},{s},{nh},{hd}] ─T─→ [{b},{nh},{s},{hd}]",
        b=batch, s=seq_len, h=hidden_size, qd=num_heads*head_dim, nh=num_heads, hd=head_dim);
    println!("║  [{b},{s},{h}] ─K─→ [{b},{s},{kd}] ─reshape─→ [{b},{s},{nk},{hd}] ─T─→ [{b},{nk},{s},{hd}]",
        b=batch, s=seq_len, h=hidden_size, kd=num_kv_heads*head_dim, nk=num_kv_heads, hd=head_dim);
    println!("║                                                            ║");
    println!("║  sdpa(Q,K,V) ──→ [{b},{nh},{s},{hd}] ─T─→ [{b},{s},{nh},{hd}] ─reshape─→ [{b},{s},{qd}] ─O─→ [{b},{s},{h}]",
        b=batch, s=seq_len, nh=num_heads, hd=head_dim, qd=num_heads*head_dim, h=hidden_size);
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_shapes() -> Result<()> {
        // Qwen3-0.6B config: hidden=1024, 16 heads, 8 kv_heads, head_dim=128 (total attention: 2048, but hidden is 1024, so we'll use matching dims)
        // Actually hidden=1024 but q_proj outputs num_heads*head_dim = 16*64 = 1024
        // Let's use a simplified version that matches
        visualize_attention_shapes(
            1,     // batch
            4,     // seq_len (short for readability)
            512,   // hidden_size
            8,     // num_heads
            4,     // num_kv_heads (GQA: 2:1 ratio)
            64,    // head_dim
        )
    }

    #[test]
    fn test_mha_shapes() -> Result<()> {
        // Standard MHA (no GQA) — like BERT/RoBERTa
        visualize_attention_shapes(
            1,     // batch
            6,     // seq_len
            768,   // hidden_size
            12,    // num_heads
            12,    // num_kv_heads (same = standard MHA)
            64,    // head_dim
        )
    }
}

fn main() -> Result<()> {
    println!("=== Q4B: Attention Shape Visualization ===\n");
    println!("--- Qwen3-style GQA (8 Q heads, 4 KV heads) ---\n");
    visualize_attention_shapes(1, 4, 512, 8, 4, 64)?;
    println!("\n\n--- Standard MHA (12 heads, like BERT) ---\n");
    visualize_attention_shapes(1, 6, 768, 12, 12, 64)?;
    Ok(())
}
