use std::sync::Arc;

use anyhow::{Error as E, Result};
use axum::{extract::State, routing::post, Json, Router};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

use candle_tutorial::chat::{apply_chat_template, ChatMessage};
use candle_tutorial::generate::{generate, SamplingParams};
use candle_tutorial::models::qwen3::{Qwen3Config, Qwen3ForCausalLM};

// ---------------------------------------------------------------------------
// Device selection: CUDA > Metal > CPU
// ---------------------------------------------------------------------------

fn select_device() -> Result<Device> {
    // Try CUDA first (NVIDIA GPU)
    if let Ok(device) = Device::new_cuda(0) {
        println!("[Device] Using CUDA (NVIDIA GPU)");
        return Ok(device);
    }

    // Try Metal (Apple Silicon GPU)
    if let Ok(device) = Device::new_metal(0) {
        println!("[Device] Using Metal (Apple Silicon GPU)");
        return Ok(device);
    }

    // Fallback to CPU
    println!("[Device] No GPU available, falling back to CPU");
    Ok(Device::Cpu)
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct AppState {
    model: Mutex<Qwen3ForCausalLM>,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: Qwen3Config,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.9
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Load model from HF Hub
// ---------------------------------------------------------------------------

fn load_model_and_tokenizer() -> Result<(Qwen3ForCausalLM, Tokenizer, Qwen3Config)> {
    let device = select_device()?;

    // Use BF16 on GPU (Metal/CUDA support it natively), F32 on CPU
    let dtype = device.bf16_default_to_f32();
    println!("[Device] dtype: {:?}", dtype);

    let model_id = "DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored".to_string(); //DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored | Qwen/Qwen3-0.6B
    let repo = Repo::with_revision(model_id, RepoType::Model, "main".to_string());

    let api = Api::new()?;
    let api = api.repo(repo);

    let config_path = api.get("config.json")?;
    let tokenizer_path = api.get("tokenizer.json")?;
    let weights_path = api.get("model.safetensors")?;

    println!("Config:    {}", config_path.display());
    println!("Tokenizer: {}", tokenizer_path.display());
    println!("Weights:   {}", weights_path.display());

    let config: Qwen3Config =
        serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)?
    };
    let model = Qwen3ForCausalLM::load(vb, &config)?;

    println!("Model loaded on {:?} successfully!", device);
    Ok((model, tokenizer, config))
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    println!("\n============================================================");
    println!("[Request] POST /v1/chat/completions");
    println!("[Request] messages={}, max_tokens={}, temperature={}, top_p={}",
        req.messages.len(), req.max_tokens, req.temperature, req.top_p);
    for (i, msg) in req.messages.iter().enumerate() {
        println!("[Request]   msg[{}]: role={}, content_len={}", i, msg.role, msg.content.len());
    }

    let prompt = apply_chat_template(&req.messages);
    println!("[Chat] Template applied, prompt_len={} chars", prompt.len());

    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenization failed");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = input_ids.len();
    println!("[Tokenizer] Encoded {} tokens from prompt", prompt_len);

    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        eos_token_id: 151645, // <|im_end|>
    };

    let generated_ids = {
        let mut model = state.model.lock().await;
        let device = model.device().clone();
        let input_tensor = Tensor::new(input_ids.as_slice(), &device)
            .expect("tensor creation failed")
            .unsqueeze(0)
            .expect("unsqueeze failed");
        generate(&mut model, &input_tensor, &params).expect("generation failed")
    };

    let completion_len = generated_ids.len();

    // Filter out EOS tokens from output
    let output_ids: Vec<u32> = generated_ids
        .into_iter()
        .filter(|&id| id != params.eos_token_id)
        .collect();

    let content = state
        .tokenizer
        .decode(&output_ids, true)
        .unwrap_or_default();

    let finish_reason = if completion_len >= req.max_tokens {
        "length"
    } else {
        "stop"
    };

    println!("[Response] finish_reason={}, prompt_tokens={}, completion_tokens={}, total={}",
        finish_reason, prompt_len, completion_len, prompt_len + completion_len);
    println!("[Response] content ({} chars): {:?}",
        content.len(),
        if content.len() > 200 { format!("{}...", &content[..200]) } else { content.clone() }
    );

    Json(ChatCompletionResponse {
        id: "chatcmpl-candle".to_string(),
        object: "chat.completion".to_string(),
        model: "Qwen/Qwen3-0.6B".to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading Qwen3-0.6B model...");
    let (model, tokenizer, config) = load_model_and_tokenizer()?;

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        config,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    println!("Server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
