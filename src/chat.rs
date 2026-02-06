use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Format messages using the ChatML template used by Qwen3.
///
/// Each message becomes:
///   <|im_start|>role\ncontent<|im_end|>\n
///
/// The final output ends with:
///   <|im_start|>assistant\n
pub fn apply_chat_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
