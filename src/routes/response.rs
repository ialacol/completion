use serde::Serialize;

#[derive(Serialize, Debug)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Serialize, Debug)]
pub struct Choices {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choices>,
    usage: Option<Usage>,
}
