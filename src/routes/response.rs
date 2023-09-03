use serde::Serialize;

#[derive(Serialize, Debug)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Serialize, Debug)]
pub struct Choices {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<()>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choices>,
    pub usage: Option<Usage>,
}
