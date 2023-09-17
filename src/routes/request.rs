use std::collections::HashMap;

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub enum LogitBias {
    TokenIds,
    Tokens,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub(crate) struct CompletionRequest {
    pub prompt: String,
    pub suffix: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub mirostat_mode: Option<usize>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub echo: Option<bool>,
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub last_n_tokens: Option<usize>,
    pub logit_bias_type: Option<LogitBias>,
    pub model: Option<String>,
    pub n: Option<usize>,
    pub best_of: Option<usize>,
    pub seed: Option<u64>,
    pub user: Option<String>,
}

