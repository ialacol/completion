use std::sync::Arc;

use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

pub struct AppState {
    pub tokenizer: Arc<std::sync::Mutex<Tokenizer>>,
    pub model_weights: Arc<std::sync::Mutex<ModelWeights>>,
    pub seed: u64,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
}
