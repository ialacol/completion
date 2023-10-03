use std::sync::Arc;
use tokio::sync::Mutex;

use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct AppState {
    pub tokenizer: Arc<Tokenizer>,
    pub model_weights: Arc<Mutex<ModelWeights>>,
    pub seed: u64,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
}
