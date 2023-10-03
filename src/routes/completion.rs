use crate::routes::request::CompletionRequest;
use crate::routes::response::{Choices, CompletionResponse};
use crate::state::AppState;
use crate::token::token_to_text;
use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::MAX_SEQ_LEN;
use futures::stream::Stream;
use serde_json::json;
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) async fn completion(
    State(state): State<AppState>,
    Json(body): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let stream = async_stream::stream! {
        let tokenizer = &state.tokenizer;
        let model_weights = &state.model_weights;

        let seed = match body.seed {
            Some(value) => value,
            None => {state.seed}
        };
        let repeat_last_n = match body.last_n_tokens {
            Some(value) => value,
            None => {state.repeat_last_n}
        };
        let repeat_penalty = match body.repeat_penalty {
            Some(value) => value,
            None => {state.repeat_penalty}
        };

        let prompt = body.prompt;

        let tokens = tokenizer.encode(prompt, true).unwrap();
        let pre_prompt_tokens = vec![];
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = body.n.unwrap().saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens: Vec<u32> = vec![];
        let mut logits_processor = LogitsProcessor::new(seed, body.temperature, body.top_p);
        let mut locked_model_weights = model_weights.lock().await;
        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            let logits = locked_model_weights.forward(&input, 0).unwrap();
            let logits = logits.squeeze(0).unwrap();
            logits_processor.sample(&logits).unwrap()
        };
        all_tokens.push(next_token);
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            let logits = locked_model_weights.forward(&input, prompt_tokens.len() + index).unwrap();
            let logits = logits.squeeze(0).unwrap();
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            let _ = candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            );
            next_token = logits_processor.sample(&logits).unwrap();
            all_tokens.push(next_token);
            let text = token_to_text(next_token, &tokenizer);
            let obj = json!(CompletionResponse{
                id: "id".to_string(),
                object: "text.completion.chunk".to_string(),
                created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() ,
                model: "_".to_string(),
                choices: vec![Choices {
                    text: text,
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            });
            let json = serde_json::to_string(&obj).unwrap();
            yield Ok(SseEvent::default().data(json));
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}
