use crate::process::process_prompt;
use crate::routes::request::CompletionRequest;
use crate::routes::response::{Choices, CompletionResponse};
use crate::state::AppState;
use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use serde_json::json;
use std::convert::Infallible;
use std::time::{UNIX_EPOCH, SystemTime};

pub(crate) async fn completion(State(state): State<AppState>, Json(body): Json<CompletionRequest>
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let stream = async_stream::stream! {
        let prompt = body.prompt;
        let sender = state.sender.clone();
        let mut receiver = process_prompt(&sender, prompt).await;
        while let Some(text) = receiver.recv().await {
            let obj = json!(CompletionResponse {
                id: "id".to_string(),
                object: "text.completion.chunk".to_string(),
                created: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: "_".to_string(),
                choices: vec![Choices {
                    text: text.to_string(),
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            });
            let json = serde_json::to_string(&obj).unwrap();
            yield Ok(axum::response::sse::Event::default().data(json));
        }

    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}