use crate::routes::request::CompletionRequest;
use crate::routes::response::{Choices, CompletionResponse};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use serde_json::json;
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) async fn completion(
    Json(body): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    print!("{:?}", body);
    let prompt = body.prompt;
    let model: String = body.model;
    let stream = async_stream::stream! {
        let obj = json!(CompletionResponse{
            id: "id".to_string(),
            object: "text.completion.chunk".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() ,
            model: model.clone(),
            choices: vec![Choices {
                text: "".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        });
        let json = serde_json::to_string(&obj).unwrap();
        yield Ok(SseEvent::default().data(json));
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}
