type Responder<T> = tokio::sync::mpsc::Sender<T>;

#[derive(Debug)]
pub enum Command {
    Prompt {
        prompt: String,
        responder: Responder<String>,
    },
}
