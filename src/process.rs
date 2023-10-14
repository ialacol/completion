use tokio::sync::mpsc;

use crate::cmd::Command;
use tracing::info;

pub async fn process_prompt(sender: &mpsc::Sender<Command>, prompt: String) -> tokio::sync::mpsc::Receiver<String> {
  let (responder, receiver) = mpsc::channel(8);

  info!("sending prompt to model");
  sender.send(Command::Prompt {
      prompt,
      responder,
  }).await.unwrap();

  return receiver
}