
use crate::cmd::Command;

#[derive(Clone)]
pub struct AppState {
    pub sender: tokio::sync::mpsc::Sender<Command>
}
