use axum::{Router, routing::post};
use clap::Parser;
pub mod routes;
use crate::routes::completion::completion;

#[derive(Parser)]
#[command(name = "completion")]
#[command(
    about = "Run a copilot server in your terminal",
    long_about = "Run a copilot server in your terminal which exposes a REST API POST /v1/:engine/completion to interact with the model."
)]
struct Args {
    /// The port to bind the server on, default to 9090.
    #[arg(short = 'p', long = "port", default_value = "9090")]
    port: u32,
    /// The host to bind the server on, default to 0.0.0.0.
    #[arg(short = 'o', long = "host", default_value = "0.0.0.0")]
    host: String,
}

// Plan
// 1. Create a CLI.
// 2. Download the model to ./model.
// 3. Run a server serving the model.
#[tokio::main]
async fn main() {
    let args: Args = Args::parse();
    let addr = args.host.clone() + ":" + args.port.to_string().as_str();

    let app = Router::new().route("/v1/completions", post(completion));

    axum::Server::bind(&addr.parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
