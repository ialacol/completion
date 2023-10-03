use std::sync::Arc;
use tracing_subscriber::fmt::format::FmtSpan;
use tokio::sync::Mutex;

use axum::{routing::post, Router};
use candle_transformers::models::quantized_llama::ModelWeights;
use clap::Parser;

use candle_core::quantized::gguf_file;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub mod routes;
mod state;
mod token;
use std::net::SocketAddr;
use crate::routes::completion::completion;
use tracing::Instrument;
use tracing::info;
use tracing::info_span;

#[derive(Parser)]
#[command(name = "completion")]
#[command(
    about = "Run a copilot server in your terminal",
    long_about = "Run a copilot server in your terminal which exposes a REST API POST /v1/:engine/completion to interact with the model."
)]
struct Args {
    /// The port to bind the server on, default to 9090.
    #[arg(short = 'p', long = "port", default_value = "9090")]
    port: u16,
    /// The seed to use when generating random samples, default to 299792458, only used when no seed is provided in the request.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    /// Penalty to be applied for repeating tokens, 1. means no penalty, only used when no repeat_penalty is provided in the request.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty, only used when no last_n_tokens is provided in the request.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    /// HG tokenizer repo id, default to "hf-internal-testing/llama-tokenizer" (https://huggingface.co/hf-internal-testing/llama-tokenizer)
    #[arg(long, default_value = "hf-internal-testing/llama-tokenizer")]
    tokenizer_repo_id: String,
    /// HG tokenizer repo revision, default to "main"
    #[arg(long, default_value = "main")]
    tokenizer_repo_revision: String,
    /// HG tokenizer repo tokenizer file, default to "tokenizer.json"
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer_file: String,
    // HG model repo id, default to "TheBloke/CodeLlama-7B-GGUF" (https://huggingface.co/TheBloke/CodeLlama-7B-GGUF)
    #[arg(long, default_value = "TheBloke/CodeLlama-7B-GGUF")]
    model_repo_id: String,
    /// HG model repo revision, default to "main"
    #[arg(long, default_value = "main")]
    model_repo_revision: String,
    /// HG model repo GGMl/GGUF file, default to "codellama-7b.Q2_K.gguf"
    #[arg(long, default_value = "codellama-7b.Q2_K.gguf")]
    model_file: String,
}

// Plan
// 1. Create a CLI.
// 2. Download the model to ./model.
// 3. Run a server serving the model.
#[tokio::main]
async fn main() {
    // The tracing crate is a framework for instrumenting Rust programs to 
    // collect structured, event-based diagnostic information.
    // https://github.com/tokio-rs/tracing
    // https://tokio.rs/tokio/topics/tracing
    // Start configuring a `fmt` subscriber
    let subscriber = tracing_subscriber::fmt()
        // Use a more compact, abbreviated log format
        .compact()
        // Display source code file paths
        .with_file(true)
        // Display source code line numbers
        .with_line_number(true)
        // Display the thread ID an event was recorded on
        .with_thread_ids(true)
        // Display the event's target (module path)
        .with_target(true)
        // Add span events
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        // Build the subscriber
        .finish();

    // Set the subscriber as the default
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(_) => (),
        Err(error) => panic!("error setting default tracer as `fmt` subscriber {:?}", error),
    };

    let args = Args::parse();

    // Init the HF API.
    // https://github.com/huggingface/hf-hub
    let hf_api = match hf_hub::api::tokio::Api::new() {
        Ok(hf_hub) => hf_hub,
        Err(error) => panic!("error init hf_hub {:?}", error)
    };

    // Create the tokenizer repo.
    let tokenizer_repo = hf_api.repo(Repo::with_revision(
        args.tokenizer_repo_id,
        RepoType::Model,
        args.tokenizer_repo_revision,
    ));

    // Download and load/init the tokenizer.
    let tokenizer_file = args.tokenizer_file;
    let tokenizer_filename = match tokenizer_repo.get(&tokenizer_file).instrument(info_span!("tokenizer_repo.get")).await {
        Ok(filename) => filename,
        Err(error) => panic!("error attempt fetching the tokenizer {:?}", error)
    };
    let tokenizer = match info_span!("Tokenizer::from_file").in_scope(|| Tokenizer::from_file(tokenizer_filename)) {
        Ok(tokenizer) => tokenizer,
        Err(error) => panic!("error init tokenizer {:?}", error),
    };

    // Create the model repo.
    let model_repo = hf_api.repo(Repo::with_revision(
        args.model_repo_id,
        RepoType::Model,
        args.model_repo_revision,
    ));
    // Download and init the GGUF model weights.
    let model_file = args.model_file;
    let model_filename = match model_repo.get(&model_file).instrument(info_span!("model_repo.get")).await {
        Ok(filename) => filename,
        Err(error) => panic!("error attempt fetching the model file {:?}", error)
    };
    let mut model_file = match info_span!("std::fs::File::open").in_scope(|| std::fs::File::open(&model_filename)) {
        Ok(model_file) => model_file,
        Err(error) => panic!("Failed to open model file {:?}", error)
    };
    let model_content = match info_span!("gguf_file::Content::read").in_scope(|| gguf_file::Content::read(&mut model_file)) {
        Ok(model_content) => model_content,
        Err(error) => panic!("Failed gguf_file::Content::read {:?}", error)
    };
    let model_weights = match info_span!("ModelWeights::from_gguf").in_scope(|| ModelWeights::from_gguf(model_content, &mut model_file)) {
        Ok(model_content) => model_content,
        Err(error) => panic!("Failed creating model weights from gguf {:?}", error)
    };

    let state = state::AppState {
        tokenizer: Arc::new(tokenizer),
        model_weights: Arc::new(Mutex::new(model_weights)),
        seed: args.seed,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
    };

    let app = Router::new()
      .route("/v1/completions", post(completion))
      .with_state(state);

    let port = args.port;
    let address = SocketAddr::from(([0, 0, 0, 0], port));

    info!("Server listening on {}", address);
    match axum::Server::bind(&address)
        .serve(app.into_make_service())
        .await {
            Ok(_) => (),
            Err(error) => panic!("error running the server {:?}", error)
        };
}
