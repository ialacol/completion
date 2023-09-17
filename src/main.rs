use std::sync::{Arc, Mutex};

use axum::{routing::post, Router};
use candle_transformers::models::quantized_llama::ModelWeights;
use clap::Parser;

use candle_core::quantized::gguf_file;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub mod routes;
mod state;
mod token;
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
    let args: Args = Args::parse();
    let addr = args.host.clone() + ":" + args.port.to_string().as_str();

    let tokenizer_api = Api::new().unwrap();
    let tokenizer_repo = tokenizer_api.repo(Repo::with_revision(
        args.tokenizer_repo_id,
        RepoType::Model,
        args.tokenizer_repo_revision,
    ));
    let tokenizer_filename = tokenizer_repo.get(args.tokenizer_file.as_str()).unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

    let model_api = Api::new().unwrap();
    let model_repo = model_api.repo(Repo::with_revision(
        args.model_repo_id,
        RepoType::Model,
        args.model_repo_revision,
    ));
    let model_filename = model_repo.get(args.model_file.as_str()).unwrap();
    let mut model_file = std::fs::File::open(&model_filename).unwrap();
    let model_content = gguf_file::Content::read(&mut model_file).unwrap();
    let model_weights = ModelWeights::from_gguf(model_content, &mut model_file).unwrap();

    let app = Router::new()
        .route("/v1/completions", post(completion))
        .with_state(Arc::new(state::AppState {
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            model_weights: Arc::new(Mutex::new(model_weights)),
            seed: args.seed,
            repeat_last_n: args.repeat_last_n,
            repeat_penalty: args.repeat_penalty,
        }));

    axum::Server::bind(&addr.parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
