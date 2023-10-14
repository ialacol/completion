#[cfg(feature = "accelerate")]
extern crate accelerate_src; // cargo run --release --features accelerate

use candle_core::Device;
use candle_core::Tensor;
use candle_core::utils::get_num_threads;
use candle_core::utils::has_accelerate;

use candle_core::utils::has_mkl;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::MAX_SEQ_LEN;
use tokio::sync::mpsc;
use tracing_subscriber::fmt::format::FmtSpan;

use axum::{routing::post, Router};
use candle_transformers::models::quantized_llama::ModelWeights;
use clap::Parser;

use candle_core::quantized::gguf_file;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub mod routes;
mod state;
mod token;
mod cmd;
mod process;
use std::net::SocketAddr;
use crate::routes::completion::completion;
use crate::token::token_to_text;
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
    /// The seed to use when generating random samples, default to 1.0
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// The seed to use when generating random samples, default to 299792458
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty, default to 64
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
    if has_accelerate() {
        info!("candle was compiled with 'accelerate' support")
    }
    if has_mkl() {
        info!("candle was compiled with 'mkl' support")
    }
    info!("number of thread: {:?} used by candle", get_num_threads());

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
    info!("model_filename: {:?}", model_filename);
    let model_content = match info_span!("gguf_file::Content::read").in_scope(|| gguf_file::Content::read(&mut model_file)) {
        Ok(model_content) => model_content,
        Err(error) => panic!("Failed gguf_file::Content::read {:?}", error)
    };
    let mut model_weights = match info_span!("ModelWeights::from_gguf").in_scope(|| ModelWeights::from_gguf(model_content, &mut model_file)) {
        Ok(model_content) => model_content,
        Err(error) => panic!("Failed creating model weights from gguf {:?}", error)
    };

    let (sender, mut receiver) = mpsc::channel(32);

    let manager = tokio::spawn(async move {
        let seed = args.seed;
        let temperature: f64 = args.temperature;
        let top_p = 1.1;
        let n = 10;
        let repeat_last_n = 64;
        let repeat_penalty = 1.1;
        while let Some(cmd) = receiver.recv().await {
            match cmd {
                // handle Command::Prompt from tx.send().await;
                cmd::Command::Prompt { prompt, responder } => {
                    info!("prompt {:?}", &prompt);
                    let tokens = tokenizer.encode(prompt, true).unwrap();
                    info!("tokenized prompt: {:?}", tokens.get_ids());
                    let pre_prompt_tokens = vec![];
                    let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
                    let to_sample = n;
                    let prompt_tokens = if prompt_tokens.len() + to_sample > MAX_SEQ_LEN - 10 {
                        let to_remove = prompt_tokens.len() + to_sample + 10 - MAX_SEQ_LEN;
                        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
                    } else {
                        prompt_tokens
                    };
                    let mut all_tokens: Vec<u32> = vec![];
                    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));
                    let mut next_token = {
                        let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
                        let logits = model_weights.forward(&input, 0).unwrap();
                        let logits = logits.squeeze(0).unwrap();
                        logits_processor.sample(&logits).unwrap()
                    };
                    all_tokens.push(next_token);
                    info!("to_sample: {:?}", to_sample);
                    for index in 0..to_sample {
                        let input = Tensor::new(&[next_token], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
                        let logits = model_weights.forward(&input, prompt_tokens.len() + index).unwrap();
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
                        info!("text: {:?}", text);
                        responder.send((text).to_string()).await.unwrap();
                    }
                }
            }
        }
    });

    let state = state::AppState { sender };
    let app = Router::new()
      .route("/v1/completions", post(completion))
      .with_state(state);

    let port = args.port;
    let address = SocketAddr::from(([0, 0, 0, 0], port));

    tokio::select! {
        _ = axum::Server::bind(&address).serve(app.into_make_service()) => {
            info!("Server stopped.");
        }
        _ = manager => {
            info!("Prompt processing manager stops.");
        }
    }
}
