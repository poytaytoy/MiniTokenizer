use std::time::Instant;
use token_coder::encode::TokenEncoder;

fn main() {
    println!("Loading TokenEncoder...");
    let mut encoder = TokenEncoder::new("test_tokenizer.json").expect("Failed to load tokenizer");
    
    // 1. Benchmark short text (naive merge)
    let short_text = "hello world. this is a short sentence designed to benchmark BPE encoding speed.";
    println!("\nBenchmarking short text (uses naive_merge):");
    println!("Text length: {} bytes", short_text.len());
    
    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encoder.encode(short_text);
    }
    let duration = start.elapsed();
    let avg_duration = duration / iterations;
    println!("Total time for {} iterations: {:?}", iterations, duration);
    println!("Average time per encode: {:?}", avg_duration);
    println!("Throughput: {:.2} MB/s", (short_text.len() * iterations as usize) as f64 / duration.as_secs_f64() / 1_000_000.0);

    // 2. Benchmark long text (optimized merge)
    let long_text = "ab".repeat(1200);
    println!("\nBenchmarking long repeating text (uses optimized_merge):");
    println!("Text length: {} bytes", long_text.len());
    
    let iterations_long = 1_000;
    let start_long = Instant::now();
    for _ in 0..iterations_long {
        let _ = encoder.encode(&long_text);
    }
    let duration_long = start_long.elapsed();
    let avg_duration_long = duration_long / iterations_long;
    println!("Total time for {} iterations: {:?}", iterations_long, duration_long);
    println!("Average time per encode: {:?}", avg_duration_long);
    println!("Throughput: {:.2} MB/s", (long_text.len() * iterations_long as usize) as f64 / duration_long.as_secs_f64() / 1_000_000.0);
}
