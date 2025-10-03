use clap::Parser;
use rust_survival::data::{SurvivalData, Factor};

/// Modify numeric mappings in factor JSON files
#[derive(Parser, Debug)]
#[clap(author="Rust Survival CLI", version="0.1", about="Modify factor JSON files")]
struct Args {
    /// Path to factor JSON file
    #[clap(short, long)]
    factors_file: String,

    /// Optional factor name pattern to match (substring)
    #[clap(short, long)]
    factor: Option<String>,

    /// Comma-separated levels to modify
    #[clap(short, long, default_value="")]
    levels: String,

    /// Comma-separated numeric values to assign
    #[clap(short, long)]
    numeric: String,

    /// Overwrite all numeric values for matching factors
    #[clap(long)]
    overwrite: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut data = SurvivalData::default();
    data.load_factors(&args.factors_file)?;

    let levels_to_change: Vec<String> = if args.levels.is_empty() {
        vec![]
    } else {
        args.levels.split(',').map(|s| s.to_string()).collect()
    };
    let optional_levels = if levels_to_change.is_empty() {
        None
    }else {
        Some(levels_to_change.as_slice())
    };
    // Parse numeric values as integers
    let numeric_values: Vec<f64> = args.numeric
        .split(',')
        .map(|v| v.parse::<f64>().expect("Invalid numeric value") as f64)
        .collect();

    for (col_name, factor) in data.factors.iter_mut() {
        let matches_factor = match &args.factor {
            Some(pattern) => col_name.contains(pattern),
            None => true,
        };

        if matches_factor {
            factor.modify_levels( &numeric_values, optional_levels ).unwrap();
        }
    }

    
    // Save back to file
    data.save_factors(&args.factors_file)?;
    println!("Factors updated successfully in {}", args.factors_file);
    Ok(())
}