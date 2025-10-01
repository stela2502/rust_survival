use clap::Parser;
use rust_survival::data::{SurvivalData};

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
    // Parse numeric values as integers
    let numeric_values: Vec<f64> = args.numeric
        .split(',')
        .map(|v| v.parse::<i64>().expect("Invalid numeric value: must be an integer") as f64)
        .collect();

    for (col_name, factor) in data.factors.iter_mut() {
        let matches_factor = match &args.factor {
            Some(pattern) => col_name.contains(pattern),
            None => true,
        };

        if matches_factor {
            if !levels_to_change.is_empty() {
                // Modify only the specified levels
                for (lvl, &val) in levels_to_change.iter().zip(numeric_values.iter()) {
                    if let Some(_idx) = factor.levels.iter().position(|x| x == lvl) {
                        factor.level_to_index.insert(lvl.clone(), val);
                    }
                }
            } else if args.overwrite {
                // Overwrite all levels in order
                for (lvl, &val) in factor.levels.iter().zip(numeric_values.iter()) {
                    factor.level_to_index.insert(lvl.clone(), val);
                }
            }
        }
    }

    
    // Save back to file
    data.save_factors(&args.factors_file)?;
    println!("Factors updated successfully in {}", args.factors_file);
    Ok(())
}