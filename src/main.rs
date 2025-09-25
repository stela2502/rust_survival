// src/main.rs

use clap::Parser;
use std::collections::HashMap;

mod data;
mod rsf;
mod cox;
mod points;

use data::load_csv;
use rsf::{RSFConfig, fit_rsf};
use cox::fit_cox;
use points::{assign_points, total_points};

/// CLI arguments
#[derive(Parser, Debug)]
#[clap(author="Rust Survival CLI", version="0.1", about="RSF -> Cox -> Points")]
struct Args {
    /// Path to CSV dataset
    #[clap(short, long)]
    file: String,

    /// Name of survival time column
    #[clap(short, long)]
    time_col: String,

    /// Name of event status column (0/1)
    #[clap(short, long)]
    status_col: String,

    /// Comma-separated categorical columns
    #[clap(short, long, default_value="")]
    categorical: String,

    /// Number of trees for RSF
    #[clap(short, long, default_value="100")]
    n_trees: usize,

    /// Minimum node size in RSF trees
    #[clap(short, long, default_value="5")]
    min_node_size: usize,

    /// Number of top variables to select for Cox
    #[clap(long, default_value="5")]
    top_n: usize,

    /// Base hazard ratio for 1 point
    #[clap(long, default_value="1.2")]
    base_hr: f64,

    /// CSV delimiter: default is tab (\t)
    #[arg(short = 'd', long, default_value = "\t")]
    delimiter: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let categorical_cols: Vec<String> = if args.categorical.is_empty() {
        vec![]
    } else {
        args.categorical.split(',').map(|s| s.to_string()).collect()
    };

    let delim_byte = args.delimiter.as_bytes()[0];

    // Load dataset
    let (header, data) = load_csv(&args.file, &args.time_col, &args.status_col, categorical_cols, delim_byte)?;

    // --- Step 1: RSF ---
    let rsf_config = RSFConfig {
        n_trees: args.n_trees,
        min_node_size: args.min_node_size,
        max_features: None,
        seed: 42,
    };

    let time_idx = feature_names.iter().position(|n| n == time_col)
        .ok_or("Time column not found")?;
    let status_idx = feature_names.iter().position(|n| n == status_col)
        .ok_or("Status column not found")?;

    let top_n = args.top_n.min( headr.len() - 2 );
    
    let time: Vec<f64> = data.rows.iter().map(|r| r[time_idx]).collect();
    
    let status: Vec<f64> = data.rows.iter().map(|r| r[status_idx]).collect();
    
    let rsf_model = fit_rsf(&data.features, &time, &status, &rsf_config);

    // Sort features by importance
    let mut feature_importance: Vec<(&usize, &f64)> = rsf_model.feature_importance.iter().collect();
    feature_importance.sort_by(|a,b| b.1.partial_cmp(a.1).unwrap());

    println!("Top RSF variables (by importance):");
    for (i, (idx, imp)) in feature_importance.iter().enumerate().take(args.top_n) {
        println!("{}: {} (importance={})", i+1, header[**idx], imp);
    }

    // Select top variables
    let top_indices: Vec<usize> = feature_importance.iter().take(top_n).map(|(idx, _)| **idx).collect();
    let top_features: Vec<String> = top_indices.iter().map(|&i| header[i].clone()).collect();

    // Prepare feature matrix for Cox
    let mut cox_data = ndarray::Array2::<f64>::zeros((data.nrows(), top_features.len()));
    for (j, &idx) in top_indices.iter().enumerate() {
        for i in 0..data.features.nrows() {
            cox_data[[i,j]] = data.features[[i,idx]];
        }
    }

    // --- Step 2: Cox ---
    let cox_model = fit_cox(&cox_data, &time, &status, &top_features, 100, 1e-6);

    println!("\nCox model hazard ratios:");
    for var in &top_features {
        let hr = cox_model.hr.get(var).unwrap();
        println!("{} -> HR = {:.3}", var, hr);
    }

    // --- Step 3: Assign points ---
    let points_map = assign_points(&cox_model, args.base_hr);

    println!("\nPoints per variable:");
    for (var, pt) in &points_map {
        println!("{} -> {} points", var, pt);
    }

    // --- Optional: calculate total points for a new patient ---
    // Example: use first patient in dataset
    let mut patient: HashMap<String,f64> = HashMap::new();
    for (j, &idx) in top_indices.iter().enumerate() {
        patient.insert(data.feature_names[idx].clone(), data.features[[0,j]]);
    }

    let total = total_points(&patient, &points_map);
    println!("\nTotal points for first patient: {}", total);

    Ok(())
}

