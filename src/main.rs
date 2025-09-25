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
use std::collections::HashSet;
use ndarray::Array2;

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
    #[clap(short='d', long, default_value="\t")]
    delimiter: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let categorical_cols: HashSet<String> = if args.categorical.is_empty() {
        HashSet::new()
    } else {
        args.categorical.split(',').map(|s| s.to_string()).collect()
    };
    
    let delim_byte = args.delimiter.as_bytes()[0];

    // Load CSV
    let (headers, data) = load_csv(
        &args.file,
        &args.time_col,
        &args.status_col,
        &categorical_cols,
        delim_byte
    )?;

    let n_rows = data.len();
    let n_cols = headers.len();

    let mut array = Array2::<f64>::zeros((n_rows, n_cols));

    for i in 0..n_rows {
        for j in 0..n_cols {
            array[[i, j]] = data[i][j];
        }
    }


    // Identify time and status columns
    let time_idx = headers.iter().position(|h| h == &args.time_col)
        .ok_or(format!("Time column '{}' not found", args.time_col))?;
    let status_idx = headers.iter().position(|h| h == &args.status_col)
        .ok_or(format!("Status column '{}' not found", args.status_col))?;

    // Extract time, status, and features
    let time: Vec<f64> = data.iter().map(|r| r[time_idx]).collect();
    let status: Vec<u8> = data.iter().map(|r| r[status_idx] as u8).collect();

    // Features: all columns except time/status
    let feature_indices: Vec<usize> = (0..n_cols)
        .filter(|&i| i != time_idx && i != status_idx)
        .collect();

    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for row in &data {
        let mut feat_row = Vec::with_capacity(feature_indices.len());
        for &i in &feature_indices {
            feat_row.push(row[i]);
        }
        features.push(feat_row);
    }

    // --- Step 1: RSF ---
    let rsf_config = RSFConfig {
        n_trees: args.n_trees,
        min_node_size: args.min_node_size,
        max_features: None,
        seed: 42,
    };

    let rsf_model = fit_rsf(&array, &time, &status, &rsf_config);

    // Sort by importance
    let mut importance_vec: Vec<(usize, f64)> = rsf_model.feature_importance
        .values()
        //.iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val))  // <- dereference the f64
        .collect();
    importance_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_n = args.top_n.min(importance_vec.len());
    println!("Top RSF variables:");
    for (i, (idx, imp)) in importance_vec.iter().take(args.top_n).enumerate() {
        let col_name = &headers[feature_indices[*idx]];
        println!("{}: {} (importance={})", i+1, col_name, imp);
    }

    // Select top features
    let top_indices: Vec<usize> = importance_vec.iter().take(args.top_n).map(|(idx, _)| *idx).collect();
    let top_features: Vec<String> = top_indices.iter().map(|&i| headers[feature_indices[i]].clone()).collect();

    // Prepare Cox matrix
    let mut cox_matrix = ndarray::Array2::<f64>::zeros((n_rows, top_features.len()));
    for (j, &feat_idx) in top_indices.iter().enumerate() {
        for i in 0..n_rows {
            cox_matrix[[i, j]] = features[i][feat_idx];
        }
    }

    // --- Step 2: Cox ---
    let cox_model = fit_cox(&cox_matrix, &time, &status, &top_features, 100, 1e-6);

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

    // --- Example: total points for first patient ---
    let mut patient: HashMap<String, f64> = HashMap::new();
    for (j, &feat_idx) in top_indices.iter().enumerate() {
        patient.insert(top_features[j].clone(), features[0][feat_idx]);
    }

    let total = total_points(&patient, &points_map);
    println!("\nTotal points for first patient: {}", total);

    Ok(())
}
