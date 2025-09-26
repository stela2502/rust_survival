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
use ndarray:: {Array2, Axis};


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

    // --- Parse categorical columns ---
    let categorical_cols: HashSet<String> = if args.categorical.is_empty() {
        HashSet::new()
    } else {
        args.categorical.split(',').map(|s| s.to_string()).collect()
    };
    
    let delim_byte = args.delimiter.as_bytes()[0];

    // --- Load CSV ---
    let (headers, data) = load_csv(
        &args.file,
        &args.time_col,
        &args.status_col,
        &categorical_cols,
        delim_byte
    )?;

    let n_rows = data.len();
    let n_cols = headers.len();

    // --- Identify time and status columns ---
    let time_idx = headers.iter().position(|h| h == &args.time_col)
        .ok_or_else(|| format!("Time column '{}' not found\nAvailable colnames: {:?}", args.time_col, headers))?;
    let status_idx = headers.iter().position(|h| h == &args.status_col)
        .ok_or_else(|| format!("Status column '{}' not found\nAvailable colnames: {:?}", args.status_col, headers))?;

    // --- Feature columns: all except time/status ---
    let feature_indices: Vec<usize> = (0..n_cols)
        .filter(|&i| i != time_idx && i != status_idx)
        .collect();

    let n_features = feature_indices.len();

    // --- Build feature Array2 ---
    let mut feature_array = Array2::<f64>::zeros((n_rows, n_features));
    for (i, row) in data.iter().enumerate() {
        for (j, &col_idx) in feature_indices.iter().enumerate() {
            feature_array[[i, j]] = row[col_idx];
        }
    }

    // --- Feature names ---
    let feature_names: Vec<String> = feature_indices.iter().map(|&i| headers[i].clone()).collect();

    // --- Extract time and status ---
    let time: Vec<f64> = data.iter().map(|r| r[time_idx]).collect();
    let status: Vec<u8> = data.iter().map(|r| r[status_idx] as u8).collect();

    // --- Step 1: RSF ---
    let rsf_config = RSFConfig {
        n_trees: args.n_trees,
        min_node_size: args.min_node_size,
        max_features: None,
        seed: 42,
    };

    let rsf_model = fit_rsf(&feature_array, &time, &status, &rsf_config);

    // --- Sort by importance ---
    let mut importance_vec: Vec<(String, f64)> = rsf_model.feature_importance
        .values()
        .enumerate()
        .map(|(idx, &val)| (feature_names[idx].clone(), val))
        .collect();
    importance_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_n = args.top_n.min(importance_vec.len());

    println!("Top RSF variables:");
    for (i, (col_name, imp)) in importance_vec.iter().take(top_n).enumerate() {
        println!("{}: {} (importance={})", i+1, col_name, imp);
    }

    // --- Select top features and map to column indices in feature_array ---
    let top_features: Vec<String> = importance_vec.iter().take(top_n).map(|(name, _)| name.clone()).collect();
    let top_indices: Vec<usize> = top_features.iter()
        .map(|feat| feature_names.iter().position(|h| h == feat)
            .expect("Top feature not found in feature_names"))
        .collect();

    // --- Prepare Cox matrix (only top features) ---
    let mut cox_matrix = Array2::<f64>::zeros((n_rows, top_indices.len()));
    for (j, &feat_col) in top_indices.iter().enumerate() {
        let column = feature_array.index_axis(Axis(1), feat_col);
        cox_matrix.column_mut(j).assign(&column);
    }

    // --- Step 2: Cox ---
    println!("Still alive!");
    let cox_model = fit_cox(&cox_matrix, &time, &status, &top_features, 100, 1e-6);

    // --- Step 3: Assign points ---
    let points_map = assign_points(&cox_model, args.base_hr);

    println!("\nPoints per variable:");
    for (var, pt) in &points_map {
        println!("{} -> {} points", var, pt);
    }
    let results = cox_model.predict( &feature_array, &feature_names , &points_map);
    println!("\nidx\thazard\tpoints");
    for (idx, (harzard, points)) in results.iter().enumerate(){
        println!("{idx}\t{harzard}\t{points}")
    }
    Ok(())
}

