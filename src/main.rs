// src/main.rs

use clap::Parser;
use std::collections::HashMap;

mod data;
mod rsf;
mod cox;
mod points;

use rsf::{RSFConfig, fit_rsf};
use points::{assign_points, total_points};
use std::collections::HashSet;
use ndarray:: {Array2, Axis};
use crate::cox::CoxModel;
use crate::data::SurvivalData;


#[derive(Parser, Debug)]
#[clap(author="Rust Survival CLI", version="0.2", about="RSF -> Cox -> Points")]
enum Command {
    /// Train a model from CSV dataset
    Train {
        /// Path to CSV dataset
        #[clap(short, long)]
        file: String,

        /// Name of the Pateient ID column: default first column
        #[clap(short, long)]
        patient_col: Option<String>,

        /// Name of survival time column
        #[clap(short, long)]
        time_col: String,

        /// Name of event status column (0/1)
        #[clap(short, long)]
        status_col: String,

        /// future measurements after diagnosis
        #[clap(short, long, value_delimiter = ',')]
        exclude_cols: Option<Vec<String>>,

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

        /// CSV delimiter
        #[clap(short='d', long, default_value="\t")]
        delimiter: String,

        /// File path to save trained model
        #[clap(short='m', long)]
        model: String,
    },

    /// Apply a saved model to new data
    Test {
        /// Path to CSV dataset
        #[clap(short, long)]
        file: String,

        /// Name of the Pateient ID column: default first column
        #[clap(short, long)]
        patient_col: Option<String>,

        /// Path to saved Cox/RSF model
        #[clap(short='m', long)]
        model: String,

        /// Optional output CSV
        #[clap(short='o', long)]
        output: Option<String>,

        /// CSV delimiter
        #[clap(short='d', long, default_value="\t")]
        delimiter: String,

        /// Comma-separated categorical columns
        #[clap(short, long, default_value="")]
        categorical: String,

        /// Base hazard ratio for 1 point
        #[clap(long, default_value="1.2")]
        base_hr: f64,

    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cmd = Command::parse();

    match cmd {
        Command::Train { file, patient_col, time_col, status_col, categorical, exclude_cols, n_trees, min_node_size, top_n, base_hr, delimiter, model } => {
            run_train(&file, patient_col, &time_col, &status_col, &categorical, exclude_cols, n_trees, min_node_size, 
                top_n, base_hr, delimiter.as_bytes()[0], &model)?;
        }
        Command::Test { file, patient_col, delimiter, categorical, base_hr, model, output } => {
            run_test(&file, patient_col, delimiter.as_bytes()[0], &categorical, base_hr, &model, output)?;
        }
    }

    Ok(())
}
fn run_train(file: &str, patient_col:Option<String>, time_col: &str, status_col: &str, categorical: &str, exclude_cols:Option<Vec<String>>, 
             n_trees: usize, min_node_size: usize, top_n: usize, base_hr: f64,
             delimiter: u8, model: &str) -> Result<(), Box<dyn std::error::Error>> {
//fn main() -> Result<(), Box<dyn std::error::Error>> {
//    let args = Args::parse();

    // --- Parse categorical columns ---
    let categorical_cols: HashSet<String> = if categorical.is_empty() {
        HashSet::new()
    } else {
        categorical.split(',').map(|s| s.to_string()).collect()
    };
    
    let delim_byte = delimiter;


    let mut future_cols: HashSet<String> = exclude_cols
        .unwrap_or_default()
        .into_iter()
        .collect();
    future_cols.insert( time_col.to_string() );
    future_cols.insert( status_col.to_string() );


    let mut survival_data = SurvivalData::from_file(&file, delim_byte, 10)?;
    let patient_col = patient_col.unwrap_or_else(|| survival_data.headers[0].clone());

    survival_data.filter_na( &time_col );
    let filtered = survival_data.filter_low_var( 1e-20 );
    if filtered > 0 {
        println!("removed {filtered} columns dure to low varianze of the data");
    }
    

    
    let mut feature_names: Vec<String> = survival_data.filter_features_by_na( 0.10);
    feature_names.retain(|name| !future_cols.contains(name));

    let min_common = ((feature_names.len() as f64) * 0.7).ceil() as usize;
    survival_data.impute_knn( 3, min_common, true );

    survival_data.filter_all_na_rows();


    let headers = &survival_data.headers;
    let n_rows = survival_data.numeric_data.nrows();
    let n_cols = headers.len();

    let time_idx = headers.iter()
        .position(|h| h == &time_col)
        .ok_or(format!("Time column '{}' not found.\nAvailable column: {:?}", time_col, &survival_data.headers))?;

    let status_idx = headers.iter()
        .position(|h| h == &status_col)
        .ok_or(format!("Status column '{}' not found.\nAvailable column: {:?}", status_col, &survival_data.headers))?;
    

    let n_features = feature_names.len();
    let feature_indices: Vec<usize> = feature_names.iter()
        .filter_map(|name| survival_data.headers.iter().position(|h| h == name))
        .collect();

    // --- get the feature Array2 ---
    let feature_array = survival_data.as_array2(Some(&feature_names));

    // --- Extract time and status ---
    let time:  Vec<f64> = survival_data.as_vec_f64( &time_col );

    let status_raw: Vec<u8> = survival_data.as_vec_u8( &status_col );
    let min_val = *status_raw.iter().min().unwrap_or(&0);
    let max_val = *status_raw.iter().max().unwrap_or(&1);

    println!("Status column range: min={} max={}", min_val, max_val);

    if min_val != 0 {
        eprintln!("Warning: status column minimum is {}. Cox model expects 0/1 coding.", min_val);
    }

    // Now you can safely coerce values > 1 to 1
    let mut status = status_raw.clone();
    for val in status.iter_mut() {
        if *val == min_val{
            *val = 0;
        }else if *val > 1 {
            *val = 1;
        }
    }


    // --- Step 1: RSF ---
    let rsf_config = RSFConfig {
        n_trees: n_trees,
        min_node_size: min_node_size,
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

    let top_n = top_n.min(importance_vec.len());

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
    #[cfg(debug_assertions)]
    {
        println!("Identified {} top features:\n{:?}", top_features.len(), top_features);
        let tmp: Vec<String> = top_indices.iter()
        .map(|&idx| feature_names[idx].clone())   // or survival_data.headers[idx]
        .collect();
        println!("And this is the same from the feature_array colnames:\n{:?}", tmp);
    }
    // --- Prepare Cox matrix (only top features) ---
    let mut cox_matrix = Array2::<f64>::zeros((n_rows, top_indices.len()));
    for (j, &feat_col) in top_indices.iter().enumerate() {
        let column = feature_array.index_axis(Axis(1), feat_col);
        cox_matrix.column_mut(j).assign(&column);
    }

    // --- Step 2: Cox ---
    println!("fit COX model!");
    let cox_model = CoxModel::fit(&cox_matrix, &time, &status, &top_features, 100, 1e-6);

    cox_model.to_file( model);
    println!("The cox model:{}", cox_model);

    println!("Get points_map");
    // --- Step 3: Assign points ---
    let points_map = assign_points(&cox_model, base_hr);

    println!("\nPoints per variable:");
    for (var, pt) in &points_map {
        println!("{} -> {} points", var, pt);
    }

    #[cfg(debug_assertions)]
    {
        let patient_ids = survival_data.as_vec_string( &patient_col );
        let results = cox_model.predict( &feature_array, &feature_names , &points_map, &patient_ids );
        println!("\nidx\thazard\tpoints");

        for  (id, harzard, points) in results.iter(){
            println!("{id}\t{harzard}\t{points}")
        }
    }

    println!("Cox Model saved to file {}", model);
    Ok(())
}



// --- TEST ---
fn run_test(file: &str, patient_col:Option<String>, delimiter: u8, categorical: &str, base_hr: f64, model_file: &str, output: Option<String>) 
    -> Result<(), Box<dyn std::error::Error>> {

    // --- Parse categorical columns ---
    let categorical_cols: HashSet<String> = if categorical.is_empty() {
        HashSet::new()
    } else {
        categorical.split(',').map(|s| s.to_string()).collect()
    };
    
    let delim_byte = delimiter;

    // --- Load CSV ---
    
    let mut survival_data = SurvivalData::from_file(&file, delim_byte, 10)?;

    let min_common = ((survival_data.headers.len() as f64) * 0.7).ceil() as usize;
    survival_data.impute_knn( 3, min_common, true );

    let patient_col = patient_col.unwrap_or_else(|| survival_data.headers[0].clone());  
    let headers = &survival_data.headers;
    let n_rows = survival_data.numeric_data.nrows();
    let n_cols = headers.len();

    let cox_model = CoxModel::from_file( model_file ).unwrap();

    let feature_array = survival_data.as_array2(Some(&cox_model.coefficients));

    // --- Step 3: Assign points ---
    let points_map = assign_points(&cox_model, base_hr);

    let patient_ids = survival_data.as_vec_string( &patient_col );
    let results = cox_model.predict( &feature_array, &cox_model.coefficients , &points_map, &patient_ids);
    println!("\nidx\thazard\tpoints");
    for  (id, harzard, points) in results.iter(){
        println!("{id}\t{harzard}\t{points}")
    }
    Ok(())
}
