// src/main.rs

use clap::{Parser};
 use clap::builder::ValueParser;

use std::collections::HashSet;

use ndarray:: {Array2, Axis};
use rust_survival::cox::CoxModel;
use rust_survival::data::{SurvivalData, Factor};

use rust_survival::points::Points;
use rust_survival::rsf::{RSFConfig, fit_rsf};

use std::path::PathBuf;
use std::fs;

use plotters::prelude::*;

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

        /// Split data into in a train and test fraction (default 0.7) 
        #[clap(long, default_value = "0.7", value_parser = ValueParser::new(|s: &str| {
            let val: f64 = s.parse().map_err(|_| "Not a valid number")?;
            if val < 0.1 || val > 0.9 {
                Err("Value must be between 0.1 and 0.9")
            } else {
                Ok(val)
            }
        }) )]
        split: f64,

        /// future measurements after diagnosis
        #[clap(short, long, value_delimiter = ',')]
        exclude_cols: Option<Vec<String>>,

        /// Comma-separated categorical columns
        #[clap(short, long, default_value="")]
        categorical: String,

        /// Number of trees for RSF
        #[clap(short, long, default_value="100")]
        n_trees: usize,

        /// Path to factor JSON file
        #[clap(long)]
        factors_file: Option<String>,

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

        /// Summary Stats for one column name
        #[clap( long,)]
        summary:Option<String>
    },

    /// Apply a saved model to new data
    Test {
        /// Path to CSV dataset
        #[clap(short, long)]
        file: String,

        /// Path to factor JSON file
        #[clap(short, long)]
        factors_file: Option<String>,

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
        Command::Train { file, patient_col, time_col, status_col, categorical, exclude_cols, split, n_trees, factors_file, top_n, base_hr, delimiter, model, summary } => {
            run_train(&file, patient_col, &time_col, &status_col, &categorical, exclude_cols, split, n_trees, factors_file,
                top_n, base_hr, delimiter.as_bytes()[0], &model, summary )?;
        }
        Command::Test { file, patient_col, factors_file, delimiter, categorical, base_hr, model, output } => {
            run_test(&file, patient_col, factors_file, delimiter.as_bytes()[0], &categorical, base_hr, &model, output)?;
        }
    }

    Ok(())
}
fn run_train(file: &str, patient_col:Option<String>, time_col: &str, status_col: &str, categorical: &str, exclude_cols:Option<Vec<String>>, 
            split: f64, n_trees: usize, factors_file: Option<String>, top_n: usize, base_hr: f64,
             delimiter: u8, model: &str, summary:Option<String> ) -> Result<(), Box<dyn std::error::Error>> {
//fn main() -> Result<(), Box<dyn std::error::Error>> {
//    let args = Args::parse();

    // --- Parse categorical columns ---
    let mut categorical_cols: HashSet<String> = if categorical.is_empty() {
        HashSet::new()
    } else {
        categorical.split(',').map(|s| s.to_string()).collect()
    };
    categorical_cols.insert( status_col.to_string() );
    let factors_file = match factors_file{
        Some(f) => f,
        None => {
            let mut path = PathBuf::from(file);
            path.set_extension(""); // remove existing extension
            path = path.with_file_name(format!("{}_factors.json", path.file_stem().unwrap().to_string_lossy()));
            format!("{}", path.display())
        }
    };

    let mut output = PathBuf::from(file);
    output.set_extension(""); // remove existing extension
    output = output.with_file_name(format!("{}_as_the_tool_sees_the_data.csv", output.file_stem().unwrap().to_string_lossy()));

    //println!("These columns will be forced to be categorials: {:?}",categorical_cols);
    let delim_byte = delimiter;


    let mut future_cols: HashSet<String> = exclude_cols
        .unwrap_or_default()
        .into_iter()
        .collect();
    
    future_cols.insert( time_col.to_string() );
    future_cols.insert( status_col.to_string() );

    let mut survival_data = SurvivalData::from_file(&file, delim_byte, categorical_cols, &factors_file)?;

    println!("Data loaded: {} patients {} data columns", survival_data.numeric_data.nrows(), survival_data.numeric_data.ncols() -1 );

    let futures:Vec<String> = future_cols.iter().cloned().collect();
    for future_col in futures {
        if let Some(factor) = survival_data.factors.get( &future_col ){
            for add_on in factor.all_column_names() {
                future_cols.insert( add_on );
            }          
        }
    }

    //println!("Data summary directly after read:");
    //survival_data.data_summary();

    let patient_col = patient_col.unwrap_or_else(|| survival_data.headers[0].clone());

    survival_data.filter_na( &time_col );
    let filtered = survival_data.filter_low_var( 1e-20 );
    if filtered > 0 {
        println!("removed {filtered} columns due to low varianze in the data");
    }
    
    //println!("Data summary after filter nas:");
    //survival_data.data_summary();
    
    let mut feature_names: Vec<String> = survival_data.filter_features_by_na( 0.10);
    feature_names.retain(|name| ! ( future_cols.contains(name) || survival_data.exclude.contains( name ) ) );

    //println!("Data summary after filter by na:");
    //survival_data.data_summary();

    let min_common = ((feature_names.len() as f64) * 0.7).ceil() as usize;
    survival_data.impute_knn( 3, min_common, true );

    survival_data.filter_all_na_rows(&feature_names);

    let (train_data, test_data) = survival_data.train_test_split( split );

    println!( "Data split into {} train- and {} test-rows", train_data.numeric_data.nrows(), test_data.numeric_data.nrows());

    //println!("Data summary after filter_all_na_rows:");
    //survival_data.data_summary();
    
    let n_rows = train_data.numeric_data.nrows();

    // --- get the feature Array2 ---
    let feature_array = train_data.as_array2(Some(&feature_names));


    /*for j in 0..feature_array.ncols() {
        // check every data rwo
        let col = feature_array.column(j);
        let unique_vals: HashSet<String> = col.iter().map(|v| v.to_string() ).collect();
        println!("Col {}: {} unique values: top 10 {:?}", feature_names[j], unique_vals.len(), unique_vals.iter().take( unique_vals.len().min(10) ).cloned().collect::<Vec<String>>());
    }*/
    // --- Extract time and status ---
    let time:  Vec<f64> = train_data.as_vec_f64( &time_col );

    let status_raw: Vec<u8> = train_data.as_vec_u8( &status_col );
    let min_val = *status_raw.iter().min().unwrap_or(&0);
    let max_val = *status_raw.iter().max().unwrap_or(&1);

    //println!("Status column range: min={} max={}", min_val, max_val);

    if min_val != 0 {
        panic!("Error: status column minimum is {} (max ={}). Both RFS and Cox models need 0/1 coding.", min_val, max_val);
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

    //println!("Data summary before random forest:");
    //survival_data.data_summary();

    // --- Step 1: RSF ---
    let rsf_config = RSFConfig {
        n_trees: n_trees,
        min_node_size: 20,
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
        println!("{}: {} (importance={:.2})", i+1, col_name, imp);
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

    println!("The cox model:{}", cox_model);


    println!("Get points_map");
    // --- Step 3: Assign points ---
    //let points_map = assign_points(&cox_model, base_hr);

    let pts = Points::new(&cox_model, base_hr);

    println!("Points model:\n{pts}");

    let test_stats = pts.summary(&test_data, &cox_model, summary.as_ref() );
    let train_stats = pts.summary(&train_data, &cox_model, summary.as_ref());

    println!("The results on the training data (n={}):\n{}", train_stats.0, train_data.numeric_data.nrows());
    println!("And here the results on the test data (n={}):\n{}", test_stats.0,  test_data.numeric_data.nrows());

    let mut hazard_file = PathBuf::from(model);
    hazard_file.set_extension(""); // remove existing extension
    hazard_file = hazard_file.with_file_name(format!("{}_hazard_values.svg", hazard_file.file_stem().unwrap().to_string_lossy()));

    Points::plot_raw_summary( &test_stats.1, Some(&train_stats.1), &Factor::new("unknown", false), "raw Hazard Values for the training session", &hazard_file);


    let mut points_file = PathBuf::from(model);
    points_file.set_extension(""); // remove existing extension
    points_file = points_file.with_file_name(format!("{}_points_values.svg", points_file.file_stem().unwrap().to_string_lossy()));

    Points::plot_raw_summary( &test_stats.2, Some(&train_stats.2), &Factor::new("unknown", false),"raw Points Values for the training session", &points_file);


    #[cfg(debug_assertions)]
    {
        let patient_ids = survival_data.as_vec_string( &patient_col );
        let results = cox_model.predict( &feature_array, &feature_names , &points_map, &summary );
        println!("\nidx\thazard\tpoints");

        for  (id, harzard, points) in results.iter(){
            println!("{id}\t{harzard}\t{points}")
        }
    }

    //println!("Data summary before data export:");
    //survival_data.data_summary();
    survival_data.to_file( output, delim_byte );

    println!( "Saving the cox model {:?}",cox_model.to_file( model));
    println!("Cox Model saved to file {}", model);
    Ok(())
}



// --- TEST ---
fn run_test(file: &str, patient_col:Option<String>, factors_file: Option<String>, delimiter: u8, 
    categorical: &str, base_hr: f64, model_file: &str, output: Option<String>) 
    -> Result<(), Box<dyn std::error::Error>> {

    // --- Parse categorical columns ---
    let categorical_cols: HashSet<String> = if categorical.is_empty() {
        HashSet::new()
    } else {
        categorical.split(',').map(|s| s.to_string()).collect()
    };
    
    let delim_byte = delimiter;

    // --- Load CSV ---

    let factors_file = match factors_file{
        Some(f) => f,
        None => {
            let mut path = PathBuf::from(file);
            path.set_extension(""); // remove existing extension
            path = path.with_file_name(format!("{}_factors.json", path.file_stem().unwrap().to_string_lossy()));
            format!("{}", path.display())
        }
    };
    
    let mut survival_data = SurvivalData::from_file(&file, delim_byte, categorical_cols, &factors_file)?;

    let min_common = ((survival_data.headers.len() as f64) * 0.7).ceil() as usize;
    survival_data.impute_knn( 3, min_common, true );

    let patient_col = patient_col.unwrap_or_else(|| survival_data.headers[0].clone());  


    let cox_model = CoxModel::from_file( model_file ).unwrap();

    let feature_array = survival_data.as_array2(Some(&cox_model.coefficients));

    // --- Step 3: Assign points ---
    // let points_map = assign_points(&cox_model, base_hr);

    let patient_ids = survival_data.as_vec_string( &patient_col );

    /*let stats = pts.summary(&survival_data, &cox_model, Some(status_col));
    println!("{stats:#?}");*/
    let pts = Points::new(&cox_model, base_hr);

    // Pass Some(&patient_ids) if you have them, otherwise None.
    pts.save_predictions(
        &cox_model,
        &feature_array,
        &cox_model.coefficients,
        patient_ids.as_deref(),        // patient IDs
        output,   // output file
    )?;

    Ok(())
}
