// src/cox.rs

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve; // for linear algebra
use std::collections::HashMap;

use std::fs::{self, File};
use std::path::PathBuf;
use std::io::{BufWriter};
use serde::{Serialize, Deserialize};
use serde_json;
use indicatif::{ProgressBar, ProgressStyle};

use ndarray_linalg::Inverse;

/// Cox model result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxModel {
    pub coefficients: Vec<String>,
    pub hr: HashMap<String, f64>,          // hazard ratios
    pub se: HashMap<String, f64>,          // standard errors
}

use std::fmt;

impl fmt::Display for CoxModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CoxModel:")?;
        writeln!(f, "Features: {:?}", self.coefficients)?;
        writeln!(f, "Hazard ratios (HR):")?;
        for feat in &self.coefficients {
            let hr = self.hr.get(feat).unwrap_or(&0.0);
            writeln!(f, "  {} -> {:.4}", feat, hr)?;
        }
        Ok(())
    }
}


impl CoxModel {

    /// Fit a Cox proportional hazards model
    /// 
    /// # Arguments
    /// * `data` - feature matrix (n_samples x n_features)
    /// * `time` - survival times
    /// * `status` - 0/1 event indicator
    /// * `feature_names` - names of columns
    /// * `max_iter` - maximum iterations for Newton-Raphson
    /// * `tol` - tolerance for convergence
    pub fn fit(
        data: &Array2<f64>,
        time: &Vec<f64>,
        status: &Vec<u8>,
        feature_names: &Vec<String>,
        max_iter: usize,
        tol: f64,
    ) -> Self {

        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut beta = Array1::<f64>::zeros(n_features);
        let _status_f: Array1<f64> = Array1::from(status.iter().map(|&x| x as f64).collect::<Vec<f64>>());

        // Sort ascending by time
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| time[i].partial_cmp(&time[j]).unwrap());

        let mut final_hessian = Array2::<f64>::zeros((n_features, n_features));

        let pb = ProgressBar::new(max_iter as u64);
        pb.set_style(
            ProgressStyle::with_template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        pb.set_prefix("Cox iterations");

        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                if data[[i,j]].is_nan() {
                    println!("NaN at row {}, col {}", i, j);
                }
            }
        }
        for _ in 0..max_iter {
            pb.inc(1);

            let mut gradient = Array1::<f64>::zeros(n_features);
            let mut hessian = Array2::<f64>::zeros((n_features, n_features));

            for &i in &indices {
                if status[i] == 0 {
                    continue; 
                }

                // Risk set: all with time >= time[i]
                let risk_indices: Vec<usize> = (0..n_samples).filter(|&j| time[j] >= time[i]).collect();
                if risk_indices.is_empty() { 
                    println!("Risk index id empty! {i}");
                    continue; 
                }

                // Compute exp(x beta) safely
                let xb_risk: Vec<f64> = risk_indices.iter().map(|&j| {
                    let xb = data.row(j).dot(&beta);
                    if xb.is_nan() || xb > 700.0 { 700.0 } // clamp to avoid overflow
                    else if xb < -700.0 { -700.0 } 
                    else { xb }
                }).map(|x| x.exp()).collect();



                //println!("xb_risk for patient {i} == {:?}",xb_risk);

                let sum_exp: f64 = xb_risk.iter().sum();

                // Weighted mean of covariates
                let mut weighted_mean = Array1::<f64>::zeros(n_features);
                for (k, &j) in risk_indices.iter().enumerate() {
                    weighted_mean += &(&data.row(j) * xb_risk[k]);
                }
                weighted_mean /= sum_exp;

                // Gradient
                let xi = data.row(i).to_owned();
                gradient += &(xi - &weighted_mean);

                // Hessian
                let mut outer_sum = Array2::<f64>::zeros((n_features, n_features));
                for (k, &j) in risk_indices.iter().enumerate() {
                    let xj = data.row(j).to_owned();
                    let diff = &xj - &weighted_mean;
                    outer_sum += &(diff.view().insert_axis(Axis(1)).dot(&diff.view().insert_axis(Axis(0))) * xb_risk[k]);
                }
                hessian.scaled_add(1.0 / sum_exp, &outer_sum);
            }

            // Regularize Hessian for stability
            let hessian_reg = &hessian + Array2::eye(n_features) * 1e-8;

            // Newton-Raphson step
            let delta = hessian_reg.solve_into(gradient.clone()).unwrap();
            beta += &delta;

            final_hessian.assign(&hessian);

            if delta.iter().map(|x| x.abs()).sum::<f64>() < tol { break; }
        }
        pb.finish_with_message("done");

        // Prepare outputs
        let mut coefficients= Vec::with_capacity(feature_names.len());
        let mut hr = HashMap::new();
        let mut se = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            coefficients.push(name.clone());
            hr.insert(name.clone(), beta[i].exp());
        }

        // Standard errors from inverted Hessian
        if let Ok(inv_h) = final_hessian.clone().inv() {
            for (i, name) in feature_names.iter().enumerate() {
                se.insert(name.clone(), inv_h[[i, i]].sqrt());
            }
        }

        CoxModel {
            coefficients: feature_names.clone(),
            hr,
            se,
        }
    }

    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let output = PathBuf::from(path.as_ref());
        if let Some(parent) = output.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?; // creates all missing parent directories
            }
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let _ = serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let model = serde_json::from_reader(file)?;
        Ok(model)
    }
}