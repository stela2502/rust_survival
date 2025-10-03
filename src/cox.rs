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
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;



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
            writeln!(f, "hr_values  {} -> {:.4}", feat, hr)?;
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
    ) -> CoxModel {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut beta = Array1::<f64>::zeros(n_features);

        // Sort indices ascending by time
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| time[i].partial_cmp(&time[j]).unwrap());

        let mut final_hessian = Array2::<f64>::zeros((n_features, n_features));

        // Precompute safe mask for valid rows (no NaNs)
        let valid_mask: Vec<bool> = (0..n_samples)
            .map(|i| (0..n_features).all(|j| !data[[i,j]].is_nan()))
            .collect();

        let pb = ProgressBar::new(max_iter as u64);
        pb.set_style(
            ProgressStyle::with_template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );
        pb.set_prefix("Cox iterations");

        let max_step = 1.0;           // maximum allowed delta per iteration
        let min_risk_set = 3;         // minimum patients in risk set

        for _ in 0..max_iter {
            pb.inc(1);

            let mut gradient = Array1::<f64>::zeros(n_features);
            let mut hessian = Array2::<f64>::zeros((n_features, n_features));

            // Parallel contributions
            let contributions: Vec<_> = indices.par_iter()
                .filter(|&&i| status[i] != 0 && valid_mask[i])
                .filter_map(|&i| {
                    // Risk set: patients with time >= time[i] and valid
                    let risk_indices: Vec<usize> = (0..n_samples)
                        .filter(|&j| valid_mask[j] && time[j] >= time[i])
                        .collect();

                    if risk_indices.len() < min_risk_set {
                        return None; // skip tiny risk sets
                    }

                    // Compute exp(x beta) safely
                    let xb_risk: Vec<f64> = risk_indices.iter().map(|&j| {
                        let xb = data.row(j).dot(&beta);
                        xb.clamp(-700.0, 700.0).exp()
                    }).collect();

                    let sum_exp: f64 = xb_risk.iter().sum();
                    if sum_exp == 0.0 {
                        return None; // avoid division by zero
                    }

                    // Weighted mean of covariates
                    let mut weighted_mean = Array1::<f64>::zeros(n_features);
                    for (k, &j) in risk_indices.iter().enumerate() {
                        weighted_mean += &(&data.row(j) * xb_risk[k]);
                    }
                    weighted_mean /= sum_exp;

                    // Gradient contribution
                    let mut grad = &data.row(i) - &weighted_mean;
                    grad.mapv_inplace(|x| x.clamp(-100.0, 100.0)); // clip extremes

                    // Hessian contribution
                    let mut hess = Array2::<f64>::zeros((n_features, n_features));
                    for (k, &j) in risk_indices.iter().enumerate() {
                        let diff = &data.row(j) - &weighted_mean;
                        let diff_clipped = diff.mapv(|x| x.clamp(-100.0, 100.0));
                        hess += &(diff_clipped.view().insert_axis(Axis(1)).dot(&diff_clipped.view().insert_axis(Axis(0))) * xb_risk[k]);
                    }
                    hess /= sum_exp;

                    Some((grad, hess))
                })
                .collect();

            // Reduce contributions
            for (g, h) in contributions {
                gradient += &g;
                hessian += &h;
            }

            // Regularize Hessian for stability
            let hessian_reg = &hessian + Array2::eye(n_features) * 1e-4;

            // Newton-Raphson step
            let delta = hessian_reg.solve_into(gradient.clone()).unwrap();
            let delta_clipped = delta.mapv(|x| x.clamp(-max_step, max_step));
            beta += &delta_clipped;

            final_hessian.assign(&hessian);

            // Check convergence (both sum and max)
            if delta_clipped.iter().map(|x| x.abs()).sum::<f64>() < tol &&
               delta_clipped.iter().cloned().fold(0./0., f64::max).abs() < tol
            {
                break;
            }
        }

        pb.finish_with_message("done");

        // Prepare output
        let mut hr = HashMap::new();
        let mut se = HashMap::new();

        for (i, name) in feature_names.iter().enumerate() {
            hr.insert(name.clone(), beta[i].exp());
        }

        if let Ok(inv_h) = final_hessian.clone().inv() {
            for (i, name) in feature_names.iter().enumerate() {
                let val = inv_h[[i,i]];
                if val > 0.0 {
                    se.insert(name.clone(), val.sqrt());
                } else {
                    se.insert(name.clone(), 0.0);
                }
            }
        }

        CoxModel {
            coefficients: feature_names.clone(),
            hr,
            se,
        }
    }
/*
    pub fn fit(
        data: &Array2<f64>,
        time: &Vec<f64>,
        status: &Vec<u8>,
        feature_names: &Vec<String>,
        max_iter: usize,
        tol: f64,
    ) -> CoxModel {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut beta = Array1::<f64>::zeros(n_features);

        // Sort indices ascending by time
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

        // Check for NaNs before starting
        for i in 0..n_samples {
            for j in 0..n_features {
                if data[[i,j]].is_nan() {
                    println!("NaN at row {}, col {}", i, j);
                }
            }
        }

        for _ in 0..max_iter {
            pb.inc(1);

            let mut gradient = Array1::<f64>::zeros(n_features);
            let mut hessian = Array2::<f64>::zeros((n_features, n_features));

            // Parallel computation of contributions
            let contributions: Vec<_> = indices.par_iter()
                .filter(|&&i| status[i] != 0)
                .map(|&i| {
                    // Risk set: all patients with time >= time[i]
                    let risk_indices: Vec<usize> = (0..n_samples).filter(|&j| time[j] >= time[i]).collect();
                    if risk_indices.is_empty() { return None; }

                    // Compute exp(x beta) safely
                    let xb_risk: Vec<f64> = risk_indices.iter().map(|&j| {
                        let xb = data.row(j).dot(&beta);
                        xb.clamp(-700.0, 700.0).exp()
                    }).collect();

                    let sum_exp: f64 = xb_risk.iter().sum();

                    // Weighted mean of covariates
                    let mut weighted_mean = Array1::<f64>::zeros(n_features);
                    for (k, &j) in risk_indices.iter().enumerate() {
                        weighted_mean += &(&data.row(j) * xb_risk[k]);
                    }
                    weighted_mean /= sum_exp;

                    // Gradient contribution
                    let xi = data.row(i).to_owned();
                    let grad = &xi - &weighted_mean;

                    // Hessian contribution
                    let mut hess = Array2::<f64>::zeros((n_features, n_features));
                    for (k, &j) in risk_indices.iter().enumerate() {
                        let xj = data.row(j).to_owned();
                        let diff = &xj - &weighted_mean;
                        hess += &(diff.view().insert_axis(Axis(1)).dot(&diff.view().insert_axis(Axis(0))) * xb_risk[k]);
                    }
                    hess /= sum_exp;

                    Some((grad, hess))
                })
                .filter_map(|x| x)
                .collect();

            // Reduce into final gradient and Hessian
            for (g, h) in contributions {
                gradient += &g;
                hessian += &h;
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
        let mut hr = HashMap::new();
        let mut se = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            hr.insert(name.clone(), beta[i].exp());
        }

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
    }*/

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