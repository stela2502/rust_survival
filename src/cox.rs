// src/cox.rs

use ndarray::{Array1, Array2, Axis, ArrayView1};
use ndarray_linalg::Solve; // for linear algebra
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write, Read};
use bincode::{Encode, Decode, config};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

/// Cox model result
#[derive(Debug, Encode, Decode)]
pub struct CoxModel {
    pub coefficients: Vec<String>,
    pub hr: HashMap<String, f64>,          // hazard ratios
    pub se: HashMap<String, f64>,          // standard errors
}

impl CoxModel {
    /// Predict relative risk (exp(hazard)) and total points for a new dataset
    pub fn predict(
        &self,
        new_data: &Array2<f64>,
        headers: &[String],
        points_map: &HashMap<String, i32>,
    ) -> Vec<(f64, i32)> {
        let n_rows = new_data.nrows();
        let mut results = Vec::with_capacity(n_rows);

        // Map feature name -> column index
        let feature_indices: Vec<usize> = self.coefficients.iter()
            .map(|feat| headers.iter().position(|h| h == feat)
                .expect("Feature missing in headers"))
            .collect();

        for i in 0..n_rows {
            // Linear predictor: sum β_i * x_i, β_i = ln(HR_i)
            let hazard: f64 = self.coefficients.iter().enumerate()
                .map(|(j, feat)| {
                    let beta = self.hr.get(feat).map(|hr| hr.ln()).unwrap_or(0.0);
                    let x = new_data[[i, feature_indices[j]]];
                    beta * x
                })
                .sum();

            let relative_risk = hazard.exp();

            // Total points
            let total_points: i32 = self.coefficients.iter()
                .map(|feat| *points_map.get(feat).unwrap_or(&0))
                .sum();

            results.push((relative_risk, total_points));
        }

        results
    }
    /// Save model to a binary file using bincode
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let config = config::standard();
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::encode_into_std_write(self, &mut writer, config)?;
        Ok(())
    }

    /// Load model from a binary file using bincode
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config::standard();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let (model, _len) : (Self, usize) = bincode::decode_from_std_read(&mut reader, config)?;
        Ok(model)
    }
}

/// Fit a Cox proportional hazards model
/// 
/// # Arguments
/// * `data` - feature matrix (n_samples x n_features)
/// * `time` - survival times
/// * `status` - 0/1 event indicator
/// * `feature_names` - names of columns
/// * `max_iter` - maximum iterations for Newton-Raphson
/// * `tol` - tolerance for convergence
pub fn fit_cox(
    data: &Array2<f64>,
    time: &Vec<f64>,
    status: &Vec<u8>,
    feature_names: &Vec<String>,
    max_iter: usize,
    tol: f64,
) -> CoxModel {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    let mut beta = Array1::<f64>::zeros(n_features); // initial coefficients

    // Convert status to f64
    let status_f: Array1<f64> = Array1::from(status.iter().map(|&x| x as f64).collect::<Vec<f64>>());

    // Sort by descending time
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_by(|&i, &j| {
        match time[j].partial_cmp(&time[i]) {
            Some(ordering) => ordering,
            None => std::cmp::Ordering::Equal, // or choose Less/Greater to push NaNs to one side
        }
    });
    let mut final_hessian = Array2::<f64>::zeros((n_features, n_features));

    let pb = ProgressBar::new(max_iter as u64);
    pb.set_style(
        ProgressStyle::with_template(
            // a full-width bar with percent and counter
            "{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)"
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ") // smoother fill
    );
    pb.set_prefix("Cox iterations");

    for _ in 0..max_iter {
        pb.inc(1);
        let mut gradient = Array1::<f64>::zeros(n_features);
        let mut hessian  = Array2::<f64>::zeros((n_features, n_features));
        let mut log_likelihood = 0.0;

        for &i in &indices {
            // skip censored or bad rows
            if status[i] == 0 || time[i].is_nan() || data.row(i).iter().any(|v| v.is_nan()) {
                continue;
            }

            // risk set: subjects with time >= time[i] and finite covariates
            let risk_indices: Vec<usize> = (0..n_samples)
                .filter(|&j| {
                    !time[j].is_nan()
                        && time[j] >= time[i]
                        && data.row(j).iter().all(|v| v.is_finite())
                })
                .collect();
            if risk_indices.is_empty() { continue; }

            // exp(xβ) for risk set
            let mut xb_risk: Vec<f64> = Vec::with_capacity(risk_indices.len());
            let mut valid_rows: Vec<ArrayView1<f64>> = Vec::with_capacity(risk_indices.len());
            let mut valid_indices: Vec<usize> = Vec::with_capacity(risk_indices.len());
            for &j in &risk_indices {
                let row = data.row(j);
                if row.iter().any(|v| !v.is_finite()) || beta.iter().any(|v| !v.is_finite()) {
                    continue; // skip rows with NaN
                }
                let xb = row.dot(&beta).exp();
                if xb.is_finite() {
                    xb_risk.push(xb);
                    valid_rows.push(row);
                    valid_indices.push(j);
                }
            }
            if xb_risk.is_empty() { continue; }

            let mut weighted_mean = Array1::<f64>::zeros(n_features);
            for (k, row) in valid_rows.iter().enumerate() {
                weighted_mean += &row.mapv(|x| x * xb_risk[k]); // safe: row and xb are valid
            }
            let sum_exp: f64 = xb_risk.iter().sum();
            weighted_mean /= sum_exp;

            // gradient
            let xi = data.row(i).to_owned();
            for f in 0..n_features {
                let x_val = xi[f];
                let w_val = weighted_mean[f];
                if x_val.is_finite() && w_val.is_finite() {
                    gradient[f] += x_val - w_val;
                }
            }

            // Hessian
            let mut outer_sum = Array2::<f64>::zeros((n_features, n_features));
            for (k, &j) in risk_indices.iter().enumerate() {
                let row = data.row(j);
                let mut diff_masked = Array1::<f64>::zeros(n_features);

                for f in 0..n_features {
                    let val = row[f] - weighted_mean[f];
                    if val.is_finite() {
                        diff_masked[f] = val;
                    } else {
                        diff_masked[f] = 0.0; // or skip, but 0.0 works for outer product
                    }
                }

                let dv = diff_masked.view();
                outer_sum += &(dv.insert_axis(Axis(1)).dot(&dv.insert_axis(Axis(0))) * xb_risk[k]);
            }
            hessian.scaled_add(1.0 / sum_exp, &outer_sum);

            // log-likelihood
            let xb = xi.dot(&beta);
            if xb.is_finite() {
                log_likelihood += xb - sum_exp.ln();
            }
        }

        // Newton–Raphson step
        if let Ok(delta) = hessian.solve_into(gradient.clone()) {
            if delta.iter().all(|v| v.is_finite()) {
                beta += &delta;
                final_hessian.assign(&hessian);
                if delta.iter().map(|x| x.abs()).sum::<f64>() < tol { break; }
            }
        }
    }
    pb.finish_with_message("done");

    // Prepare outputs
    let mut coefficients = Vec::new();
    let mut hr = HashMap::new();
    let mut se = HashMap::new();

    for (i, name) in feature_names.iter().enumerate() {
        coefficients.push(name.clone());
        hr.insert(name.clone(), beta[i].exp());
        se.insert(name.clone(), (1.0 / final_hessian[[i,i]]).sqrt()); // approximate SE
    }

    CoxModel { coefficients, hr, se }
}

