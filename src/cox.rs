// src/cox.rs

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve; // for linear algebra
use std::collections::HashMap;

/// Cox model result
#[derive(Debug, Serialize, Deserialize)]
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
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load model from a binary file using bincode
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
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
    indices.sort_by(|&i, &j| time[j].partial_cmp(&time[i]).unwrap());
    let mut final_hessian = Array2::<f64>::zeros((n_features, n_features));

    for _ in 0..max_iter {
        let mut gradient = Array1::<f64>::zeros(n_features);
        let mut hessian = Array2::<f64>::zeros((n_features, n_features));
        let mut log_likelihood = 0.0;

        for &i in &indices {
            if status[i] == 0 { continue; }

            // Risk set: all individuals with time >= time[i]
            let risk_indices: Vec<usize> = (0..n_samples).filter(|&j| time[j] >= time[i]).collect();

            // Compute sum of exp(x beta) in risk set
            let xb_risk: Vec<f64> = risk_indices.iter()
                .map(|&j| (data.row(j).dot(&beta)).exp())
                .collect();
            let sum_exp = xb_risk.iter().sum::<f64>();

            // Compute weighted mean of covariates in risk set
            let mut weighted_mean = Array1::<f64>::zeros(n_features);
            for (k, &j) in risk_indices.iter().enumerate() {
                weighted_mean += &(&data.row(j) * xb_risk[k]);
            }
            weighted_mean /= sum_exp;

            // Gradient update
            let xi = data.row(i).to_owned();
            gradient += &(xi.clone()- &weighted_mean);

            // Hessian update
            let mut outer_sum = Array2::<f64>::zeros((n_features, n_features));
            for (k, &j) in risk_indices.iter().enumerate() {
                let xj = data.row(j).to_owned();
                let diff = &xj - &weighted_mean;
                let diff_view = diff.view();
                outer_sum += &(diff_view.insert_axis(Axis(1)).dot(&diff_view.insert_axis(Axis(0))) * xb_risk[k]);
            }

            hessian.scaled_add(1.0 / sum_exp, &outer_sum);
            //hessian += &outer_sum / sum_exp;

            // Log-likelihood
            log_likelihood += xi.dot(&beta) - sum_exp.ln();
        }

        // Newton-Raphson step: beta_new = beta_old + H^{-1} * gradient
        let delta = hessian.solve_into(gradient.clone()).unwrap();
        beta += &delta;

        // Save this Hessian in case of convergence
        final_hessian.assign(&hessian);

        if delta.iter().map(|x| x.abs()).sum::<f64>() < tol {
            break;
        }
    }

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

