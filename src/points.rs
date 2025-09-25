// src/points.rs

use std::collections::HashMap;
use crate::cox::CoxModel;

/// Assign points based on Cox hazard ratios
///
/// # Arguments
/// * `cox_model` - fitted CoxModel
/// * `base_hr` - the HR that corresponds to 1 point (typical: 1.2)
///
/// # Returns
/// HashMap with variable name -> integer points
pub fn assign_points(cox_model: &CoxModel, base_hr: f64) -> HashMap<String, i32> {
    let mut points: HashMap<String, i32> = HashMap::new();

    for (var, &hr) in &cox_model.hr {
        // Convert HR to log scale, divide by log(base_hr), round to nearest integer
        let pt = ((hr.ln()) / base_hr.ln()).round() as i32;
        points.insert(var.clone(), pt);
    }

    points
}

/// Compute total points for a new patient
///
/// # Arguments
/// * `patient` - HashMap of variable name -> value (0/1 for binary, 1,2,... for categorical)
/// * `points_map` - HashMap from assign_points
///
/// # Returns
/// Integer total point score
pub fn total_points(patient: &HashMap<String,f64>, points_map: &HashMap<String,i32>) -> i32 {
    let mut total = 0;
    for (var, &pt) in points_map {
        if let Some(&val) = patient.get(var) {
            // Multiply points by patient value (for continuous: scale linearly)
            total += (pt as f64 * val).round() as i32;
        }
    }
    total
}

