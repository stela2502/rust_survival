// src/points.rs
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use ndarray::Array2;
use crate::cox::CoxModel;
use crate::data::SurvivalData;


/// Single summary object for factor-level statistics
#[derive(Debug, Clone)]
pub struct SummaryStat {
    pub data: HashMap<String, (usize, f64, f64, f64, f64)>, 
    // (hazard_mean, hazard_var, points_mean, points_var)
}

// Optional: implement Display for pretty printing
impl fmt::Display for SummaryStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sort keys for consistent column order
        let mut keys: Vec<&String> = self.data.keys().collect();
        keys.sort();

        // Prepare each statistic row
        let mut n_row = vec!["n".to_string()];
        let mut haz_mean_row = vec!["hazard_mean".to_string()];
        let mut haz_var_row = vec!["hazard_var".to_string()];
        let mut pts_mean_row = vec!["points_mean".to_string()];
        let mut pts_var_row = vec!["points_var".to_string()];

        for k in &keys {
            let (n, haz_mean, haz_var, pts_mean, pts_var) = self.data[*k];
            n_row.push(format!("{:.0}", n));
            haz_mean_row.push(format!("{:.2e}", haz_mean));
            haz_var_row.push(format!("{:.2e}", haz_var));
            pts_mean_row.push(format!("{:.2e}", pts_mean));
            pts_var_row.push(format!("{:.2e}", pts_var));
        }

        // Write each row as tab-separated
        writeln!(f, "\t{}", keys.into_iter().map(|x| format!("'{}'",x)).collect::<Vec<String>>().join("\t"))?;
        writeln!(f, "{}", n_row.join("\t"))?;
        writeln!(f, "{}", haz_mean_row.join("\t"))?;
        writeln!(f, "{}", haz_var_row.join("\t"))?;
        writeln!(f, "{}", pts_mean_row.join("\t"))?;
        writeln!(f, "{}", pts_var_row.join("\t"))?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Points {
    pub base_hr: f64,
    pub points_map: HashMap<String, i32>,
}

impl Points {
    pub fn new(cox_model: &CoxModel, base_hr: f64) -> Self {
        let mut points_map = HashMap::new();
        for (var, &hr) in &cox_model.hr {
            let pt = ((hr.ln()) / base_hr.ln()).round() as i32;
            points_map.insert(var.clone(), pt);
        }
        Self { base_hr, points_map }
    }

    /// Predict per-patient hazard and total points
    pub fn predict(
        &self,
        cox_model: &CoxModel,
        data: &Array2<f64>,
        headers: &[String],
    ) -> Vec<(f64, i32)> {
        let feature_indices: Vec<usize> = cox_model
            .coefficients
            .iter()
            .map(|feat| headers
                .iter()
                .position(|h| h == feat)
                .expect("Feature missing in headers"))
            .collect();

        (0..data.nrows())
            .map(|i| {
                let hazard: f64 = cox_model.coefficients.iter().enumerate()
                    .map(|(j, feat)| {
                        let beta = cox_model.hr[feat].ln();
                        let x = data[[i, feature_indices[j]]];
                        beta * x
                    })
                    .sum::<f64>()
                    .exp();

                let total_points: i32 = self.points_map.iter()
                    .map(|(f, &p)| {
                        let idx = headers.iter().position(|h| h == f).unwrap();
                        (p as f64 * data[[i, idx]]).round() as i32
                    })
                    .sum();
                (hazard, total_points)
            })
            .collect()
    }

    /// Save predictions to CSV using patient IDs if provided.
    /// `patient_ids` must be same length as data rows.
    pub fn save_predictions<P: AsRef<Path>>(
        &self,
        cox_model: &CoxModel,
        data: &Array2<f64>,
        headers: &[String],
        patient_ids: Option<&[String]>,
        ofile: Option<P>,
    ) -> std::io::Result<()> {
        if let Some(path) = ofile {
            let preds = self.predict(cox_model, data, headers);
            let mut file = File::create(path)?;
            writeln!(file, "patient_id,hazard,total_points")?;
            match patient_ids {
                Some( pids ) => {
                    for id in 0..preds.len(){
                        writeln!(file, "{},{:.6},{:.0}", pids[id], preds[id].0, preds[id].1)?;
                    }
                },
                None => {
                    for id in 0..preds.len(){
                        writeln!(file, "{},{:.6},{:.0}", id, preds[id].0, preds[id].1)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Summarize mean and variance of hazards and points.
    /// If `factor_col` is given, compute stats per factor level.
    pub fn summary(
        &self,
        survival_data: &SurvivalData,
        cox_model: &CoxModel,
        factor_col: Option<String>,
    ) -> SummaryStat {
        let results = self.predict(
            cox_model,
            &survival_data.numeric_data,
            &survival_data.headers,
        );
   
        fn mean_var(xs: &[f64]) -> (f64, f64) {
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            (mean, var)
        }

        let mut out = HashMap::<String, (usize, f64, f64, f64, f64)>::new();
        if let Some(col) = factor_col {
            let values= survival_data.as_vec_f64( &col );

            match survival_data.factors.get( &col) {
                Some(fact) => {
                    let levels = fact.get_levels();
                    println!("The recorded levels for factor '{col}': {:?}",levels );
                    let mut haz: Vec<Vec<f64>> = (0..levels.len())
                        .map(|_| Vec::with_capacity(results.len()))
                        .collect();
                    let mut points: Vec<Vec<f64>> = (0..levels.len())
                        .map(|_| Vec::with_capacity(results.len()))
                        .collect();    
                    for (i, (h, p)) in results.into_iter().enumerate() {
                        haz[ values[i] as usize].push( h );
                        points [values[i] as usize].push( p as f64);
                    }
                    for (id,group) in levels.into_iter().enumerate() {
                        let (hm, hv) = mean_var(&haz[id]);
                        let (pm, pv) = mean_var(&points[id]);
                        out.insert( group.to_string(), ( haz[id].len(), hm, hv, pm, pv ) );
                    }
                },
                None => {
                    println!("The column '{col}' does not seam to be a factor in the survival data:\n{:?}",survival_data.headers );
                    // Split results into hazard and points
                    let hazards: Vec<f64> = results.iter().map(|(h, _)| *h).collect();
                    let points: Vec<f64>  = results.iter().map(|(_, p)| *p as f64).collect();

                    let (hm, hv) = mean_var(&hazards);
                    let (pm, pv) = mean_var(&points);
                    out.insert("ALL".to_string(), (hazards.len(), hm, hv, pm, pv));
                }
            }
        } else {
            // Split results into hazard and points
            let hazards: Vec<f64> = results.iter().map(|(h, _)| *h).collect();
            let points: Vec<f64>  = results.iter().map(|(_, p)| *p as f64).collect();

            let (hm, hv) = mean_var(&hazards);
            let (pm, pv) = mean_var(&points);
            out.insert("ALL".to_string(), (hazards.len(),hm, hv, pm, pv));
        }
        SummaryStat {data:out}
    }
}

/// Pretty-print the internal points mapping.
impl fmt::Display for Points {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Points assignment (base HR = {}):", self.base_hr)?;
        let mut vars: Vec<_> = self.points_map.iter().collect();
        vars.sort_by_key(|(k, _)| *k);
        for (var, pts) in vars {
            writeln!(f, "  {:20} -> {:3}", var, pts)?;
        }
        Ok(())
    }
}

