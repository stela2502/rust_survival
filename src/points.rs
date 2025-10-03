// src/points.rs
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use ndarray::Array2;
use crate::cox::CoxModel;
use crate::data::SurvivalData;
use plotters::prelude::*;
use crate::data::Factor;


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
    /// returns the SummaryStats for the model and the grouped raw hazard and points values.
    /// These can be used to create respective the plots using plot_raw_summary
    pub fn summary(
        &self,
        survival_data: &SurvivalData,
        cox_model: &CoxModel,
        factor_col: Option<&String>,
    ) -> (SummaryStat, HashMap<String, Vec<f64>>, HashMap<String, Vec<f64>>)  {
        let results = self.predict(
            cox_model,
            &survival_data.numeric_data,
            &survival_data.headers,
        );
   
        fn mean_var(xs: &[f64]) -> (f64, f64) {
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            (mean as f64, var as f64)
        }

        let mut out = HashMap::<String, (usize, f64, f64, f64, f64)>::new();
        let mut summary_raw_points: HashMap<String, Vec<f64>> = HashMap::new();
        let mut summary_raw_haz: HashMap<String, Vec<f64>> = HashMap::new();

        if let Some(col) = factor_col {
            let values= survival_data.as_vec_f64( col );

            match survival_data.factors.get( col) {
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
                        haz[ values[i] as usize].push( h as f64 );
                        points [values[i] as usize].push( p as f64);
                    }
                    for (id,group) in levels.into_iter().enumerate() {
                        let (hm, hv) = mean_var(&haz[id]);
                        let (pm, pv) = mean_var(&points[id]);
                        out.insert( group.to_string(), ( haz[id].len(), hm, hv, pm, pv ) );
                        summary_raw_points.insert(group.to_string(), points[id].clone() );
                        summary_raw_haz.insert(group.to_string(), haz[id].clone() );
                    }
                },
                None => {
                    println!("The column '{col}' does not seam to be a factor in the survival data:\n{:?}",
                        survival_data.headers(20) );
                    // Split results into hazard and points
                    let hazards: Vec<f64> = results.iter().map(|(h, _)| *h as f64).collect();
                    let points: Vec<f64>  = results.iter().map(|(_, p)| *p as f64).collect();

                    let (hm, hv) = mean_var(&hazards);
                    let (pm, pv) = mean_var(&points);
                    out.insert("ALL".to_string(), (hazards.len(), hm, hv, pm, pv));
                    summary_raw_points.insert("ALL".to_string(), points );
                    summary_raw_haz.insert("ALL".to_string(), hazards );
                }
            }
        } else {
            // Split results into hazard and points
            let hazards: Vec<f64> = results.iter().map(|(h, _)| *h as f64).collect();
            let points: Vec<f64>  = results.iter().map(|(_, p)| *p as f64).collect();

            let (hm, hv) = mean_var(&hazards);
            let (pm, pv) = mean_var(&points);
            out.insert("ALL".to_string(), (hazards.len(),hm, hv, pm, pv));
            summary_raw_points.insert("ALL".to_string(), points );
            summary_raw_haz.insert("ALL".to_string(), hazards );
        }
        (SummaryStat {data:out}, summary_raw_haz, summary_raw_points)
    }


    /// Draw boxplots for train/test summary with raw values per factor level
    pub fn plot_raw_summary(
        train_raw: &HashMap<String, Vec<f64>>,          // hazards or points per level
        test_raw: Option<&HashMap<String, Vec<f64>>>,   // optional test set
        factor: &Factor, // the factor from the survival data for this.
        title: &str,
        output_file: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let levels: Vec<String> = train_raw.keys().cloned().collect();
        let n_levels = levels.len();

        // Flatten all values to determine Y-axis range
        let mut all_values: Vec<f64> = train_raw.values().flat_map(|v| v.iter().copied()).collect();
        if let Some(test) = test_raw {
            all_values.extend(test.values().flat_map(|v| v.iter().copied()));
        }

        // minimum across train + optional test
        let (y_min, y_max) = match test_raw {
            Some(test_raw) => {
                let min_train = train_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::INFINITY, f64::min))
                    .fold(f64::INFINITY, f64::min);

                let min_test = test_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::INFINITY, f64::min))
                    .fold(f64::INFINITY, f64::min);

                let max_train = train_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                    .fold(f64::NEG_INFINITY, f64::max);

                let max_test = test_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                    .fold(f64::NEG_INFINITY, f64::max);

                (min_train.min(min_test), max_train.max(max_test))
            },
            None => {
                let min_train = train_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::INFINITY, f64::min))
                    .fold(f64::INFINITY, f64::min);

                let max_train = train_raw.values()
                    .map(|vals| vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                    .fold(f64::NEG_INFINITY, f64::max);

                (min_train, max_train)
            }
        };

        let root = SVGBackend::new(output_file, (1400, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..(n_levels*2), y_min as f32..y_max as f32) ?;

        chart.configure_mesh()
            .x_labels(n_levels)
            .x_label_formatter(&|x| {
                let lvl_idx = x / 2;
                if lvl_idx < levels.len() { levels[lvl_idx].clone() } else { "".to_string() }
            })
            .y_desc(title)
            .draw()?;

        for (i, level) in levels.iter().enumerate() {
            // Train
            if let Some(values) = train_raw.get(level) {
                let mut sorted = values.clone();
                sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                if n > 0 {
                    let q1 = sorted[n/4];
                    let q3 = sorted[(3*n)/4];
                    let median = sorted[n/2];
                    let min = sorted[0];
                    let max = sorted[n-1];
                    let quartiles = Quartiles::new(&[min, q1, median , q3, max]);

                    let bp = Boxplot::new_vertical(i*2, &quartiles).style(BLUE.filled()).width(15_u32);
                    chart.draw_series(std::iter::once(bp))?;
                }
            }

            // Test (optional)
            if let Some(test) = test_raw {
                if let Some(values) = test.get(level) {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
                    let n = sorted.len();
                    if n > 0 {
                        let q1 = sorted[n/4];
                        let q3 = sorted[(3*n)/4];
                        let median = sorted[n/2];
                        let min = sorted[0];
                        let max = sorted[n-1];
                        let quartiles = Quartiles::new(&[min, q1, median , q3, max]);
                        let  bp= Boxplot::new_vertical(i*2 +1, &quartiles).style(RED.filled()).width(15_u32);
                        chart.draw_series(std::iter::once( bp ))?;
                    }
                }
            }
        }

        Ok(())
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

