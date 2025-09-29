use std::collections::{HashMap};
use std::fs::File;
use std::path::Path;
use csv::{ReaderBuilder, WriterBuilder, StringRecord};
use ndarray::{Array2, s};
use anyhow::Result;


#[derive(Debug, Clone)]
pub struct Factor {
    indices: Vec<f64>,                // 0.0..n-1.0 for levels, NaN for missing
    levels: Vec<String>,              // level labels
    level_to_index: HashMap<String, f64>, // fast lookup
}

impl Factor {
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            levels: Vec::new(),
            level_to_index: HashMap::new(),
        }
    }

    pub fn with_empty(fill: usize) -> Self {
        Self {
            indices: vec![f64::NAN; fill],
            levels: Vec::new(),
            level_to_index: HashMap::new(),
        }
    }

    pub fn push(&mut self, value: &str) -> f64 {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
            let idx = f64::NAN;
            self.indices.push(idx);
            return idx;
        }

        let idx = if let Some(&i) = self.level_to_index.get(trimmed) {
            i
        } else {
            let new_idx = self.levels.len() as f64;
            self.levels.push(trimmed.to_string());
            self.level_to_index.insert(trimmed.to_string(), new_idx);
            new_idx
        };

        self.indices.push(idx);
        idx
    }

    pub fn push_missing(&mut self) {
        self.indices.push(f64::NAN);
    }

    pub fn get_index(&self, i: usize) -> Option<f64> {
        self.indices.get(i).copied()
    }

    pub fn get_value(&self, i: usize) -> String {
        match self.indices.get(i) {
            Some(&idx) if idx.is_nan() => "NA".to_string(),
            Some(&idx) => self.levels.get(idx as usize)
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string()),
            None => "NA".to_string(),
        }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn get_levels(&self) -> Vec<String> {
        self.levels.clone()
    }
}

#[derive(Debug, Clone)]
pub struct SurvivalData {
    pub headers: Vec<String>,
    pub numeric_data: Array2<f64>,
    pub factors: HashMap<String, Factor>,
    pub max_levels: usize,
}

impl SurvivalData {
    /// Read CSV file and build numeric + factor representation
    pub fn from_file<P: AsRef<Path>>(file_path: P, delimiter: u8, max_levels: usize) -> Result<Self> {
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .trim(csv::Trim::All)
            .flexible(true)
            .from_path(&file_path)?;

        let headers: Vec<String> = rdr
            .headers()?
            .iter()
            .map(|s| s.to_string())
            .collect();
        let n_cols = headers.len();

        // Temporary storage
        let mut factors: Vec<Option<Factor>> = vec![None; n_cols];
        let mut raw_rows: Vec<Vec<f64>> = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let mut row: Vec<f64> = Vec::with_capacity(n_cols);

            for (i, value) in record.iter().enumerate() {
                let trimmed = value.trim();

                // Missing value
                if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                    if let Some(factor) = &mut factors[i] {
                        factor.push_missing();
                    }
                    row.push(f64::NAN);
                    continue;
                }

                match trimmed.parse::<f64>() {
                    Ok(num) => row.push(num),
                    Err(_) => {
                        // This column is categorical
                        let factor = factors[i].get_or_insert(Factor::with_empty(raw_rows.len()));
                        let _idx = factor.push(trimmed);
                        row.push(_idx);
                    }
                }
            }
            raw_rows.push(row);
        }

        let n_rows = raw_rows.len();
        let mut numeric_data = Array2::<f64>::zeros((n_rows, n_cols));

        for (i, row) in raw_rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                numeric_data[[i, j]] = val;
            }
        }

        // Build factor map and remove columns exceeding max_levels
        let mut factor_map: HashMap<String, Factor> = HashMap::new();
        for (i, f_opt) in factors.into_iter().enumerate() {
            if let Some(f) = f_opt {
                if f.n_levels() <= max_levels {
                    factor_map.insert(headers[i].clone(), f);
                }
            }
        }

        Ok(Self {
            headers,
            numeric_data,
            factors: factor_map,
            max_levels,
        })
    }

    /// Impute missing values in `numeric_data` using K-nearest neighbours.
    ///
    /// * `k` – number of neighbours to use (e.g. 3 for “mean of 3 closest”).
    /// * `min_common` – minimum number of shared non-NA features required
    ///                  between two rows to consider them neighbours.
    /// * `weighted` – if `true`, use distance-weighted mean;
    ///                if `false`, simple mean of the k neighbours.
    ///
    /// Missing values are represented as `f64::NAN`.
    pub fn impute_knn(&mut self, k: usize, min_common: usize, weighted: bool) {
        let eps = 1e-8_f64;
        let (n_rows, n_cols) = (self.numeric_data.nrows(), self.numeric_data.ncols());

        // ---------- z-score scale copy for distance calculations ----------
        let mut scaled = self.numeric_data.clone();
        let mut col_means = vec![0.0; n_cols];
        let mut col_stds  = vec![1.0; n_cols];

        for j in 0..n_cols {
            let col = self.numeric_data.column(j);
            let vals: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            if vals.is_empty() { continue; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var  = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            let std  = var.sqrt().max(eps);
            col_means[j] = mean;
            col_stds[j]  = std;
            for i in 0..n_rows {
                let v = self.numeric_data[[i, j]];
                if !v.is_nan() {
                    scaled[[i, j]] = (v - mean) / std;
                }
            }
        }

        // helper to compute Euclidean distance using only shared non-NA cols
        let distance = |a: usize, b: usize| -> Option<f64> {
            let mut sum = 0.0;
            let mut cnt = 0;
            for j in 0..n_cols {
                let va = scaled[[a, j]];
                let vb = scaled[[b, j]];
                if va.is_nan() || vb.is_nan() { continue; }
                let d = va - vb;
                sum += d * d;
                cnt += 1;
            }
            if cnt >= min_common { Some((sum / cnt as f64).sqrt()) } else { None }
        };

        // ---------- main loop over rows with NAs ----------
        for i in 0..n_rows {
            // which columns are missing for this row?
            let missing: Vec<_> = (0..n_cols)
                .filter(|&j| self.numeric_data[[i, j]].is_nan())
                .collect();
            if missing.is_empty() { continue; }

            // collect neighbours and their distances
            let mut neigh = Vec::new();
            for j in 0..n_rows {
                if j == i { continue; }
                if let Some(d) = distance(i, j) {
                    neigh.push((j, d));
                }
            }
            neigh.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if neigh.is_empty() { continue; }

            // impute each missing column
            for &col in &missing {
                let mut vals = Vec::new();
                for &(row_idx, dist) in &neigh {
                    let v = self.numeric_data[[row_idx, col]];
                    if !v.is_nan() {
                        vals.push((v, dist));
                        if vals.len() >= k { break; }
                    }
                }
                if vals.is_empty() { continue; }

                let imputed = if weighted {
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for (v, d) in vals {
                        let w = 1.0 / (d + eps);
                        num += w * v;
                        den += w;
                    }
                    num / den
                } else {
                    vals.iter().map(|(v, _)| v).sum::<f64>() / vals.len() as f64
                };
                self.numeric_data[[i, col]] = imputed;
            }
        }
    }

    /// Remove any rows where `column` has NaN in the numeric data
    pub fn filter_na(&mut self, column: &str) {
        // Find the column index
        let col_idx = self.headers
            .iter()
            .position(|h| h == column)
            .expect("Column not found");

        // Determine which rows to keep
        let keep_rows: Vec<usize> = self.numeric_data
            .column(col_idx)
            .indexed_iter()
            .filter_map(|(i, &val)| if !val.is_nan() { Some(i) } else { None })
            .collect();

        // Rebuild numeric_data with only the kept rows
        let mut filtered = Array2::<f64>::zeros((keep_rows.len(), self.numeric_data.ncols()));
        for (new_i, &old_i) in keep_rows.iter().enumerate() {
            filtered
                .row_mut(new_i)
                .assign(&self.numeric_data.slice(s![old_i, ..]));
        }
        self.numeric_data = filtered;

        // Filter factors: keep only entries for kept rows
        for factor in self.factors.values_mut() {
            let mut new_indices = Vec::with_capacity(keep_rows.len());
            for &old_i in &keep_rows {
                let v = *factor.indices.get(old_i).unwrap_or(&f64::NAN);
                new_indices.push(v);
            }
            factor.indices = new_indices;
        }
    }
        /// Compute fraction of NAs per feature column
    pub fn feature_na_fraction(&self) -> HashMap<String, f64> {
        let mut na_frac: HashMap<String, f64> = HashMap::new();
        for (idx, name) in self.headers.clone().into_iter().enumerate() {
            let col = self.numeric_data.column(idx);
            let n_total = col.len();
            let n_na = col.iter().filter(|v| v.is_nan()).count();
            na_frac.insert(name.clone(), n_na as f64 / n_total as f64);
        }
        na_frac
    }

    /// Return features filtered by max allowed NA fraction
    pub fn filter_features_by_na(&self, max_na_frac: f64) -> Vec<String> {
        self.feature_na_fraction()
            .into_iter()
            .filter(|(_, frac)| *frac <= max_na_frac)
            .map(|(name, _)| name)
            .collect()
    }

    /// Return filtered numeric data and corresponding indices for allowed features
    pub fn as_array2_filtered(&self, max_na_frac: f64) -> (Array2<f64>, Vec<usize>) {
        let allowed_features = self.filter_features_by_na( max_na_frac);
        let mut indices = Vec::new();
        for name in &allowed_features {
            if let Some(idx) = self.headers.iter().position(|h| h == name) {
                indices.push(idx);
            }
        }

        let n_rows = self.numeric_data.nrows();
        let n_cols = indices.len();
        let mut arr = Array2::<f64>::zeros((n_rows, n_cols));

        for (j, &col_idx) in indices.iter().enumerate() {
            arr.column_mut(j).assign(&self.numeric_data.column(col_idx));
        }

        (arr, indices)
    }

    /// Return numeric data as ndarray, optionally selecting columns
    pub fn as_array2(&self, columns: Option<&[String]>) -> Array2<f64> {
        match columns {
            Some(cols) => {
                let indices: Vec<usize> = cols
                    .iter()
                    .map(|c| {
                        self.headers
                            .iter()
                            .position(|h| h == c)
                            .expect("Column not found")
                    })
                    .collect();
                let mut arr = Array2::<f64>::zeros((self.numeric_data.nrows(), indices.len()));
                for (j, &col_idx) in indices.iter().enumerate() {
                    arr.column_mut(j)
                        .assign(&self.numeric_data.column(col_idx));
                }
                arr
            }
            None => self.numeric_data.clone(),
        }
    }

    /// Return a single column as Vec<f64>
    pub fn as_vec_f64(&self, column: &str) -> Vec<f64> {
        let idx = self
            .headers
            .iter()
            .position(|h| h == column)
            .expect("Column not found");
        self.numeric_data.column(idx).to_vec()
    }

    /// Return a single column as Vec<f64>
    pub fn as_vec_u8(&self, column: &str) -> Vec<u8> {
        let idx = self
            .headers
            .iter()
            .position(|h| h == column)
            .expect("Column not found");
        self.numeric_data.column(idx).iter()
        .map(|&v| v as u8)  // cast f64 -> u8
        .collect()
    }

    /// Write data (numeric + factor) to CSV
    pub fn to_file<P: AsRef<std::path::Path>>(&self, file_path: P, delimiter: u8) -> Result<()> {
        let mut wtr = WriterBuilder::new()
            .delimiter(delimiter)
            .from_path(file_path)?;

        // Write header
        wtr.write_record(&self.headers)?;

        for i in 0..self.numeric_data.nrows() {
            let mut record: Vec<String> = Vec::with_capacity(self.headers.len());

            for (j, col_name) in self.headers.iter().enumerate() {
                let val = self.numeric_data[[i, j]];

                if let Some(factor) = self.factors.get(col_name) {
                    if val.is_nan() {
                        record.push("NA".to_string());
                    } else {
                        let idx = val as usize;
                        let lvl = factor.levels.get(idx)
                            .unwrap_or(&"NA".to_string())
                            .clone();
                        record.push(lvl);
                    }
                } else {
                    // Numeric column
                    if val.is_nan() {
                        record.push("NA".to_string());
                    } else {
                        record.push(val.to_string());
                    }
                }
            }

            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        Ok(())
    }

}