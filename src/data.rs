use std::collections::{HashMap, HashSet};
use std::path::{Path};
use csv::{ WriterBuilder};
use ndarray::{Array2, s};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json;
use std::fs::File;
use std::io::{BufWriter, BufReader};


#[derive(Debug, Clone)]
pub struct Factor {
    column_name: String,
    indices: Vec<f64>,                // 0.0..n-1.0 for levels, NaN for missing
    pub levels: Vec<String>,              // level labels
    pub level_to_index: HashMap<String, f64>, // fast lookup
    matching:Option<Vec<String>>, // this could match to multiple column names. Like SNP or something
    one_hot: bool, // NEW
}

/// Only save labels in JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorJson {
    column: String,
    levels: Vec<String>,
    numeric: Option<Vec<f64>>,
    matching:Option<Vec<String>>, 
    one_hot: bool, // NEW
}

impl Factor {
    /*
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            levels: Vec::new(),
            level_to_index: HashMap::new(),
        }
    }*/

    pub fn with_empty(fill: usize, column_name:&str) -> Self {
        Self {
            column_name: column_name.to_string(),
            indices: vec![f64::NAN; fill],
            levels: Vec::new(),
            level_to_index: HashMap::new(),
            matching:None,
            one_hot: false, // NEW
        }
    }
    pub fn extra_columns(&self) -> usize {
        if self.one_hot {
            self.levels.len()
        }else {
            0
        }
    }

    pub fn all_column_names(&self ) -> Vec<String> {
        if self.one_hot {
             self
                .levels
                .iter()
                .map(|lvl| self.build_one_hot_column(lvl)).collect()
        }else {
            vec![ self.column_name.to_string() ]
        }
    }

    fn build_one_hot_column( &self, value:&str) -> String {
        format!("{}_{}", self.column_name, value)
    }

    /// Push a value for this factor.
    /// Returns:
    /// - f64: the numeric index for this value
    /// - String the column name the value should be added to
    /// - Option<Vec<String>>: all one-hot columns if one-hot
    pub fn push(&mut self, value: &str) -> (f64, String,  Option<Vec<String>>) {
        let trimmed = value.trim();

        // Handle one-hot encoding
        let ret = if self.one_hot {
            println!("See we have a one_hot here! {} - trimmed {}", self.column_name, trimmed);
            let idx = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                f64::NAN
            }else {
                1.0
            };
            // all other levels become zero columns
            let zero_cols: Vec<String> = self.all_column_names( );
            let zero_cols_option = if zero_cols.len() == 1 { None } else { Some(zero_cols) };
            (idx, self.build_one_hot_column( trimmed ) , zero_cols_option)
        } else {
            let idx = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                f64::NAN
            }else {
                // Determine numeric index
                if let Some(&i) = self.level_to_index.get(trimmed) {
                    i
                } else {
                    let new_idx = self.levels.len() as f64;
                    self.levels.push(trimmed.to_string());
                    self.level_to_index.insert(trimmed.to_string(), new_idx);
                    new_idx
                }
            };
            (idx, self.column_name.to_string() , None)
        };
        println!("We {} return a value of {}", ret.1, ret.0);
        ret
    }

    pub fn push_missing(&mut self) {
        self.indices.push(f64::NAN);
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

    pub fn get_f64( &self, trimmed: &str ) ->f64  {
       *self.level_to_index.get(trimmed).unwrap_or( &f64::NAN )
    }

    pub fn get_levels(&self) -> Vec<String> {
        self.levels.clone()
    }

    /// Create Factor from FactorDef
    pub fn from_def(def: &FactorJson) -> Self {
        let level_to_index: std::collections::HashMap<String, f64> = match &def.numeric {
            Some( numbers ) => {
                def.levels
                .iter()
                .cloned()
                .zip(numbers.iter().cloned())
                .collect()
            },
            None => {
                def.levels.iter()
                .enumerate()
                .map(|(i, s)| (s.clone(),i  as f64))
                .collect()
            }
        };

        Factor {
            column_name: def.column.to_string(),
            indices: Vec::new(),
            levels: def.levels.clone(),
            level_to_index,
            matching: def.matching.clone(),
            one_hot: def.one_hot,
        }
    }

    /// Convert Factor into JSON representation
    pub fn as_json(&self, column: &str) -> FactorJson {
        let numeric: Vec<f64> = self.levels
            .iter()
            .map(|name| {
                self.level_to_index.get(name).copied().unwrap_or_else(|| {
                    panic!(
                        "Factor::as_json error: level '{}' not found in mapping for column '{}'",
                        name, column
                    )
                })
            })
            .collect();

        FactorJson {
            column: self.column_name.to_string(),
            levels: self.levels.clone(),
            numeric: Some(numeric),
            matching: self.matching.clone(),
            one_hot: self.one_hot,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurvivalData {
    pub headers: Vec<String>,
    pub numeric_data: Array2<f64>,
    pub factors: HashMap<String, Factor>,
    pub exclude: HashSet<String>,
    //pub max_levels: usize,
}

impl Default for SurvivalData {
    fn default() -> Self {
        SurvivalData {
            headers: Vec::new(),
            numeric_data: Array2::zeros((0, 0)),
            factors: HashMap::new(),
            exclude: HashSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_one_hot() {
        let mut f = Factor::with_empty(0, "Color");
        assert_eq!( f.push("Red"), (0.0,"Color".to_string(), None), "new factor Red gets 0");
        assert_eq!( f.push("Blue"), (1.0,"Color".to_string(), None), "new factor Blue gets 1");
        let (idx, col, opt) = f.push("NA");
        assert!(idx.is_nan(), "new factor NA should be NaN");
        assert_eq!(col, "Color");
        assert_eq!(opt, None);
        f.one_hot = true;

        // Push first value "Red"
        let (idx, col_to_add, all_cols) = f.push("Red");
        assert_eq!(idx, 1.0);
        assert_eq!(col_to_add, "Color_Red");
        assert_eq!(all_cols.unwrap(), vec!["Color_Red".to_string(),"Color_Blue".to_string() ]);

        // Push another value "Blue"
        let (idx, col_to_add, all_cols) = f.push("Blue");
        assert_eq!(idx, 1.0);
        assert_eq!(col_to_add, "Color_Blue");
        assert_eq!(all_cols.unwrap(), vec!["Color_Red".to_string(),"Color_Blue".to_string() ]);


        // Now push a missing value
        let (idx, col_to_add, all_cols) = f.push("NA");
        assert!(idx.is_nan());
        assert_eq!(col_to_add, "Color_NA");
        assert_eq!(all_cols.unwrap(), vec!["Color_Red".to_string(),"Color_Blue".to_string() ]);
    }

    #[test]
    fn test_push_categorical_indexed() {


        let mut f = Factor::with_empty(0, "Color");
        assert_eq!( f.push("Red"), (0.0,"Color".to_string(), None), "new factor Red gets 0");
        assert_eq!( f.push("Blue"), (1.0,"Color".to_string(), None), "new factor Blue gets 1");
        let (idx, col, opt) = f.push("NA");
        assert!(idx.is_nan(), "new factor NA should be NaN");
        assert_eq!(col, "Color");
        assert_eq!(opt, None);

        assert_eq!( f.push("Red"), (0.0,"Color".to_string(), None), "2# new factor Red gets 0");
        assert_eq!( f.push("Blue"), (1.0,"Color".to_string(), None), "2# new factor Blue gets 1");
        let (idx, col, opt) = f.push("NA");
        assert!(idx.is_nan(), "2# new factor NA should be NaN");
        assert_eq!(col, "Color");
        assert_eq!(opt, None);
    }
}

impl SurvivalData {

    /// Read CSV file and build numeric + factor representation
    pub fn from_file<P: AsRef<Path> + std::fmt::Debug, FF: AsRef<Path> + std::fmt::Debug>
    (file_path: P, delimiter: u8, categorical_cols: HashSet<String>, factors_file: FF) -> Result<Self> {

        // 1. Start with an empty SurvivalData
        let mut ret = SurvivalData::default();

        // 2. Load factors if the file exists, otherwise keep empty
        if factors_file.as_ref().exists() {
            ret.load_factors(&factors_file)?;
        }

        // 3. Open the CSV
        let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_path(file_path)?;

        let headers: Vec<String> = rdr
            .headers()?
            .iter()
            .flat_map(|s| {
                if let Some(fact) = ret.factors.get(s) {
                    let mut cols = vec![s.to_string()];
                    cols.extend(fact.all_column_names());
                    cols
                } else {
                vec![s.to_string()] // wrap the single name in a Vec
            }})
        .collect();

        let header_lookup: HashMap< String, usize> = headers
            .iter()
            .enumerate()
            .map(|(id, name)|(name.clone(), id))
        .collect();

        for header in &headers {
            if let Some(fact) = ret.factors.get(header) {
                if fact.one_hot {
                    ret.exclude.insert( header.to_string() );
                }
            }
        }

        let n_cols = headers.len();
        let mut raw_rows: Vec<Vec<f64>> = Vec::new();

        for header in &headers {
            if categorical_cols.contains(header) {
                // Insert a Factor for this categorical column if it does not exist yet
                ret.factors.entry(header.clone())
                .or_insert_with(|| Factor::with_empty(0, &header));
            }
        }

        // 4. lead the data and handle the factors
        for result in rdr.records() {
            let record = result?;
            let mut row: Vec<f64> = Vec::with_capacity(n_cols);
            let mut expanded = 0;
            for (i, value) in record.iter().enumerate() {
                let trimmed = value.trim().trim_matches('"');
                /*
                    if let Some(factor) = ret.factors.get_mut(&headers[i]) {
                        factor.push_missing();
                    }
                    row.push(f64::NAN);
                    continue;
                }*/

                match trimmed.replace(',', ".").parse::<f64>() {
                    Ok(num) => {
                        //println!("We actually identifed a number here '{}' :'{}'", headers[i], num);
                        if let Some(factor) = ret.factors.get_mut(&headers[i]) {
                            //println!("But we also have a factor for that column '{}'!", headers[i]);
                            let (idx, col_to_add, all_cols) = factor.push( &num.to_string() );
                            let alt = if idx.is_nan() { f64::NAN }else { 0.0 };
                            match all_cols {
                                Some(cols) => {
                                    expanded += cols.len();
                                    // fill the original column!
                                    row.push( factor.get_f64(trimmed) ); 
                                    for cname in cols.iter().cloned(){
                                        if cname == col_to_add { 
                                            row.push(idx);
                                        } else {
                                            row.push(alt); 
                                        }
                                    }
                                },
                                None => {
                                    row.push(idx);
                                },
                            }
                        } else {
                            //println!("Push the value {} to column {} at row count {}", num,  headers[i+expanded], row.len());
                            row.push(num as f64);
                        }
                    }
                    Err(_) => {
                        //println!("Not a number here '{}' :'{}'", headers[i], trimmed);
                        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                            if !ret.factors.contains_key(&headers[i+expanded]) {
                                row.push(f64::NAN);
                                continue;
                            }
                        }
                        // Treat as categorical
                        let factor = ret.factors
                        .entry(headers[i+expanded].clone())
                        .or_insert_with(|| Factor::with_empty(raw_rows.len(), &headers[i+expanded]) );
                        let (idx, col_to_add, all_cols)= factor.push(trimmed);
                        let alt = if idx.is_nan() { f64::NAN }else { 0.0 };
                        match all_cols {
                            Some(cols) => {
                                expanded += cols.len();
                                // fill the original column!
                                row.push( factor.get_f64(trimmed) ); 
                                for cname in cols.iter(){
                                    if *cname == col_to_add { 
                                        row.push(idx) 
                                    } else { 
                                        row.push(alt) 
                                    }
                                }
                            },
                            None=>{
                                if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                                    row.push(f64::NAN);
                                }else {
                                    row.push(idx);
                                }
                            }
                        }
                    }
                }
            }
            raw_rows.push(row);
        }

        // 5. create the numeric_data from the Vec<Vec<f64>>
        ret.headers = headers;
        let n_rows = raw_rows.len();
        let mut numeric_data = Array2::<f64>::zeros((n_rows, n_cols));

        for (i, row) in raw_rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                numeric_data[[i, j]] = val;
            }
        }
        // 6. store the Array2 in the object
        ret.numeric_data = numeric_data;        


        // 7. Save factors if the file does not exists
        if ! factors_file.as_ref().exists() {
            println!("Saved a new factors file to fine tune the factors: '{:?}'", &factors_file);
            ret.save_factors(&factors_file)?;
        }

        Ok(ret)
    }

    pub fn data_summary( &self ) {
        return ;
        eprintln!("Shape: {} rows x {} columns", self.numeric_data.nrows(), self.numeric_data.ncols());

        // 2️⃣ Check first few rows
        eprintln!("First 5 rows:");
        for i in 0..self.numeric_data.nrows().min(5) {
            eprintln!("{:?}", self.numeric_data.row(i).to_vec());
        }

        // 3️⃣ Column-wise summaries
        for j in 0..self.numeric_data.ncols().min(10) {
            let col = self.numeric_data.column(j);
            let n_total = col.len();
            let n_na = col.iter().filter(|v| v.is_nan()).count();
            let mean = col.iter().filter(|v| !v.is_nan()).sum::<f64>() / (n_total - n_na) as f64;
            eprintln!(
                "Col {}: NA fraction = {:.2}, mean of non-NA = {:.2}", 
                j, n_na as f64 / n_total as f64, mean
            );
        }
    }

    fn factors_extra_columns( &self ) -> usize {
        let mut ret = 0;
        for factor in &self.factors {
            ret += factor.1.extra_columns()
        }
        ret
    }

    /// Remove all rows that contain any NaN in numeric_data
    pub fn filter_all_na_rows(&mut self, usable:&Vec<String>) {
        let n_rows = self.numeric_data.nrows();
        let n_cols = self.numeric_data.ncols();

        // Build a lookup of usable column indices
        let usable_indices: Vec<usize> = usable
            .iter()
            .filter_map(|col| self.headers.iter().position(|h| h == col))
            .collect();

        // Determine which rows to keep
        let keep_rows: Vec<usize> = (0..n_rows)
            .filter(|&i| {
                !usable_indices
                    .iter()
                    .any(|&j| self.numeric_data[[i, j]].is_nan())
            })
            .collect();


        // Rebuild numeric_data with only the kept rows
        let mut filtered = Array2::<f64>::zeros((keep_rows.len(), n_cols));
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

        println!(
            "Filtered out {} rows containing NaNs. Remaining rows: {}",
            n_rows - keep_rows.len(),
            keep_rows.len()
            );
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
        let scaled = self.numeric_data.clone();
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
            for j in 0..n_cols {
                if self.factors.contains_key(&self.headers[j]) {
                    // factor column → use mode of non-NA values
                    let mut counts = std::collections::HashMap::new();
                    for &v in &vals { *counts.entry(v as usize).or_insert(0) += 1; }
                    let &mode = counts.iter().max_by_key(|(_, c)| *c).unwrap().0;
                    col_means[j] = mode as f64;  // for later distance scaling, can still normalize if needed
                } else {
                    // numeric → mean
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    col_means[j] = mean;
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

                let imputed = if self.factors.contains_key(&self.headers[col]) {
                    // Factor column → mode of neighbors
                    let mut counts = std::collections::HashMap::new();
                    for &(v, _) in &vals {
                        let level = v.round() as usize; // round to nearest valid factor index
                        *counts.entry(level).or_insert(0) += 1;
                    }
                    *counts.iter().max_by_key(|(_, c)| *c).unwrap().0 as f64
                } else {
                    // Numeric → weighted or regular mean
                    if weighted {
                        let mut num = 0.0;
                        let mut den = 0.0;
                        for (v, d) in vals {
                            let w = 1.0 / (d + eps);
                            num += w * v;
                            den += w;
                        }
                        num / den
                    } else {
                        vals.iter().map(|(v, _)| *v).sum::<f64>() / vals.len() as f64
                    }
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

    /// Remove columns with variance below `threshold`
    pub fn filter_low_var(&mut self, threshold: f64) -> usize{
        return 0;
        let mut keep_cols = Vec::new();
        let mut filtered = 0;
        for j in 0..self.numeric_data.ncols() {
            let col = self.numeric_data.column(j);
            let vals: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            if vals.is_empty() {
                filtered+=1;
                println!("Dropping empty column: {}", self.headers[j]);
                continue; 
            }

            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var  = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;

            if var > threshold {
                keep_cols.push(j);
            } else {
                filtered+=1;
                println!("Dropping low-variance column (mean {mean:2e}, var {var:2e}): {}", self.headers[j]);
            }
        }

        // Rebuild numeric_data and headers
        let mut filtered_data = Array2::<f64>::zeros((self.numeric_data.nrows(), keep_cols.len()));
        let mut new_headers = Vec::with_capacity(keep_cols.len());

        for (new_j, &old_j) in keep_cols.iter().enumerate() {
            filtered_data.column_mut(new_j).assign(&self.numeric_data.column(old_j));
            new_headers.push(self.headers[old_j].clone());
        }

        self.numeric_data = filtered_data;
        self.headers = new_headers;

        // Remove factors whose columns were dropped
        self.factors.retain(|name, _| self.headers.contains(name));
        filtered
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
    /*
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
    }*/

    /// Return numeric data as ndarray, optionally selecting columns
    #[allow(dead_code)]
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
        let idx = self.headers.iter().position(|h| h == column)
            .unwrap_or_else(|| panic!("Column '{}' not found in the dataset; all columns: \n{}", column, self.headers.join("\n")));
        self.numeric_data.column(idx).to_vec()
    }

    /// Return a single column as Vec<u8>
    pub fn as_vec_u8(&self, column: &str) -> Vec<u8> {
        let idx = self.headers.iter().position(|h| h == column)
            .unwrap_or_else(|| panic!("Column '{}' not found in the dataset; all columns: \n{}", column, self.headers.join("\n")));
        self.numeric_data.column(idx).iter().map(|&v| v as u8).collect()
    }

    /// Return a single column as Option<Vec<String>> - if it is a factor
    pub fn as_vec_string(&self, column: &str) -> Option<Vec<String>> {
        let idx = self
        .headers
        .iter()
        .position(|h| h == column)
        .expect("Column not found");
        if self.factors.contains_key( column ){
            let fact = self.factors.get( column ).unwrap();
            //println!("I will use the factor {fact:?} to translate the columns ids like {:?}",  self.numeric_data.column(idx).iter().take(10).collect::<Vec<_>>() );
            Some( self.numeric_data.column(idx).iter()
                    .map(|&v| fact.get_value( v as usize) )  // translate f64 -> String
                    .collect())
        }else {
            //eprintln!("Column '{column}' is no Factor here\n{:?}\nFactors I have: \n{:?}\n", self.headers, self.factors.keys() );
            None
        }
        
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
                if val.is_nan() {
                    record.push("NA".to_string());
                }else {
                    record.push(val.to_string());
                }
            }

            wtr.write_record(&record)?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Save all factors to a JSON file
    pub fn save_factors<P: AsRef<Path>>(&self, path: P) -> Result<()>  {
        let defs: Vec<FactorJson> = self.factors.iter()
        .map(|(col, factor)| factor.as_json( col ) )
        .collect();
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &defs)?;
        Ok(())
    }

    /// Load factors from a JSON file
    pub fn load_factors<P: AsRef<Path>>(&mut self, path: P) -> Result<()>  {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let defs: Vec<FactorJson> = serde_json::from_reader(reader)?;
        for def in defs {
            self.factors.insert(def.column.clone(), Factor::from_def(&def));
        }
        Ok(())
    }

}