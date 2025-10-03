use std::collections::{HashMap, HashSet};
use std::path::{Path};
use csv::{ WriterBuilder};
use ndarray::{Array2, Axis, s};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::fmt;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::data::{Factor, factor::FactorJson};


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

impl fmt::Display for SurvivalData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SurvivalData Summary:")?;
        writeln!(f, "Rows: {}, Columns: {}", self.numeric_data.nrows(), self.numeric_data.ncols())?;
        writeln!(f, "Headers:")?;
        for (idx, header) in self.headers.iter().enumerate() {
            writeln!(f, "  {}: {}", idx, header)?;
        }
        writeln!(f, "Factors:")?;
        for (_name, factor) in &self.factors {
            writeln!(f,"{}",factor );
        }
        writeln!(f, "Excluded columns: {:?}", self.exclude)
    }
}


impl SurvivalData {

    /// Read CSV file and build numeric + factor representation
    pub fn from_file<P: AsRef<Path> + std::fmt::Debug, FF: AsRef<Path> + std::fmt::Debug>(file_path: P, delimiter: u8, 
        categorical_cols: HashSet<String>, factors_file: FF) -> Result<Self> {

        // 1. Start with an empty SurvivalData
        let mut ret = SurvivalData::default();

        // 2. Load factors if the file exists, otherwise keep empty
        if factors_file.as_ref().exists() {
            println!("Factors are loaded from file");
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
                    if fact.one_hot {
                        let mut c = vec![s.to_string()];
                        c.extend(fact.all_column_names());
                        c
                    } else {
                        vec![s.to_string()]      // just the original column
                    }
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
                println!("Forcing header {header} to be a factor");
                ret.factors.entry(header.clone())
                .or_insert_with(|| Factor::new( &header, false));
            }
        }
        ret.headers = headers.clone();

        // 4. lead the data and handle the factors
        for result in rdr.records() {
            let record = result?;
            let mut row: Vec<f64> = Vec::with_capacity(n_cols);
            let mut expanded = 0;
            for (i, value) in record.iter().enumerate() {
                let trimmed = value.trim().trim_matches('"');
                match trimmed.replace(',', ".").parse::<f64>() {
                    Ok(num) => {
                        //println!("We actually identifed a number here '{}' :'{}'", headers[i+expanded], num);
                        if let Some(factor) = ret.factors.get_mut(&headers[i+expanded]) {
                            //println!("   But we also have a factor for that column '{}'!", headers[i+expanded]);
                            //let mut parts= Vec::<String>::with_capacity( factor.levels.len() +1);
                            
                            let (idx, col_to_add, all_cols) = factor.push( &num.to_string() );
                            let alt = if idx.is_nan() { f64::NAN }else { 0.0 };

                            match all_cols {
                                Some(cols) => {
                                    expanded += cols.len();
                                    // fill the original column!
                                    //parts.push(format!("dense factor: {}", factor.get_f64(trimmed) ) );
                                    row.push( factor.get_f64(trimmed) ); 
                                    for cname in cols.iter().cloned(){
                                        if cname == col_to_add { 
                                            //parts.push(format!("hot: {}", idx ) );
                                            row.push(idx);
                                        } else {
                                            //parts.push(format!("hot: {}", alt ) );
                                            row.push(alt); 
                                        }
                                    }
                                },
                                None => {
                                    //parts.push(format!("No hot columns!? dense it is then : {}", idx ) );
                                    row.push(idx);
                                },
                            }
                            //println!("   Adding the values {}", parts.join(", "));
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
                        //eprintln!("looking for a not hot column name at {} + {}: {}",i, expanded, i+expanded);
                        if headers.len() == i+expanded {
                            eprintln!( "i {}; expanded {}", i, expanded );
                            panic!( "{}", ret.header_error( i, expanded ) );
                        }
                        let factor = ret.factors
                        .entry(headers[i+expanded].clone())
                        .or_insert_with(|| Factor::new(&headers[i+expanded], false) );
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
            assert_eq!(row.len() , record.len() + expanded, "After reading one row: Row length and expeceted row length do not aligne {} vs {}:\n{}\n and the row values:\n{:?}", 
                row.len() , record.len() + expanded, ret.header_error( record.len() , expanded ), row );
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
            panic!("
Please review and update the factors file so that it accurately reflects the logic in the data.

The factors file is a JSON-formatted file, for example:

[
  {{
    'column': 'status2',
    'levels': [
      '1',
      '0'
    ],
    'numeric': [
      0.0,
      1.0
    ],
    'matching': null,
    'one_hot': false
  }}
]
In this example, there is an error: the numeric values do not match the actual data. They should be [1.0, 0.0]. Once corrected, the factor will work as expected.


The one_hot option allows the factor to be expanded into multiple 0.0/1.0 columns—two in this case. This is particularly useful when the factor levels have no inherent numeric order or relationship:


Factor: tp53_mutation_type
Levels: Missense, Nonsense, Frameshift, Splice_site, Silent
Notes: This factor is categorical with no inherent order, so it’s a good candidate for one-hot encoding in a model. Each level would be represented as a separate binary column (0/1) if one-hot encoding is used.

                ");
        }

        Ok(ret)
    }

    /// Split into train and test by a fraction (e.g. 0.7 = 70% train, 30% test).
    pub fn train_test_split(&self, train_fraction: f64) -> (SurvivalData, SurvivalData) {
        assert!(train_fraction > 0.0 && train_fraction < 1.0);

        let n_rows = self.numeric_data.nrows();
        let mut indices: Vec<usize> = (0..n_rows).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let train_size = (n_rows as f64 * train_fraction).round() as usize;

        let train_idx = &indices[..train_size];
        let test_idx = &indices[train_size..];

        let train_data = self.numeric_data.select(Axis(0), train_idx);
        let test_data = self.numeric_data.select(Axis(0), test_idx);

        let make_subset = |data: Array2<f64>| SurvivalData {
            headers: self.headers.clone(),
            numeric_data: data,
            factors: self.factors.clone(),
            exclude: self.exclude.clone(),
        };

        (make_subset(train_data), make_subset(test_data))
    }

    fn header_error( &self, i: usize, expanded: usize) -> String {
        let mut parts:Vec<String> = Vec::with_capacity( self.factors.len() + 1 );
        for factor in self.factors.values() {
            parts.push( format!("{}", factor ) );
        }
        parts.push(format!("Headers have the length {} and we would want id {}:\n{}", 
            self.headers.len(), i+expanded, 
            self.headers(20).join("\n") ) );

        parts.join("\n")
    }

    pub fn data_summary( &self ) {
        return ;
        println!("Shape: {} rows x {} columns", self.numeric_data.nrows(), self.numeric_data.ncols());

        // 2️⃣ Check first few rows
        println!("First 5 rows:");
        for i in 0..self.numeric_data.nrows().min(5) {
            println!("{:?}", self.numeric_data.row(i).to_vec());
        }

        // 3️⃣ Column-wise summaries
        for j in 0..self.numeric_data.ncols().min(10) {
            let col = self.numeric_data.column(j);
            let n_total = col.len();
            let n_na = col.iter().filter(|v| v.is_nan()).count();
            let mean = col.iter().filter(|v| !v.is_nan()).sum::<f64>() / (n_total - n_na) as f64;
            println!(
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
        println!("filter_all_na_rows got {} rows and {} columns and checks {} of these columns for na's", 
            n_rows, n_cols, usable.len() );

        // Build a lookup of usable column indices
        let usable_indices: Vec<usize> = usable
            .iter()
            .filter_map(|col| self.headers.iter().position(|h| h == col))
            .collect();
        if usable_indices.len() == n_rows {
            println!("No rows containing NA values found");
            return
        }
        // Determine which rows to keep
        let keep_rows: Vec<usize> = (0..n_rows)
            .filter(|&i| {
                !usable_indices
                    .iter()
                    .any(|&j| self.numeric_data[[i, j]].is_nan())
            })
            .collect();
        if keep_rows.len() == n_rows {
            println!("No na's found in the {} columns", usable.len() );
            return
        }

        // Rebuild numeric_data with only the kept rows
        let mut filtered = Array2::<f64>::zeros((keep_rows.len(), n_cols));
        for (new_i, &old_i) in keep_rows.iter().enumerate() {
            filtered
            .row_mut(new_i)
            .assign(&self.numeric_data.slice(s![old_i, ..]));
        }
        self.numeric_data = filtered;

        // Filter factors: keep only entries for kept rows
        let mut new_factors = HashMap::<String,Factor>::new();

        for factor in self.factors.values() {
            if let Some(col_id) = self.headers.iter().position(|h| h == &factor.column_name) {
                new_factors.insert( 
                    factor.column_name.clone(), 
                    factor.subset( &self.numeric_data, col_id ) 
                );
            }else {
                panic!("a previousely known column has vanished?! {}", factor.column_name );
            }
        }
        self.factors = new_factors;

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
            .unwrap_or_else(|| panic!("Column '{}' not found in the dataset; all columns: \n{}", 
                column, self.headers(20).join("\n")));
        self.numeric_data.column(idx).to_vec()
    }

    /// Return a single column as Vec<u8>
    pub fn as_vec_u8(&self, column: &str) -> Vec<u8> {
        let idx = self.headers.iter().position(|h| h == column)
            .unwrap_or_else(|| 
                panic!("Column '{}' not found in the dataset; all columns: \n{}\nColumn '{}' not found in the dataset", 
                    column, self.headers(20).join("\n") ,column ));
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
                    .map(|&v| fact.get_string( v ) )
                    .collect())
        }else {
            //println!("Column '{column}' is no Factor here\n{:?}\nFactors I have: \n{:?}\n", self.headers, self.factors.keys() );
            None
        }
        
    }

    pub fn headers(&self, max: usize) -> Vec<&str> {
        self.headers
            .iter()                      // borrow each String
            .take(self.headers.len().min(max))  // take at most `max`
            .map(|s| s.as_str())         // convert &String → &str
            .collect()                   // collect into Vec<&str>
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

#[cfg(test)]
mod tests_one_hot_factors {
    use super::*;
    use std::fs;
    use std::collections::HashSet;

    #[test]
    fn test_save_and_load_factors() -> Result<(), Box<dyn std::error::Error>> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let factors_path = dir.as_path().join("factors.json");

        // --- Create Factors ---
        let mut color = Factor::new("Color", false);
        let _ = color.push("Red");
        let _ = color.push("Blue");
        let _ = color.push("Green");
        color.one_hot = true;

        let mut number = Factor::new("Number", false);
        let _ = number.push("1");
        let _ = number.push("2");
        let _ = number.push("3");
        number.one_hot = true;

        // --- Insert into SurvivalData ---
        let mut data = SurvivalData::default();
        data.factors.insert("Color".to_string(), color);
        data.factors.insert("Number".to_string(), number);

        // --- Save to JSON ---
        data.save_factors(&factors_path)?;

        // --- Load from JSON ---
        data.factors = HashMap::new();
        data.load_factors(&factors_path)?;
        let loaded_factors = &data.factors;

        // Check factors exist
        assert!(loaded_factors.contains_key("Color"));
        assert!(loaded_factors.contains_key("Number"));

        // Check levels
        let color = loaded_factors.get("Color").unwrap();
        assert_eq!(color.get_levels(), vec!["Red", "Blue", "Green"]);
        assert!(color.one_hot);

        let number = loaded_factors.get("Number").unwrap();
        assert_eq!(number.get_levels(), vec!["1", "2", "3"]);
        assert!(number.one_hot);

        Ok(())
    }
}

#[cfg(test)]
use std::path::PathBuf;

mod tests_survival_data_from_file_one_hot_factors {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::collections::HashSet;
    use tempfile::tempdir;

    #[test]
    fn test_survivaldata_one_hot_factors() -> Result<(), Box<dyn std::error::Error>> {
        // --- Prepare temporary directory ---
        let dir =  PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let csv_path = dir.as_path().join("test_factors.csv");
        let factors_path = dir.as_path().join("test_factors_factors.json");

        // --- Write CSV file ---
        // Two categorical columns: Color, Number
        // Values: Color = Red Blue Blue Green Blue Red Na Na
        //         Number = 1 2 1 3 2 1 Na Na
        let csv_content = "\
Color,Number
Red,1
Blue,2
Blue,1
Green,3
Blue,2
Red,1
Na,Na
Na,Na
";
        let mut file = File::create(&csv_path)?;
        file.write_all(csv_content.as_bytes())?;

        let mut color = Factor::new("Color", false);
        let _ = color.push("Red");
        let _ = color.push("Blue");
        let _ = color.push("Green");
        color.one_hot = true;
        
        let mut number = Factor::new("Number", false);
        let _ = number.push("1");
        let _ = number.push("2");
        let _ = number.push("3");
        number.one_hot = true;
        let mut temp = SurvivalData::default();
        temp.factors.insert( "Color".to_string(), color);
        temp.factors.insert( "Nubmer".to_string(), number);

        temp.save_factors(&factors_path); //default location
        let categorical_cols= HashSet::<String>::new();
        // --- Load SurvivalData ---
        let data = SurvivalData::from_file(&csv_path, b',', categorical_cols, &factors_path)?;

        // --- Assertions ---

        // 1️⃣ Check original factor columns contain indices / NaN
        println!("The data object: {data}");
        let color_idx_col = data.headers.iter().position(|h| h == "Color").unwrap();
        let number_idx_col = data.headers.iter().position(|h| h == "Number").unwrap();
        assert_eq!(color_idx_col, 0, "expected color_idx_col 0 - have these cols: {:?}", &data.headers);
        assert_eq!(number_idx_col, 4, "expected number_idx_col 4 - have these cols: {:?}", &data.headers);

        assert_eq!(data.numeric_data[[0, color_idx_col]], 0.0); // Red first index 0
        assert_eq!(data.numeric_data[[1, color_idx_col]], 1.0); // Blue index 1
        assert!(data.numeric_data[[6, color_idx_col]].is_nan()); // Na

        assert_eq!(data.numeric_data[[0, number_idx_col]], 0.0); // 1 -> index 0
        assert_eq!(data.numeric_data[[1, number_idx_col]], 1.0); // 2 -> index 1
        assert!(data.numeric_data[[6, number_idx_col]].is_nan()); // Na

        assert_eq!(data.headers[1], "Color_Red");


        // Find the indices of the one-hot columns
        let color_red_idx = data.headers.iter().position(|h| h == "Color_Red").unwrap();
        let color_blue_idx = data.headers.iter().position(|h| h == "Color_Blue").unwrap();
        let color_green_idx = data.headers.iter().position(|h| h == "Color_Green").unwrap();

        let number_1_idx = data.headers.iter().position(|h| h == "Number_1").unwrap();
        let number_2_idx = data.headers.iter().position(|h| h == "Number_2").unwrap();
        let number_3_idx = data.headers.iter().position(|h| h == "Number_3").unwrap();

        // --- Expected one-hot values per row ---
        // Rows: Color = Red, Blue, Blue, Green, Blue, Red, NA, NA
        //       Number = 1,2,1,3,2,1,NA,NA
        let expected_color = vec![
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Blue
            (0.0, 1.0, 0.0), // Blue
            (0.0, 0.0, 1.0), // Green
            (0.0, 1.0, 0.0), // Blue
            (1.0, 0.0, 0.0), // Red
            (f64::NAN, f64::NAN, f64::NAN), // NA
            (f64::NAN, f64::NAN, f64::NAN), // NA
        ];

        let expected_number = vec![
            (1.0, 0.0, 0.0), // 1
            (0.0, 1.0, 0.0), // 2
            (1.0, 0.0, 0.0), // 1
            (0.0, 0.0, 1.0), // 3
            (0.0, 1.0, 0.0), // 2
            (1.0, 0.0, 0.0), // 1
            (f64::NAN, f64::NAN, f64::NAN), // NA
            (f64::NAN, f64::NAN, f64::NAN), // NA
        ];

        for row in 0..data.numeric_data.nrows() {
            // Color
            let val_red = data.numeric_data[[row, color_red_idx]];
            let val_blue = data.numeric_data[[row, color_blue_idx]];
            let val_green = data.numeric_data[[row, color_green_idx]];

            let (exp_red, exp_blue, exp_green) = expected_color[row];
            if exp_red.is_nan() {
                assert!(val_red.is_nan());
                assert!(val_blue.is_nan());
                assert!(val_green.is_nan());
            } else {
                assert_eq!(val_red, exp_red);
                assert_eq!(val_blue, exp_blue);
                assert_eq!(val_green, exp_green);
            }

            // Number
            let val_1 = data.numeric_data[[row, number_1_idx]];
            let val_2 = data.numeric_data[[row, number_2_idx]];
            let val_3 = data.numeric_data[[row, number_3_idx]];

            let (exp_1, exp_2, exp_3) = expected_number[row];
            if exp_1.is_nan() {
                assert!(val_1.is_nan(), "expected NA value for Number_1 at row {row}; all data {}",data.numeric_data);
                assert!(val_2.is_nan(), "expected NA value for Number_2 at row {row}; all data {}",data.numeric_data);
                assert!(val_3.is_nan(), "expected NA value for Number_3 at row {row}; all data {}",data.numeric_data);
            } else {
                assert_eq!(val_1, exp_1, "Number_1 mismatch at row {}/{}: {} != {} all data {}", number_1_idx, row, val_1, exp_1,data.numeric_data);
                assert_eq!(val_2, exp_2, "Number_2 mismatch at row {}/{}: {} != {} all data {}", number_2_idx, row, val_2, exp_2,data.numeric_data);
                assert_eq!(val_3, exp_3, "Number_3 mismatch at row {}/{}: {} != {} all data {}", number_3_idx, row, val_3, exp_3,data.numeric_data);
            }
        }

        Ok(())
    }

    #[test]
    fn test_survivaldata_train_test_split_unique() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;
        use tempfile::tempdir;

        // --- Prepare temporary directory and files ---
        let dir = tempdir()?;
        let csv_path = dir.path().join("unique_data.csv");
        let factors_path = dir.path().join("unique_factors.json");

        // --- Generate CSV data with 10 rows, 10 columns, unique values 1..100 ---
        let nrows = 10;
        let ncols = 10;
        let mut csv_content = String::new();
        // header
        csv_content.push_str(&(0..ncols).map(|i| format!("Col{}", i)).collect::<Vec<_>>().join(","));
        csv_content.push('\n');
        // data
        let mut val = 1;
        for _ in 0..nrows {
            let row = (0..ncols).map(|_| {
                let s = val.to_string();
                val += 1;
                s
            }).collect::<Vec<_>>().join(",");
            csv_content.push_str(&row);
            csv_content.push('\n');
        }

        // write CSV
        let mut f = File::create(&csv_path)?;
        f.write_all(csv_content.as_bytes())?;

        // --- Create empty factors JSON ---
        let mut f_factors = File::create(&factors_path)?;
        f_factors.write_all(b"[]")?;

        // --- Load data ---
        let categorical_cols = HashSet::<String>::new();
        let data = SurvivalData::from_file(&csv_path, b',', categorical_cols, &factors_path)?;

        // --- Split ---
        let (train, test) = data.train_test_split(0.7);

        // --- Assertions ---
        assert_eq!(train.numeric_data.nrows() + test.numeric_data.nrows(), nrows, "Row counts must add up");
        assert_eq!(train.headers, data.headers);
        assert_eq!(test.headers, data.headers);

        // Check that all rows are unique across train and test
        let train_rows: Vec<Vec<f64>> = train.numeric_data.rows().into_iter().map(|r| r.to_vec()).collect();
        let test_rows: Vec<Vec<f64>> = test.numeric_data.rows().into_iter().map(|r| r.to_vec()).collect();

        for row in &train_rows {
            assert!(!test_rows.contains(row), "Row appeared in both train and test!");
        }

        Ok(())
    }
}