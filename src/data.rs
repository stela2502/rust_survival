use std::collections::{HashMap, HashSet};
use std::path::{Path};
use csv::{ WriterBuilder};
use ndarray::{Array2, s};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Factor {
    column_name: String,
    indices: Vec<f64>,                // 0.0..n-1.0 for levels, NaN for missing
    pub levels: Vec<String>,              // level labels
    pub level_to_index: HashMap<String, f64>, // fast lookup
    matching:Option<Vec<String>>, // this could match to multiple column names. Like SNP or something
    one_hot: bool, // NEW
}


impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Factor '{}':", self.column_name)?;
        writeln!(f, "  One-hot: {}", self.one_hot)?;
        writeln!(f, "  Levels: {:?}", self.levels)?;
        writeln!(f, "  Matching: {:?}", self.matching)?;
        writeln!(f, "  All columns: {:?}", self.all_column_names())
    }
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
    
    /// Create a new Factor with a column name and one_hot flag
    pub fn new(column_name: &str, one_hot: bool) -> Self {
        Factor {
            column_name: column_name.to_string(),
            indices: Vec::new(),
            levels: Vec::new(),
            level_to_index: HashMap::new(),
            matching: None,
            one_hot,
        }
    }

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

    /// returns all one_hot column names or the original colum, name only
    pub fn all_column_names(&self ) -> Vec<String> {
        if self.one_hot {
             self
                .levels
                .iter()
                .map(|lvl| self.build_one_hot_column(lvl)).collect()
        }else {
            //vec![]
            vec![ self.column_name.to_string() ]
        }
    }

    fn build_one_hot_column( &self, value:&str) -> String {
        if self.one_hot {
            format!("{}_{}", self.column_name, value)
        }else {
            self.column_name.clone()
        }
        
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
            //println!("See we have a one_hot here! {} - trimmed {}", self.column_name, trimmed);
            let idx = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                f64::NAN
            }else {
                1.0
            };
            // all other levels become zero columns
            let zero_cols = self.all_column_names( );
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
        //println!("   We '{}' return a value of {}", ret.1, ret.0);
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
                .or_insert_with(|| Factor::with_empty(0, &header));
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

    fn header_error( &self, i: usize, expanded: usize) -> String {
        let mut parts:Vec<String> = Vec::with_capacity( self.factors.len() + 1 );
        for factor in self.factors.values() {
            parts.push( format!("{}", factor ) );
        }
        parts.push(format!("Headers have the length {} and we would want id {}:\n{}", self.headers.len(), i+expanded, self.headers
                            .iter()
                            .enumerate()
                            .map(|(idx, h)| format!("{}: {}", idx, h))
                            .collect::<Vec<_>>()
                            .join("\n") ) );

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
            //println!("Column '{column}' is no Factor here\n{:?}\nFactors I have: \n{:?}\n", self.headers, self.factors.keys() );
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

#[cfg(test)]
mod tests_one_hot_factors {
    use super::*;
    use std::fs;
    use std::collections::HashSet;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_factors() -> Result<(), Box<dyn std::error::Error>> {
        // --- Temp directory ---
        let dir =  PathBuf::from(env!("CARGO_MANIFEST_DIR"));
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

        // --- Save factors to JSON ---
        data.save_factors(&factors_path)?;

        // --- Read JSON file back ---
        data.factors = HashMap::new();
        let json_content = fs::read_to_string(&factors_path)?;
        println!("Saved JSON:\n{}", json_content);
        data.load_factors(&factors_path)?;

        // --- Deserialize to check ---
        let loaded_factors = data.factors;

        // 1️⃣ Check that both factors exist
        assert!(loaded_factors.contains_key("Color"), "Color factor exists after load");
        assert!(loaded_factors.contains_key("Number"), "Number factor exists after load");

        // 2️⃣ Check levels for Color
        let color = loaded_factors.get("Color").unwrap();
        assert_eq!(color.levels, vec!["Red", "Blue", "Green"], "Color levels match");
        assert_eq!(color.one_hot, true, "Color one_hot flag preserved");

        // 3️⃣ Check levels for Number
        let number = loaded_factors.get("Number").unwrap();
        assert_eq!(number.levels, vec!["1", "2", "3"], "Number levels match");
        assert_eq!(number.one_hot, true, "Number one_hot flag preserved");

        // 4️⃣ Check level_to_index maps
        for (idx, lvl) in color.levels.iter().enumerate() {
            let mapped_idx = color.level_to_index.get(lvl).unwrap();
            assert_eq!(*mapped_idx, idx as f64, "Color level_to_index correct for {}", lvl);
        }

        for (idx, lvl) in number.levels.iter().enumerate() {
            let mapped_idx = number.level_to_index.get(lvl).unwrap();
            assert_eq!(*mapped_idx, idx as f64, "Number level_to_index correct for {}", lvl);
        }

        // 5️⃣ Check that indices vector is empty (not yet populated)
        assert!(color.indices.is_empty(), "Color indices empty after load");
        assert!(number.indices.is_empty(), "Number indices empty after load");

        assert_eq!( color.all_column_names(), vec![ "Color_Red".to_string(), "Color_Blue".to_string(),"Color_Green".to_string()] ,"Color all headers");
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
}