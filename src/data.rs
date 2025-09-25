use csv::{ReaderBuilder, StringRecord, Trim};
use std::collections::HashSet;
use std::error::Error;
use std::path::Path;

/// Load a CSV/TSV dataset for survival analysis.
/// 
/// # Arguments
/// - `file_path`: path to the CSV/TSV file
/// - `time_col`: column name for survival time
/// - `status_col`: column name for event/censoring status
/// - `categorical_cols`: set of columns to treat as categorical
/// - `delim`: delimiter byte (default: tab)
/// 
/// # Returns
/// Vector of rows (each row is a vector of f64)
pub fn load_csv(
    file_path: &str,
    time_col: &str,
    status_col: &str,
    categorical_cols: &HashSet<String>,
    delim: u8,
) -> Result<(Vec<String>, Vec<Vec<f64>>), Box<dyn Error>> {

    if !Path::new(file_path).exists() {
        return Err(format!("File not found: {}", file_path).into());
    }

    let mut rdr = ReaderBuilder::new()
        .delimiter(delim)
        .trim(Trim::All)
        .flexible(true)
        .from_path(file_path)
        .map_err(|e| format!("Failed to open CSV file '{}': {}", file_path, e))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("Failed to read CSV header: {}", e))?
        .clone();

    // Check that time and status columns exist
    if !headers.iter().any(|h| h == time_col) {
        return Err(format!("Time column '{}' not found in CSV headers", time_col).into());
    }
    if !headers.iter().any(|h| h == status_col) {
        return Err(format!("Status column '{}' not found in CSV headers", status_col).into());
    }

    let mut data = Vec::new();
    let mut na_rows = 0;

    for (i, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| format!("Failed to read row {}: {}", i+2, e))?;
        if record.len() != headers.len() {
            return Err(format!(
                "Row length mismatch at line {}: expected {} fields, found {}",
                i+2,
                headers.len(),
                record.len()
            ).into());
        }
        if record.iter().any(|v| v == "NA" || v.is_empty()) {
            na_rows +=1;
            continue;
        }
        let mut row = Vec::with_capacity(record.len());
        for (j, field) in record.iter().enumerate() {
            let col_name = &headers[j];
            let value: f64 = if categorical_cols.contains(col_name) {
                field.parse::<f64>().map_err(|_| format!(
                    "Failed to parse categorical column '{}' at row {}: '{}'",
                    col_name, i+2, field
                ))?
            } else {
                field.parse::<f64>().map_err(|_| format!(
                    "Failed to parse numeric column '{}' at row {}: '{}'",
                    col_name, i+2, field
                ))?
            };
            row.push(value);
        }
        data.push(row);
    }

    if na_rows != 0 {
        eprintln!("{na_rows} rows contained NA - skipped");
    }

    Ok((headers.into_iter().map(|s| s.to_string()).collect(), data))
}

