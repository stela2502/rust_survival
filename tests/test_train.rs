use assert_cmd::Command;
use std::fs;
use rust_survival::cox::CoxModel;


const TRAIN_CSV: &str = "tests/data/survival_lung.csv";
const MODEL_JSON: &str = "tests/data/survival_lung_factors.json";
const PRED_CSV: &str = "tests/data/tmp_predictions.csv";

#[test]
fn integration_train_and_test() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure no leftover files
    let _ = fs::remove_file(MODEL_JSON);
    let _ = fs::remove_file(PRED_CSV);

    let top_n = "5";

    // --- TRAIN ---
    let mut train_cmd = Command::cargo_bin("rust_survival")?;
    train_cmd.args(&[
        "train",
        "--file", TRAIN_CSV,
        "--top-n,", top_n,
        "--time-col", "time",
        "--status-col", "status",
        "--model", MODEL_JSON,
    ]);

    train_cmd.assert()
        .success();

    let model: CoxModel = CoxModel::from_file(MODEL_JSON).expect("Failed to load model");
    assert!(!model.coefficients.is_empty(), "Model must have coefficients");
    assert_eq!(model.coefficients.len(), model.hr.len(), "HRs must match coefficients");
    assert_eq!(model.coefficients.len(), 5, "got exacly as many coefficients as I asked for");


    // Check that model file was created
    assert!(fs::metadata(MODEL_JSON)?.is_file());

    // Check that predictions file was created
    assert!(fs::metadata(PRED_CSV)?.is_file());

    Ok(())
}


#[test]
fn integration_test_only() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure predictions file is removed before the test
    let _ = fs::remove_file(PRED_CSV);

    // --- TEST ---
    let mut test_cmd = Command::cargo_bin("rust_survival")?;
    test_cmd.args(&[
        "test",
        "--file", TRAIN_CSV,
        "--model", MODEL_JSON,
        "--output", PRED_CSV,
    ]);

    test_cmd.assert()
        .success();
    // Check that predictions CSV was created
    assert!(fs::metadata(PRED_CSV)?.is_file());

    Ok(())
}
