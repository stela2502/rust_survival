use assert_cmd::Command;
use std::fs;
use rust_survival::cox::CoxModel;
use std::path::PathBuf;


fn pretty_cmd(cmd: &Command) -> String {

    let mut parts = vec![];
    if let prog = cmd.get_program() {
        parts.push(format!("{}",prog.display())) ;
    }

    // arguments
    for arg in cmd.get_args() {
        parts.push( format!("{}", arg.display()) );
    }
    parts.join(" ")
}
#[test]

fn integration_train() -> Result<(), Box<dyn std::error::Error>> {

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let train_csv = &format!("{}", base_dir.join("tests/data/survival_lung.csv").display());
    let factors_json = &format!("{}", base_dir.join("tests/data/survival_lung_factors.json").display());
    let model_json = &format!("{}", base_dir.join("tests/data/out/survival_lung_model.json").display());
    let pred_csv = &format!("{}", base_dir.join("tests/data/out/tmp_predictions.csv").display());

    // Ensure no leftover files

    let _ = fs::remove_file(factors_json);
    let _ = fs::remove_file(model_json);
    let _ = fs::remove_file(pred_csv);

    let top_n = "5";

    let exec =  if !cfg!(debug_assertions) {
        "target/release/rust_survival"
    }else {
        "target/debug/rust_survival"
    };

    // --- TRAIN ---
    let mut train_cmd = Command::new(exec);
    train_cmd.args(&[
        "train",
        "--file", &train_csv,
        "--top-n", &top_n,
        "--time-col", "time",
        "--status-col", "status2",
        "--model", &model_json,
        "-c", "status2",
    ]);

    println!("the command:\n{:?}", pretty_cmd(&train_cmd) );

    train_cmd.assert()
        .failure(); // the first time this fails as it creates the factors data!

    train_cmd.assert()
        .success(); // should succeed after the first round 
    let model: CoxModel = CoxModel::from_file(model_json).expect("Failed to load model");
    assert!(!model.coefficients.is_empty(), "Model must have coefficients");
    assert_eq!(model.coefficients.len(), model.hr.len(), "HRs must match coefficients");
    assert_eq!(model.coefficients.len(), 5, "got exacly as many coefficients as I asked for");


    // Check that model file was created
    assert!(fs::metadata(model_json)?.is_file());


    // --- TEST ---
    let mut test_cmd = Command::new(exec);
    test_cmd.args(&[
        "test",
        "--file", &train_csv,
        "--model", &model_json, // use the one created above
        "--output", &pred_csv,
    ]);
    println!("the command:\n{:?}", pretty_cmd(&test_cmd) );
    test_cmd.assert()
        .success();
    // Check that predictions CSV was created
    assert!(fs::metadata(pred_csv)?.is_file());

    Ok(())
}
