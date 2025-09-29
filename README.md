# rust_survival

`rust_survival` is a Rust-based survival analysis tool that supports RSF, Cox models, and point scoring.

## Overview

The workflow:

1. **RSF** – Random Survival Forest for feature selection  
2. **Cox** – Fit a Cox proportional hazards model  
3. **Points** – Assign point scores based on hazard ratios  

## Installation

Currently available from GitHub:

```bash
git clone https://github.com/stela2502/rust_survival/
cd rust_survival
cargo install --path .
```

## CLI Usage

Run `rust_survival` to see main commands:

```bash
rust_survival -h
```

### Output:

```text
RSF -> Cox -> Points

Usage: rust_survival <COMMAND>

Commands:
  train  Train a model from CSV dataset
  test   Apply a saved model to new data
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```


## Train a Model

Check the options for training:

```bash
rust_survival train -h
```

### Output:

```text
Train a model from CSV dataset

Usage: rust_survival train [OPTIONS] --file <FILE> --time-col <TIME_COL> --status-col <STATUS_COL> --model <MODEL>

Options:
  -f, --file <FILE>                    Path to CSV dataset
  -p, --patient-col <PATIENT_COL>      Name of the Pateient ID column: default first column
  -t, --time-col <TIME_COL>            Name of survival time column
  -s, --status-col <STATUS_COL>        Name of event status column (0/1)
  -e, --exclude-cols <EXCLUDE_COLS>    Future measurements after diagnosis
  -c, --categorical <CATEGORICAL>      Comma-separated categorical columns [default: ]
  -n, --n-trees <N_TREES>              Number of trees for RSF [default: 100]
  -m, --min-node-size <MIN_NODE_SIZE>  Minimum node size in RSF trees [default: 5]
      --top-n <TOP_N>                  Number of top variables to select for Cox [default: 5]
      --base-hr <BASE_HR>              Base hazard ratio for 1 point [default: 1.2]
  -d, --delimiter <DELIMITER>          CSV delimiter [default: "\t"]
  -m, --model <MODEL>                  File path to save trained model
      --summary <SUMMARY>              Summary Stats for one column name
  -h, --help                           Print help
```


## Test a Saved Model


⚠️ **Important:** To test a model, your dataset **must include all columns used in the original Cox model**. The required column names can be found in the module file `rust_survival train` did produce (json format). 

Make sure the column names in your CSV match exactly (case, spaces, and punctuation) to avoid errors during model training.

Check the options for testing:

```bash
rust_survival test -h
```

### Output:

```text
Apply a saved model to new data

Usage: rust_survival test [OPTIONS] --file <FILE> --model <MODEL>

Options:
  -f, --file <FILE>                Path to CSV dataset
  -p, --patient-col <PATIENT_COL>  Name of the Pateient ID column: default first column
  -m, --model <MODEL>              Path to saved Cox/RSF model
  -o, --output <OUTPUT>            Optional output CSV
  -d, --delimiter <DELIMITER>      CSV delimiter [default: "\t"]
  -c, --categorical <CATEGORICAL>  Comma-separated categorical columns [default: ]
      --base-hr <BASE_HR>          Base hazard ratio for 1 point [default: 1.2]
  -h, --help                       Print help
```


## Notes

- The patient column is optional; if not provided, row indices are used as IDs.  
- Categorical columns are automatically encoded and handled during model training.  
- The `--exclude-cols` option allows you to exclude features measured **after diagnosis**.  
- Summary statistics can be generated for any factor column using `--summary`.  

---

For the latest updates, visit the GitHub repository:  
[https://github.com/stela2502/rust_survival/](https://github.com/stela2502/rust_survival/)