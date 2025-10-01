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


## Usage Example

You can dowanoad (the MATABRIC data from kaggle)[www.kaggle.com/datasets/gunesevitan/breast-cancer-metabric].


The command to train a model from this data is this:
```bash
rust_survival train -f ~/Downloads/Breast_Cancer_METABRIC.csv -p "Patient ID" -m ~/Downloads/Breast_Cancer_METABRIC_COX_model.json  -d , --status-col "Overall Survival Status" --n-trees 1000 --time-col "Overall Survival (Months)" --top-n 30 --summary "Patient's Vital Status" --exclude-cols "Type of Breast Surgery,Chemotherapy,Pam50 + Claudin-low subtype,Integrative Cluster,Lymph nodes examined positive ,Mutation Count,Nottingham prognostic index,Overall Survival (Months),Overall Survival Status,Relapse Free Status (Months),Relapse Free Status,Patient's Vital Status"
```

You see we are exluding many columns from the model building, as they might not be available at an early timepoint.
And a predictive model would be best if it would usable early on - right?

When you run this the first time the tool will create a features.json file ``~/Downloads/Breast_Cancer_METABRIC_features.json`` that describes how the not numeric data is translated into purely numeric values needed for the model building (fatorization).

There are two main startegies for building a factor:

1. the not numeric values have a numeric conection like e.g "serverity" "low", "medium" and "high". Thouse could easily be translated into 1,2 and 3.
2. Unrelated measurements like Cancer subtypes. These cancer subtypes should not be used as a single factor, but broken up into multiple 0/1 categories.

The Json file is a simple text file and can be modifies with any text editor like gedit or notepad.
For case 1 you would simply adjust the "numeric" value to the apprpriate number. 
For case 2 you can simply set the "one_hot" value to true.

Afterwards you can simply re-run the model testing.
THis will select likely predictive features using a RandForestSurvival model and later on use these likely predictive columns to build the COX model and from this the hazard values and points system.

Later on you can use a table with only the predictve column to "test" the patients for hazard value and points.

This system is untested and under development until tested, but the results look promising in the way that the model does find differences in the training data. I have not even tested it with fresh data.

Likely close to everything apart from data loading and factorization is up for change.


## Test Case

This repo contains a test dataset exoported from the R Survival package: ``tests/data/survival_lung.csv``.
You can use this data to train a model like this:

```bash


## Implementation

This work heavily relied on ChatGPTs input.
I am not 100\% sure that it performs correctly, but it seams to find reliable connections and can produce somewhat goot results.
Unfortunately I am no expert in this field either.

**log-rank statistic**.  
This is the standard version used in survival trees (and RSF). It uses the *Mantel–Haenszel log-rank test* with variance.  

### Formula (per unique event time \(t\))  
- Let \(Y_{L}(t), Y_{R}(t)\) = number at risk in left/right at \(t\).  
- Let \(d_{L}(t), d_{R}(t)\) = number of events at \(t\) in left/right.  
- Total at risk: \(Y(t) = Y_{L}(t) + Y_{R}(t)\).  
- Total events: \(d(t) = d_{L}(t) + d_{R}(t)\).  
- Expected events in left:  
  \[
  E_L(t) = d(t) \cdot \frac{Y_{L}(t)}{Y(t)}
  \]  
- Variance contribution:  
  \[
  V(t) = \frac{Y_L(t) Y_R(t) d(t) (Y(t) - d(t))}{Y(t)^2 (Y(t) - 1)}
  \]  

Then accumulate:  

\[
Z = \frac{\sum_t (d_L(t) - E_L(t))}{\sqrt{\sum_t V(t)}}
\]

This \(Z\) is the log-rank statistic.  


## Notes

- The patient column is optional; if not provided, row indices are used as IDs.  
- Categorical columns are automatically encoded and handled during model training.  
- The `--exclude-cols` option allows you to exclude features measured **after diagnosis**.  
- Summary statistics can be generated for any factor column using `--summary`.  

---

For the latest updates, visit the GitHub repository:  
[https://github.com/stela2502/rust_survival/](https://github.com/stela2502/rust_survival/)