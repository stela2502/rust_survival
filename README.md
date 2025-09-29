# rust_survival

USE AT YOUR OWN RISK

**RSF → Cox → Points**  

A Rust command-line tool to run **Random Survival Forests (RSF)** on a dataset, select top variables via feature importance, fit a **Cox proportional hazards model**, and assign points for a risk scoring system.

---

## Installation

```bash
git clone <your-repo-url>
cd rust_survival
cargo build --release
```

The binary will be available at `target/release/rust_survival`.

---

## Usage

```bash
rust_survival [OPTIONS] --file <FILE> --time-col <TIME_COL> --status-col <STATUS_COL>
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --file <FILE>` | Path to CSV dataset | — |
| `-t, --time-col <TIME_COL>` | Name of survival time column | — |
| `-s, --status-col <STATUS_COL>` | Name of event status column (0/1) | — |
| `-c, --categorical <CATEGORICAL>` | Comma-separated categorical columns | `""` |
| `-n, --n-trees <N_TREES>` | Number of trees for RSF | `100` |
| `-m, --min-node-size <MIN_NODE_SIZE>` | Minimum node size in RSF trees | `5` |
| `--top-n <TOP_N>` | Number of top variables to select for Cox | `5` |
| `--base-hr <BASE_HR>` | Base hazard ratio for 1 point | `1.2` |
| `-d, --delimiter <DELIMITER>` | CSV delimiter | `"\t"` |
| `-h, --help` | Print help | — |
| `-V, --version` | Print version | — |

---

## Example

Run survival analysis on a CSV dataset:

```bash
rust_survival -f tests/data/survival_lung.csv -t time -s status -c sex,rx -n 200 -m 10 --top-n 5 --base-hr 1.2
```

- **Step 1:** Fits a Random Survival Forest (RSF) to select top variables.  
- **Step 2:** Fits a Cox proportional hazards model using top variables.  
- **Step 3:** Assigns points to each variable based on hazard ratios.  

You can optionally specify a delimiter if your CSV is not tab-separated:

```bash
rust_survival -f data.csv -t time -s status -d ","
```

## Output Example

Running the tool on the `survival_lung.csv` dataset:

```bash
rust_survival -f tests/data/survival_lung.csv -t time -s status
```

Produces:

```
61 rows contained NA - skipped

Top RSF variables:
1: pat.karno (importance=22410)
2: age (importance=20869)
3: ph.karno (importance=19437)
4: meal.cal (importance=19164)
5: ph.ecog (importance=17758)

Cox model hazard ratios:
pat.karno -> HR = 0.991
age -> HR = 1.002
ph.karno -> HR = 1.022
meal.cal -> HR = 1.000
ph.ecog -> HR = 1.596

Points per variable:
ph.karno -> 0 points
pat.karno -> 0 points
ph.ecog -> 3 points
age -> 0 points
meal.cal -> 0 points

Total points for first patient: 0
```

**Notes:**

- Rows containing `NA` in numeric columns are automatically skipped.
- The top RSF variables are ranked by feature importance.
- Hazard ratios (HR) are computed via the Cox model.
- Points are assigned for each variable based on HR, and summed for a total score.

---

## Notes

- Missing numeric values (`NA`) in the dataset will currently cause a parse error. Consider preprocessing your data or using tab/empty strings appropriately.  
- Categorical columns are optional and should be specified by name.  
- The first row of your CSV **must contain column headers**.  

---

## License

MIT / Apache-2.0
<<<<<<< HEAD

=======
>>>>>>> devel2
