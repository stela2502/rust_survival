# Install required packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if (!requireNamespace("TCGAbiolinks", quietly = TRUE))
    BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)

# -----------------------------
# Step 1: Query TCGA-GBM gene expression
# -----------------------------
query <- GDCquery(
    project = "TCGA-GBM",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    experimental.strategy = "RNA-Seq"
)

# -----------------------------
# Step 2: Download & prepare data
# -----------------------------
GDCdownload(query, method = "api")  # may take some time
data_se <- GDCprepare(query)

# Extract expression matrix
expr_matrix <- assay(data_se)

# Transpose: samples as rows, genes as columns
expr_df <- as.data.frame(t(expr_matrix))

# Optionally, add patient barcodes as a column
expr_df$patient_id <- rownames(expr_df)

# -----------------------------
# Step 3: Add clinical info
# -----------------------------
clin <- GDCquery_clinic(project = "TCGA-GBM", type = "clinical")

# Merge clinical data with expression
# Make sure to match patient IDs
expr_df <- expr_df %>%
    left_join(clin, by = c("patient_id" = "bcr_patient_barcode"))

# -----------------------------
# Step 4: Save CSV
# -----------------------------
# Optional: keep size under 1 GB by sampling columns/genes
set.seed(123)
max_genes <- 5000  # or adjust based on memory
keep_genes <- sample(colnames(expr_matrix), max_genes)
keep_cols <- c(keep_genes, "patient_id", setdiff(colnames(clin), "bcr_patient_barcode"))

expr_df <- expr_df[, keep_cols]

write.csv(expr_df, "TCGA_GBM_subset.csv", row.names = FALSE)
cat("Saved TCGA-GBM CSV with", nrow(expr_df), "patients and", length(keep_genes), "genes.\n")

