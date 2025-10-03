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

var = apply( expr_matrix, 1 , var)

genes = names(sort(var, decreasing = TRUE) [1:min(c(2000, nrow(expr_matrix)))])

# Transpose: samples as rows, genes as columns
expr_df <- as.data.frame(t(expr_matrix[genes,]))

# Optionally, add patient barcodes as a column
expr_df$patient_id <- rownames(expr_df)

# -----------------------------
# Step 3: Add clinical info
# -----------------------------
clin <- GDCquery_clinic(project = "TCGA-GBM", type = "clinical")

# clean out useless columns
clin = clin[, colSums(!is.na(clin)) > 10]

# Function to keep only the last timepoint for each patient
keep_last_all <- function(df, sep = ";") {
  df[] <- lapply(df, function(col) {
    # Only process character/factor columns containing semicolons
    if (is.character(col) || is.factor(col)) {
      col <- as.character(col)
      has_sep <- grepl(sep, col)
      col[has_sep] <- sapply(strsplit(col[has_sep], sep), function(parts) tail(parts, 1))
      return(col)
    } else {
      return(col)
    }
  })
  return(df)
}


clin = keep_last_all( clin) 
# Merge clinical data with expression

merge_tcga_flexible <- function(exprs, clin, exprs_col_name = "patient_id", clin_col_name = "submitter_id") {
  expr_ids <- exprs[[exprs_col_name]]
  clin_ids <- clin[[clin_col_name]]
  
  # For each clinical ID, find which expr rows start with it
  matches <- unlist(sapply(clin_ids, function(cid) which(startsWith(expr_ids, cid))))
  matches = matches[matches != 0 ]

  return( cbind( clin[match( names(matches), clin[,clin_col_name] ),],  exprs[ matches, ] ) )
}


expr_df = merge_tcga_flexible(expr_df, clin )

# -----------------------------
# Step 4: Save CSV
# -----------------------------

expr_df [,'status'] = unlist( lapply( expr_df['days_to_death'], function(x) ifelse(is.na(x), 0, 1) ) )

write.csv(expr_df, "TCGA_GBM_subset.csv", row.names = FALSE)
cat("Saved TCGA-GBM CSV with", nrow(expr_df), "patients and", ncol(expr_df), "genes / clinical data.\n")

