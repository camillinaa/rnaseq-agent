import sqlite3
import pandas as pd
from pathlib import Path
import argparse

# -----------------------------
# 1. Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Inject nfcore/rnaseq results into SQLite database")
parser.add_argument(
    "--base_dir", type=str, required=True,
    help="Path to the root folder containing nfcore/rnaseq downstream results results"
)
parser.add_argument(
    "--db_path", type=str, required=True,
    help="Path where the SQLite database will be created"
)

args = parser.parse_args()

base_dir = Path(args.base_dir)
db_path = Path(args.db_path).expanduser()  # expand ~ if used
db_path.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directories exist

print(f"Using base_dir: {base_dir}")
print(f"Using database: {db_path}")

# -----------------------------
# 2. Connect to database and create tables
# -----------------------------
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS deseq2_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_subset TEXT,
    comparison TEXT,
    gene_name TEXT,
    baseMean REAL,
    log2FoldChange REAL,
    lfcSE REAL,
    stat REAL,
    pvalue REAL,
    padj REAL,
    significance TEXT,
    geneid TEXT,
    chr TEXT,
    start INTEGER,
    end INTEGER,
    strand TEXT,
    length INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS enrichment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_subset TEXT,
    comparison TEXT,
    analysis_type TEXT,
    gene_set TEXT,
    Term TEXT,
    Overlap TEXT,
    P_value REAL,
    Adjusted_P_value REAL,
    Old_P_value REAL,
    Old_Adjusted_P_value REAL,
    Odds_Ratio REAL,
    Combined_Score REAL,
    Genes TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS dea_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_subset TEXT,
    comparison TEXT,
    description TEXT
)
""")

conn.commit()

# -----------------------------
# 3. Differential expression files
# -----------------------------
for subset_dir in base_dir.glob("dea_*"):
    if not subset_dir.is_dir():
        continue
    subset_name = subset_dir.name.replace("dea_", "")

    for comparison_dir in subset_dir.glob("dea_*"):
        if not comparison_dir.is_dir():
            continue
        comparison_name = comparison_dir.name.replace("dea_", "")

        # ---- DESeq2 results ----
        for deseq_file in comparison_dir.glob("deseq2_toptable.*.txt"):
            df = pd.read_csv(deseq_file, sep="\t")
            df.insert(0, "comparison", comparison_name)
            df.insert(0, "sample_subset", subset_name)
            df.to_sql("deseq2_results", conn, if_exists="append", index=False)
            print(f"Imported DESeq2 table: {deseq_file}")

        # ---- Enrichment results ----
        for enrich_file in comparison_dir.glob("*_all.xlsx"):
            if enrich_file.name.startswith("deseq2"):
                continue

            if enrich_file.name.startswith("enrichr"):
                analysis_type = "enrichr"
            elif enrich_file.name.startswith("gsea"):
                analysis_type = "gsea"
            elif enrich_file.name.startswith("ora"):
                analysis_type = "ora"
            else:
                continue

            xls = pd.ExcelFile(enrich_file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.insert(0, "analysis_type", analysis_type)
                df.insert(0, "gene_set", sheet_name)
                df.insert(0, "sample_subset", subset_name)
                df.insert(0, "comparison", comparison_name)
                # Rename columns to be SQLite-friendly
                df.rename(columns={
                    "P.value": "P_value",
                    "Adjusted.P.value": "Adjusted_P_value",
                    "Old.P.value": "Old_P_value",
                    "Old.Adjusted.P.value": "Old_Adjusted_P_value",
                    "Odds.Ratio": "Odds_Ratio",
                    "Combined.Score": "Combined_Score"
                }, inplace=True)
                df.to_sql("enrichment_results", conn, if_exists="append", index=False)

# DEA supporting table

dea_pairs = set()

for subset_dir in base_dir.glob("dea_*"):
    if not subset_dir.is_dir():
        continue
    sample_subset = subset_dir.name.replace("dea_", "")

    for comparison_dir in subset_dir.glob("dea_*"):
        if not comparison_dir.is_dir():
            continue
        comparison = comparison_dir.name.replace("dea_", "")
        dea_pairs.add((sample_subset, comparison))

for sample_subset, comparison in dea_pairs:
    cursor.execute("""
        INSERT INTO dea_metadata (sample_subset, comparison)
        VALUES (?, ?)
    """, (sample_subset, comparison))
    
print(f"Populated dea_metadata with {len(dea_pairs)} unique subset/comparison pairs")

# -----------------------------
# 4. Other files 
# -----------------------------
# Correlation matrix
corr_file = base_dir / "samples_correlation_table.txt"
if corr_file.exists():
    df = pd.read_csv(corr_file, sep="\t", index_col=0)  # assuming rownames as index
    df.to_sql("correlation_matrix", conn, if_exists="replace")
    print(f"Imported correlation matrix: {corr_file}")
else:
    print(f"Correlation matrix file not found: {corr_file}")

# Dimensionality reduction
dim_dir = base_dir / "dim_reduction"
if dim_dir.exists():
    for table_name in ["MDS_scores.txt", "PCA_scores.txt"]:
        file_path = dim_dir / table_name
        if file_path.exists():
            df = pd.read_csv(file_path, sep="\t", index_col=0)
            sql_table_name = table_name.replace("_scores.txt", "_scores").lower()  # mds_scores / pca_scores
            df.to_sql(sql_table_name, conn, if_exists="replace")
            print(f"Imported dimensionality reduction table: {file_path}")
else:
    print(f"Dimensionality reduction directory not found: {dim_dir}")

# Normalization
norm_dir = base_dir / "normalization"
if norm_dir.exists():
    # CPM normalized counts
    cpm_file = norm_dir / "cpm.txt"
    if cpm_file.exists():
        df = pd.read_csv(cpm_file, sep="\t", index_col=0)
        df.to_sql("normalized_counts_matrix", conn, if_exists="replace")
        print(f"Imported normalized counts: {cpm_file}")
    # Library size factors
    lib_file = norm_dir / "lib_size_factors.txt"
    if lib_file.exists():
        df = pd.read_csv(lib_file, sep="\t", index_col=0)
        df.to_sql("library_size", conn, if_exists="replace")
        print(f"Imported library size factors: {lib_file}")
else:   
    print(f"Normalization directory not found: {norm_dir}")
        
conn.commit()
conn.close()
print("✅ Database populated from files in", base_dir)
