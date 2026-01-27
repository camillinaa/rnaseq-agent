import sqlite3
import pandas as pd
from pathlib import Path
import argparse

# -----------------------------
# 1. Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Inject nfcore/rnaseq results into SQLite database")
parser.add_argument("--base_dir", type=str, required=True, help="Path to the root folder containing nfcore/rnaseq downstream results")
parser.add_argument("--db_path", type=str, required=True, help="Path where the SQLite database will be created (e.g., results.db)")
parser.add_argument("--metadata", type=str, required=True, help="Path to the metadata.csv file used as input to the pipeline")
args = parser.parse_args()

base_dir = Path(args.base_dir)
db_path = Path(args.db_path).expanduser()
db_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Using base_dir: {base_dir}")
print(f"Using database: {db_path}")

# Load metadata and detect comparison variable
metadata_path = Path(args.metadata)
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
metadata_df = pd.read_csv(metadata_path)
print(f"Loaded metadata with columns: {list(metadata_df.columns)}")

exclude_cols = {'Sample_ID', 'Sample_Name'}
potential_comparison_vars = [col for col in metadata_df.columns if col not in exclude_cols]

comparison_variable = None
dea_dirs = list(base_dir.glob("dea_*"))
if dea_dirs and dea_dirs[0].is_dir():
    nested_dirs = list(dea_dirs[0].glob("dea_*"))
    if nested_dirs:
        first_comparison_name = nested_dirs[0].name.replace("dea_", "")
        for col in potential_comparison_vars:
            if col in first_comparison_name:
                comparison_variable = col
                print(f"Detected comparison variable: {comparison_variable}")
                break

if not comparison_variable:
    print("WARNING: Could not auto-detect comparison variable")

# -----------------------------
# 2. Connect to database and create tables
# -----------------------------
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS deseq2_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_subset TEXT,
    comparison_variable TEXT,
    comparison1 TEXT,
    comparison2 TEXT,
    gene_name TEXT,
    baseMean REAL,
    log2FoldChange REAL,
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
    sample_subset TEXT,
    comparison_variable TEXT,
    comparison1 TEXT,
    comparison2 TEXT,
    gene_set TEXT,
    analysis_type TEXT,
    ID TEXT,
    Description TEXT,
    Term TEXT,
    Overlap TEXT,
    P_value REAL,
    Adjusted_P_value REAL,
    Old_P_value REAL,
    Old_Adjusted_P_value REAL,
    Odds_Ratio REAL,
    Combined_Score REAL,
    Genes TEXT,
    GeneRatio TEXT,
    BgRatio TEXT,
    RichFactor REAL,
    FoldEnrichment REAL,
    zScore REAL,
    pvalue REAL,
    p_adjust REAL,
    qvalue REAL,
    geneID TEXT,
    Count INTEGER,
    setSize INTEGER,
    enrichmentScore REAL,
    NES REAL,
    rank TEXT,
    leading_edge TEXT,
    core_enrichment TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS dea_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_subset TEXT,
    comparison_variable TEXT,
    comparison1 TEXT,
    comparison2 TEXT
)
""")

conn.commit()

# -----------------------------
# 3. Differential expression files
# -----------------------------
dirs_to_process = [
    (subset_dir, subset_dir.name.replace("dea_", ""), comparison_dir)
    for subset_dir in base_dir.glob("dea_*") if subset_dir.is_dir()
    for comparison_dir in subset_dir.glob("dea_*") if comparison_dir.is_dir()
]

dea_pairs = set()

for parent_dir, subset_name, comparison_dir in dirs_to_process:
    deseq_files = list(comparison_dir.glob("deseq2_toptable.*.txt"))
    if not deseq_files:
        continue
    
    basename = deseq_files[0].stem.replace("deseq2_toptable.", "")
    parts = basename.split("_vs_")
    
    comp_var = comparison_variable if comparison_variable else ""
    if comp_var and parts[0].startswith(comp_var + "_"):
        comparison1 = parts[0][len(comp_var) + 1:]
    else:
        comparison1 = parts[0]
    comparison2 = parts[1] if len(parts) > 1 else ""
    
    dea_pairs.add((subset_name, comp_var, comparison1, comparison2))

    # Import DESeq2 results
    for deseq_file in comparison_dir.glob("deseq2_toptable.*.txt"):
        df = pd.read_csv(deseq_file, sep="\t", comment='#')
        df.insert(0, "sample_subset", subset_name)
        df.insert(1, "comparison_variable", comp_var)
        df.insert(2, "comparison1", comparison1)
        df.insert(3, "comparison2", comparison2)
        df.to_sql("deseq2_results", conn, if_exists="append", index=False)
        print(f"Imported DESeq2 table: {deseq_file}")

    # Import enrichment results
    for enrich_file in comparison_dir.glob("*.xlsx"):
        if enrich_file.name.startswith("deseq2"):
            continue

        if (enrich_file.name.startswith("enrichr.") and enrich_file.name.endswith("_all.xlsx")) or \
           (enrich_file.name.startswith("gsea.") and not any(x in enrich_file.name for x in [".c2.", ".c5.", ".h."])) or \
           (enrich_file.name.startswith("ora_CP.") and enrich_file.name.endswith(".all.xlsx")):
            
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
                df.insert(0, "sample_subset", subset_name)
                df.insert(1, "comparison_variable", comp_var)
                df.insert(2, "comparison1", comparison1)
                df.insert(3, "comparison2", comparison2)
                df.insert(4, "gene_set", sheet_name)
                df.insert(5, "analysis_type", analysis_type)
                df.rename(columns={
                    "P.value": "P_value",
                    "Adjusted.P.value": "Adjusted_P_value",
                    "Old.P.value": "Old_P_value",
                    "Old.Adjusted.P.value": "Old_Adjusted_P_value",
                    "Odds.Ratio": "Odds_Ratio",
                    "p.adjust": "p_adjust",
                    "Combined.Score": "Combined_Score"
                }, inplace=True)
                df.to_sql("enrichment_results", conn, if_exists="append", index=False)
                print(f"Imported enrichment: {enrich_file.name} (sheet: {sheet_name})")


# Populate dea_metadata
for sample_subset, comparison_variable, comparison1, comparison2 in dea_pairs:
    cursor.execute("INSERT INTO dea_metadata (sample_subset, comparison_variable, comparison1, comparison2) VALUES (?, ?, ?, ?)",
                   (sample_subset, comparison_variable, comparison1, comparison2))
print(f"Populated dea_metadata with {len(dea_pairs)} unique subset/comparison pairs")


# -----------------------------
# 4. Other files 
# -----------------------------
corr_file = base_dir / "samples_correlation_table.txt"
if corr_file.exists():
    pd.read_csv(corr_file, sep="\t", index_col=0).to_sql("correlation_matrix", conn, if_exists="replace")
    print(f"Imported correlation matrix")

dim_dir = base_dir / "dim_reduction"
if dim_dir.exists():
    for table_name in ["MDS_scores.txt", "PCA_scores.txt"]:
        file_path = dim_dir / table_name
        if file_path.exists():
            sql_table_name = table_name.replace("_scores.txt", "_scores").lower()
            pd.read_csv(file_path, sep="\t", index_col=0).to_sql(sql_table_name, conn, if_exists="replace")
            print(f"Imported {sql_table_name}")

norm_dir = base_dir / "normalization"
if norm_dir.exists():
    cpm_file = norm_dir / "cpm.txt"
    if cpm_file.exists():
        pd.read_csv(cpm_file, sep="\t", index_col=0).to_sql("normalized_counts_matrix", conn, if_exists="replace")
        print(f"Imported normalized counts")
    lib_file = norm_dir / "lib_size_factors.txt"
    if lib_file.exists():
        pd.read_csv(lib_file, sep="\t", index_col=0).to_sql("library_size", conn, if_exists="replace")
        print(f"Imported library size factors")

# -----------------------------
# 5. Close connection
# -----------------------------
conn.commit()
conn.close()
print("✅ Database populated from files in", base_dir)
