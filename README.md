# RNAseq Analysis Agent

A comprehensive AI-powered agent for analyzing RNAseq data with interactive plotting capabilities. This agent can query SQLite databases containing nfcore/rnaseq output and generate various types of visualizations to help researchers interpret their data.

## Features

### 🔍 **Database Operations**
- Connect to SQLite databases containing nfcore/rnaseq output
- Execute SQL queries with safety checks
- Retrieve database schema information
- Handle multiple table types (counts matrix, correlation, dimensionality reduction, differential expression, pathway enrichment using various gene sets, etc.)

### 🤖 **Conversational AI-Powered Analysis**
- Natural language query processing - understanding and answering
- Intelligent data retrieval from vast RNAseq knowledge base
- Intelligent interactive plot type selection and generation based on question asked and data
- In depth biological context interpretation
- Statistical significance assessment

### 📊 **Interactive Plotting**
- **Volcano Plots**: Visualize differential expression results
- **MA Plots**: Show relationship between expression level and fold change
- **Pathway Enrichment Plots**: Display enriched biological pathways
- **Histograms**: Show distribution of values
- **Scatter Plots**: Explore relationships between variables
- **Box Plots**: Display data distributions and outliers
- **Heatmaps**: Visualize correlation matrices
- **Bar Plots**: Show categorical data comparisons

## Installation

### Prerequisites
- Python 3.11+
- Required packages (conda env - container tbd):

```bash
pip install pandas numpy logging typing os re json dotenv datetime sqlite3 langchain langchain-mistralai matplotlib seaborn plotly dash
```

### Environment Setup
This project requires a .env file with the following variables:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
MODEL_NAME=mistralai_model_used
DB_PATH=path_to_your_database
```

To run the app:
1. Create a .env file in the project root.
2. Add the required keys as shown above.
3. Save the file.
4. Run the Streamlit app as usual.

### File Structure
```
├── README.md               # Project documentation
├── data/
│   └── rnaseq.db           # SQLite database file
├── plots/                  # Directory for generated plots
├── src/
│   ├── agent.py            # Langchain agent orchestrator
│   ├── main.py             # Minimal CLI agent runner
│   ├── app.py              # Dash web conversational interface for agent interaction
│   ├── plotter.py          # Plot generation tools for the agent
│   └── database.py         # Database tools for the agent
└── tests/
    └── test.py             # Placeholder for unit tests
```

## Quick Start

### 1. Web App Interface

To explore your RNA-seq results interactively, run the Dash app:

```bash
python src/app.py
```

This launches a browser-based UI for asking questions and visualizing answers from your RNA-seq data using the agent.

### 2. Programmatic Usage

If you want to interact with the agent in a script or notebook, use the following. 

```python
from agent import RNAseqAgent

# Initialize the agent
agent = RNAseqAgent(
    db_path="your_rnaseq.db",
    mistral_api_key="your_mistral_api_key"
)

# Ask questions about your data
response = agent.ask("What are the top upregulated genes?")
print(response)

# Clean up
agent.close()
```

Note: This is the same logic used in main.py, which serves as a minimal script to initialize and run the agent without a GUI.

### 3. Running Tests - WIP

```bash
python test_agent.py
```

This will:
- Create a sample database with RNAseq data
- Test all plotting functionalities
- Generate example visualizations
- Validate the complete workflow

## Database Schema

The agent expects SQLite databases with tables respecting the naming convention of nfcore/rnaseq output.
In particular, the agent expects Deseq2, GSEA, and ORA tables following this naming convention:
`{sample_subset}_{comparison}_{analysis_type}_{gene_set}`

### Example Tables

#### Differential Expression Table
```sql
CREATE TABLE NS_flattening_yes_vs_no_deseq2 (
    gene_id TEXT PRIMARY KEY,
    gene_name TEXT,
    basemean REAL,
    log2fc REAL,
    lfcse REAL,
    stat REAL,
    pvalue REAL,
    padj REAL
);
```

#### Pathway Enrichment Table
```sql
CREATE TABLE NS_flattening_yes_vs_no_ora_hallmark (
    pathway TEXT PRIMARY KEY,
    description TEXT,
    enrichment REAL,
    pvalue REAL,
    padj REAL,
    genes_in_pathway INTEGER,
    genes_found INTEGER
);
```

## Usage Examples

### Example 1: Differential Expression Analysis
```python
# Query for significantly upregulated genes
response = agent.ask("""
Show me the top 10 upregulated genes with padj < 0.05 and log2fc > 1.
Also create a volcano plot to visualize the results.
""")
```

### Example 2: Pathway Analysis
```python
# Analyze enriched pathways
response = agent.ask("""
What are the most significantly enriched pathways in the hallmark gene set?
Create a pathway enrichment plot showing the top 15 pathways.
""")
```

### Example 3: Data Exploration
```python
# Explore data distribution
response = agent.ask("""
Show me the distribution of log2 fold changes and create a histogram.
Also show me the correlation between different statistical measures.
""")
```

## API Reference

### RNAseqAgent Class

#### `__init__(db_path, mistral_api_key)`
Initialize the agent with database path and API key.

#### `ask(question)`
Process a natural language question and return analysis results.

#### `close()`
Clean up database connections and resources.

### RNAseqDatabase Class

#### `connect()`
Establish database connection.

#### `execute_query(query)`
Execute SQL query with safety checks.

#### `get_table_info()`
Retrieve database schema information.

### RNAseqPlotter Class

#### `store_query_data(data, query_info)`
Store query results for plotting.

#### `create_plot(plot_type, **kwargs)`
Generate interactive plots from stored data.

## Configuration

### Environment Variables
- `MISTRAL_API_KEY`: Your Mistral AI API key

### Plot Settings
- Output directory: `plots/` (configurable)
- Plot format: HTML (interactive Plotly plots)
- Default theme: `plotly_white`

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check if the database file exists
   - Verify file permissions
   - Ensure SQLite3 is available

2. **API Key Issues**
   - Verify your Mistral API key is valid
   - Check API rate limits
   - Ensure internet connectivity

3. **Plot Generation Errors**
   - Ensure data is loaded before plotting
   - Check column names match expected format
   - Verify numeric data types for calculations

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Changelog

### Version 1.0.0
- Initial release
- Complete plotting functionality
- Database operations
- AI-powered query processing
- Comprehensive test suite

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test examples
3. Create an issue with detailed error information

---

**Note**: This agent requires a valid Mistral AI API key for natural language processing. Mistral API provides a free tier with extensive use at: https://docs.mistral.ai/getting-started/quickstart/. 
