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
- **Dot Plots**: Display enriched biological pathways as dot plot
- **Heatmaps**: Visualize correlation matrices
- **Scatter Plots**: Explore relationships between variables
- **PCA Plots**: Visualise the principal components 
- **MDS Plots**: Visualise the MDS scores
- **Histograms**: Show distribution of values
- **Box Plots**: Display data distributions and outliers
- **Bar Plots**: Show categorical data comparisons

## Installation

### Prerequisites
- Python 3.11+
- Conda (for environment management)


### Environment Setup

This project uses a conda environment defined in my_env.yaml.

```bash
# Create the conda environment from the environment file
conda env create -f my_env.yaml

# Activate the environment
conda activate my_env
```
This project requires a .env file in the project root with the following variables:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash
DB_PATH=data/rnaseq_results.db
```

### File Structure
```
├── assets/
│   ├── fonts/                      # Aptos font 
│   └── plots/                      # Directory for generated plots
├── config/
│   └── plot_instructions.yaml      # Directory for generated plots
├── data/
│   └── rnaseq_results.db           # SQLite database file
├── src/
│   ├── agent.py                    # Langchain agent orchestrator
│   ├── app.py                      # Dash web conversational interface for agent interaction
│   ├── database.py                 # Database tools for the agent
│   ├── main.py                     # Minimal CLI agent runner
│   ├── plotter.py                  # Plot generation tools for the agent
│   ├── tools.py                    # Tool orchestrator for database.py and plotter.py 
│   └── utils.py                    # Functions for the agent
├── utils/
│   └── dir_to_sql.py               # Automated ETL pipeline for rnaseq results
├── my_env.yaml                     # Requirements file for conda environment
└── README.md                       # Project documentation
```

## Quick Start

### Web App Interface

To explore your RNA-seq results interactively, run the Dash app:

```bash
python src/app.py
```

This launches a browser-based UI for asking questions and visualizing answers from your RNA-seq data using the agent.

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

#### `__init__(db_path, gemini_api_key)`
Initialize the agent with API key and model name.

#### `ask(question)`
Process a natural language question and return analysis results.

### RNAseqDatabase Class

#### `__init__(db_path, gemini_api_key)`
Initialize the database with its path.

#### `connect()`
Establish database connection.

#### `execute_query(query)`
Execute SQL query with safety checks.

#### `get_table_names(query)`
Retrieve a list of all table names.

#### `get_table_info()`
Retrieve database schema information.

#### `close()`
Close database connection.

### RNAseqPlotter Class

#### `init(data, query_info)`
Initialize the plotter with an LLM and output.

#### `store_query_data(data, query_info)`
Store query results for plotting.

#### `create_plot(plot_type, **kwargs)`
Generate interactive plots from stored data.

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Gemini AI API key

### Plot Settings
- Output directory: `assets/plots/` (configurable)
- Plot format: HTML (interactive Plotly plots)
- Default theme: `plotly_white`

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check if the database file exists
   - Verify file permissions
   - Ensure SQLite3 is available

2. **API Key Issues**
   - Verify your Gemini API key is valid
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

**Note**: To use the Gemini API, you need an API key. You can create a key for free with a few clicks in Google AI Studio: https://aistudio.google.com/app/apikey.
