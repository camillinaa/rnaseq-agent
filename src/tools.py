import logging
import os
import yaml
import pandas as pd
from typing import List, Dict, Any
from langchain.tools import Tool
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Global state management for data passed between tools.
# NOTE: This is not thread-safe. A production application should use a different method.
LAST_QUERY_DATA = {"data": None, "query_info": "", "timestamp": None}

def store_query_data(data: List[Dict], query_info: str = ""):
    """Store data and metadata from a SQL query for later use by plotting tools."""
    LAST_QUERY_DATA["data"] = data
    LAST_QUERY_DATA["query_info"] = query_info
    LAST_QUERY_DATA["timestamp"] = datetime.now()
    logger.info(f"Data stored successfully for plotting. {len(data)} rows available.")

def get_plot_instructions():
    """Load plot instructions from the config file."""
    plot_instructions_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'plot_instructions.yaml')
    try:
        with open(plot_instructions_path, 'r') as file:
            plot_instructions = yaml.safe_load(file)
            return plot_instructions, list(plot_instructions.keys())
    except FileNotFoundError:
        logger.error(f"Error: plot_instructions.yaml not found at {plot_instructions_path}")
        return {}, []

PLOT_INSTRUCTIONS, ALLOWED_PLOTS = get_plot_instructions()

def create_tools(db) -> List[Tool]:
    """Create and return a list of tools for the RNA-seq agent."""
    def sql_query_tool(query: str) -> str:
        """Execute a read-only SQL query on the database and return a summary of results."""
        logger.info(f"[SQL_TOOL] Executing query: {query}")
        result = db.execute_query(query)
        
        if "error" in result:
            error_msg = f"Query failed: {result['error']}\n"
            if "no such table" in result["error"].lower():
                error_msg += "RECOMMENDATION: Use Database_Schema tool first to understand the data structure, then use Sample_Column_Values tool to see actual data values before writing queries."
            return error_msg

        if result.get("row_count", 0) > 0:
            store_query_data(result["data"], query)

        max_rows = 15
        output = f"Query returned {result['row_count']} rows. "
        if result['row_count'] > max_rows:
            output += f"Showing first {max_rows} rows:\n"
        else:
            output += "Here are all the results:\n"

        if result.get("data"):
            columns = result["columns"]
            output += "\n" + " | ".join(columns) + "\n"
            output += "-" * (len(" | ".join(columns))) + "\n"
            for row in result["data"][:max_rows]:
                output += " | ".join([str(row.get(col, "")) for col in columns]) + "\n"
            output += "\nThis is the actual data from the database. Use this to answer the user's question."

        return output

    def database_schema_tool(input_str: str) -> str:
        """Return the schema of all available tables in the database."""
        logger.info("[SCHEMA_TOOL] Retrieving database schema")
        result = db.get_table_info()
        if "error" in result:
            return f"Error retrieving schema: {result['error']}"
        output = "Available tables and their schemas:\n\n"
        displayed_tables = 0
        table_count = len(result.get("tables", {}))
        for table_name, table_info in result["tables"].items():
            if displayed_tables >= 10:
                output += f"... and {table_count - displayed_tables} more tables\n"
                break
            output += f"Table: {table_name}\nKey columns:\n"
            for i, col in enumerate(table_info["columns"]):
                if isinstance(col, dict) and 'name' in col and 'type' in col:
                    output += f"  - {col['name']} ({col['type']})\n"
                else:
                    logger.error(f"Unexpected column format: {col}")
                    output += "  - Unexpected column format\n"
            output += f"Sample query: {table_info.get('sample_query','')}\n\n"
            displayed_tables += 1
        return output

    def sample_column_values_tool(query: str = "") -> str:
        """Retrieve a list of distinct, unique values from text columns."""
        logger.info("[SAMPLE_VALUES_TOOL] Retrieving sample column values")
        all_sample_values = {}
        table_names = db.get_table_names()
        if not table_names:
            return "Error: Could not retrieve table names."
        
        for table in table_names:
            try:
                info_result = db.execute_query(f"PRAGMA table_info('{table}');")
                if info_result.get("error"):
                    continue
                text_columns = [d['name'] for d in info_result.get('data', []) if isinstance(d, dict) and 'name' in d and 'type' in d and 'text' in d.get('type', '').lower()]
                for col in text_columns:
                    values_result = db.execute_query(f'SELECT DISTINCT "{col}" FROM "{table}" LIMIT 5;')
                    if values_result.get("data"):
                        all_sample_values[f"{table}.{col}"] = [d.get(col) for d in values_result['data']]
            except Exception as e:
                logger.error(f"Error fetching sample values from {table}: {e}")
                continue
        
        if not all_sample_values:
            return "Could not find any text columns with sample values."
        
        output = "Here are a few sample values from key text columns:\n"
        for column, values in all_sample_values.items():
            output += f"- {column}: {', '.join([str(v) for v in values])}\n"
        return output
    
    def create_plot_tool(plot_request: str) -> str:
        """Generate a Plotly plot based on the last retrieved data."""
        logger.info(f"[PLOT_TOOL] Attempting to create plot with request: '{plot_request}'")
        
        if not LAST_QUERY_DATA["data"]:
            return "Plot creation failed: No data available. You must run a SQL_Query first to get data."
        
        data_age = (datetime.now() - LAST_QUERY_DATA["timestamp"]).total_seconds()
        if data_age > 120:
            return "Plot creation failed: The data from the last query is too old. Please run a new query."

        try:
            parts = plot_request.split("|")
            plot_type = parts[0].strip().lower()
            plot_params = {}
            for param in parts[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    plot_params[key.strip()] = value.strip()

            if plot_type not in ALLOWED_PLOTS:
                return f"Plot type '{plot_type}' is not allowed. Allowed types are: {', '.join(ALLOWED_PLOTS)}"

            df = pd.DataFrame(LAST_QUERY_DATA["data"])
            fig = None
            
            # --- Using a dispatcher for clean plot creation logic ---
            plot_functions = {
                'scatter': lambda params: px.scatter(
                    df,
                    x=params.get('x_column'),
                    y=params.get('y_column'),
                    color=params.get('color_column'),
                    size=params.get('size_column'),
                    hover_data=df.columns,
                    title=params.get('title', 'Scatter Plot')
                ),
                'pca': lambda params: px.scatter(
                    df,
                    x=params.get('x_column'),
                    y=params.get('y_column'),
                    color=params.get('color_column'),
                    size=params.get('size_column'),
                    hover_data=df.columns,
                    title=params.get('title', 'PCA Plot')
                ),
                'volcano': lambda params: px.scatter(
                    df,
                    x=params.get('x_column'),
                    y=params.get('y_column'),
                    color='significant', # Enforced color column
                    hover_data=df.columns,
                    title=params.get('title', 'Volcano Plot')
                ),
                'heatmap': lambda params: px.imshow(
                    df.set_index(df.columns[0]).apply(pd.to_numeric, errors='coerce'),
                    text_auto=True,
                    aspect="auto",
                    title=params.get('title', 'Heatmap')
                ),
                'bar': lambda params: px.bar(
                    df,
                    x=params.get('x_column'),
                    y=params.get('y_column'),
                    color=params.get('color_column'),
                    title=params.get('title', 'Bar Plot')
                )
            }
            
            # Special data preparation for certain plots
            if plot_type == 'volcano':
                df['significant'] = df[plot_params.get('y_column')].apply(lambda p: 'Significant' if pd.to_numeric(p, errors='coerce') < 0.05 else 'Not Significant')
                
            # Call the appropriate function from the dictionary
            plot_function = plot_functions.get(plot_type)
            if plot_function:
                fig = plot_function(plot_params)
            else:
                return f"Plot creation failed: '{plot_type}' is not a valid plot type. Allowed types are: {', '.join(ALLOWED_PLOTS)}"

            # Save the plot
            if fig:
                assets_plots_dir = os.path.join("assets", "plots")
                os.makedirs(assets_plots_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"{plot_type}_{timestamp}.html"
                full_path = os.path.join(assets_plots_dir, plot_filename)
                fig.write_html(full_path)
                logger.info(f"Plot saved to: {full_path}")
                return plot_filename
            else:
                return "Plot creation failed: An unexpected error occurred during plot generation."

        except Exception as e:
            logger.error(f"[PLOT_TOOL] Plot creation failed: {str(e)}")
            return f"Plot creation failed: {str(e)}. Please check your query and parameters and try again."
    
    return [
    Tool(
        name="SQL_Query",
        description=(
            "Execute SQL queries against the RNAseq database to retrieve information "
            "about genes, samples, differential expression results, and metadata. "
            "This is the primary tool for data retrieval. "
            "Input should be a complete, valid SQL SELECT statement. "
            "Always use this tool first when a user asks a question that requires data "
            "from the database, such as 'list all differentially expressed genes,' "
            "'show the top 10 genes by log2FoldChange,' or 'find data for a specific gene.'"
        ),
        func=sql_query_tool
    ),
    Tool(
        name="Database_Schema",
        description=(
            "Inspect the database schema to understand the structure of tables and "
            "the columns they contain. Use this to determine which table holds "
            "the information needed to answer a user's question. "
            "This tool does not return data itself; it only provides metadata about the database structure. "
            "The input is not required; simply pass an empty string."
        ),
        func=database_schema_tool
    ),
    Tool(
        name="Sample_Column_Values",
        description=(
            "Retrieve a list of distinct, unique values from text columns in all tables. "
            "Use this tool to discover available comparison variables, sample names, or "
            "other categorical data points that can be used to construct a SQL query. "
            "This is useful for questions like 'what comparisons do we have?' or 'what types of samples are there?' "
            "The input is not required; simply pass an empty string."
        ),
        func=sample_column_values_tool
    ),
    Tool(
        name="Create_Plot",
        description=(
            "Generate plots from the data retrieved by the 'SQL_Query' tool. "
            "This tool requires data to be available from a preceding SQL query. "
            "Do not call this tool until after a successful 'SQL_Query' has been executed. "
            "The input must be a specific plot type followed by parameters in a 'key=value' format, "
            "separated by '|'. "
            "Example input: 'volcano|x_column=log2FoldChange|y_column=padj|title=Volcano Plot'. "
            "Allowed plot types are: 'scatter', 'pca', 'heatmap', 'volcano', and 'bar'."
        ),
        func=create_plot_tool
    )
]