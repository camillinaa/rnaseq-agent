import logging
import os
import io
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
    
    def create_csv_report_tool(query: str = "") -> str:
        """Generate a CSV report based on the last retrieved data."""
        logger.info(f"[CSV_REPORT_TOOL] Attempting to create a CSV report'")
        if not LAST_QUERY_DATA["data"]:
            return "CSV report creation failed: No data available. You must run a SQL_Query first to get data."
        
        # Check the age of the data
        data_age = (datetime.now() - LAST_QUERY_DATA["timestamp"]).total_seconds()
        if data_age > 120:
            return "CSV report creation failed: The data from the last query is too old. Please run a new query."
        
        try:
            df = pd.DataFrame(LAST_QUERY_DATA["data"])
            assets_reports_dir = os.path.join("assets", "reports")
            os.makedirs(assets_reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.csv"
            full_path = os.path.join(assets_reports_dir, report_filename)
            df.to_csv(full_path, index=False)            
            return report_filename
        
        except Exception as e:
            logger.info(f"[CSV_REPORT_TOOL] CSV report creation failed with error: {e}")
            return f"CSV report creation failed: An error occurred while generating the report. Error: {e}"
                
    def create_plot_tool(plot_request: str) -> str:
        """Generate a Plotly plot based on the last retrieved data."""
        logger.info(f"[PLOT_TOOL] Attempting to create plot with request: '{plot_request}'")
        
        try:
            # Check if data is available
            if not LAST_QUERY_DATA["data"]:
                logger.error("[PLOT_TOOL] No data available for plotting")
                return "No data available. Please run a SQL query first to get data."
            
            # Check data freshness
            data_age = (datetime.now() - LAST_QUERY_DATA["timestamp"]).total_seconds()
            if data_age > 120:
                logger.warning(f"[PLOT_TOOL] Data is {data_age} seconds old")
                return "Data from the last query is too old. Please run a new query."

            # Parse plot request
            parts = plot_request.split("|")
            if not parts:
                logger.error("[PLOT_TOOL] Invalid plot request format")
                return "Invalid plot request format. Please specify plot type and parameters."
                
            plot_type = parts[0].strip().lower()
            plot_params = {}
            
            for param in parts[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    plot_params[key.strip()] = value.strip()

            # Validate plot type
            if plot_type not in ALLOWED_PLOTS:
                logger.error(f"[PLOT_TOOL] Invalid plot type: {plot_type}")
                return f"Plot type '{plot_type}' is not allowed. Allowed types are: {', '.join(ALLOWED_PLOTS)}"

            # Create DataFrame from data
            df = pd.DataFrame(LAST_QUERY_DATA["data"])
            if df.empty:
                logger.error("[PLOT_TOOL] DataFrame is empty")
                return "No valid data to plot."

            # Validate required columns exist
            required_columns = _get_required_columns(plot_type, plot_params)
            missing_columns = [col for col in required_columns if col and col not in df.columns]
            if missing_columns:
                logger.error(f"[PLOT_TOOL] Missing required columns: {missing_columns}")
                return f"Missing required columns: {', '.join(missing_columns)}. Available columns: {', '.join(df.columns)}"

            # Create the plot
            fig = _create_plot(plot_type, df, plot_params)
            
            if fig is None:
                logger.error(f"[PLOT_TOOL] Failed to create {plot_type} plot")
                return f"Failed to create {plot_type} plot. Please check your parameters."

            # Save the plot
            plot_filename = _save_plot(fig, plot_type)
            if plot_filename:
                logger.info(f"[PLOT_TOOL] Successfully created plot: {plot_filename}")
                return plot_filename
            else:
                logger.error("[PLOT_TOOL] Failed to save plot")
                return "Failed to save the plot."
                
        except KeyError as e:
            logger.error(f"[PLOT_TOOL] Missing required parameter or column: {str(e)}")
            return f"Missing required parameter or column. Please check your plot request."
        
        except ValueError as e:
            logger.error(f"[PLOT_TOOL] Invalid data or parameter values: {str(e)}")
            return "Invalid data or parameter values. Please check your data types and parameters."
        
        except Exception as e:
            logger.error(f"[PLOT_TOOL] Unexpected error during plot creation: {str(e)}")
            return "An unexpected error occurred while creating the plot. Please try again with different parameters."


    def _get_required_columns(plot_type: str, plot_params: dict) -> list:
        """Get required columns for each plot type."""
        column_requirements = {
            'scatter': [plot_params.get('x_column'), plot_params.get('y_column')],
            'pca': [plot_params.get('x_column'), plot_params.get('y_column')],
            'volcano': [plot_params.get('x_column'), plot_params.get('y_column')],
            'heatmap': [],  # Uses first column as index
            'bar': [plot_params.get('x_column'), plot_params.get('y_column')],
            'enrichment': [plot_params.get('x_column'), plot_params.get('y_column')],
            'dot': [plot_params.get('x_column'), plot_params.get('y_column')]
        }
        return column_requirements.get(plot_type, [])


    def _create_plot(plot_type: str, df: pd.DataFrame, plot_params: dict):
        """Create the appropriate plot based on type and parameters."""
        try:
            if plot_type == 'scatter':
                return px.scatter(
                    df,
                    x=plot_params.get('x_column'),
                    y=plot_params.get('y_column'),
                    color=plot_params.get('color_column') if plot_params.get('color_column') != 'None' else None,
                    size=plot_params.get('size_column') if plot_params.get('size_column') != 'None' else None,
                    hover_data=df.columns.tolist(),
                    title=plot_params.get('title', 'Scatter Plot')
                )
                
            elif plot_type == 'pca':
                return px.scatter(
                    df,
                    x=plot_params.get('x_column'),
                    y=plot_params.get('y_column'),
                    color=plot_params.get('color_column') if plot_params.get('color_column') != 'None' else None,
                    size=plot_params.get('size_column') if plot_params.get('size_column') != 'None' else None,
                    hover_data=df.columns.tolist(),
                    title=plot_params.get('title', 'PCA Plot')
                )
                
            elif plot_type == 'volcano':
                # Create significance column
                y_col = plot_params.get('y_column')
                df_copy = df.copy()
                df_copy['significant'] = df_copy[y_col].apply(
                    lambda p: 'Significant' if pd.to_numeric(p, errors='coerce') < 0.05 else 'Not Significant'
                )
                return px.scatter(
                    df_copy,
                    x=plot_params.get('x_column'),
                    y=y_col,
                    color='significant',
                    hover_data=df.columns.tolist(),
                    title=plot_params.get('title', 'Volcano Plot')
                )
                
            elif plot_type == 'heatmap':
                # Use first column as index and convert to numeric
                numeric_df = df.set_index(df.columns[0]).apply(pd.to_numeric, errors='coerce')
                return px.imshow(
                    numeric_df,
                    text_auto=True,
                    aspect="auto",
                    title=plot_params.get('title', 'Heatmap')
                )
                
            elif plot_type == 'bar':
                return px.bar(
                    df,
                    x=plot_params.get('x_column'),
                    y=plot_params.get('y_column'),
                    color=plot_params.get('color_column') if plot_params.get('color_column') != 'None' else None,
                    title=plot_params.get('title', 'Bar Plot')
                )
                
            elif plot_type == 'enrichment':
                fig = px.bar(
                    df,
                    x=plot_params.get('x_column'),
                    y=plot_params.get('y_column'),
                    color=plot_params.get('color_column') if plot_params.get('color_column') != 'None' else None,
                    orientation='h',
                    title=plot_params.get('title', 'Enrichment Plot'),
                    color_continuous_scale='viridis_r'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                return fig
                
            elif plot_type == 'dot':
                fig = px.scatter(
                    df,
                    x=plot_params.get('x_column'),
                    y=plot_params.get('y_column'),
                    size=plot_params.get('size_column') if plot_params.get('size_column') != 'None' else None,
                    color=plot_params.get('color_column') if plot_params.get('color_column') != 'None' else None,
                    title=plot_params.get('title', 'Dot Plot'),
                    color_continuous_scale='viridis_r',
                    size_max=20
                )
                fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                return fig
                
            else:
                logger.error(f"[PLOT_TOOL] Unknown plot type: {plot_type}")
                return None
                
        except Exception as e:
            logger.error(f"[PLOT_TOOL] Error creating {plot_type} plot: {str(e)}")
            return None

    def _save_plot(fig, plot_type: str) -> str:
        """Save the plot to file and return filename."""
        try:
            assets_plots_dir = os.path.join("assets", "plots")
            os.makedirs(assets_plots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{plot_type}_{timestamp}.html"
            full_path = os.path.join(assets_plots_dir, plot_filename)
            
            fig.write_html(full_path)
            logger.info(f"[PLOT_TOOL] Plot saved to: {full_path}")
            return plot_filename
            
        except Exception as e:
            logger.error(f"[PLOT_TOOL] Error saving plot: {str(e)}")
            return None
        
    
    return [
    Tool(
        name="SQL_Query",
        description=(
            "Execute SQL queries against the RNAseq database to retrieve information "
            "about genes, samples, differential expression results, and metadata. "
            "This is the primary tool for data retrieval. "
            "Always use this tool first when a user asks a question that requires data "
            "from the database, such as 'list all differentially expressed genes,' "
            "'show the top 10 genes by log2FoldChange,' or 'find data for a specific gene.'"
            "Action_Input advice: "
            "Input should be a complete, valid SQL SELECT statement. "
            "ORDER BY clauses should come after UNION ALL not before. "
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
        name="Create_Report",
        description=(
            "Generate a CSV report from the data retrieved by the 'SQL_Query' tool. "
            "This tool requires data to be available from a preceding SQL query. "
            "Do not call this tool until after a successful 'SQL_Query' has been executed. "
            "The input to this tool is a simple string that triggers the report generation. "
            "Example input: 'generate report'. "
        ),
        func=create_csv_report_tool
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
            f"Allowed plot types are: {', '.join(ALLOWED_PLOTS)}."
        ),
        func=create_plot_tool
    )
]