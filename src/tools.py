import logging
import json
import re
import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from langchain.tools import Tool
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

LAST_QUERY_DATA = {"data": None, "query_info": "", "timestamp": None}

def store_query_data(data: List[Dict], query_info: str = ""):
    LAST_QUERY_DATA["data"] = data
    LAST_QUERY_DATA["query_info"] = query_info
    LAST_QUERY_DATA["timestamp"] = datetime.now()
    logger.info(f"Data stored successfully for plotting. {len(data)} rows available.")

def get_plot_instructions():
    prompts_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'prompts.yaml')
    try:
        with open(prompts_path, 'r') as file:
            config = yaml.safe_load(file)
            return config.get("plot_instructions"), config.get("allowed_plots")
    except FileNotFoundError:
        logger.error(f"Error: prompts.yaml not found at {prompts_path}")
        return {}, []

PLOT_INSTRUCTIONS, ALLOWED_PLOTS = get_plot_instructions()

def create_tools(db) -> List[Tool]:
    def sql_query_tool(query: str) -> str:
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
                logger.warning(f"[SAMPLE_VALUES_TOOL] Error with table {table}: {e}")
        return json.dumps(all_sample_values)

    def create_plot_tool(plot_request: str) -> str:
        logger.info(f"[PLOT_TOOL] Creating plot: {plot_request}")
        
        if not LAST_QUERY_DATA["data"]:
            return "Error: No data available for plotting. Please run a SQL query first."
            
        try:
            parts = plot_request.split("|")
            plot_type = parts[0].strip()
            plot_params = {}
            for param in parts[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    plot_params[key.strip()] = value.strip()
            
            if plot_type not in ALLOWED_PLOTS:
                return f"Plot type '{plot_type}' is not allowed. Allowed types: {ALLOWED_PLOTS}"
                
            data = LAST_QUERY_DATA["data"]
            df = pd.DataFrame(data)
            
            template_code = PLOT_INSTRUCTIONS.get(plot_type, {}).get("template", "")
            if not template_code:
                return f"Error: No template found for plot type '{plot_type}'."
                
            filled_code = template_code.format(**plot_params)
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            plot_filename = f"plots/{plot_type}_{timestamp}.html"
            os.makedirs("plots", exist_ok=True)
            
            exec_globals = {
                'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np, 'os': os,
                'plotly_filename': plot_filename,
                '__builtins__': {'__import__': __import__, 'print': print, 'open': open, 'ValueError': ValueError}
            }
            exec(filled_code, exec_globals)
            
            return f"Plot of type '{plot_type}' created successfully. Plot saved to: {plot_filename}"
            
        except Exception as e:
            logger.error(f"[PLOT_TOOL] Plot creation failed: {str(e)}")
            return f"Plot creation failed: {str(e)}"
    
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
            "Allowed plot types are: 'scatter', 'pca', 'heatmap', and 'volcano'."
        ),
        func=create_plot_tool
    )
]