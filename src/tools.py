from langchain.agents import Tool
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def sql_query_tool_factory(db, plotter):
    def sql_query_tool(query: str) -> str:
        logger.info(f"EXECUTING SQL QUERY: {query}")
        result = db.execute_query(query)
        logger.info(f"QUERY RETURNED {len(result.get('data', []))} rows")

        if "error" in result:
            return f"Error: {result['error']}"

        if result["row_count"] > 0:
            plotter.store_query_data(result["data"], query)

        if result["row_count"] == 0:
            return "Query executed successfully but returned no results."

        max_rows = 20
        data = result["data"][:max_rows]
        columns = result["columns"]

        output = f"Query returned {result['row_count']} rows. "
        output += "Showing first 20 rows:\n" if result["row_count"] > max_rows else "Here are all the results:\n"
        output += "\n" + " | ".join(columns) + "\n"
        output += "-" * len(" | ".join(columns)) + "\n"

        for row in data:
            output += " | ".join(str(row.get(col, "")) for col in columns) + "\n"

        output += "\nThis data has been stored and is available for plotting."

        return output

    return sql_query_tool


def database_schema_tool_factory(db):
    def database_schema_tool(_: str) -> str:
        logger.info("DATABASE_SCHEMA_TOOL called")  
        
        result = db.get_table_info()
        
        max_length = 500  # max number of characters to log
        summary = str(result)
        if len(summary) > max_length:
            summary = summary[:max_length] + " ... [truncated]"
        logger.info(f"Schema result: {summary}")

        if "error" in result:
            return f"Error: {result['error']}"

        output = "Available tables and their schemas:\n\n"

        for table_name, table_info in result["tables"].items():
            output += f"Table: {table_name}\nColumns:\n"
            for col in table_info["columns"]:
                output += f"  - {col['name']} ({col['type']})\n"
            output += f"Sample query: {table_info['sample_query']}\n\n"

        max_output_length = 1000  # Set your desired max output length here
        if len(output) > max_output_length:
            output = output[:max_output_length] + "\n... [output truncated]"

        return output

    return database_schema_tool


def sample_column_values_tool_factory(db):
    def sample_column_values_tool(input_str: str) -> str:
        """Get a list of sample values from each column in each table"""
        logger.info("SAMPLE_COLUMN_VALUES_TOOL called")  

        output = "Sample values for selected columns:\n\n"

        for table_name in db.get_table_names():
            output += f"Table: {table_name}\n"
            try:
                df = db.run(f"SELECT DISTINCT * FROM {table_name} LIMIT 20;")  # pull small sample
                if isinstance(df, list):
                    df = pd.DataFrame(df)
            except Exception as e:
                logger.error(f"Could not fetch data from {table_name}: {e}")
                continue

            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 0:
                    continue
                # Only show string-like values
                if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                    vals_preview = ", ".join(map(str, unique_vals[:5]))  # limit to 5
                    output += f"  {col}: [{vals_preview}]\n"
            output += "\n"

        return output

    return sample_column_values_tool



def plot_tool_factory(plotter):
    def plot_tool(plot_request: str) -> str:
        if not plotter.last_query_data:
            return "Error: No data available for plotting. Please run a SQL query first."

        try:
            parts = plot_request.split("|")
            plot_type = parts[0].strip()
            additional_info = " ".join(parts[1:]) if len(parts) > 1 else ""

            result = plotter.create_plot(plot_type, additional_info=additional_info)

            if "error" in result:
                return f"Plot creation failed: {result['error']}"

            return f"""
            {result['summary']}
            The plot was saved to: {result['plot_filename']}

            Now that the plot is created, return a biologically rich final answer. Include:
            - Summary statistics (range, min, max, mean)
            - Notable samples, genes, or pathways
            - Implications of observed patterns
            - Hypotheses or biological context
            - Relevance for follow-up analysis or validation
            """

        except Exception as e:
            return f"Plot creation error: {str(e)}"

    return plot_tool
