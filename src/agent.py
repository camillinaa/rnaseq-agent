import logging
import yaml
import re
import os
from typing import List
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from utils import invoke_with_retry

logger = logging.getLogger(__name__)

class RNAseqAgent:
    """Main RNAseq analysis agent"""

    def __init__(self, database, plotter, llm):
        self.db = database
        self.plotter = plotter
        self.llm = llm
        
        with open("config/prompts.yaml", "r") as file:
            prompts = yaml.safe_load(file)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="output", 
            return_messages=True
        )

        # Create tools
        self.tools = self._create_tools()

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=60,
            agent_kwargs={
                "system_message": prompts["system_message"]
            }
        )

        # Connect to database
        if not self.db.connect():
            raise Exception("Failed to connect to database")

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""

        def sql_query_tool(query: str) -> str:
            """Execute SQL query against RNAseq database"""
            logger.info(f"EXECUTING SQL QUERY: {query}")
            result = self.db.execute_query(query)
            logger.info(f"QUERY RETURNED {len(result.get('data', []))} rows")

            if "error" in result:
                return f"Error: {result['error']}"

            # Store data for plotting
            if result["row_count"] > 0:
                store_result = self.plotter.store_query_data(result["data"], query)

            # Format result for LLM
            if result["row_count"] == 0:
                return "Query executed successfully but returned no results."

            # Limit output for large results
            max_rows = 20
            data = result["data"][:max_rows]

            output = f"Query returned {result['row_count']} rows. "
            if result["row_count"] > max_rows:
                output += f"Showing first {max_rows} rows:\n"
            else:
                output += "Here are all the results:\n"

            # Format as table-like string
            if data:
                columns = result["columns"]
                output += "\n" + " | ".join(columns) + "\n"
                output += "-" * (len(" | ".join(columns))) + "\n"

                for row in data:
                    row_values = [str(row.get(col, "")) for col in columns]
                    output += " | ".join(row_values) + "\n"

                output += f"\nThis is the actual data from the database. Use this to answer the user's question with specific details."
                output += f"\nNOTE: This data has been stored and is available for plotting if visualization would be helpful."

            return output

        def database_schema_tool(input_str: str) -> str:
            """Get information about database tables and their schemas"""
            logger.info("DATABASE_SCHEMA_TOOL called")  
            result = self.db.get_table_info()
            logger.info(f"Schema result: {result}")  

            if "error" in result:
                return f"Error: {result['error']}"

            output = "Available tables and their schemas:\n\n"
            for table_name, table_info in result["tables"].items():
                output += f"Table: {table_name}\n"
                output += "Columns:\n"
                for col in table_info["columns"]:
                    output += f"  - {col['name']} ({col['type']})\n"
                output += f"Sample query: {table_info['sample_query']}\n\n"

            return output
        
        def sample_column_values_tool(input_str: str) -> str:
            """Get a list of sample values from each column in each table"""
            logger.info("SAMPLE_COLUMN_VALUES_TOOL called")  

            output = "Sample values for selected columns:\n\n"

            for table_name in self.db.get_table_names():
                output += f"Table: {table_name}\n"
                try:
                    df = self.db.run(f"SELECT DISTINCT * FROM {table_name} LIMIT 20;")  # pull small sample
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

        def plot_tool(plot_request: str) -> str:
            """Create plots from data. Format: 'plot_type|column_info|additional_params'"""
            try:
                if not self.plotter.last_query_data:
                    return "Error: No data available for plotting. Please run a SQL query first."

                # Parse the request - could be just "volcano" or "bar|title=My Title"  
                parts = plot_request.split("|")
                plot_type = parts[0].strip()
                
                # Extract any additional parameters
                additional_info = ""
                if len(parts) > 1:
                    additional_info = " ".join(parts[1:])

                result = self.plotter.create_plot(plot_type, additional_info=additional_info)

                if "error" in result:
                    return f"Plot creation failed: {result['error']}"

                return {
                    "success": True,
                    "summary": result["summary"],
                    "plot_filename": result["plot_filename"],
                    "generated_code": result["generated_code"],
                }

            except Exception as e:
                return f"Plot creation error: {str(e)}"
            
        return [
            Tool(
                name="SQL_Query",
                description="Execute SQL queries against the RNAseq database. Use this to get specific data. Input should be a valid SQL SELECT statement.",
                func=sql_query_tool
            ),
            Tool(
                name="Database_Schema",
                description="Get information about available tables and their column structures. Use this to understand what data is available before writing queries.",
                func=database_schema_tool
            ),
            Tool(
                name="Sample_Column_Values",
                description="Get sample values from each column in each table. Use this to match natural language references to possible values in the database.",
                func=sample_column_values_tool
            ),
            Tool(
                name="Create_Plot",
                description="Create plots. Input format: 'plot_type' or 'plot_type|additional_parameters'. Example: 'bar' or 'volcano|title=My Plot'",
                func=plot_tool
            )
        ]

    def ask(self, question: str) -> str:
        """Process user question and return response"""
        try:
            # Add system context about RNAseq analysis
            system_context = """
            You are an expert RNAseq data analyst. You have access to an RNAseq database with tools to:
            1. Query the database using SQL
            2. Get database schema information
            3. Create visualizations

            When analyzing RNAseq data:
            - Always check the database schema first if you're unsure about available tables/columns
            - Use appropriate significance thresholds (e.g., padj < 0.05, |log2fc| > 1)
            - Provide biological context in your interpretations

            Start by understanding what data is available, then query appropriately to answer the question.
            """

            # Run the agent
            contextualized_question = f"{system_context}\n\nUser question: {question}"
            result = invoke_with_retry(self.agent, contextualized_question)
            answer = result.get("output", "")

            # Search intermediate steps for plot_filename
            plot_filename = None
            for action, observation in result.get("intermediate_steps", []):
                if isinstance(observation, dict) and "plot_filename" in observation:
                    plot_filename = observation["plot_filename"]
                    break

            return answer, plot_filename
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    def close(self):
        """Clean up resources"""
        self.db.close()
