import logging
import yaml
import re
import os
from typing import List, Dict, Any
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

        # Context state tracking
        self.context_state = {
            "stage": "start",
            "last_query_successful": False,
            "current_data_context": None,
            "conversation_count": 0,
        }

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="output", 
            return_messages=True,
            max_token_limit=4000,  # Limit to prevent overflow
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
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=75,
            agent_kwargs={
                "system_message": prompts["system_message"]
            }
        )

        # Connect to database
        logger.info("Initializing RNAseq Agent...")
        if not self.db.connect():
            logger.error("Failed to connect to database")
            raise Exception("Failed to connect to database")
        logger.info("Database connection established successfully")

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""

        def sql_query_tool(query: str) -> str:
            """Execute SQL query against RNAseq database"""
            logger.info(f"[SQL_TOOL] Executing query: {query}")

            try:
                result = self.db.execute_query(query)

                if "error" in result:
                    logger.error(f"[SQL_TOOL] Database error: {result['error']}")
                    self.context_state["last_query_successful"] = False
                    return f"Error: {result['error']}"

                # Store data for plotting
                if result["row_count"] > 0:
                    self.context_state["last_query_successful"] = True
                    self.context_state["current_data_context"] = {
                            "query": query,
                            "row_count": result["row_count"],
                            "columns": result["columns"]
                        }
                    store_result = self.plotter.store_query_data(result["data"], query)
                    logger.info(f"[SQL_TOOL] Data stored for plotting - {result['row_count']} rows, {len(result['columns'])} columns")
                else:
                    self.context_state["last_query_successful"] = False
                    logger.warning(f"[SQL_TOOL] Query returned no results")

                # Format result for LLM
                if result["row_count"] == 0:
                    return "Query executed successfully but returned no results."

                # Limit output for large results
                max_rows = 15
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
        
            except Exception as e:
                logger.error(f"[SQL_TOOL] Unexpected error: {str(e)}")
                self.context_state["last_query_successful"] = False
                return f"Unexpected error executing query: {str(e)}"

        def database_schema_tool(input_str: str) -> str:
            """Get information about database tables and their schemas"""
            logger.info("[SCHEMA_TOOL] Retrieving database schema")
            
            # Simple caching to avoid repeated calls
            if hasattr(self, '_cached_schema'):
                logger.info("[SCHEMA_TOOL] Using cached schema")
                return self._cached_schema
            
            try:
                result = self.db.get_table_info()

                if "error" in result:
                    logger.error(f"[SCHEMA_TOOL] Error retrieving schema: {result['error']}")
                    return f"Error: {result['error']}"
                
                table_count = len(result.get("tables", {}))
                logger.info(f"[SCHEMA_TOOL] Found {table_count} tables")

                output = "Available tables and their schemas:\n\n"
                displayed_tables = 0
                
                for table_name, table_info in result["tables"].items():
                    if displayed_tables >= 10:  # Limit to prevent context overflow
                        remaining = table_count - displayed_tables
                        output += f"... and {remaining} more tables (use Sample_Column_Values for specific table details)\n"
                        break
                        
                    output += f"Table: {table_name}\n"
                    output += "Key columns:\n"
                    
                    # Limit columns shown per table
                    columns_shown = 0
                    for col in table_info["columns"]:
                        if columns_shown >= 8:
                            remaining_cols = len(table_info["columns"]) - columns_shown
                            output += f"  ... and {remaining_cols} more columns\n"
                            break
                        output += f"  - {col['name']} ({col['type']})\n"
                        columns_shown += 1
                        
                    output += f"Sample query: {table_info['sample_query']}\n\n"
                    displayed_tables += 1

                # Cache the result
                self._cached_schema = result
                logger.info("[SCHEMA_TOOL] Schema cached for future use")
                return output

            except Exception as e:
                logger.error(f"[SCHEMA_TOOL] Unexpected error: {str(e)}")
                return f"Error retrieving schema: {str(e)}"

        
        def sample_column_values_tool(input_str: str) -> str:
            """Get sample values from columns with focused output and error handling"""
            logger.info("[SAMPLE_VALUES_TOOL] Retrieving sample column values")

            output = "Sample values for key columns:\n\n"
            tables_processed = 0
            
            try:
                table_names = self.db.get_table_names()
                logger.info(f"[SAMPLE_VALUES_TOOL] Processing {len(table_names)} tables")
                
                for table_name in table_names[:8]:  # Limit tables to prevent overflow
                    tables_processed += 1
                    output += f"Table: {table_name}\n"
                    
                    try:
                        df = self.db.run(f"SELECT * FROM {table_name} LIMIT 10;")
                        if isinstance(df, list):
                            df = pd.DataFrame(df)
                        
                        if df.empty:
                            output += "  (No data available)\n\n"
                            continue
                            
                        logger.debug(f"[SAMPLE_VALUES_TOOL] Table {table_name}: {len(df)} rows, {len(df.columns)} columns")
                        
                    except Exception as e:
                        logger.warning(f"[SAMPLE_VALUES_TOOL] Could not fetch data from {table_name}: {e}")
                        output += f"  (Error accessing table: {str(e)})\n\n"
                        continue

                    columns_shown = 0
                    for col in df.columns:
                        if columns_shown >= 6:  # Limit columns per table
                            remaining_cols = len(df.columns) - columns_shown
                            output += f"  ... and {remaining_cols} more columns\n"
                            break
                            
                        try:
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) == 0:
                                continue
                                
                            # Focus on categorical/string columns
                            if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                                vals_preview = ", ".join(map(str, unique_vals[:4]))  # Limit to 4 values
                                if len(unique_vals) > 4:
                                    vals_preview += f", ... ({len(unique_vals)} total)"
                                output += f"  {col}: [{vals_preview}]\n"
                                columns_shown += 1
                        except Exception as e:
                            logger.debug(f"[SAMPLE_VALUES_TOOL] Error processing column {col}: {e}")
                            continue
                            
                    output += "\n"

                logger.info(f"[SAMPLE_VALUES_TOOL] Processed {tables_processed} tables successfully")
                return output

            except Exception as e:
                logger.error(f"[SAMPLE_VALUES_TOOL] Unexpected error: {str(e)}")
                return f"Error retrieving sample values: {str(e)}"

        def plot_tool(plot_request: str) -> str:
            """Create plots with enhanced logging and error handling"""
            logger.info(f"[PLOT_TOOL] Creating plot: {plot_request}")
            
            try:
                if not self.plotter.last_query_data:
                    logger.warning("[PLOT_TOOL] No data available for plotting")
                    return "Error: No data available for plotting. Please run a SQL query first to retrieve data."

                # Parse the request - first part is plot type
                parts = plot_request.split("|")
                plot_type = parts[0].strip()
                
                # Parse the request - second part is additional parameters
                additional_info = ""
                if len(parts) > 1:
                    additional_info = " ".join(parts[1:])

                logger.info(f"[PLOT_TOOL] Plot type: {plot_type}, Additional info: {additional_info}")

                result = self.plotter.create_plot(plot_type, additional_info=additional_info)

                if "error" in result:
                    logger.error(f"[PLOT_TOOL] Plot creation failed: {result['error']}")
                    return f"Plot creation failed: {result['error']}"

                # Update context
                self.context_state["last_plot_created"] = plot_type
                logger.info(f"[PLOT_TOOL] Successfully created {plot_type} plot: {result.get('plot_filename', 'unknown')}")

                return {
                    "success": True,
                    "summary": result["summary"],
                    "plot_filename": result["plot_filename"],
                    "generated_code": result.get("generated_code", ""),
                }

            except Exception as e:
                logger.error(f"[PLOT_TOOL] Unexpected error: {str(e)}")
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

    def _should_clear_memory(self) -> bool:
        """Determine if memory should be cleared to prevent context overflow"""
        self.context_state["conversation_count"] += 1
        
        # Clear memory every 25 exchanges or if memory is getting large
        if (self.context_state["conversation_count"] % 25 == 0 or 
            len(str(self.memory.chat_memory.messages)) > 8000):
            logger.info(f"[MEMORY] Clearing memory - conversation count: {self.context_state['conversation_count']}")
            return True
        return False


    def ask(self, question: str) -> str:
        """Process user question and return response"""
        logger.info(f"[ASK] Processing question: '{question[:100]}{'...' if len(question) > 100 else ''}'")

        try:
            if self._should_clear_memory():
                self.memory.clear()
                self.context_state["conversation_count"] = 0
                # clear schema cache when memory is cleared
                if hasattr(self, '_cached_schema'):
                    del self._cached_schema
                    logger.info("[MEMORY] Cleared cached schema")
                    
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
            
            # Update context state
            self.context_state["last_response_successful"] = True
            logger.info(f"[ASK] Successfully processed question - response length: {len(answer)} chars")
            
            return answer, plot_filename
            
        except Exception as e:
            logger.error(f"[ASK] Agent execution error: {str(e)}")
            self.context_state["last_response_successful"] = False
            
            # Provide helpful fallback response
            fallback = f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about a specific aspect of the RNAseq analysis."
            return fallback, None

    def get_context_summary(self) -> Dict[str, Any]:
        """Get current context state for debugging"""
        return {
            "context_state": self.context_state.copy(),
            "memory_length": len(self.memory.chat_memory.messages),
            "has_cached_schema": hasattr(self, '_cached_schema'),
            "database_connected": True  # Assuming connection since we got this far
        }

    def reset_context(self):
        """Reset conversation context and memory"""
        logger.info("[RESET] Resetting context and memory")
        self.memory.clear()
        self.context_state = {
            "stage": "start",
            "last_intent": None,
            "last_query_successful": False,
            "current_data_context": None,
            "conversation_count": 0
        }
        if hasattr(self, '_cached_schema'):
            delattr(self, '_cached_schema')
        logger.info("[RESET] Context reset complete")

    def refresh_schema_cache(self):
        """Manually refresh schema cache"""
        if hasattr(self, '_cached_schema'):
            del self._cached_schema
            logger.info("[SCHEMA_TOOL] Schema cache cleared manually")

    def close(self):
        """Clean up resources"""
        logger.info("[CLOSE] Cleaning up agent resources")
        try:
            self.memory.clear()
            self.db.close()
            logger.info("[CLOSE] Cleanup completed successfully")
        except Exception as e:
            logger.error(f"[CLOSE] Error during cleanup: {e}")
