import logging
import json
from typing import List, Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
from classifier import IntentRecognizer, PlotRecognizer
from utils import invoke_with_retry

logger = logging.getLogger(__name__)

class RNAseqAgent:
    """Main RNAseq analysis agent"""

    def __init__(self, database, plotter, code_llm, response_llm):
        self.db = database
        self.plotter = plotter
        self.code_llm = code_llm
        self.response_llm = response_llm
        
        # Initialize intent recognizer
        self.intent_recognizer = IntentRecognizer(examples_file="config/intents.json", prompts_file="config/prompts.yaml")
        self.plot_recognizer = PlotRecognizer()

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

        # The agent is initialized with a generic system message.
        # Specific instructions will be injected dynamically per query.
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.code_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            return_intermediate_steps=True,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=75,
            agent_kwargs={
                "system_message": 
                    """  You are an expert RNA-seq data analyst. Your role is to provide concrete answers using actual 
                    data from the database—never simulated or imagined, and interpret them in a biological context 
                    to research scientists.

                    MANDATORY INSTRUCTIONS:
                    0. Respond to small talk politely and briefly, then restate your role.
                    1. Always use the provided tools to interact with the database and generate visualizations.
                    2. If a query fails, use schema and sample value tools to debug before correcting youself and retrying.
                    3. Always try to produce a visualization to show the user if the data is suitable.
                    4. After retrieving data, provide a final answer with in-depth biological interpretation.
                    """
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
            """Execute SQL query against RNAseq database with schema validation"""
            logger.info(f"[SQL_TOOL] Executing query: {query}")

            try:
                # First attempt to execute the query
                result = self.db.execute_query(query)

                if "error" in result:
                    logger.error(f"[SQL_TOOL] Database error: {result['error']}")
                    self.context_state["last_query_successful"] = False
                    
                    # Provide detailed error information with schema context
                    error_msg = f"Query failed: {result['error']}\n\n"
                    
                    # If table doesn't exist, show available tables
                    if "no such table" in result["error"].lower():
                        try:
                            available_tables = self.db.get_table_names()
                            if available_tables:
                                error_msg += f"Available tables: {', '.join(available_tables)}\n"
                                error_msg += "Use the Database_Schema tool to see table structures before querying.\n"
                            else:
                                error_msg += "Could not retrieve available tables.\n"
                        except Exception as e:
                            logger.warning(f"[SQL_TOOL] Could not get table names: {e}")
                            error_msg += "Could not retrieve available tables.\n"
                    
                    # If column doesn't exist, try to show available columns for the table
                    elif "no such column" in result["error"].lower():
                        try:
                            # Extract table name from query (basic parsing)
                            import re
                            table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                            if table_match:
                                table_name = table_match.group(1)
                                
                                # Get schema info for this specific table
                                schema_info = self.db.get_table_info()
                                if "tables" in schema_info and table_name in schema_info["tables"]:
                                    columns = [col["name"] for col in schema_info["tables"][table_name]["columns"]]
                                    error_msg += f"Available columns in table '{table_name}': {', '.join(columns)}\n"
                                else:
                                    error_msg += f"Could not get column information for table '{table_name}'.\n"
                            
                            error_msg += "Use the Database_Schema tool to see exact column names and types.\n"
                            
                        except Exception as e:
                            logger.warning(f"[SQL_TOOL] Could not analyze column error: {e}")
                            error_msg += "Use the Database_Schema tool to see available columns.\n"
                    
                    # For any database error, suggest using schema tools
                    error_msg += "\nRECOMMENDATION: Use Database_Schema tool first to understand the data structure, "
                    error_msg += "then use Sample_Column_Values tool to see actual data values before writing queries."
                    
                    return error_msg

                # Query executed successfully
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
                    return "Query executed successfully but returned no results. The query syntax was correct but no data matches your criteria."

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
                
                # Provide helpful guidance for unexpected errors too
                error_msg = f"Unexpected error executing query: {str(e)}\n\n"
                error_msg += "RECOMMENDATION: Use Database_Schema tool to check table structures "
                error_msg += "and Sample_Column_Values tool to see actual data before writing queries."
                
                return error_msg
            
        def database_schema_tool(input_str: str) -> str:
            """Get information about database tables and their schemas"""
            logger.info("[SCHEMA_TOOL] Retrieving database schema")

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

                return output

            except Exception as e:
                logger.error(f"[SCHEMA_TOOL] Unexpected error: {str(e)}")
                return f"Error retrieving schema: {str(e)}"

        
        def sample_column_values_tool(query: str = "") -> str:
            """
            Get a list of unique values for a column from a table, useful for
            determining which categories or samples are available.
            
            This is the corrected version of the tool.
            """
            logger.info("[SAMPLE_VALUES_TOOL] Retrieving sample column values")
            
            # Get a list of all table names
            result_tables = self.db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
            if result_tables.get('error'):
                return f"Error fetching table names: {result_tables['error']}"

            table_names = [d['name'] for d in result_tables.get('data', [])]
            logger.info(f"[SAMPLE_VALUES_TOOL] Found {len(table_names)} tables.")
            
            # Go through each table and fetch values for categorical columns
            all_sample_values = {}
            for table in table_names:
                try:
                    # Fetch top 5 values for all text columns in the table
                    info_query = f"PRAGMA table_info('{table}');"
                    info_result = self.db.execute_query(info_query)

                    if info_result.get('error'):
                        logger.warning(f"[SAMPLE_VALUES_TOOL] Could not get schema for {table}: {info_result['error']}")
                        continue
                    
                    columns = [d['name'] for d in info_result.get('data', []) if 'text' in d.get('type', '').lower()]
                    
                    for col in columns:
                        value_query = f"SELECT DISTINCT \"{col}\" FROM \"{table}\" LIMIT 5;"
                        values_result = self.db.execute_query(value_query)

                        if values_result.get('error'):
                            logger.warning(f"[SAMPLE_VALUES_TOOL] Could not fetch values for {table}.{col}: {values_result['error']}")
                            continue

                        # Correctly check for empty data
                        if values_result.get('data'):
                            unique_values = [d.get(col) for d in values_result['data']]
                            key = f"{table}.{col}"
                            all_sample_values[key] = unique_values
                except Exception as e:
                    logger.warning(f"[SAMPLE_VALUES_TOOL] An unexpected error occurred with table {table}: {e}")
            
            return json.dumps(all_sample_values)

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

    def ask(self, question: str) -> str:
        """Process user question and return response"""
        logger.info(f"[ASK] Processing question: '{question[:100]}{'...' if len(question) > 100 else ''}'")

        try:
            # 1. Recognize intent
            intent = self.intent_recognizer.recognize_intent(question)
            logger.info(f"[CLASSIFIER] Detected intent: '{intent}'")
            
            # 2. Get the task-specific prompt
            task_specific_prompt = self.intent_recognizer.get_task_specific_prompt(intent)

            # 3. Determine if visualization is needed
            needs_plot = self.plot_recognizer.needs_visualization(question)
            logger.info(f"[PLOT_INTENT] Visualization needed: {needs_plot}")

            # 4. Use Codestral agent for data retrieval and code generation only
            plot_instructions = ""
            if needs_plot:
                plot_instructions = """
            VISUALIZATION REQUIREMENT DETECTED - MANDATORY:
            - You MUST create a visualization after retrieving data
            - Use the Create_Plot tool with appropriate plot type: 'bar', 'scatter', 'heatmap', 'volcano', 'histogram', 'boxplot'
            - This is not optional - the user expects a visual output
            - If you don't create a plot, your response will be incomplete
            """

            system_context = f"""
            You are an expert RNA-seq data analyst. Your role is to provide concrete answers using actual 
            data from the database—never simulated or imagined, and interpret them in an in-depth biological
            context to research scientists.

            {task_specific_prompt}
            {plot_instructions}
            """
            logger.info(f"[CLASSIFIER] Context: '{system_context}'")

            contextualized_question = f"{system_context}\n\nUser question: {question}"
            result = invoke_with_retry(self.agent, contextualized_question)
            technical_output = result.get("output", "")

            # 4. Extract comprehensive context from all intermediate steps
            plot_filename = None
            sql_queries_executed = []
            data_retrieved = []
            plot_info = []
            errors_encountered = []
                    
            for action, observation in result.get("intermediate_steps", []):
                # Extract plot filename
                if isinstance(observation, dict) and "plot_filename" in observation:
                    plot_filename = observation["plot_filename"]
                    plot_info.append(f"Created plot: {observation.get('summary', 'Plot created')}")
                
                # Track SQL queries and their results
                if hasattr(action, 'tool') and action.tool == "SQL_Query":
                    sql_queries_executed.append(action.tool_input)
                    if isinstance(observation, str):
                        if "Query returned" in observation:
                            # Extract the actual data portion
                            data_retrieved.append(observation)
                        elif "Error" in observation or "error" in observation:
                            errors_encountered.append(observation)
                            # Track schema and sample value lookups
                elif hasattr(action, 'tool') and action.tool in ["Database_Schema", "Sample_Column_Values"]:
                    if isinstance(observation, str) and len(observation) > 0:
                        data_retrieved.append(f"Schema/Sample info: {observation[:200]}...")
            
            # 5. Always use Gemini for the natural language response
            logger.info("[ASK] Generating natural language response with Gemini")
            
            # Prepare comprehensive context for Gemini
            context_summary = {
                "original_question": question,
                "intent": intent,
                "sql_queries": sql_queries_executed,
                "data_results": data_retrieved,
                "plot_created": bool(plot_filename),
                "plot_info": plot_info,
                "errors": errors_encountered,
                "technical_output": technical_output
            }
            
            gemini_prompt = f"""
                You are an expert RNA-seq data analyst providing responses to research scientists. 
                Based on the database operations that were just performed, provide a comprehensive natural language response.

                ORIGINAL USER QUESTION: {question}

                TECHNICAL OPERATIONS PERFORMED:
                - SQL Queries executed: {len(sql_queries_executed)}
                - Data retrieved: {'Yes' if data_retrieved else 'No'}
                - Visualizations created: {'Yes' if plot_filename else 'No'}
                - Any errors: {'Yes' if errors_encountered else 'No'}

                DETAILED CONTEXT:
                
                Database Format: Differential expression (Deseq2) results are stored in tables with the nomenclature 'dea_[sample_subset]_[comparison]_deseq2'.
                Pathway enrichment results are stored in tables with the nomenclature 'dea_[sample_subset]_[comparison]_[analysis_type]_[gene_set]'.
                Normalized counts are stored in the table 'normalization'.
                Metadata about samples is stored in the table 'metadata'.
                Correlation matrices are stored in table 'correlation' in square NxN format.
                
                SQL Queries: {sql_queries_executed}

                Data Results: {data_retrieved[:2] if data_retrieved else ['No data retrieved']}

                Plot Information: {plot_info if plot_info else ['No plots created']}

                Errors (if any): {errors_encountered if errors_encountered else ['No errors']}

                Technical Output: {technical_output}

                INSTRUCTIONS:
                1. Provide a complete, natural language answer to the user's original question
                2. If data was retrieved, explain what the data shows and its significance, without mentioning the SQL queries run
                3. If biological data is involved, provide relevant biological interpretation
                4. If plots were created, describe what they show
                5. If errors occurred, explain them in user-friendly terms and suggest solutions
                6. Do NOT mention any internal tool names or technical details about the database or code
                7. Be schematic and scientifically accurate
                8. Still be conversational and engaging and prompt more exploration
                9. Focus on answering the user's specific question with the actual results obtained
                10. Do not use the conditional tense or speculate. 

                Respond as if you're directly answering the user's question with the real data that was just retrieved.
            """
            
            try:
                gemini_response = invoke_with_retry(self.response_llm, gemini_prompt)
                if isinstance(gemini_response, dict):
                    final_answer = gemini_response.get("output", gemini_response)
                elif isinstance(gemini_response, AIMessage):
                    final_answer = gemini_response.content
                else:
                    final_answer = str(gemini_response)
                                
                logger.info("[ASK] Successfully generated response with Gemini")
            
            except Exception as e:
                logger.error(f"[ASK] Failed to generate Gemini response: {str(e)}")
                final_answer = "I encountered an error while generating the response. Please try again or rephrase your question."

            # Update context state
            self.context_state["last_response_successful"] = True
            logger.info(f"[ASK] Successfully processed question - response length: {len(final_answer)} chars")

            return final_answer, plot_filename
            
        except Exception as e:
            logger.error(f"[ASK] Agent execution error: {str(e)}")
            self.context_state["last_response_successful"] = False
            
            # Provide helpful fallback response
            fallback = f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about a specific aspect of the RNAseq analysis."
            return fallback, None
