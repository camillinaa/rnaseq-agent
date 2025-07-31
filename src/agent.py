import logging
import yaml
import os
import pandas as pd
from typing import List
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import warnings
import utils


warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
logger = logging.getLogger(__name__)

class RNAseqAgent:
    """Main RNAseq analysis agent"""

    def __init__(self, database, plotter, llm):
        self.db = database
        self.plotter = plotter
        self.llm = llm
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'prompts.yaml')
        with open(prompts_path, 'r') as file:
            self.prompts_path = yaml.safe_load(file)

        # Initialize memory and context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="output", 
            return_messages=True
        )
        self.context_state = {
            "stage": "start"  # initial stage
        }

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
            agent_kwargs={"system_message":self.prompts_path.get('system message', 'No prompt available.')} 
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
                description=f'{self.prompts_path.get("sql_query_tool", "No prompt available.")}',
                func=sql_query_tool
            ),
            Tool(
                name="Database_Schema",
                description=f'{self.prompts_path.get("database_schema_tool", "No prompt available.")}',
                func=database_schema_tool
            ),
            Tool(
                name="Sample_Column_Values",
                description=f'{self.prompts_path.get("sample_column_values_tool", "No prompt available.")}',
                func=sample_column_values_tool
            ),
            Tool(
                name="Create_Plot",
                description= f'{self.prompts_path.get("plot_tool", "No prompt available.")}',
                func=plot_tool
            )
        ]

    def detect_intent(self, user_input: str) -> str:
        prompt = f"""
        You are a classification system. Categorize the user's message into **only one** of the following three categories:

        - chitchat → greetings, personal questions (e.g., "Who are you?"), or thanks  
        - query → questions about RNAseq data, analysis, or database structure  
        - explanation → questions about theoretical concepts or methods
        Only return one word: **chitchat**, **query**, or **explanation**.
        Message: "{user_input.strip()}"
        """
        
        response = utils.invoke_with_retry(self.agent, {"input": prompt}) 
        print(f"DEBUG: Raw response from intent detection: {response}")
        intent = response.get("output", "").strip().lower()
        
        print(f"DEBUG: Intent full response: {response}") 
        print(f"DEBUG: Detected intent: {intent}") 

        if "chitchat" in intent:
            return "chitchat"
        if "query" in intent:
            return "query"
        if "explanation" in intent:
            return "explanation"
        else:
            logger.warning(f"Unrecognized intent: {intent}, falling back to 'explanation'")
            return "explanation"
    
    def ask(self, question: str) -> str:
        """Process user question and return response"""
        try:
            intent = self.detect_intent(question)
            self.context_state["last_intent"] = intent
            
            # Choose system prompt based on intent
            if intent == "query":
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

            elif intent == "explanation":
                system_context = """
                You are an RNAseq data analyst explaining bioinformatics concepts to a non-technical research scientist audience.
                The user is trying to understand a technical concept related to RNAseq, such as normalization, PCA, log2FC, adjusted p-values, or quality control steps.
                Your job is to explain clearly, using analogies if useful, and always relate the concept back to how it's used in practice in RNAseq.
                Keep it concise but insightful.
                """
            
            elif intent == "chitchat":
                system_context = """
                You are an RNAseq chatbot with a friendly personality.
                The user may ask personal or playful questions like "Who are you?", "Do you sleep?", or just want to talk.
                Stay in character and respond warmly but keep the tone relevant to RNAseq and your capabilities as a data assistant.
                """
                
            # Run the agent
            contextualized_question = f"{system_context}\n\nUser: {question}"
            result = utils.invoke_with_retry(self.agent, {"input": contextualized_question}) # calls the invoke method of the agent, but has retry logic in place for error 429
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
