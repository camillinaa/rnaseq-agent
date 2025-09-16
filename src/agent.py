import logging
import json
import re
from typing import List
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.schema import AIMessage
from classifier import IntentRecognizer, PlotRecognizer
from utils import invoke_with_retry
from tools import create_tools

logger = logging.getLogger(__name__)

class RNAseqAgent:
    """Main RNAseq analysis agent using ConversationBufferMemory"""

    def __init__(self, database, code_llm):
        self.db = database
        self.code_llm = code_llm
        # The plotter is no longer a separate instance.

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            max_token_limit=4000
        )

        self.tools = create_tools(self.db)
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.code_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=30,
            agent_kwargs={
                "system_message": 
                    """
                    You are an expert RNA-seq data analyst and a helpful assistant.

                    Your primary function is to answer questions using the provided tools. Only use your internal knowledge for questions that do not require tool usage (e.g., greetings, general definitions of biological terms).

                    You must think step-by-step.
                    1.  First, check if the user's question requires data analysis. If not, provide a conversational response.
                    2.  If it requires data, your first action must be to use the `Database_Schema` and `Sample_Column_Values` tools to understand the data's structure.
                    3.  After understanding the data, use the `SQL_Query` tool to retrieve the necessary information.
                    4.  If a plot is requested, use the `Create_Plot` tool after a successful `SQL_Query`.
                    5.  Finally, provide a concise, biologically-informed answer based on the retrieved data.

                    DATA-SPECIFIC INSTRUCTIONS:
                    - When querying gene expression, use the `normalized_counts_matrix` table.
                    - In `normalized_counts_matrix`, the columns named "gene_name" and "gene_id" are for gene symbols and Ensembl IDs, respectively. All other columns are **SAMPLE COLUMNS** containing expression values.
                    - There are NO generic 'sample_id' or 'expression_value' columns. The column headers themselves are the sample names.
                    - To find the expression of a gene in a specific sample, you must query for the gene and then select the specific sample's column.
                    - If the user asks for metadata of a specific sample, use the `study_metadata` table, where sample names are in the 'Sample' column.
                    - For differential expression analysis, use the `deseq2_results` table. This table contains columns such as 'gene_name', 'log2FoldChange', and 'padj'.

                    ERROR PREVENTION:
                    - Do not guess table names, column names, or data values. Use the discovery tools first.
                    - If a query fails with a 'no such table' or 'no such column' error, your next action MUST be to use the `Database_Schema` tool.
                    - Do not generate SQL queries that alter or modify the database. Only use SELECT statements.                    """
            }
        )

    def ask(self, question: str):
        """Process user question and return response"""
        logger.info(f"[ASK] Processing question: '{question[:100]}{'...' if len(question) > 100 else ''}'")
        try:
            result = self.agent.invoke({"input": question})
            final_answer = result.get("output", "I was unable to provide a response.")
            
            plot_filename = None
            if isinstance(result.get("intermediate_steps"), list):
                for action, observation in result["intermediate_steps"]:
                    if isinstance(observation, str) and "Plot saved to:" in observation:
                        match = re.search(r"plots/[\w\d_.-]+\.html", observation)
                        if match:
                            plot_filename = match.group(0)

            logger.info("[ASK] Successfully processed question.")
            return final_answer, plot_filename

        except Exception as e:
            logger.error(f"[ASK] Agent execution error: {str(e)}")
            fallback = f"I encountered an error while processing your question: {str(e)}."
            return fallback, None