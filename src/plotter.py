import logging
import os
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import utils 

logger = logging.getLogger(__name__)

class RNAseqPlotter:
    """Handle plot generation for RNAseq data using Plotly"""

    def __init__(self, llm=None, output_dir: str = "plots"):
        self.llm = llm
        self.output_dir = output_dir
        self.last_query_data = None  # Store data from last query for plotting
        os.makedirs(output_dir, exist_ok=True)
        pio.templates.default = "plotly_white" # Set default plotly theme
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'prompts.yaml')
        
        # Load plotting instructions
        with open(prompts_path, 'r') as file:
            self.plot_instructions = yaml.safe_load(file)["plot_instructions"]

    def store_query_data(self, data: List[Dict], query_info: str = ""):
        """Store data from SQL query for potential plotting"""
        self.last_query_data = {
            "data": data,
            "timestamp": datetime.now(),
            "query_info": query_info
        }
        return f"Data stored successfully for plotting. {len(data)} rows available."

    def create_plot(self, plot_type: str, **kwargs) -> Dict[str, Any]:
        """Let LLM generate and execute plotting code"""
        try:
            if not self.last_query_data or not self.last_query_data["data"]:
                return {"error": "No data available for plotting"}

            data = self.last_query_data["data"]
            df = pd.DataFrame(data)
            available_columns = list(df.columns)
            
            logger.info(f"[PLOTTER] Creating {plot_type} plot with {len(df)} rows and {len(df.columns)} columns")
            logger.debug(f"[PLOTTER] Available columns: {available_columns}")
            logger.debug(f"[PLOTTER] Data sample:\n{df.head(3).to_string(index=False)}")

            now = datetime.now()
            timestamp = now.strftime("%m_%d_%H_%M_%S")
            plot_filename = f"plots/{plot_type}_{timestamp}.html"

            # Ask LLM to generate plotting code
            code_prompt = f"""
            Generate Python plotly code to create a {plot_type} plot.
            
            Available columns: {available_columns}
            Data sample (first few rows): {df.head(2).to_dict()}
            
            Requirements:
            - Use plotly.express or plotly.graph_objects
            - The dataframe is already loaded as 'df'
            - Use a variable named plot_filename to save the plot, do not hardcode any filename string in fig.write_html(). The variable plot_filename will be passed into the execution environment.
            - Return just the Python code, no explanations
            - Choose appropriate columns based on the data type and plot type
            
            Follow these instructions for the {plot_type} plot specifically: {self.plot_instructions.get(plot_type, '')}
            """
            
            logger.info(f"[PLOTTER] Code prompt:\n{code_prompt}")
            
            # Get code from LLM
            generated_code = self.llm.invoke(code_prompt).content
            cleaned_code = utils.clean_generated_code(generated_code)
            logger.info(f"[PLOTTER] Generated code (cleaned):\n{cleaned_code}")
            
            # Execute the generated code
            exec_globals = {
                'df': df,
                'px': px,
                'go': go,
                'pd': pd,
                'plot_filename': plot_filename
            }
            
            exec(cleaned_code, exec_globals)
            
            logger.debug(f"[PLOTTER] Generated code for {plot_type}:\n{cleaned_code}")

            return {
                "summary": f"{plot_type} plot created successfully",
                "plot_filename": plot_filename,
                "generated_code": cleaned_code
            }
            
        except Exception as e:
            logger.error(f"[PLOTTER] Exception during plot generation: {str(e)}")
            return {"error": f"Plot generation failed: {str(e)}"}
