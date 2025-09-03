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
            config = yaml.safe_load(file)
            self.plot_instructions = config.get("plot_instructions")
            self.allowed_plots = config.get("allowed_plots", [])

    def store_query_data(self, data: List[Dict], query_info: str = ""):
        """Store data from SQL query for potential plotting"""
        self.last_query_data = {
            "data": data,
            "timestamp": datetime.now(),
            "query_info": query_info
        }
        return f"Data stored successfully for plotting. {len(data)} rows available."

    def create_plot(self, plot_type: str, additional_info: str = "", **kwargs) -> Dict[str, Any]:
        """Let LLM generate and execute plotting code"""
        try:
            if not self.last_query_data or not self.last_query_data["data"]:
                return {"error": "No data available for plotting"}

            # Check if plot_type is allowed
            if plot_type not in self.allowed_plots:
                return {"error": f"Plot type '{plot_type}' is not allowed. Allowed types: {self.allowed_plots}"}
            
            data = self.last_query_data["data"]
            df = pd.DataFrame(data)
            available_columns = list(df.columns)
            
            logger.info(f"[PLOTTER] Creating {plot_type} plot with {len(df)} rows and {len(df.columns)} columns")
            logger.debug(f"[PLOTTER] Available columns: {available_columns}")
            logger.debug(f"[PLOTTER] Data sample:\n{df.head(3).to_string(index=False)}")

            # Get the template
            template = self.plot_instructions[plot_type]["template"]
            
            # Ask LLM to select parameters for the template
            param_prompt = f"""
            You need to fill in parameters for a {plot_type} plot template.
            
            Available columns: {available_columns}
            Data sample: {df.head(2).to_dict()}
            Additional context: {additional_info}
            
            Template needs these parameters filled in: {self._extract_template_vars(template)}
            
            Return ONLY a JSON object with the parameter values.
            Use 'None' for any optional parameters you don't want to use.
            Choose appropriate columns based on their names and the plot type.
            
            Example: {{"x_column": "gene_name", "y_column": "expression", "color_column": "None", "title": "Gene Expression"}}
            """
            
            logger.info(f"[PLOTTER] Parameter prompt:\n{param_prompt}")
            
            # Get parameters from LLM
            llm_response = self.llm.invoke(param_prompt).content
            parameters = self._parse_parameters(llm_response, template)

            logger.info(f"[PLOTTER] Selected parameters: {parameters}")
            
            # Fill template with parameters
            filled_code = template.format(**parameters)
            
            # Execute the code
            now = datetime.now()
            timestamp = now.strftime("%m_%d_%H_%M_%S")
            plot_filename = f"plots/{plot_type}_{timestamp}.html"

            exec_globals = {
                'df': df,
                'px': px,
                'go': go,
                'pd': pd,
                'plot_filename': plot_filename,
                'np': __import__('numpy')
            }
            
            exec(filled_code, exec_globals)
            
            logger.debug(f"[PLOTTER] Generated code:\n{filled_code}")

            return {
                "summary": f"{plot_type} plot created successfully",
                "plot_filename": plot_filename,
                "generated_code": filled_code
            }
            
        except Exception as e:
            logger.error(f"[PLOTTER] Exception during plot generation: {str(e)}")
            return {"error": f"Plot generation failed: {str(e)}"}

    def _extract_template_vars(self, template: str) -> list:
        """Extract variable names from template string"""
        import string
        return [fname for _, fname, _, _ in string.Formatter().parse(template) if fname]

    def _parse_parameters(self, llm_response: str, template: str) -> dict:
        """Parse LLM response and ensure all template vars have values"""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                parameters = json.loads(json_match.group())
            else:
                parameters = json.loads(llm_response)
        except json.JSONDecodeError:
            logger.warning(f"[PLOTTER] Failed to parse parameters: {llm_response}")
            parameters = {}
        
        # Ensure all template variables have values to prevent KeyError
        template_vars = self._extract_template_vars(template)
        for var in template_vars:
            if var not in parameters:
                parameters[var] = 'None'  # Safe default
        
        return parameters