import pandas as pd
import logging
import time
import random
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

###############################
#        AGENT FUNCTIONS      #
###############################

def invoke_with_retry(agent, input_dict: Dict[str, str], max_retries: int = 5) -> Dict[str, Any]:
    """Invoke agent with retry logic for 429 errors and save full API response for debugging."""
    
    for attempt in range(max_retries):
        try:
            result = agent.invoke(input_dict)
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str or "capacity" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)
                    print(f"Gemini at capacity, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    # All retries exhausted - return in the same format as normal agent response
                    logger.error(f"All {max_retries} retries exhausted for API call.")
                    return {
                        "output": "Gemini is currently at capacity. Please try again in a few minutes or contact us.",
                        "intermediate_steps": []
                    }
            else:
                # Not a rate limit error, re-raise
                logger.error(f"Non-retryable error during agent invocation: {e}")
                raise e

# def should_clear_memory(self) -> bool:
#     """Determine if memory should be cleared to prevent context overflow"""
#     self.context_state["conversation_count"] += 1
    
#     # Clear memory every 25 exchanges or if memory is getting large
#     if (self.context_state["conversation_count"] % 25 == 0 or 
#         len(str(self.memory.chat_memory.messages)) > 8000):
#         logger.info(f"[MEMORY] Clearing memory - conversation count: {self.context_state['conversation_count']}")
#         return True
#     return False

# def get_context_summary(agent: Any) -> Dict[str, Any]:
#     """Get current context state for debugging"""
#     return {
#         "context_state": agent.context_state.copy(),
#         "memory_length": len(agent.memory.chat_memory.messages),
#         "database_connected": True  # Assuming connection since we got this far
#     }

def reset_memory(agent: Any):
    """Reset conversation context and memory for a given agent instance."""
    logger.info("[RESET] Resetting context and memory")

    if not hasattr(agent, 'memory'):
        logger.error("[RESET] Agent instance is missing 'memory' attribute.")
        return
    
    agent.memory.clear()
    logger.info("[RESET] Memory cleared.")


###############################
#        SQL FUNCTIONS        #
###############################

def clean_generated_code(code):
        """Clean LLM generated code"""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "")
        if code.startswith("```"):
            code = code.replace("```", "")
        
        # Remove explanatory text before the code
        lines = code.strip().split('\n')
        clean_lines = []
        code_started = False
        
        for line in lines:
            if line.strip().startswith('import') or line.strip().startswith('fig') or code_started:
                code_started = True
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)

def find_column(df: pd.DataFrame, possible_names: List[str]) -> str:
    """Find column by checking possible names (case insensitive)"""
    df_cols_lower = [col.lower() for col in df.columns]
    for name in possible_names:
        if name.lower() in df_cols_lower:
            return df.columns[df_cols_lower.index(name.lower())]
    return None
