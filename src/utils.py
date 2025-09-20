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
            
            # Gemini FinishReason enum errors
            if "'int' object has no attribute 'name'" in str(e):
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt + random.uniform(0, 1), 10)
                    print(f"Gemini API FinishReason error, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Persistent Gemini API error after {max_retries} retries.")
                    return {
                        "output": "I encountered a technical issue with the Gemini API. Please try your question again.",
                        "intermediate_steps": []
                    }
                    
            # Gemini rate limit errors
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

def reset_memory(agent: Any):
    """Reset conversation context and memory for a given agent instance."""
    logger.info("[RESET] Resetting context and memory")

    if not hasattr(agent, 'memory'):
        logger.error("[RESET] Agent instance is missing 'memory' attribute.")
        return
    
    agent.memory.clear()
    logger.info("[RESET] Memory cleared.")