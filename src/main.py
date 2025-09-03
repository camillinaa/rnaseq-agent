# main.py
import logging
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from database import RNAseqDatabase
from plotter import RNAseqPlotter
from agent import RNAseqAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def create_agent():
    db_path = "data/rnaseq.db"
    code_llm = ChatGoogleGenerativeAI(
        model=os.getenv("CODE_MODEL_NAME"), 
        api_key=os.getenv("GEMINI_API_KEY"), 
        temperature=0, 
        max_retries=10)
    response_llm = ChatGoogleGenerativeAI(
        model=os.getenv("RESPONSE_MODEL_NAME"), 
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    db = RNAseqDatabase(db_path)
    plotter = RNAseqPlotter(code_llm)
    return RNAseqAgent(db, plotter, code_llm, response_llm)

def run_cli():
    agent = create_agent()
    questions = [
        #"Plot the differentially expressed ORA pathways between flattening yes and no using go gene set",
        "What genes are upregulated in flattening yes vs no?"
    ]
    try:
        for question in questions:
            print(f"\nQ: {question}")
            print(f"A: {agent.ask(question)}")
    finally:
        agent.close()

if __name__ == "__main__":
    bot = create_agent()
    bot.refresh_schema_cache()
    #run_cli()
