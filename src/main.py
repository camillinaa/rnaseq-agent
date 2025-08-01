# main.py
import logging
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from database import RNAseqDatabase
from plotter import RNAseqPlotter
from agent import RNAseqAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def create_agent():
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("MISTRAL_API_KEY")
    db_path = "data/rnaseq.db"
    llm = ChatMistralAI(model=model_name, api_key=api_key, temperature=0, max_retries=10)
    db = RNAseqDatabase(db_path)
    plotter = RNAseqPlotter(llm)
    return RNAseqAgent(db, plotter, llm)

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
