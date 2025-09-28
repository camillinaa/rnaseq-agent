import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import RNAseqAgent
from database import RNAseqDatabase

load_dotenv()

def create_agent():
    db_path = os.getenv("DB_PATH", "data/rnaseq.db")
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME"), 
        api_key=os.getenv("GEMINI_API_KEY"), 
        temperature=0.3, 
        max_retries=10
    )
    db = RNAseqDatabase(db_path)
    return RNAseqAgent(db, llm)

if __name__ == "__main__":
    pass