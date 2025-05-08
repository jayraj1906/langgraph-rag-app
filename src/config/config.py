# config.py
import os
from dotenv import load_dotenv
load_dotenv()

ASTRA_DB_ID=os.getenv("ASTRA_DB_ID")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
HF_KEY=os.getenv("HF_KEY")
GROQ_API_KEY=os.getenv('GROQ_API_KEY')