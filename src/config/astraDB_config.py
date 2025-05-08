from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
import cassio
from src.logger import logging
from src.exception import MyException
import sys
from src.config.config import ASTRA_DB_ID,ASTRA_DB_APPLICATION_TOKEN
class AstraDBConnection:
    def __init__(self, table_name="qa_mini_demo", model_name="sentence-transformers/all-MiniLM-L6-v2",keyspace=None):
        self.db_id = ASTRA_DB_ID
        self.db_token = ASTRA_DB_APPLICATION_TOKEN
        self.keyspace = keyspace 
        self.table_name = table_name
        self.model_name = model_name
        self.embedding = None
        self.vector_store = None

    def initiate(self):
        try:
            logging.info("Connection initiated")
            cassio.init(token=self.db_token, database_id=self.db_id)
            self.embedding = HuggingFaceEmbeddings(model_name=self.model_name)
            self.vector_store = Cassandra(
                embedding=self.embedding,
                table_name=self.table_name,
                keyspace=self.keyspace
            )
            logging.info("Connection completed")
        except Exception as e:
            logging.error(f"Something went wrong during establishing the connection: {e}")
            raise MyException(e,sys)

    def get_vector_store(self):
        if self.vector_store is None:
            logging.error(f"Vector store seems to be none")
            raise MyException("Connection not initialized. Call initiate() first.",sys)
        return self.vector_store

            
