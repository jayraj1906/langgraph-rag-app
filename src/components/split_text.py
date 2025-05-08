from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from typing import List
from src.logger import logging
from src.exception import MyException
import sys

class SplitTexts:
    def __init__(self,urls=List[str]):
        self.urls=urls
        self.docs=[WebBaseLoader(url).load() for url in self.urls]
        self.docs_list = [item for sublist in self.docs for item in sublist]

    def split_text(self):
        try:
            logging.info("Splitting initiated")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
            doc_splits = text_splitter.split_documents(self.docs_list)
            logging.info("Splitting Finished")
            return doc_splits
        except Exception as e:
            logging.error(f"Something went went wrong while splitting the text: {e}")
            raise MyException(e,sys)


