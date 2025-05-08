from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from src.config.config import GROQ_API_KEY

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class QueryRouter:
    def __init__(self):
        self.llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Gemma2-9b-It")
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.question_router = None 

    def initiate_query_route(self):
        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or wikipedia.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        self.question_router = route_prompt | self.structured_llm_router
        