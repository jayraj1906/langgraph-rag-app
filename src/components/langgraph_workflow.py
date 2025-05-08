from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from src.components.retrieve_search import RetrieveSearch
from src.components.wiki_searh_function import WikiSearch
from src.components.router_question_function import RouteQuestion

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

class WorkFlow:
    def __init__(self):
        self.lang_app=None
    def initiate_workflow(self,app):
        retriever = app.state.retriever
        wiki=app.state.wiki
        wiki_search=WikiSearch(wiki)
        retrieve_search = RetrieveSearch(retriever)
        rq=RouteQuestion()
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("wiki_search", wiki_search.wiki_search)  # web search
        workflow.add_node("retrieve", retrieve_search.retrieve)  # retrieve

        # Build graph
        workflow.add_conditional_edges(
            START,
            rq.route_question,
            {
                "wiki_search": "wiki_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge( "retrieve", END)
        workflow.add_edge( "wiki_search", END)
        # Compile
        self.lang_app = workflow.compile()

    def run(self, question: str) -> GraphState:
        if not self.lang_app:
            raise RuntimeError("Workflow not initialized. Call initiate_workflow(app) first.")
        return self.lang_app.invoke({"question": question, "generation": "", "documents": []})
    
    def run_streaming(self, question: str) -> dict:
        if not self.lang_app:
            raise RuntimeError("Workflow not initialized. Call initiate_workflow(app) first.")

        inputs = {"question": question, "generation": "", "documents": []}
        final_state = {}

        for output in self.lang_app.stream(inputs):
            for key, value in output.items():
                final_state = value  # the final value after all steps

        # Safely extract description if available
        try:
            if final_state.get("documents"):
                doc = final_state["documents"][0]
                # Assume doc is a LangChain Document
                metadata = getattr(doc, "metadata", {})
                description = metadata.get("description", "No description")
                final_state["final_description"] = description
        except Exception as e:
            final_state["final_description"] = f"Error extracting description: {e}"

        return final_state