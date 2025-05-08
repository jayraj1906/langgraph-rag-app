from src.components.query_router import QueryRouter
class RouteQuestion:
    def __init__(self):
        self.qr = QueryRouter()
        self.qr.initiate_query_route()

    def route_question(self,state):
        """
        Route question to wiki search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.qr.question_router.invoke({"question": question})
        if source.datasource == "wiki_search":
            print("---ROUTE QUESTION TO Wiki SEARCH---")
            return "wiki_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"