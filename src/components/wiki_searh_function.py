from langchain.schema import Document

class WikiSearch:
    def __init__(self,wiki):
        self.wiki=wiki
    def wiki_search(self,state):
        """
        wiki search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---wikipedia---")
        print("---HELLO--")
        question = state["question"]

        # Wiki search
        docs = self.wiki.invoke({"query": question})
        #print(docs["summary"])
        print(docs)
        wiki_results = docs
        wiki_results = Document(page_content=wiki_results)

        return {"documents": [wiki_results], "question": question}