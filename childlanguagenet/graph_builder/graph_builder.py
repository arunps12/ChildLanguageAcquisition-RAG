"""Graph builder for LangGraph RAG workflow."""

from langgraph.graph import StateGraph, END

from childlanguagenet.state.rag_state import RAGState
#from childlanguagenet.node.rag_nodes import RAGNodes
from childlanguagenet.node.react_node import RAGNodes 


class GraphBuilder:
    """Builds and manages the LangGraph workflow for RAG."""

    def __init__(self, retriever, llm):
        """
        Initialize graph builder.

        Args:
            retriever: Retriever instance from VectorStore
            llm: Language model instance (e.g., from Config.get_llm())
        """
        self.nodes = RAGNodes(retriever=retriever, llm=llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph.

        Returns:
            Compiled LangGraph instance
        """
        builder = StateGraph(RAGState)

        # ---- Nodes ----
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("generate", self.nodes.generate_answer)

        # ---- Flow ----
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str):
        """
        Run the RAG workflow.

        Args:
            question: User query

        Returns:
            Final RAGState as dict (answer + citations)
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
