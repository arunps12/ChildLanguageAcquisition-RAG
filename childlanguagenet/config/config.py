"""
Configuration module for Child Language Acquisition RAG system.

Centralizes model, data, and pipeline configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()


class Config:
    """Global configuration for the RAG system."""

    # -----------------------
    # Paths
    # -----------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    DATA_DIR = PROJECT_ROOT / "data"
    METADATA_FILE = DATA_DIR / "metadata.json"

    PDF_DIR = DATA_DIR / "pdf"
    INDEX_DIR = DATA_DIR / "index" / "faiss"

    # -----------------------
    # API / Keys
    # -----------------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_API_KEY is None:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. "
            "Set it in your environment or in a .env file."
        )

    # -----------------------
    # LLM configuration
    # -----------------------
    LLM_MODEL = "openai:gpt-4o-mini"  
    TEMPERATURE = 0.2                 
    MAX_TOKENS = 1024

    # -----------------------
    # Embeddings
    # -----------------------
    #EMBEDDING_MODEL = "text-embedding-3-small"

    # -----------------------
    # Document processing
    # -----------------------
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # -----------------------
    # Retrieval
    # -----------------------
    DEFAULT_TOP_K = 8  

    # -----------------------
    # Factory methods
    # -----------------------
    @classmethod
    def get_llm(cls):
        """Initialize and return the chat LLM."""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(
            cls.LLM_MODEL,
            temperature=cls.TEMPERATURE,
            max_tokens=cls.MAX_TOKENS,
        )

    @classmethod
    def get_embeddings(cls):
        """Initialize and return the embedding model."""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return OpenAIEmbeddings(model=cls.EMBEDDING_MODEL)
