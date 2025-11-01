"""
rag.py

This module implements Retrieval-Augmented Generation (RAG) logic specifically designed for
content generation that relies on information gathered from web crawling.

Using:
- HuggingFace Embeddings for text embedding
- FAISS for vector storage and similarity search
- Groq LLM for generating final content based on retrieved information
- Prompts defined in lib/prompt.py for guiding the LLM on how to process and format the content
"""

import faiss
import os
from typing import Dict

from langchain_core.messages import HumanMessage 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_groq import ChatGroq 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

from lib.prompt import prompt__ppt_content, prompt__pdf_content
from shared.config import Config
import shared.logger as logger

class RAGRunner:
    """RAG Runner using HuggingFace embeddings, manual FAISS, and Groq LLM"""

    def __init__(self) -> None:
        """Initialize the RAG (embedding model, LLM, and FAISS vector store) instance"""
        os.environ["GROQ_API_KEY"] = Config.GROQ_API_KEY
        
        # Initialize HuggingFace Embedder
        self.embedder = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        
        # Set up vector store using FAISS and in-memory document store
        index = faiss.IndexFlatL2(Config.EMBEDDING_DIM)
        self.vector_store = FAISS(
            embedding_function=self.embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Initialize LLM
        self.llm = ChatGroq(
            model=Config.LLM_MODEL_NAME,
            temperature=0.3,
            max_retries=2
        )

        logger.info("Initialized RAG with HuggingFace, manual FAISS, and Groq")

    def run(self, input_data: Dict[str, str]) -> str:
        """
        Process user input for a Retrieval-Augmented Generation (RAG) workflow.

        This method:
        1. Validates incoming request data (document type, text content).
        2. Splits the raw text into smaller overlapping chunks for vector search.
        3. Adds the chunks to a FAISS vector store for similarity search.
        4. Finds the most relevant chunks based on the user's query.
        5. Constructs a final prompt for the Large Language Model (LLM).
        6. Sends the prompt to the LLM (Groq) and returns its generated output.

        Args:
            input_data (Dict[str, str]):
                - type: Document type ("pdf" or "ppt").
                - text: The raw text from crawler result.
                - prompt: User's custom instruction or question.
                - topic: The topic associated with the document.

        Returns:
            str: Generated response from the LLM for PDF/PPT content.
        
        Raises:
            ValueError: If `type` is not "pdf" or "ppt", or if `text` is empty.
            Exception: For any runtime errors in text splitting, FAISS operations,
                    similarity search, or LLM invocation. These are caught and
                    converted to a string message before being returned.
        """
        # Define each input
        type = input_data.get("type", "").lower()
        text_inject = input_data.get("text_inject", "")
        query = input_data.get("query", "")
        topic = input_data.get("topic", "")

        if type not in ("pdf", "ppt"):
            logger.error(f"Invalid document type: {type}")
            raise ValueError("Invalid type. Must be 'pdf' or 'ppt'.")
            
        if not text_inject.strip():
            logger.error("No input text provided")
            raise ValueError("No text provided.")

        logger.info(f"Running RAG for type: {type}, topic: {topic}")

        try:
            # Split the text into chunks
            splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=70)
            documents = splitter.create_documents([text_inject])
            logger.debug(f"Split text into {len(documents)} documents")
            
            # Add documents to FAISS
            self.vector_store.add_documents(documents)
            logger.debug("Documents added to FAISS vector store")
            
            # Query similar documents
            query = f"{query} (Topik: {topic})"
            top_docs = self.vector_store.similarity_search(query, k=10)
            context = "\n".join([doc.page_content for doc in top_docs])
            logger.debug(f"Retrieved top {len(top_docs)} similar documents")

            # Prepare final prompt
            base_prompt = prompt__pdf_content if type == "pdf" else prompt__ppt_content
            full_prompt = base_prompt.format(query=query, topic=topic, context=context)
            
            # Query LLM
            logger.info("Sending prompt to Groq LLM")
            response = self.llm.invoke([HumanMessage(content=full_prompt)])

            return response.content.strip()
        
        # Log and return any processing errors
        except Exception as e:
            logger.error(f"RAG processing error: {str(e)}")
            return f"Groq LLM error: {str(e)}"




