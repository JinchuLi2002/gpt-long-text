from langchain.vectorstores import Pinecone as lc_pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import os


class VectorStore:
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        pinecone_index_name: str,
    ):
        self.PINECONE_API_KEY, self.PINECONE_ENV, self.pinecone_index_name = pinecone_api_key, pinecone_env, pinecone_index_name
        pinecone.init(api_key=self.PINECONE_API_KEY,
                      environment=self.PINECONE_ENV)

        self.pinecone_index = pinecone.Index(
            index_name=self.pinecone_index_name)
        self.embedding = OpenAIEmbeddings()
