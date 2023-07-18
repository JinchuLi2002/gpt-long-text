from langchain.chat_models import ChatOpenAI
from vecstore import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int = 1300,
    chunk_overlap: int = 100,
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.create_documents([text])
    return docs


class LangChainClient:
    def __init__(
        self,
        openai_api_key: str,
        vector_store: VectorStore,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=self.openai_api_key,
            model=model,
        )
        self.vectorstore = vector_store

    def get_summary(
        self,
        docs: list,
        chain_type: str = "map_reduce",
        verbose: bool = False,
    ) -> str:
        summarize_chain = load_summarize_chain(
            self.llm, chain_type=chain_type, verbose=verbose)
        summary = summarize_chain.run(docs)
        return summary

    def override_index_w_summary(
        self,
        docs: list,
        chain_type: str = "map_reduce",
        verbose: bool = False,
        metadata: dict = {"metadata": "test_summary"},
    ) -> None:
        summary = self.get_summary(docs, chain_type, verbose)
        summary = "the summary of the article/paper is as follows: " + summary
        docs.append(Document(page_content=summary, metadata=metadata))
        self.override_index(docs)

    def override_index(self, docs: list):
        self.vectorstore.clear()
        self.vectorstore.add_docs(docs)
        self.docsearch = self.vectorstore.db

    def clear_index(self):
        self.vectorstore.clear()

    def get_chain(self, chain_type: str = "stuff", verbose: bool = False):
        if hasattr(self, "qa"):
            return self.qa
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.docsearch.as_retriever(),
            chain_type=chain_type,
            verbose=verbose,
            return_source_documents=True,
        )
        return self.qa
