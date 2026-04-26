import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class AIEngine:
    def __init__(self, client_id: str):
        self.client_id = client_id

        # embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Use in-memory Chroma for Railway (ephemeral filesystem)
        # Each client_id gets its own in-memory vector store
        self.vector_db = Chroma(
            collection_name=f"client_{client_id}",
            embedding_function=self.embeddings
        )

        # Initialize LLM with API key validation
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it in Railway variables."
            )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=groq_api_key
        )

    # =========================
    # TRAIN
    # =========================
    def train_on_data(self, text_content: str):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        docs = splitter.create_documents([text_content])

        self.vector_db.add_documents(docs)

        return "تم تدريب البوت بنجاح على بياناتك!"

    # =========================
    # CHAT
    # =========================
    def get_chat_response(self, question: str, history: list):

        retriever = self.vector_db.as_retriever()

        # format history safely
        history_text = ""
        for h in history:
            history_text += f"User: {h.get('user','')}\nBot: {h.get('bot','')}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant.\n"
             "Use the context and chat history to answer clearly.\n\n"
             "History:\n{history}\n\n"
             "Context:\n{context}"),
            ("human", "{input}")
        ])

        # FIXED context retrieval (safe + stable)
        def get_context(_):
            docs = retriever.invoke(question)
            if not docs:
                return "No relevant context found."
            return "\n\n".join([d.page_content for d in docs])

        chain = (
            {
                "context": get_context,
                "input": RunnablePassthrough(),
                "history": lambda _: history_text
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)
