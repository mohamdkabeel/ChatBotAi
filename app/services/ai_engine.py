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

        self.persist_directory = f"./data/{client_id}"

        # ✅ FIX: ما تعتمدش على env دلوقتي عشان المشكلة اللي حصلت
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
             api_key=os.getenv("GROQ_API_KEY")  # أو حطه مباشرة لو عايز test
        )

        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
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

        Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

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
