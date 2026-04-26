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


# 🧠 هنا داتا كل عميل (مؤقتًا - بعدين نخليها DB)
CLIENT_PROFILES = {
    "test": {
        "name": "Test Store",
        "tone": "friendly sales assistant",
        "instructions": "You help users buy products and answer questions clearly."
    }
}


class AIEngine:
    def __init__(self, client_id: str):
        self.client_id = client_id

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_db = Chroma(
            collection_name=f"client_{client_id}",
            embedding_function=self.embeddings
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
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

        return "Training completed"

    # =========================
    # CHAT
    # =========================
    def get_chat_response(self, question: str, history: list):

        retriever = self.vector_db.as_retriever()

        profile = CLIENT_PROFILES.get(self.client_id, {
            "name": "Store",
            "tone": "helpful assistant",
            "instructions": "Answer clearly and concisely."
        })

        history_text = "\n".join(
            [f"User: {h.get('user','')} | Bot: {h.get('bot','')}" for h in history]
        )

        def get_context(_):
            docs = retriever.invoke(question)
            return "\n\n".join([d.page_content for d in docs]) if docs else "No context"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""
You are AI assistant for: {profile['name']}
Style: {profile['tone']}
Rules: {profile['instructions']}

CHAT HISTORY:
{history_text}

CONTEXT:
{{context}}

IMPORTANT:
- Answer like a real ecommerce assistant
- Be short, smart, and helpful
- If product exists in context, recommend it
"""),
            ("human", "{input}")
        ])

        chain = (
            {
                "context": get_context,
                "input": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)