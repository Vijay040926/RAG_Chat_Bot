import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)
my_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
my_database = FAISS.load_local("medquad_faiss_index", my_embeddings, allow_dangerous_deserialization=True)
groq_key = os.getenv("GROQ_API_KEY")
my_ai_brain = ChatGroq(temperature=0.1, groq_api_key=groq_key, model_name="llama-3.1-8b-instant")
memory_instructions = "Look at the chat history and the new question. Rewrite the question so it makes sense on its own."
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", memory_instructions),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
smart_searcher = create_history_aware_retriever(my_ai_brain, my_database.as_retriever(search_kwargs={"k": 3}), memory_prompt)
bot_rules = """You are a helpful medical chatbot. Use the retrieved facts to answer the user. Do not guess. If you don't know, say you don't know. Facts: {context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", bot_rules),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(my_ai_brain, qa_prompt)
final_chatbot = create_retrieval_chain(smart_searcher, qa_chain)
class UserInput(BaseModel):
    message: str
    history: list
@app.post("/chat")
def chat_with_bot(data: UserInput):
    formatted_history = []
    for past_msg in data.history:
        if past_msg["role"] == "user":
            formatted_history.append(HumanMessage(content=past_msg["content"]))
        else:
            formatted_history.append(AIMessage(content=past_msg["content"]))
    result = final_chatbot.invoke({
        "input": data.message, 
        "chat_history": formatted_history
    })
    return {"reply": result["answer"]}
@app.get("/")
def read_root():
    return {"status": "MediChat API is Live and Running!"}
