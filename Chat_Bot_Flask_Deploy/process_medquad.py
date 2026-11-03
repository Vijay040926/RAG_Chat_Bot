import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def process_medquad_dataset(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return
    except Exception as e:
        return
    
    if 'question' not in df.columns or 'answer' not in df.columns:
        return
    
    df = df.dropna(subset=['question', 'answer'])
    combined_texts = df.apply(lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", axis=1).tolist()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False)
    chunks = text_splitter.create_documents(combined_texts)
    return chunks


def build_faiss_index(chunks, save_path="medquad_faiss_index"):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore


if __name__ == "__main__":
    csv_file_path = "medquad.csv"
    chunks = process_medquad_dataset(csv_file_path)
    if chunks:
        build_faiss_index(chunks)