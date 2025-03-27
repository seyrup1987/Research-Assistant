import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from uuid import uuid4
from models import Models
from langchain_core.documents import Document

load_dotenv()

models = Models()
embeddings = models.embeddings

data_folder = "C:/Users/seyru/Research_Assistant/Data"
chunk_size = 1000
chunk_overlap = 200
check_interval = 10

vector_store = Chroma(
    collection_name = "documents",
    embedding_function = embeddings,
    persist_directory = "./db/chroma_langchain_db"
)

def ingest_file(file_path):
    if not file_path.lower().endswith("pdf"):
        print(f'Skipping non-pdf files: {file_path}')
        return
    
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()

    # Semantic Chunking
    text_splitter2 = SemanticChunker(
        embeddings,
        breakpoint_threshold_type = "percentile", #standard_deviation, interquartile
        breakpoint_threshold_amount = 95.0
    )

    chunks = text_splitter2.split_documents(loaded_documents)

    print(f"Processing File : {os.path.basename(file_path)}")
    print(f"Document split into {len(chunks)} chunks")
    
    processed_chunks = []
    for i, chunk in enumerate(chunks, 1):
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
            clean_text = text.encode("utf-8", "ignore").decode("utf-8")
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            processed_chunk = Document(
                page_content = clean_text,
                metadata = metadata
            )
            processed_chunks.append(processed_chunk)

            if i % 10 == 0 or i == len(chunks):
                print(f"Progress: {i}/{len(chunks)} chunks Processed ({(i/len(chunks)*100):.1f}%)")


    uuids = [str(uuid4()) for _ in range(len(processed_chunks))]
    print(f"Ingesting {len(processed_chunks)} chunks into vector store . . . ")
    vector_store.add_documents(documents = processed_chunks, ids = uuids)
    print(f"Successfully ingested {len(processed_chunks)} chunks from {os.path.basename(file_path)}")

def main_loop():
    print(f"Starting Document Ingestion Service | Monitoring folder: {data_folder}")
    print(f"Documents will checked every {check_interval} seconds.")
    print(f"Add PDF files to this folder to ingest them into the vector database")
    print("=" * 80)

    while True:
        files = os.listdir(data_folder)
        pdf_files = [f for f in files if f.endswith(".pdf") and not f.startswith("_")]
        pdf_count = len(pdf_files)

        if pdf_count > 0:
            print(f"Found {pdf_count} PDF Files in {data_folder}")

            processed_count = 0
            for filename in pdf_files:
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_"+filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
                processed_count += 1

            print(f"Processed {processed_count} PDF Files.")
            print("=" * 80)
            print(f"All Documentshave been processed successfully.")
            print(f"Processed documents are prefixed with '_' and remain in the data folder.")
            print(f"text check in {check_interval} seconds ... ")
            print("=" * 80)

        else:
            print(f"[{time.strftime('%y-%m-%d %H:%M:%S')}] No new documents found. Waiting for Documents to be added to {data_folder}")
            print(f"Add PDF files to {data_folder} for procesing ... ")
            print("=" * 80)

        time.sleep(check_interval)

if __name__ == "__main__":
    main_loop()