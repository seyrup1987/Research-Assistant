import os
import logging
import base64
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from io import BytesIO
from tqdm import tqdm
import threading
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.schema import HumanMessage
from langchain_experimental.text_splitter import SemanticChunker
from models import Models
from PIL import Image
from uuid import uuid4

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_PATH = os.path.join("output", "temp")
DATA_PATH = os.path.join('..', 'Data')
CHECK_INTERVAL = 10
PROMPT_PATH = os.path.join('..', 'config', 'prompt4propositions.txt')
IMAGE_PROMPT_PATH = os.path.join('..', 'config', 'prompt4ImageSumNProp.txt')
TABLE_PROMPT_PATH = os.path.join('..', 'config', 'prompt4TableSumNProp.txt')

# Initialize Models
embeddings = Models().embeddings

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db"
)

text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type = "percentile", #standard_deviation, interquartile
    )

# Graceful Shutdown Handling
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    logging.warning("Interrupt received! Shutting down gracefully...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def summerizeImages(images):
    """
    Summarize a list of Base64-encoded images using a multimodal LLM.
    Includes retry logic for API failures and limits concurrent requests to optimize server load.
    
    Args:
        images (list): List of Base64-encoded image strings.
    
    Returns:
        list: List of summaries or error messages for each image.
    """
    if not images:
        logging.info("No images provided for summarization.")
        return []

    model = Models().multimodal_llm
    summaries = []
    max_retries = 3  # Number of retry attempts for failed API calls
    max_workers = 5  # Limit concurrent API calls to prevent server overload

    def process_image(img_b64):
        """
        Process a single Base64-encoded image and return its summary.
        
        Args:
            img_b64 (str): Base64-encoded image string.
        
        Returns:
            str: Summary of the image or an error message.
        """
        for attempt in range(max_retries):
            try:
                if interrupted:
                    logging.warning("Stopping text extraction due to interrupt.")
                    break
                
                propositions = []
                with open(IMAGE_PROMPT_PATH, 'r') as promptFile:
                    prompt = promptFile.read()
                
                logging.info("Extracting Propositions from Image Summary . . .")


                # Decode Base64 to image
                img_data = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_data))

                messages = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    )
                ]

                # Invoke the model
                response = model.invoke(messages)
                del response.response_metadata['message']
                logging.debug(f"Successfully summarized image: {response.content.strip()}")
                print("DEBUG LOG | Response: ", response.content.strip())
                return Document(
                    page_content = response.content.strip(),
                    metadata = response.response_metadata
                    )

            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Failed to summarize image after {max_retries} attempts: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
                else:
                    logging.warning(
                        f"Image summarization failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying..."
                    )
                    time.sleep(1)  # Wait 1 second before retrying

    # Use ThreadPoolExecutor to process images concurrently with a limit on workers
    logging.info(f"Starting summarization of {len(images)} images with {max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        summaries = list(executor.map(process_image, images))

    logging.info(f"Completed summarization of {len(images)} images.")
    return summaries

def summerizeTables(tables):
    if not tables:
        logging.info("No tables provided for summarization.")
        return []

    model = Models().multimodal_llm
    summaries = []
    max_retries = 3  # Number of retry attempts for failed API calls
    max_workers = 5  # Limit concurrent API calls to prevent server overload

    def process_table(table):
        for attempt in range(max_retries):
            try:
                if interrupted:
                    logging.warning("Stopping text extraction due to interrupt.")
                    break

                propositions = []
                with open(TABLE_PROMPT_PATH, 'r') as promptFile:
                    prompt = promptFile.read()
                
                logging.info("Extracting Proposition from Table SUmmary . . .")
                full_prompt = f"{prompt}{table}"
                messages = [
                    HumanMessage(
                        content=full_prompt
                    )
                ]

                # Invoke the model
                response = model.invoke(messages)
                logging.debug(f"Successfully summarized table: {response.content.strip()}")
                print("DEBUG LOG | Response: ", response.content.strip())
                del response.response_metadata['message']
                return Document(
                    page_content = response.content.strip(),
                    metadata = response.response_metadata
                    )
            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Failed to summarize table after {max_retries} attempts: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
                else:
                    logging.warning(
                        f"Table summarization failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying..."
                    )
                    time.sleep(1)  # Wait 1 second before retrying
    # Use ThreadPoolExecutor to process images concurrently with a limit on workers
    logging.info(f"Starting summarization of {len(tables)} tables with {max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        summaries = list(executor.map(process_table, tables))
        logging.info(f"Summarized {len(summaries)} tables: {[doc.page_content[:50] for doc in summaries if isinstance(doc, Document)]}")
    
    
    logging.info(f"Completed summarization of {len(tables)} tables.")
    return summaries

# Function to Extract Images as Base64
def get_images_base64(chunks):
    images_b64 = []
    for chunk in tqdm(chunks, colour='CYAN', desc="Extracting images"):
        if interrupted:
            logging.warning("Stopping image extraction due to interrupt.")
            break
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def get_tables(chunks):
    tables = []
    for chunk in tqdm(chunks, colour='CYAN', desc="Extracting tables"):
        if interrupted:
            logging.warning("Stopping table extraction due to interrupt.")
            break
        if "Table" in str(type(chunk)):
            tables.append(chunk)
    return tables

def process_text(chunks):
    texts = []
    logging.info(f"Processing text chunks from PDF")
    for chunk in tqdm(chunks, colour='CYAN', desc="Extracting text"):
        if interrupted:
            logging.warning("Stopping text extraction due to interrupt.")
            break
        chunk_type = str(type(chunk))
        if any(t in chunk_type for t in ["CompositeElement", "Text", "NarrativeText", "Title"]):
            texts.append(str(chunk))
    
    complete_text = ' '.join(texts)
    logging.info(f"Extracted text: {complete_text[:500]}...")
    processed_chunks = text_splitter.split_text(complete_text)
    logging.info(f"Number of Text chunks: {len(processed_chunks)}")
    propositions = getPropositions4mText(processed_chunks)
    return propositions

def getPropositions4mText(chunks):
    try:
        model = Models().llm
        propositions = []
        with open(PROMPT_PATH, 'r') as promptFile:
            prompt = promptFile.read()
        
        logging.info("Extracting Proposition from Texk Blocks . . .")
        for chunk in tqdm(chunks, colour = 'RED'):
            full_prompt = f"{prompt}{chunk}"
            messages = [
                HumanMessage(
                    content=full_prompt
                )
            ]
            
            # Invoke the model
            response = model.invoke(messages)
            print("DEBUG LOG | Response: ", response.content.strip())
            # Handle the response content
            prop_lines = response.content.strip().split('\n')
            for line in prop_lines:
                # Remove numbering (e.g., "1. ") and create a Document
                prop_text = line.strip().split('. ', 1)[-1] if '. ' in line else line.strip()
                if prop_text:
                    # Safely handle metadata
                    metadata = response.response_metadata.copy()
                    metadata.pop('message', None)  # Remove 'message' if it exists, no error if it doesn’t
                    propositions.append(Document(
                        page_content=prop_text,
                        metadata=metadata
                    ))
        return propositions
    except Exception as error:
        logging.error(f"Error extracting propositions: {str(error)}")
        return []
    


# Main PDF Processing Function
def partitionAndSummerizePDF(file_path):
    summary = {}
    if not file_path.lower().endswith("pdf"):
        logging.warning(f'Skipping non-PDF file: {file_path}')
        return {}
    
    logging.info(f"Partitioning PDF {file_path}...")
    chunks = partition_pdf(
        filename=os.path.join(DATA_PATH, file_path),
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        chunking_strategy='by_title',
        max_character=1000,
        combine_text_under_n_chars=500,
        new_after_n_chars=6000,
        extract_images_in_pdf=True,  # Ensure images are extracted
        use_ocr=True,  # Enable OCR
        ocr_languages="eng"  # Specify language
    )

    images = get_images_base64(chunks)
    logging.info(f"Number of Images in {file_path} : {len(images)}")

    tables = get_tables(chunks)
    logging.info(f"Number of Tables in {file_path} : {len(tables)}")

    ingest_text = process_text(chunks)
    logging.info(f"Number of Proposition in {file_path} : {len(ingest_text)}")

    if not interrupted:
        logging.info(f"Summarizing Tables and Images for {file_path}")
        summary['images'] = summerizeImages(images)
        summary['tables'] = summerizeTables(tables)

        if summary['images']:
            logging.info(f"Storing Image Summaries in Vector Database for {file_path}")
            uuids_images = [str(uuid4()) for _ in range(len(summary['images']))]
            print(f"Ingesting {len(summary['images'])} chunks into vector store . . . ")
            vector_store.add_documents(documents = summary['images'], ids = uuids_images)

        if summary['tables']:
            logging.info(f"Storing Table Summaries in Vector Database for {file_path}")
            uuids_tables = [str(uuid4()) for _ in range(len(summary['tables']))]
            print(f"Ingesting {len(summary['tables'])} chunks into vector store . . . ")
            vector_store.add_documents(documents = summary['tables'], ids = uuids_tables)

        if ingest_text:
            logging.info(f"Storing Text Chunks in Vector Database for {file_path}")
            uuids_text = [str(uuid4()) for _ in range(len(ingest_text))]
            print(f"Ingesting {len(ingest_text)} chunks into vector store . . . ")
            vector_store.add_documents(documents = ingest_text, ids = uuids_text)



# Main Event Loop
def main_loop(stop_event):
    logging.info(f"Starting Document Ingestion Service | Monitoring folder: {DATA_PATH}")
    logging.info(f"Documents will checked every {CHECK_INTERVAL} seconds.")
    logging.info(f"Add PDF files to this folder to ingest them into the vector database")
    logging.info(f"Monitoring folder: {DATA_PATH} (Check interval: {CHECK_INTERVAL} seconds)")
    print("=" * 80)
    while not (interrupted or stop_event.is_set()):
        files = os.listdir(DATA_PATH)
        pdf_files = [f for f in files if f.endswith(".pdf") and not f.startswith("_")]
        pdf_count = len(pdf_files)

        if pdf_count > 0:
            print(f"Found {pdf_count} PDF Files in {DATA_PATH}")

            processed_count = 0
            for filename in pdf_files:
                file_path = os.path.join(DATA_PATH, filename)
                partitionAndSummerizePDF(file_path)
                new_filename = "_"+filename
                new_file_path = os.path.join(DATA_PATH, new_filename)
                os.rename(file_path, new_file_path)
                processed_count += 1

            logging.info(f"Processed {processed_count} PDF Files.")
            logging.info("=" * 80)
            logging.info(f"All Documentshave been processed successfully.")
            logging.info(f"Processed documents are prefixed with '_' and remain in the data folder.")
            logging.info(f"text check in {CHECK_INTERVAL} seconds ... ")
            print("=" * 80)

        else:
            print(f"[{time.strftime('%y-%m-%d %H:%M:%S')}] No new documents found. Waiting for Documents to be added to {DATA_PATH}")
            print(f"Add PDF files to {DATA_PATH} for procesing ... ")
            print("=" * 80)

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    stop_event = threading.Event()
    main_loop(stop_event)
