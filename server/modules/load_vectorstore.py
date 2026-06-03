import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-assistant"

os.environ["GOOGLE_API_KEY"]= GOOGLE_API_KEY

upload_dir = "./uploaded_docs"
os.makedirs(upload_dir, exist_ok=True)

# instantiate Pinecone client 
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072, 
        metric="cosine",
        spec=spec
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# load, split, embed and upsert documents
def load_vectorstore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    file_paths = []

    # upload 
    for file in uploaded_files:
        save_path = Path(upload_dir)/file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(save_path)

    # split
    for file_path in tqdm(file_paths, desc="Processing documents"):
        loader=PyPDFLoader(str(file_path))
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = text_splitter.split_documents(documents)


        # getting the text, metadata and ids for printing the source of info in the documents later
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [ f"{Path(file_path).stem} - {i}" for i in range(len(chunks))]

        print(f"Embedding chunks for {file_paths}...")

        # embedding 
        embedding = embed_model.embed_documents(texts)

        # upsert to pinecone 
        print(f"Upserting chunks for {file_paths} to Pinecone...")
        with tqdm(total = len(chunks), desc="Upserting to Pinecone") as pbar:
            for i in range(0, len(chunks), 100):
                batch_ids = ids[i:i+100]
                batch_embeddings = embedding[i:i+100]
                batch_metadata = metadata[i:i+100]

                index.upsert(vectors=zip(batch_ids, batch_embeddings, batch_metadata))
                pbar.update(len(batch_ids))

        print(f"Upload complete for {file_paths}")
