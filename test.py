import openai
import fitz
import numpy as np
import faiss
from langchain.text_splitter import CharacterTextSplitter
import os

# Setup OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
# Step 1: Document Parsing (PyMuPDF)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Text Chunking (LangChain)
def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Step 3: Generate Embeddings (OpenAI Ada-002)
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",  # The embedding model
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    print("Embeddings done")
    return embeddings

# Step 4: Create FAISS Index
def create_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    print("FAISS Index created")
    return index


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Create client instance

# Step 5: Extract Structured Requirements
def extract_requirements(text_chunks):
    prompt = f"""
    Extrahiere aus den folgenden Textabschnitten technische Anforderungen in strukturiertem Format.
    Antworte NUR mit den Anforderungen und lass irrelevante Informationen weg.
    
    Hier sind die Textabschnitte:
    {' '.join(text_chunks)}
    
    Gib die Anforderungen in einer Liste aus:
    """

    response = client.chat.completions.create( 
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Du bist ein Experte für technische Dokumente."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# Step 6: Query the Index
def query_index(query, index, text_chunks):
    response = openai.embeddings.create(
        model="text-embedding-ada-002", input=query
    )
    query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, k=5)  # Retrieve more chunks to increase accuracy
    retrieved_chunks = [text_chunks[i] for i in I[0]]
    
    structured_requirements = extract_requirements(retrieved_chunks)
    
    return structured_requirements

# Example Usage
pdf_text = extract_text_from_pdf("/Users/youssef/Desktop/Audi/Lastenheft_Audi.pdf")
print("Extraction done")
chunks = chunk_text(pdf_text)
embeddings = generate_embeddings(chunks)
index = create_faiss_index(embeddings)

# Example Query
query = "Extrahiere 15 technische Anforderungen"
requirements = query_index(query, index, chunks)
print(requirements)
