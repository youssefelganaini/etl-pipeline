import openai
import fitz
import numpy as np
import faiss
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Setup OpenAI API
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Extract Text from PDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Step 2: Extract TOC (Table of Contents) to get chapters


def extract_toc_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    structured_toc = {}
    for entry in toc:
        level, title, page = entry
        structured_toc[title] = page
    return structured_toc

# Step 3: Chunk the Text into Smaller Pieces


def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n", " "]
    )
    return text_splitter.split_text(text)

# Step 4: Generate Embeddings for Each Chunk


def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(
            model="text-embedding-ada-002", input=chunk)
        embeddings.append(response.data[0].embedding)
    return embeddings

# Step 5: Create FAISS Index


def create_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index

# Step 6: Query the Index and Extract Structured Requirements


def extract_requirements(text_chunks):
    prompt = f"""
    Extrahiere aus den folgenden Textabschnitten technische Anforderungen und ordne sie Kategorien zu.
    Nutze die folgenden Kategorien:
    - Geometrie & Abmessungen
    - Material & Qualität
    - Elektronik & Schnittstellen
    - Mechanische Anforderungen
    - Normen & Zertifizierungen
    Antworte im JSON-Format mit dieser Struktur:
    {{
        "requirements": [
            {{
                "category": "Geometrie & Abmessungen",
                "chapter": "Kapitelname",
                "requirement": "Breite: 367mm, Tiefe: 322mm, Höhe: 106mm"
            }}
        ]
    }}
    Textabschnitte:
    {' '.join(text_chunks)}
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Du bist ein Experte für technische Dokumente."},
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)

# Step 7: Run the Pipeline


def main(pdf_path):
    print("Extrahiere Text...")
    pdf_text = extract_text_from_pdf(pdf_path)

    print("Extrahiere Inhaltsverzeichnis...")
    toc = extract_toc_from_pdf(pdf_path)

    print("Teile das Dokument in Abschnitte...")
    chunks = chunk_text(pdf_text)

    print("Generiere Embeddings...")
    embeddings = generate_embeddings(chunks)

    print("Erstelle FAISS Index...")
    index = create_faiss_index(embeddings)

    print("Extrahiere Anforderungen...")
    structured_requirements = extract_requirements(chunks)

    print("Speichere Ergebnisse in JSON-Datei...")
    with open("extracted_requirements.json", "w", encoding="utf-8") as f:
        json.dump(structured_requirements, f, indent=4, ensure_ascii=False)

    print("✅ Anforderungen extrahiert und gespeichert!")


# Example Usage
pdf_path = "/Users/youssef/Desktop/Audi/Lastenheft_Audi.pdf"
main(pdf_path)
