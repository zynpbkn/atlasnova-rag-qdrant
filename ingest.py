import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# ==================================================
# ENV
# ==================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY bulunamadı. .env dosyanı kontrol et."
    )


# ==================================================
# AYARLAR
# ==================================================

PDF_PATH = "./data/AtlasNova_Calisan_El_Kitabi.pdf"   
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "atlasnova_hr"

# ==================================================
# 1. PDF OKU
# ==================================================

print("📄 PDF okunuyor...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ {len(documents)} sayfa okundu.")


# ==================================================
# 2. CHUNK'LARA AYIR
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

print(f"✂️ {len(chunks)} chunk oluşturuldu.")


# ==================================================
# 3. METADATA EKLE
# ==================================================

for chunk in chunks:

    page = chunk.metadata.get("page", 0)

    chunk.metadata.update(
        {
            "page_number": page + 1,
            "company": "AtlasNova Teknoloji ve Danışmanlık Ltd. Şti.",
            "document_type": "employee_handbook",
        }
    )


# ==================================================
# 4. EMBEDDING MODELİ
# ==================================================

print("🧠 Embedding modeli hazırlanıyor...")

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

# ==================================================
# 5. QDRANT LOCAL MODE
# ==================================================

print("💾 Qdrant Local Mode oluşturuluyor...")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
)


# ==================================================
# TAMAMLANDI
# ==================================================

print("\n" + "=" * 60)
print("✅ INGESTION TAMAMLANDI")
print("=" * 60)

print(f"📄 Sayfa sayısı : {len(documents)}")
print(f"✂️ Chunk sayısı  : {len(chunks)}")
print(f"📦 Qdrant       : {QDRANT_PATH}")
print(f"📚 Collection   : {COLLECTION_NAME}")
   
#print(type(documents))
#print(type(documents[15]))
#print(documents[15].page_content)