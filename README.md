Süper, çıktı sayılarını da tam olarak seninkiyle eşitleyelim. `README.md` içindeki örnek ingest çıktısını ve sayfa/chunk sayılarını seninkiyle (20 sayfa, 75 chunk) güncelledim.

Bu yeni içeriği VS Code'daki `README.md` dosyasının içine yapıştırıp kaydedebilirsin:

```markdown
# AtlasNova HR – Agentic RAG Assistant

AtlasNova Teknoloji ve Danışmanlık Ltd. Şti. için geliştirilmiş, çalışanların İnsan Kaynakları politikaları ve şirket uygulamaları hakkında doğal dilde soru sorabildiği bir Agentic RAG (Retrieval-Augmented Generation) uygulamasıdır.

Uygulama, şirketin Çalışan El Kitabı içerisindeki bilgileri yerel vektör veritabanında saklar. Kullanıcıdan gelen soruları LangChain Agent aracılığıyla değerlendirir, `search_knowledge_base` aracı üzerinden bilgi tabanında arama yapar ve kaynak sayfa numaralarına dayalı doğru yanıtlar üretir.

## Proje Özellikleri

- **PDF Document Loading:** Çalışan El Kitabı'nın PDF formatında (`PyPDFLoader`) yüklenmesi.
- **Text Chunking:** `RecursiveCharacterTextSplitter` ile dokümanın anlamlı parçalara ayrılması (`chunk_size=900`, `chunk_overlap=150`).
- **Metadata Enrichment:** Her parçaya sayfa numarası (`page_number`), şirket adı (`company`) ve doküman türü (`document_type`) bilgilerinin dinamik eklenmesi.
- **Embedding:** OpenRouter altyapısı üzerinden `nvidia/nemotron-3-embed-1b:free` modeli ile metinlerin vektörlere dönüştürülmesi.
- **Vector Database:** `Qdrant Local Mode` ile vektörlerin yerel ortamda (`./qdrant_db`) saklanması (`atlasnova_hr` koleksiyonu).
- **Semantic Search:** Kullanıcı sorularına benzer içeriklerin $k=5$ parametresi ile vektör benzerliği üzerinden getirilmesi.
- **Agentic RAG:** `create_agent` ile oluşturulan LangChain agent'ının `@tool` dekoratörüyle tanımlanmış `search_knowledge_base` aracını kullanarak bilgi tabanını sorgulaması.
- **Conversation Memory:** LangGraph `InMemorySaver` ve sabit `thread_id` konfigürasyonu ile aynı oturum içerisindeki takip sorularının ve bağlamın hatırlanması.
- **Source Citation:** Yanıtların sonuna ilgili dokümanın sayfa numaralarının eklendiği kaynak gösterimi formatı.
- **LLM:** OpenRouter üzerinden `ChatOpenRouter` ile erişilen dil modelleri (`openrouter/free` veya `google/gemma-2-9b-it:free`).

## Teknolojiler

- Python
- uv
- LangChain (`langchain`, `langchain-community`, `langchain-core`, `langchain-openai`, `langchain-openrouter`, `langchain-qdrant`, `langchain-text-splitters`)
- LangGraph (`langgraph`, `InMemorySaver`)
- Qdrant Local Mode (`qdrant-client`)
- PyPDF (`pypdf`)
- python-dotenv

## Proje Yapısı

```text
atlasnova-rag-qdrant/
│
├── data/
│   └── AtlasNova_Calisan_El_Kitabi.pdf
│
├── qdrant_db/
│
├── .env
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── ingest.py
└── rag.py

```

> **Not:** `.env`, PDF dosyası ve `qdrant_db/` veritabanı klasörü `.gitignore` içerisinde tutulur ve GitHub'a gönderilmez.

## Kurulum

1. **Projeyi klonlayın:**
```bash
git clone <repository-url>
cd atlasnova-rag-qdrant

```


2. **Bağımlılıkları yükleyin:**
```bash
uv sync
# veya
uv pip install -r requirements.txt

```


3. **Sanal ortamı aktif edin:**
* Linux / WSL / macOS:
```bash
source .venv/bin/activate

```


* Windows (PowerShell):
```powershell
.venv\Scripts\activate

```




4. **Ortam değişkenlerini tanımlayın:**
Proje ana dizininde `.env` dosyası oluşturun ve OpenRouter API anahtarınızı ekleyin:
```env
OPENROUTER_API_KEY=your_openrouter_api_key

```


5. **PDF dosyasını yerleştirin:**
Çalışan El Kitabı PDF dosyasını `data/` klasörüne ekleyin. Beklenen dosya yolu:
```text
data/AtlasNova_Calisan_El_Kitabi.pdf

```



## Model Yapılandırması

### LLM – OpenRouter

Agent'ın yanıt üretmesi için `ChatOpenRouter` kullanılır:

```python
model = ChatOpenRouter(
    model="openrouter/free",  # Veya "google/gemma-2-9b-it:free"
    openrouter_api_key=OPENROUTER_API_KEY,
    temperature=0.2
)

```

### Embedding – OpenRouter (Nemotron)

PDF dokümanlarından oluşturulan metin parçalarını vektörlere dönüştürmek için `OpenAIEmbeddings` arayüzü ile OpenRouter üzerinden `nvidia/nemotron-3-embed-1b:free` modeli kullanılır:

```python
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
    check_embedding_ctx_length=False,
)

```

## Kullanım

### 1. Dokümanları Yükleyin (Ingestion)

PDF içeriğini okuyup vektörleştirerek yerel Qdrant veritabanına kaydetmek için:

```bash
python ingest.py

```

**İşlem Adımları:**

1. `data/AtlasNova_Calisan_El_Kitabi.pdf` dosyasını okur.
2. Metinleri `chunk_size=900` ve `chunk_overlap=150` ile parçalara ayırır.
3. Her parçaya sayfa numarası ve şirket metadatalarını ekler.
4. OpenRouter `nvidia/nemotron-3-embed-1b:free` embedding modeliyle vektörleri oluşturur.
5. Vektörleri yerel ortama (`./qdrant_db`) `atlasnova_hr` koleksiyonu olarak kaydeder.

**Örnek Ingestion Çıktısı:**

```text
📄 PDF okunuyor...
✅ 20 sayfa okundu.
✂️ 75 chunk oluşturuldu.
🧠 Embedding modeli hazırlanıyor...
💾 Qdrant Local Mode oluşturuluyor...

============================================================
✅ INGESTION TAMAMLANDI
============================================================
📄 Sayfa sayısı : 20
✂️ Chunk sayısı  : 75
📦 Qdrant       : ./qdrant_db
📚 Collection   : atlasnova_hr

```

### 2. HR Agent'ı Çalıştırın

Konsol arayüzü üzerinden soru sormak için:

```bash
python rag.py

```

**Örnek Konuşma Senaryosu:**

* **Siz:** *7 yıldır AtlasNova'da çalışıyorum. Kaç gün yıllık iznim var?*
* **AtlasNova HR:** *7 yıl kıdemi olan çalışanlar için yıllık izin hakkı 20 gündür.\nKaynak: Çalışan El Kitabı, Sayfa 12*
* **Siz:** *Geçen yıldan 4 günüm daha kalmıştı, toplam kaç gün olur?*
* **AtlasNova HR:** *Önceki konuşmamıza istinaden bu yılki 20 günlük hakkınız ve geçen yıldan devreden 4 gününüzle birlikte toplam 24 gün izin kullanabilirsiniz.\nKaynak: Çalışan El Kitabı, Sayfa 12*

Uygulamadan çıkmak için `exit` veya `quit` yazabilirsiniz.

## Mimari Akış

### Ingestion Pipeline (`ingest.py`)

```text
Employee Handbook (PDF)
          |
          v
     PyPDFLoader
          |
          v
 RecursiveTextSplitter (900 / 150)
          |
          v
 Metadata Enrichment (page_number, company, document_type)
          |
          v
 OpenAIEmbeddings (nvidia/nemotron-3-embed-1b:free via OpenRouter)
          |
          v
   Qdrant VectorStore (Local Mode: ./qdrant_db)

```

### Agentic RAG Pipeline (`rag.py`)

```text
     User Question
           |
           v
    LangChain Agent (create_agent)
           |
           v
 search_knowledge_base (Tool)
           |
           v
 Qdrant Retriever (k=5)
           |
           v
   Retrieved Chunks & Page Metadata
           |
           v
 ChatOpenRouter LLM (openrouter/free) + System Prompt
           |
           v
 Contextual Answer with Page Citation
           |
           v
 LangGraph Memory (InMemorySaver with thread_id)

```

## Önemli Notlar

* **Qdrant Local Mode:** `./qdrant_db` klasöründe yerel olarak çalışır; harici bir Docker konteyneri veya Qdrant sunucusu gerektirmez.
* **Yeniden Yükleme:** PDF dosyası değiştiğinde veya yenilendiğinde `python ingest.py` komutu tekrar çalıştırılmalıdır.
* **Hafıza (Memory):** `InMemorySaver` oturum bazlı çalışır; `rag.py` sonlandırıldığında hafıza sıfırlanır.
* **Güvenlik:** API anahtarları sadece `.env` dosyasında tutulmalı, kesinlikle Git deposuna eklenmemelidir.

```

Yapıştırdıktan sonra terminalden aşağıdaki komutları çalıştırarak güncellemeyi GitHub'a gönderebilirsin:

```bash
git add README.md
git commit -m "docs: update README with exact ingestion metrics"
git push origin main

```