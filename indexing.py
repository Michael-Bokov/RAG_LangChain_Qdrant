
#indexing.py
import torch
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams



# 1. Настройки подключения
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
emb_model = "BAAI/bge-m3"
COLLECTION_NAME = "solar_system_rag"

# 2. Загружаем документы из папки /docs
print("Загрузка документов...")
loader = DirectoryLoader('./docs', glob="./*.md", loader_cls=UnstructuredMarkdownLoader)
docs = loader.load()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Разбиваем текст на части (Chunks)
# Это важно: эмбеддеры плохо работают с огромными текстами. 
# Мы режем их на куски по 1000 символов с перекрытием 100 для сохранения контекста.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"Подготовлено {len(splits)} фрагментов текста.")

# 4. Создаем эмбеддер через Ollama
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model, 
    #base_url=OLLAMA_URL,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
# Лишние так как библиотека маленькая
client = QdrantClient(url=QDRANT_URL)
# Пересоздаем коллекцию 
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ Удалена старая коллекция {COLLECTION_NAME}")

print(f"🛠 Создание коллекции с HNSW индексом...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    # Настройка HNSW 
    hnsw_config=models.HnswConfigDiff(
        m=16,               # Количество связей у точки 
        ef_construct=100    # Число кандидатов при построении графа
    )
)# Для миллионов документов нужно:
#    Экономить RAM: Включить On-disk storage или Quantization (сжатие векторов) для RAM.
# 5. Инициализируем Qdrant и загружаем векторы
print("Индексация в Qdrant... (это может занять время на CPU)")
qdrant = QdrantVectorStore.from_documents(
    splits,
    embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True  # Пересоздать коллекцию при запуске
)

print(f"Успех! Данные в базе Qdrant. Коллекция: {COLLECTION_NAME}")