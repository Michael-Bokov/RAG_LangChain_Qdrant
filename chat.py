import torch
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from rich.console import Console

from rich.live import Live
from rich.markdown import Markdown

# 1. Настройки
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "solar_system_rag"
EMB_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen2.5:14b" # Твоя модель в Ollama
device = "cuda" if torch.cuda.is_available() else "cpu"

console = Console()

# 2. Инициализация эмбеддера (того же, что при индексации!)
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={'device': device}
)

# 3. Подключение к существующей базе Qdrant
#from qdrant_client.http import models as rest

# filter_condition = rest.Filter(
#     must=[
#         rest.FieldCondition(
#             key="metadata.year",
#             match=rest.MatchValue(value=2026)
#         )
#     ]
#)
# filtered_results = vectorstore.similarity_search_with_score(
#     query, 
#     k=3, 
#     filter=filter_condition
# )

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL
)
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 3,
#         "filter": filter_condition 
#     }
# )
# --- ГИБРИД ---
console.print("[yellow]Подготовка гибридного поиска (BM25 + Vector)...[/yellow]")
# 1. Получаем документы для текстового индекса (BM25)
# Мы берем первые 10000 (всего 82, так что подтянет всё)
all_docs = vectorstore.similarity_search("", k=10000) 

# 2. Создаем текстовый ретривер
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 3

vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, # Берем 3 лучших куска
                   "score_threshold": 0.5 # Степень уверенности
                   }) 
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], 
    weights=[0.6, 0.4]
)

# 4. Настройка LLM (Ollama)
llm = ChatOllama(model=LLM_MODEL, 
                 base_url="http://localhost:11434", 
                 temperature=0.3,
                 streaming=True ) # Включаем поток

# 5. Создание Промпта (Инструкция для Qwen)
template = """Вы — помощник-астроном. Используйте только предоставленный контекст для ответа на вопрос. 
Если в контексте нет ответа, скажите, что вы не знаете, не пытайтесь выдумывать.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. Сборка цепочки (LangChain Expression Language - LCEL)
def format_docs(docs):
    # Собираем текст и сохраняет метаданные для вывода источников
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Цикл чата
console.print("[bold green]Система RAG по Солнечной системе готова! Задавайте вопросы.[/bold green]")

while True:
    query = input("\nВы: ")
    if query.lower() in ["exit", "quit", "выход"]: break

    # Поиск источников отдельно для вывода пользователю
    source_docs = retriever.invoke(query)

    # 2. Проверяем, нашел ли поиск хоть что-то выше порога 0.5
    if not source_docs:
        console.print("[red]Извините, в моих документах нет надежной информации по этому вопросу.[/red]")
        continue

    sources = set([doc.metadata.get("source", "Unknown") for doc in source_docs])

    console.print(f"[yellow]Нашел совпадения в: {', '.join(sources)}...[/yellow]")

    # Генерация ответа
    #response = rag_chain.invoke(query)
    
    console.print("\n[bold blue]Ответ ИИ:[/bold blue]")

    full_response = ""
    # Создаем Live-контейнер для динамического обновления
    with Live(Markdown(""), console=console, refresh_per_second=4) as live:
        for chunk in rag_chain.stream(query):
            full_response += chunk
            # Каждую итерацию обновляем Markdown-объект внутри Live
            live.update(Markdown(full_response))
            
    print() 
    #console.print(Markdown(response))
