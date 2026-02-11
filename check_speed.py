import time
import torch
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table

# --- Настройки ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "solar_system_rag"
EM_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen2.5:14b"
device = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()

# Тестовые вопросы разной сложности
test_queries = [
    "Какая температура на поверхности Венеры?", # Простой факт
    "Сравни атмосферу Земли и Марса.",           # Сравнение (нужно много контекста)
    "Расскажи про Большое Красное Пятно."         # Специфический термин
]

# 1. Загрузка моделей
console.print(f"[yellow]Загрузка моделей на {device}...[/yellow]")
embeddings = HuggingFaceEmbeddings(model_name=EM_MODEL, model_kwargs={'device': device})
llm = ChatOllama(model=LLM_MODEL, temperature=0, streaming=True)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings, collection_name=COLLECTION_NAME, url=QDRANT_URL
)

def run_benchmark(query, k_neighbors):
    stats = {}
    
    # --- ТЕСТ 1: Поиск в Qdrant (Retrieval) ---
    start_retrieval = time.perf_counter()
    docs = vectorstore.similarity_search_with_score(query, k=k_neighbors)
    stats['retrieval_time'] = time.perf_counter() - start_retrieval
    
    context = "\n\n".join([doc.page_content for doc, score in docs])
    
    # --- ТЕСТ 2: Генерация (Generation) ---
    prompt = f"Контекст: {context}\n\nВопрос: {query}\nОтвет:"
    
    tokens_count = 0
    start_gen = time.perf_counter()
    ttft = None # Time To First Token
    
    # Стримим для замера TTFT
    for chunk in llm.stream(prompt):
        if ttft is None:
            ttft = time.perf_counter() - start_gen
        tokens_count += 1
    
    total_gen_time = time.perf_counter() - start_gen
    
    stats['ttft'] = ttft
    stats['total_time'] = total_gen_time
    stats['tps'] = tokens_count / total_gen_time if total_gen_time > 0 else 0
    stats['k'] = k_neighbors
    
    return stats

# --- Запуск тестов ---
results = []
ks = [1, 3, 5] # Сравним разное количество контекста

console.print("\n[bold green]Начинаю тестирование...[/bold green]\n")

for q in test_queries:
    for k in ks:
        res = run_benchmark(q, k)
        res['query'] = q[:30] + "..."
        results.append(res)

# --- Вывод таблицы ---
table = Table(title="Результаты Benchmark (Qwen 2.5 14B on CPU)")
table.add_column("Вопрос", style="cyan")
table.add_column("k", style="magenta")
table.add_column("Поиск (сек)", style="green")
table.add_column("TTFT (1-й токен)", style="yellow")
table.add_column("Токены/сек", style="blue")

for r in results:
    table.add_row(
        r['query'], 
        str(r['k']), 
        f"{r['retrieval_time']:.3f}", 
        f"{r['ttft']:.2f}", 
        f"{r['tps']:.2f}"
    )

console.print(table)