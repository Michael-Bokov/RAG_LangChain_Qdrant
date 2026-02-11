import ollama
import os

# Список тем
topics = [
    "Солнце", "Меркурий", "Венера", "Земля", "Марс", 
    "Юпитер", "Сатурн", "Уран", "Нептун", "Плутон", 
    "Астероид", "Комета"
]

# Создаем папку, если ее нет
output_dir = "./docs"
os.makedirs(output_dir, exist_ok=True)
client = ollama.Client(host="http://localhost:11434") # или http://ollama:11434
def generate_topic(topic):
    print(f"Генерирую данные о: {topic}...")
    
    prompt = f"""
    Напиши подробную статью для энциклопедии о небесном теле: {topic}.
    Статья должна быть объемом около 1000-1500 слов (примерно 2-3 страницы текста).
    Используй формат Markdown. 
    Обязательно включи разделы:
    1. Общее описание и физические характеристики.
    2. История открытия и исследований.
    3. Атмосфера и геология (если применимо).
    4. Интересные и малоизвестные факты.
    Пиши научным, но доступным языком.
    """

    response = client.generate(
        model='qwen2.5:14b', # Ваша квантованная модель
        prompt=prompt,
        options={
            "num_ctx": 4096, # Увеличим контекст для длинного текста
            "temperature": 0.7
        }
    )
    return response['response']

for t in topics:
    file_path = os.path.join(output_dir, f"{t.lower()}.md")
    
    # Проверка, чтобы не генерировать заново, если файл есть
    if os.path.exists(file_path):
        print(f"Файл {t} уже существует, пропускаю.")
        continue
        
    content = generate_topic(t)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Готово: {file_path}")

print("\nВсе документы успешно созданы в папке /docs!")