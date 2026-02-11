Создаем сеть
docker network create rag-network

создаем контейнер с оламой и моделью
docker build -f Dockerfile.ollama -t имя_образа .

Запускаем контейнер (скрпит entrypoint.sh скачает qwen2.5)

docker run -d \
  --name ollama \
  --network rag-network \
  -p 11434:11434 \
  -v ollama_models:/root/.ollama \
  --restart unless-stopped
 имя_образа

создаем контейнер с базой
docker build -f Dockerfile.qdrant -t имя_образа .

запускаем контейнер
docker run -d \
  --name qdrant \
  --network rag-network \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  michaelbokov/qdrant:latest

  Генерим документы (в папку docs) docs_generator.py 

  Индексация в базу indexing.py

  Чат бот chat.py



