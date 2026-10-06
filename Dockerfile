FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ragkit ragkit
COPY corpus corpus
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    RAG_MODEL=gpt-oss:20b
EXPOSE 8000
CMD ["uvicorn", "ragkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
