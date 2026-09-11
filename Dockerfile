# Usar uma imagem oficial e leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instalar dependências do sistema necessárias para algumas bibliotecas de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas os requirements primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Instalar as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código fonte do backend para dentro do container
COPY . .

# Dar permissão de leitura para todos os usuários (Hugging Face roda como user 1000)
RUN chmod -R 777 /app

# Expor a porta 8000 (Fallback)
EXPOSE 8000

# Comando padrão ao rodar o container (Lê a porta dinâmica do provedor de nuvem, senão usa 8000)
CMD ["sh", "-c", "uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
