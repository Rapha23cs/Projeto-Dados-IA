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

# Copiar todo o código fonte para dentro do container
# (O .dockerignore vai impedir que arquivos pesados/db entrem aqui)
COPY . .

# Expor a porta 8000 para a API (FastAPI)
EXPOSE 8000

# Comando padrão ao rodar o container (Inicializar a API do FastAPI)
CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
