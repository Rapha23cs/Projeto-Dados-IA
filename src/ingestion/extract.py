import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def get_db_connection():
    """
    Cria a conexão com o banco de dados. 
    Usa PostgreSQL se disponível no .env, senão faz fallback para um SQLite local 
    na pasta data/ para garantir que rodará sem infraestrutura externa.
    """
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        print("Conectando ao banco PostgreSQL/Supabase configurado no .env...")
        engine = create_engine(database_url)
    else:
        print("Nenhum banco PostgreSQL (DATABASE_URL) configurado no .env.")
        # Fallback para SQLite local
        sqlite_path = Path("data/projeto_dados.db").resolve()
        print(f"Fazendo fallback para banco SQLite local em: {sqlite_path}")
        engine = create_engine(f"sqlite:///{sqlite_path}")
        
    return engine

def extract_data():
    """
    Função principal de extração de dados:
    1. Baixa o CSV remoto.
    2. Salva em data/raw/.
    3. Salva no banco de dados.
    """
    # URL pública do dataset Telco Customer Churn (IBM) no GitHub
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    print(f"Baixando dados do Telco Customer Churn de: {url}")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Erro ao baixar os dados: {e}")
        return None
        
    print(f"Dados baixados com sucesso! Formato: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    # 1. Salvar na pasta data/raw/
    raw_path = Path("data/raw/telco_churn_raw.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Dataset bruto salvo fisicamente em: {raw_path}")
    
    # 2. Inserir no Banco de Dados
    engine = get_db_connection()
    table_name = "raw_telco_churn"
    
    try:
        # if_exists="replace" vai recriar a tabela toda vez que o script rodar.
        # index=False evita que o índice do pandas vire uma coluna no banco.
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        print(f"Dados inseridos com sucesso na tabela '{table_name}' do banco de dados!")
    except Exception as e:
        print(f"Erro ao inserir no banco de dados: {e}")
        return None
    
    print("Processo de ingestão finalizado com sucesso.")
    return df

if __name__ == "__main__":
    # Garante que o script está sendo rodado a partir da raiz do projeto
    if not Path("data").exists():
        print("AVISO: Execute este script a partir do diretório raiz do projeto (onde está a pasta 'data').")
    else:
        extract_data()
