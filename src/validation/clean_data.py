import os
import pandas as pd
import great_expectations as ge
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path

# Carrega as variáveis de ambiente
load_dotenv()

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return create_engine(database_url)
    else:
        sqlite_path = Path("data/projeto_dados.db").resolve()
        return create_engine(f"sqlite:///{sqlite_path}")

def clean_and_validate_data():
    print("Iniciando Fase 2: Processamento e Validação...")
    
    # 1. Leitura dos dados da tabela bruta
    engine = get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM raw_telco_churn", engine)
        print(f"Dados brutos lidos do banco. Shape: {df.shape}")
    except Exception as e:
        print(f"Erro ao ler do banco de dados: {e}")
        return
        
    # 2. Processamento (Pandas)
    print("Aplicando regras de limpeza...")
    
    # Remove coluna inútil
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print(" - Coluna 'customerID' removida.")
        
    # Tratamento de TotalCharges (substitui espaços vazios por 0.0)
    if "TotalCharges" in df.columns:
        # errors='coerce' converte os espaços em NaN
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        print(" - Coluna 'TotalCharges' convertida para numérico (nulos = 0.0).")
        
    # Binarização da coluna alvo (Churn)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        print(" - Coluna 'Churn' convertida para 1 (Yes) e 0 (No).")
        
    # 3. Validação (Great Expectations)
    print("Executando regras de validação (Great Expectations)...")
    # Converte o DataFrame pandas para um Dataset do Great Expectations
    gdf = ge.from_pandas(df)
    
    # Regra 1: Churn só pode ser 0 ou 1
    res_churn = gdf.expect_column_values_to_be_in_set("Churn", [0, 1])
    assert res_churn["success"], f"Falha na validação: Valores inesperados em Churn. Detalhes: {res_churn}"
    
    # Regra 2: TotalCharges não pode ser negativo
    res_charges = gdf.expect_column_values_to_be_between("TotalCharges", min_value=0.0)
    assert res_charges["success"], "Falha na validação: TotalCharges menor que zero encontrado."
    
    # Regra 3: Nenhuma coluna pode ter valores nulos (já tratamos)
    for col in df.columns:
        res_null = gdf.expect_column_values_to_not_be_null(col)
        assert res_null["success"], f"Falha na validação: Valores nulos encontrados na coluna {col}."
        
    print("Sucesso! Todos os dados passaram nos testes de qualidade.")
    
    # 4. Armazenamento Processado
    processed_path = Path("data/processed/telco_churn_clean.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Dados limpos salvos em: {processed_path}")
    
    # Salvar no banco
    table_name = "processed_telco_churn"
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    print(f"Dados processados inseridos na tabela '{table_name}'.")
    
if __name__ == "__main__":
    if not Path("data").exists():
        print("AVISO: Execute este script a partir do diretório raiz do projeto.")
    else:
        clean_and_validate_data()
