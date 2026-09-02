import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def train_model():
    print("Iniciando Fase 3: Treinamento do Modelo IA...")
    
    # 1. Carregar Dados Limpos
    processed_path = Path("data/processed/telco_churn_clean.csv")
    if not processed_path.exists():
        print(f"Erro: Arquivo {processed_path} não encontrado. Execute a Fase 2 primeiro.")
        return
        
    df = pd.read_csv(processed_path)
    print(f"Dados carregados. Shape: {df.shape}")
    
    # 2. Separar Features (X) e Target (y)
    if "Churn" not in df.columns:
        print("Erro: Coluna 'Churn' não encontrada.")
        return
        
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    
    # 3. One-Hot Encoding das variáveis categóricas
    # Variáveis do tipo 'object' ou 'category' serão transformadas em colunas numéricas (0 ou 1)
    categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    print(f"Aplicando One-Hot Encoding nas colunas: {categorical_cols}")
    
    # drop_first=True evita a armadilha de multicolinearidade
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Shape do X após encoding: {X_encoded.shape}")
    
    # 4. Dividir em Treino e Teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Divisão de dados: {X_train.shape[0]} amostras de treino e {X_test.shape[0]} de teste.")
    
    # 5. Treinar o modelo XGBoost
    print("Treinando o XGBClassifier...")
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    
    # 6. Avaliação
    print("Avaliando o modelo no conjunto de teste...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # 7. Salvar Modelo e as colunas utilizadas (para a API saber a estrutura exata)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "xgb_churn_model.pkl"
    features_path = models_dir / "model_features.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(X_encoded.columns.tolist(), features_path)
    
    print(f"Modelo salvo em: {model_path}")
    print(f"Lista de features salva em: {features_path}")
    print("Fase 3 concluída com sucesso!")

if __name__ == "__main__":
    # Garante que o script está sendo rodado a partir da raiz do projeto
    if not Path("data").exists():
        print("AVISO: Execute este script a partir do diretório raiz do projeto.")
    else:
        train_model()
