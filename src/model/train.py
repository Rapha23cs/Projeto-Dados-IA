import pandas as pd
import sqlite3
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def get_db_connection():
    db_path = Path("data/projeto_dados.db")
    return sqlite3.connect(db_path)

def create_features(df):
    """Engenharia de features (Novas colunas derivadas)"""
    df = df.copy()
    
    # 1. Total_Servicos_Contratados
    servicos = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    def count_services(row):
        count = 0
        for s in servicos:
            if str(row.get(s, 'No')) not in ['No', 'No internet service', 'No phone service']:
                count += 1
        return count
        
    df['Total_Servicos_Contratados'] = df.apply(count_services, axis=1)
    
    # 2. Gasto_Por_Mes_De_Vida (evitando divisão por zero)
    df['Gasto_Por_Mes_De_Vida'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 3. Custo por Serviço
    df['Custo_Por_Servico'] = df['MonthlyCharges'] / (df['Total_Servicos_Contratados'] + 1)
    
    # 4. Agrupamento de Safra (Tenure Group)
    def categorize_tenure(t):
        if t <= 12: return 'Novato'
        elif t <= 48: return 'Estavel'
        else: return 'Leal'
    df['Tenure_Group'] = df['tenure'].apply(categorize_tenure)
    
    return df

def train_model():
    print("Iniciando treinamento da IA com Otimização Extrema (XAI)...")
    
    # 1. Carregar dados do banco de dados
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM processed_telco_churn", conn)
    conn.close()
    
    print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    # 2. Engenharia de Features Avançada
    df = create_features(df)
    
    # 3. Separar X e Y
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 4. One-Hot Encoding
    categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    print(f"Aplicando One-Hot Encoding nas colunas: {categorical_cols}")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, dtype=int)
    
    # Salvar a estrutura de features ANTES do treino, para a API usar
    features = X_encoded.columns.tolist()
    features_path = Path("models/model_features.pkl")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features, features_path)
    
    # 5. Dividir em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Cálculo de Pesos para Desbalanceamento (Substituindo SMOTE)
    pos_weight = sum(y_train == 0) / sum(y_train == 1)
    print(f"Balanço de classes calculado (scale_pos_weight): {pos_weight:.2f}")
    
    # 7. Modelagem e Hyperparameter Tuning com XGBoost
    print("Iniciando GridSearchCV para hiperparâmetros do XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=pos_weight)
    
    # Grade expandida para melhor generalização
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    melhor_modelo = grid_search.best_estimator_
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    # 8. Threshold Tuning (Otimização do Ponto de Corte)
    from sklearn.metrics import precision_recall_curve
    y_pred_proba = melhor_modelo.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    fscore = (2 * precisions * recalls) / (precisions + recalls)
    fscore = [0 if pd.isna(x) else x for x in fscore] # trata NaNs se houver
    
    ix = fscore.index(max(fscore))
    best_threshold = thresholds[ix]
    print(f"Melhor Threshold (Corte) Encontrado: {best_threshold:.4f} com F1-Score: {fscore[ix]:.4f}")
    
    # Avaliação com o novo Threshold
    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_opt)
    
    print("\n=== Resultados da Otimização ===")
    print(f"Acurácia no Teste (com corte customizado): {acc*100:.2f}%")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred_opt))
    
    # 9. Serializar o modelo final e o threshold
    model_path = Path("models/xgb_churn_model.pkl")
    joblib.dump(melhor_modelo, model_path)
    
    with open("models/optimal_threshold.txt", "w") as f:
        f.write(str(best_threshold))
        
    print(f"Modelo e Limite Ótimo salvos com sucesso em: {model_path.parent}")

if __name__ == "__main__":
    if not Path("data/projeto_dados.db").exists():
        print("AVISO: Execute os scripts da Fase 1 e 2 antes do treinamento.")
    else:
        train_model()
