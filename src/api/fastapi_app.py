from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import warnings
import os
from supabase import create_client, Client
warnings.filterwarnings('ignore')

# Inicializando a API
app = FastAPI(
    title="Churn Prediction API",
    description="API de MLOps para prever probabilidade de cancelamento de clientes (Churn) via XGBoost",
    version="1.0.0"
)

# Configurando CORS para o Frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na vida real, restringiríamos aos domínios do site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializando Supabase (se as chaves estiverem configuradas no ambiente)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Conectado ao Supabase com sucesso!")
    except Exception as e:
        print(f"Aviso: Erro ao conectar no Supabase: {e}")

# Definindo o esquema de entrada de dados com Pydantic
class ClientData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Carregamento do modelo (cache global)
model_path = Path("models/xgb_churn_model.pkl")
features_path = Path("models/model_features.pkl")

if model_path.exists() and features_path.exists():
    modelo = joblib.load(model_path)
    features_treinamento = joblib.load(features_path)
else:
    modelo = None
    features_treinamento = None

# Tentar carregar o threshold otimizado, senão usa 0.5
threshold_path = Path("models/optimal_threshold.txt")
try:
    if threshold_path.exists():
        raw = threshold_path.read_text().strip()
        # Remove qualquer caracter não numérico (colchetes, espaços) que o numpy possa gerar
        import re
        nums = re.findall(r'[0-9]+\.?[0-9]*[eE]?[-+]?[0-9]*', raw)
        optimal_threshold = float(nums[0]) if nums else 0.5
    else:
        optimal_threshold = 0.5
except Exception:
    optimal_threshold = 0.5
print(f"Threshold carregado: {optimal_threshold}")

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as mesmas regras matemáticas do treino (Feature Engineering)"""
    df = df.copy()
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
    df['Gasto_Por_Mes_De_Vida'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['Custo_Por_Servico'] = df['MonthlyCharges'] / (df['Total_Servicos_Contratados'] + 1)
    
    def categorize_tenure(t):
        if t <= 12: return 'Novato'
        elif t <= 48: return 'Estavel'
        else: return 'Leal'
    df['Tenure_Group'] = df['tenure'].apply(categorize_tenure)
    
    return df

@app.post("/predict")
def predict_churn(client: ClientData):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo não encontrado no servidor.")
        
    try:
        # Converter os dados recebidos (JSON) para DataFrame do Pandas
        df_input = pd.DataFrame([client.model_dump()])
        
        # 1. Feature Engineering
        df_fe = apply_feature_engineering(df_input)
        
        # 2. One-Hot Encoding
        categorical_cols = df_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        df_encoded = pd.get_dummies(df_fe, columns=categorical_cols, dtype=int)
        
        # 3. Alinhar com a estrutura treinada (garante que não falte nenhuma coluna)
        df_final = df_encoded.reindex(columns=features_treinamento, fill_value=0)
        
        # CRÍTICO: Garantir que todas as colunas são float64 puro
        # O XGBoost/SHAP rejeitam qualquer tipo string ou object, independente da versão
        df_final = df_final.astype('float64')
        
        # 4. Predição com Threshold Dinâmico
        probabilidade = float(modelo.predict_proba(df_final)[0][1])
        classe = 1 if probabilidade >= optimal_threshold else 0
        
        # 5. Explicação com XAI (Feature Importance nativa do XGBoost)
        # Usando feature_importances_ nativo do modelo em vez do SHAP TreeExplainer,
        # que tem um bug conhecido com XGBoost 2.x onde o base_score e serializado
        # como "[5E-1]" pelo NumPy, causando crash no parser do SHAP.
        # A importancia por 'gain' e equivalente e igualmente interpretavel.
        importances = modelo.get_booster().get_score(importance_type='gain')
        
        # Mapeia features para importancias (features nao usadas ficam com 0)
        feature_names = list(df_final.columns)
        feature_importance = [
            {"feature": f, "impact": float(importances.get(f, 0.0))}
            for f in feature_names
            if importances.get(f, 0.0) != 0
        ]
        feature_importance = sorted(feature_importance, key=lambda x: abs(x["impact"]), reverse=True)[:3]
        
        resultado = {
            "churn_probability": float(probabilidade),
            "churn_class": classe,
            "risk_level": "High" if classe == 1 else "Low",
            "top_contributors": feature_importance
        }
        
        # Salvar no Supabase (se estiver configurado)
        if supabase:
            try:
                # Opcional: Adicionar identificação do cliente se vier na request
                db_data = {
                    "probability": resultado["churn_probability"],
                    "churn_class": resultado["churn_class"],
                    "risk_level": resultado["risk_level"],
                    "top_factors": resultado["top_contributors"],
                    "client_data": client.model_dump()
                }
                supabase.table("churn_predictions").insert(db_data).execute()
                print("Predição salva no Supabase com sucesso!")
            except Exception as db_err:
                print(f"Aviso: Erro ao salvar log no Supabase: {db_err}")
        
        return resultado
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Erro ao processar predição: {str(e)}\n\nTraceback:\n{tb}")
