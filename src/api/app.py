import streamlit as st
import pandas as pd
import joblib
import time
from pathlib import Path

# Configuração da página
st.set_page_config(page_title="Previsão de Churn | IA", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# CSS Avançado para UI Moderna e Premium (Dark Mode, Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
    }
    
    .stApp {
        background-image: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.1), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(37, 99, 235, 0.1), transparent 25%);
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #a855f7, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    
    h2, h3 {
        color: #f1f5f9;
        font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    .metric-card-danger {
        background: linear-gradient(145deg, #1e1b4b, #450a0a);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    .metric-card-safe {
        background: linear-gradient(145deg, #022c22, #064e3b);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    
    .prob-text {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .prob-label {
        color: #94a3b8;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stForm"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Sistema de Retenção de Clientes")
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Motor de Inteligência Artificial para identificação precoce de risco de churn (Cancelamento).</p>", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelo():
    model_path = Path("models/xgb_churn_model.pkl")
    features_path = Path("models/model_features.pkl")
    
    if not model_path.exists() or not features_path.exists():
        return None, None
        
    modelo = joblib.load(model_path)
    features = joblib.load(features_path)
    return modelo, features

modelo, features_treinamento = carregar_modelo()

if modelo is None:
    st.error("⚠️ **Alerta do Sistema:** Modelo não encontrado. Por favor, execute a pipeline de treinamento (Fase 3).")
    st.stop()

# Área Principal - Formulário
with st.form("simulador_form"):
    st.markdown("### 📊 Perfil e Parâmetros do Cliente")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Informações Pessoais")
        gender = st.selectbox("Gênero", ["Male", "Female"])
        senior = st.selectbox("Idoso (Sênior)?", [0, 1], format_func=lambda x: "Sim" if x==1 else "Não")
        partner = st.selectbox("Possui Parceiro(a)?", ["Yes", "No"], format_func=lambda x: "Sim" if x=="Yes" else "Não")
        dependents = st.selectbox("Possui Dependentes?", ["Yes", "No"], format_func=lambda x: "Sim" if x=="Yes" else "Não")
        tenure = st.slider("Tempo de Contrato (Meses)", 0, 72, 12)

    with col2:
        st.markdown("#### Serviços Contratados")
        internet = st.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Serviço de Telefonia?", ["Yes", "No"])
        multiple = st.selectbox("Múltiplas Linhas?", ["Yes", "No", "No phone service"])
        online_sec = st.selectbox("Segurança Online?", ["Yes", "No", "No internet service"])
        tech_sup = st.selectbox("Suporte Técnico?", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("#### Detalhes Financeiros")
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Fatura Digital (Paperless)?", ["Yes", "No"])
        payment = st.selectbox("Método de Pagamento", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Mensalidade ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
        total = st.number_input("Receita Total Acumulada ($)", min_value=0.0, max_value=10000.0, value=600.0, step=50.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("🧠 Processar Análise de Risco")

if submit_button:
    with st.status("🧠 Inicializando Motor de Inteligência Artificial...", expanded=True) as status:
        st.write("🔍 Extraindo perfil e parâmetros do cliente...")
        time.sleep(0.8)
        
        # Preenchendo dados ocultos para bater com o formato treinado
        dados_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": "No", 
            "DeviceProtection": "No", 
            "TechSupport": tech_sup,
            "StreamingTV": "No", 
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }
        
        df_input = pd.DataFrame([dados_dict])
        
        st.write("⚙️ Aplicando Engenharia de Features (One-Hot Encoding)...")
        time.sleep(0.8)
        categorical_cols = df_input.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, dtype=int)
        df_final = df_encoded.reindex(columns=features_treinamento, fill_value=0)
        
        st.write("⚡ Executando inferência preditiva com XGBoost...")
        time.sleep(1.2)
        probabilidade = modelo.predict_proba(df_final)[0][1]
        classe = modelo.predict(df_final)[0]
        
        st.write("📊 Consolidando resultados estatísticos...")
        time.sleep(0.5)
        
        status.update(label="Análise Concluída com Sucesso!", state="complete", expanded=False)
        
    st.markdown("---")
    st.markdown("### 🎯 Diagnóstico da Inteligência Artificial")
    
    res_col1, res_col2 = st.columns([3, 2])
    
    with res_col1:
        if classe == 1:
            st.markdown(f'''
            <div class="metric-card-danger">
                <h2 style="color: #fca5a5; margin:0; font-size: 2.2rem;">🚨 Risco Crítico de Evasão</h2>
                <p style="color: #fecaca; margin-top: 15px; font-size: 1.1rem;">Ação Imediata Recomendada. O modelo detectou padrões fortíssimos de cancelamento com base no comportamento deste cliente.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="metric-card-safe">
                <h2 style="color: #86efac; margin:0; font-size: 2.2rem;">✅ Cliente Estável e Seguro</h2>
                <p style="color: #a7f3d0; margin-top: 15px; font-size: 1.1rem;">Retenção Positiva. O comportamento atual se assemelha ao de clientes fiéis. Não há indicativos de cancelamento iminente.</p>
            </div>
            ''', unsafe_allow_html=True)
            
    with res_col2:
        cor_prob = "#f87171" if classe == 1 else "#4ade80"
        st.markdown(f'''
        <div style="text-align: center; padding: 30px; background-color: rgba(30, 41, 59, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <p class="prob-text" style="color: {cor_prob};">{probabilidade * 100:.1f}%</p>
            <p class="prob-label">Probabilidade Matemática</p>
        </div>
        ''', unsafe_allow_html=True)
