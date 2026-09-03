import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Previsão de Churn | IA", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# CSS Avançado (Dark Mode, Glassmorphism)
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .metric-card-safe {
        background: linear-gradient(145deg, #022c22, #064e3b);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(15, 23, 42, 0.95) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelo():
    model_path = Path("models/xgb_churn_model.pkl")
    features_path = Path("models/model_features.pkl")
    
    if not model_path.exists() or not features_path.exists():
        return None, None
        
    modelo = joblib.load(model_path)
    features = joblib.load(features_path)
    return modelo, features

def criar_grafico_velocimetro(probabilidade):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probabilidade * 100,
        number = {'suffix': "%", 'font': {'size': 60, 'color': 'white'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidade de Cancelamento", 'font': {'size': 20, 'color': '#94a3b8'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Barra invisível, usaremos steps
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "#10b981"},    # Verde
                {'range': [40, 70], 'color': "#f59e0b"},   # Amarelo
                {'range': [70, 100], 'color': "#ef4444"}   # Vermelho
            ],
            'threshold': {
                'line': {'color': "white", 'width': 6},
                'thickness': 0.75,
                'value': probabilidade * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

modelo, features_treinamento = carregar_modelo()

if modelo is None:
    st.error("⚠️ **Alerta:** Modelo não encontrado. Execute a pipeline de treinamento (Fase 3/5).")
    st.stop()

# ================= SIDEBAR (Parâmetros) =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10103/10103814.png", width=60)
    st.markdown("## Perfil do Cliente")
    st.markdown("Configure os parâmetros abaixo para gerar uma simulação matemática de risco.")
    
    with st.expander("👤 Informações Pessoais", expanded=True):
        gender = st.selectbox("Gênero", ["Male", "Female"])
        senior = st.selectbox("Idoso (Sênior)?", [0, 1], format_func=lambda x: "Sim" if x==1 else "Não")
        partner = st.selectbox("Possui Parceiro(a)?", ["Yes", "No"], format_func=lambda x: "Sim" if x=="Yes" else "Não")
        dependents = st.selectbox("Possui Dependentes?", ["Yes", "No"], format_func=lambda x: "Sim" if x=="Yes" else "Não")
        tenure = st.slider("Tempo de Contrato (Meses)", 0, 72, 12)
        
    with st.expander("🌐 Serviços Contratados", expanded=False):
        internet = st.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Serviço de Telefonia?", ["Yes", "No"])
        multiple = st.selectbox("Múltiplas Linhas?", ["Yes", "No", "No phone service"])
        online_sec = st.selectbox("Segurança Online?", ["Yes", "No", "No internet service"])
        tech_sup = st.selectbox("Suporte Técnico?", ["Yes", "No", "No internet service"])
        
    with st.expander("💳 Dados Financeiros", expanded=False):
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Fatura Digital (Paperless)?", ["Yes", "No"])
        payment = st.selectbox("Método de Pagamento", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Mensalidade ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
        total = st.number_input("Receita Total ($)", min_value=0.0, max_value=10000.0, value=600.0, step=50.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.button("🧠 Processar Inteligência Artificial")


# ================= TELA PRINCIPAL =================
st.title("Sistema de Retenção de Clientes")
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Motor preditivo otimizado com SMOTE e Engenharia de Features para risco de Churn.</p>", unsafe_allow_html=True)

if submit_button:
    with st.status("🧠 Inicializando Motor de Inteligência Artificial...", expanded=True) as status:
        st.write("🔍 Extraindo perfil e parâmetros do cliente...")
        time.sleep(0.5)
        
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
        
        st.write("⚙️ Calculando novas métricas (Engenharia de Features)...")
        time.sleep(0.5)
        # Replicando a engenharia de features do treino
        servicos = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        def count_services(row):
            count = 0
            for s in servicos:
                if str(row.get(s, 'No')) not in ['No', 'No internet service', 'No phone service']:
                    count += 1
            return count
            
        df_input['Total_Servicos_Contratados'] = df_input.apply(count_services, axis=1)
        df_input['Gasto_Por_Mes_De_Vida'] = df_input['TotalCharges'] / (df_input['tenure'] + 1)
        
        st.write("🔄 Aplicando One-Hot Encoding e equalizando matriz...")
        time.sleep(0.5)
        categorical_cols = df_input.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, dtype=int)
        df_final = df_encoded.reindex(columns=features_treinamento, fill_value=0)
        
        st.write("⚡ Executando inferência preditiva com hiperparâmetros tunados...")
        time.sleep(0.8)
        probabilidade = modelo.predict_proba(df_final)[0][1]
        classe = modelo.predict(df_final)[0]
        
        status.update(label="Análise Concluída com Sucesso!", state="complete", expanded=False)
        
    st.markdown("---")
    
    colA, colB = st.columns([1, 1.2])
    
    with colA:
        if classe == 1:
            st.markdown(f'''
            <div class="metric-card-danger">
                <h2 style="color: #fca5a5; margin:0; font-size: 2rem;">🚨 Risco Crítico de Evasão</h2>
                <p style="color: #fecaca; margin-top: 15px; font-size: 1.1rem;">Ação Imediata Recomendada.</p>
                <hr style="border-color: rgba(255,255,255,0.1)">
                <ul style="color: #fecaca; padding-left: 20px;">
                    <li>A probabilidade matemática aponta para cancelamento.</li>
                    <li>O modelo foi ajustado via SMOTE e apurou forte correlação de risco.</li>
                    <li>Ofereça incentivos ou retenção especializada.</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="metric-card-safe">
                <h2 style="color: #86efac; margin:0; font-size: 2rem;">✅ Cliente Estável e Seguro</h2>
                <p style="color: #a7f3d0; margin-top: 15px; font-size: 1.1rem;">Retenção Positiva.</p>
                <hr style="border-color: rgba(255,255,255,0.1)">
                <ul style="color: #a7f3d0; padding-left: 20px;">
                    <li>Baixa propensão ao cancelamento.</li>
                    <li>Perfil engajado com os serviços atuais.</li>
                    <li>Boa oportunidade para Up-Sell.</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
            
    with colB:
        # Gráfico Velocímetro com Plotly
        fig = criar_grafico_velocimetro(probabilidade)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 🧠 Raio-X da Decisão (IA Explicável com SHAP)")
    st.markdown("<p style='color: #94a3b8;'>O gráfico abaixo utiliza a Teoria dos Jogos para abrir a 'caixa preta' do XGBoost. Barras vermelhas empurram o risco para cima, e barras azuis puxam para baixo.</p>", unsafe_allow_html=True)
    
    with st.spinner("Desenhando matriz de impacto..."):
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer(df_final)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0b0f19')
        
        shap.plots.waterfall(shap_values[0], show=False, max_display=8)
        
        st.pyplot(fig)
else:
    st.info("👈 Preencha os dados na barra lateral e clique em 'Processar Inteligência Artificial' para visualizar os resultados em tempo real.")
