# 🔮 Previsão de Risco de Cancelamento (Churn) com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Concluído-success.svg)

Este é um projeto completo de Ciência de Dados e Machine Learning de ponta a ponta. O objetivo é prever com alta precisão se um cliente vai cancelar seus serviços (Churn) com base em seu perfil, histórico de faturamento e serviços contratados.

O sistema conta com um pipeline robusto, cobrindo desde a extração automatizada de dados brutos até uma interface analítica moderna e interativa em **Streamlit**.

---

## 🏗️ Arquitetura do Pipeline (Ponta a Ponta)

| Etapa | Responsabilidade | Ferramentas Utilizadas |
| :--- | :--- | :--- |
| **1. Ingestão & Armazenamento** (`src/ingestion`) | Download automático do dataset da IBM e persistência estruturada no banco de dados. | Pandas, SQLAlchemy, SQLite (Fallback) / PostgreSQL |
| **2. Processamento & Validação** (`src/validation`) | Limpeza profunda, transformação de tipos e testes de qualidade de dados. | Pandas, Great Expectations |
| **3. Camada de IA / Modelagem** (`src/model`) | Engenharia de Features (One-Hot Encoding), separação Treino/Teste e Treinamento Estatístico. | Scikit-Learn, XGBoost, Joblib |
| **4. Entrega Analítica** (`src/api`) | Dashboard web premium, escuro e responsivo para inferência de risco em tempo real. | Streamlit |

---

## 📂 Estrutura de Diretórios

```text
projeto_dados_ia/
├── data/
│   ├── raw/                 # Cópia intocável do dataset original (telco_churn_raw.csv)
│   ├── processed/           # Dados limpos e preparados para a IA (telco_churn_clean.csv)
│   └── projeto_dados.db     # Banco de Dados relacional local (SQLite)
├── models/
│   ├── xgb_churn_model.pkl  # Cérebro do projeto (Modelo XGBoost treinado)
│   └── model_features.pkl   # Estrutura exata das colunas (para engenharia reversa)
├── sql/
│   ├── ddl/                 # Schemas e criação de tabelas
│   └── queries/             # Views e consultas analíticas
├── src/
│   ├── ingestion/
│   │   └── extract.py       # Extrai e carrega no DB (Etapa 1)
│   ├── validation/
│   │   └── clean_data.py    # Limpa e valida com Great Expectations (Etapa 2)
│   ├── model/
│   │   └── train.py         # Treina a Inteligência Artificial (Etapa 3)
│   └── api/
│       └── app.py           # Dashboard Front-End Streamlit (Etapa 4)
├── tests/                   # Testes unitários do pipeline
├── config/
│   └── .env.example         # Variáveis de ambiente de Banco de Dados / API Keys
├── requirements.txt         # Lista oficial de dependências do projeto
└── README.md                # Documentação atual
```

---

## 🚀 Como Executar o Projeto Localmente

### 1. Configuração do Ambiente Virtual
Crie o ambiente virtual e ative-o no seu terminal:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # No Windows (PowerShell)
# ou
source venv/bin/activate      # No Mac/Linux
```

### 2. Instalação das Dependências
```bash
pip install -r requirements.txt
```

### 3. Rodando o Pipeline Completo
Se você quiser recriar tudo do zero, execute os scripts em ordem a partir da pasta raiz do projeto:

1. **Baixar Dados e Salvar no Banco:**
   ```bash
   python src/ingestion/extract.py
   ```
2. **Limpar Dados e Testar Qualidade:**
   ```bash
   python src/validation/clean_data.py
   ```
3. **Treinar o Modelo XGBoost (IA):**
   ```bash
   python src/model/train.py
   ```
4. **Abrir a Interface Web (Dashboard):**
   ```bash
   streamlit run src/api/app.py
   ```

A interface visual abrirá automaticamente no seu navegador no endereço: `http://localhost:8501`.

---

## 🏆 Resultados Obtidos

O modelo `XGBClassifier` alcançou uma acurácia de base superior a **80.2%** logo no primeiro treinamento rápido, mostrando excelente precisão em reter (prever quem **não** vai sair), com capacidade gigantesca de encontrar as "bandeiras vermelhas" (red flags) ocultas que humanos normalmente ignorariam.
