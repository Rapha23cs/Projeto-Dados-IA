import { useState, useEffect } from 'react'

const AnimatedGauge = ({ targetPercentage, isRisk }) => {
  const [displayPercentage, setDisplayPercentage] = useState(0);

  useEffect(() => {
    setDisplayPercentage(0);
    let startTimestamp = null;
    const duration = 1500; // 1.5 segundos de animação suave

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // Efeito de Easing (Desacelerar no final - easeOutQuart)
      const easeProgress = 1 - Math.pow(1 - progress, 4);
      
      setDisplayPercentage(targetPercentage * easeProgress);
      
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    
    window.requestAnimationFrame(step);
  }, [targetPercentage]);

  return (
    <div className="gauge-container" 
         style={{
           "--percentage": displayPercentage,
           "--color-risk": isRisk ? "#ef4444" : "#10b981"
         }}>
      <div className="gauge-inner">
        <span className="gauge-number">{displayPercentage.toFixed(1)}<span style={{fontSize: "1.5rem"}}>%</span></span>
        <span className="gauge-label">Risco de Churn</span>
      </div>
    </div>
  );
};

// Componente Premium: Segmented Control (Substitui Selects)
const SegmentedControl = ({ name, options, value, onChange }) => (
  <div className="segmented-control">
    {options.map(opt => (
      <label key={opt.value} className={`segmented-item ${value === opt.value ? 'active' : ''}`}>
        <input 
          type="radio" 
          name={name} 
          value={opt.value} 
          checked={value === opt.value} 
          onChange={onChange} 
        />
        <span>{opt.label}</span>
      </label>
    ))}
  </div>
);

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  // Estado para o formulário
  const [formData, setFormData] = useState({
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 70.0,
    TotalCharges: 840.0
  });

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Erro na predição:", error);
      alert("Erro ao conectar com a IA. Verifique se o backend está rodando no porto 8000.");
    }
    setLoading(false);
  };

  return (
    <>
      <div className="ambient-background">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
      </div>

      <header className="app-header">
        <h1 className="title-gradient">Churn Predictor SaaS</h1>
        <p className="subtitle">Motor de Inteligência Artificial para Retenção de Clientes</p>
      </header>
      
      <main className="dashboard-grid">
        
        {/* Lado Esquerdo: Formulário */}
        <section className="glass-card">
          <h2 style={{marginBottom: "1.5rem", fontWeight: 700}}>Perfil do Cliente</h2>
          <form onSubmit={handleSubmit} className="form-grid">
            
            <div className="input-group">
              <label>Tempo de Contrato (Meses)</label>
              <div className="input-icon-wrapper">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                <input type="number" name="tenure" className="glass-input with-icon" value={formData.tenure} onChange={handleChange} min="0" max="72" />
              </div>
            </div>
            
            <div className="input-group">
              <label>Tipo de Contrato</label>
              <SegmentedControl 
                name="Contract" 
                value={formData.Contract} 
                onChange={handleChange}
                options={[
                  {label: "Mensal", value: "Month-to-month"},
                  {label: "Anual", value: "One year"},
                  {label: "2 Anos", value: "Two year"}
                ]}
              />
            </div>

            <div className="input-group">
              <label>Mensalidade (US$)</label>
              <div className="input-icon-wrapper">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                <input type="number" name="MonthlyCharges" className="glass-input with-icon" value={formData.MonthlyCharges} onChange={handleChange} step="0.01" />
              </div>
            </div>

            <div className="input-group">
              <label>Receita Total (US$)</label>
              <div className="input-icon-wrapper">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                <input type="number" name="TotalCharges" className="glass-input with-icon" value={formData.TotalCharges} onChange={handleChange} step="0.01" />
              </div>
            </div>

            <div className="input-group" style={{ gridColumn: "span 2" }}>
              <label>Tipo de Internet</label>
              <SegmentedControl 
                name="InternetService" 
                value={formData.InternetService} 
                onChange={handleChange}
                options={[
                  {label: "DSL", value: "DSL"},
                  {label: "Fibra Óptica", value: "Fiber optic"},
                  {label: "Sem Internet", value: "No"}
                ]}
              />
            </div>

            <div className="input-group" style={{ gridColumn: "span 2" }}>
              <label>Possui Suporte Técnico?</label>
              <SegmentedControl 
                name="TechSupport" 
                value={formData.TechSupport} 
                onChange={handleChange}
                options={[
                  {label: "Sim", value: "Yes"},
                  {label: "Não", value: "No"},
                  {label: "Sem Internet", value: "No internet service"}
                ]}
              />
            </div>
            
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Processando IA..." : "Analise de Risco Inteligente 🧠"}
            </button>
          </form>
        </section>

        {/* Lado Direito: Resultados */}
        <section className="glass-card results-area">
          {!result && !loading && (
            <div className="empty-state">
              <div style={{fontSize: "4rem", marginBottom: "1rem"}}>🔍</div>
              <p>Preencha os dados e clique em analisar para descobrir se este cliente está em risco de cancelamento.</p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div style={{fontSize: "4rem", marginBottom: "1rem"}} className="spin">⚡</div>
              <p>Otimizando modelo e executando XGBoost...</p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Velocímetro Customizado e Animado em React/CSS */}
              <AnimatedGauge 
                targetPercentage={result.churn_probability * 100} 
                isRisk={result.churn_class === 1} 
              />

              {/* Card de Decisão */}
              {result.churn_class === 1 ? (
                <div className="info-card card-danger">
                  <h3>🚨 Ação Imediata Necessária</h3>
                  <p>Probabilidade crítica de cancelamento detectada pela Inteligência Artificial.</p>
                </div>
              ) : (
                <div className="info-card card-safe">
                  <h3>✅ Cliente Estável</h3>
                  <p>Baixa propensão de abandono. Ótima oportunidade para tentar Up-Sell.</p>
                </div>
              )}

              {/* Justificativa SHAP - Gráfico de Barras Moderno */}
              {result.top_contributors && (
                <div className="shap-container">
                  <h4 style={{marginBottom: "1.5rem", color: "var(--text-muted)", textAlign: "center"}}>Fatores Decisivos (XAI)</h4>
                  <div className="shap-chart">
                    {result.top_contributors.map((item, idx) => {
                      const isPositive = item.impact > 0;
                      const maxImpact = Math.max(...result.top_contributors.map(i => Math.abs(i.impact)));
                      const widthPercent = (Math.abs(item.impact) / maxImpact) * 100;
                      
                      let displayName = item.feature.replace(/_/g, ' ');
                      displayName = displayName.replace('Custo Por Servico', 'Cost Per Service');
                      displayName = displayName.replace('Total Servicos Contratados', 'Total Services');
                      displayName = displayName.replace('Gasto Por Mes De Vida', 'Lifetime Monthly Spend');
                      displayName = displayName.replace('Tenure Group', 'Tenure Group');

                      return (
                        <div key={idx} className="shap-bar-row">
                          <div className="shap-bar-label">
                            <span style={{fontWeight: 500}}>{displayName}</span>
                            <span style={{color: isPositive ? 'var(--danger)' : 'var(--primary)', fontWeight: 700}}>
                              {isPositive ? "+" : ""}{(item.impact).toFixed(2)}
                            </span>
                          </div>
                          <div className="shap-bar-track">
                            <div 
                              className={`shap-bar-fill ${isPositive ? 'bg-danger' : 'bg-primary'}`} 
                              style={{ "--target-width": `${widthPercent}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </section>

      </main>
    </>
  )
}

export default App
