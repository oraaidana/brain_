import React, { useState } from 'react';
import './App.css';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
const API_BASE_URL = "https://careon-r10p.onrender.com"
// --- Chart Component ---
const RiskTrendChart = ({ data }) => {
  if (!data) return null;
  
  // Transform backend data into Recharts format
  const chartData = data.labels.map((label, index) => ({
    name: label,
    unmanaged: data.unmanaged[index],
    managed: data.managed[index],
  }));

  return (
    <div style={{ width: '100%', height: 300, marginTop: '20px' }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="name" 
            stroke="rgba(255,255,255,0.5)" 
            fontSize={12} 
            tickLine={false}
          />
          <YAxis 
            stroke="rgba(255,255,255,0.5)" 
            fontSize={12} 
            tickLine={false} 
            label={{ value: 'Risk %', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
            itemStyle={{ fontSize: '12px' }}
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          
          <Line
            name="Unmanaged Trend"
            type="monotone"
            dataKey="unmanaged"
            stroke="#ff4d4d"
            strokeDasharray="5 5"
            strokeWidth={3}
            dot={{ r: 4, fill: '#ff4d4d' }}
          />
          
          <Line
            name="Intervention Path"
            type="monotone"
            dataKey="managed"
            stroke="#4caf50"
            strokeWidth={3}
            dot={{ r: 4, fill: '#4caf50' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// --- Main App Component ---
function App() {
  const [step, setStep] = useState('auth'); 
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [patientHistory, setPatientHistory] = useState([]); // To store history
  const [userData, setUserData] = useState({
    full_name: '', email: '', age: '', password: ''
  });

  const [diagnosticFiles, setDiagnosticFiles] = useState({
    mri_baseline: null,
    mri_current: null,
    ecg_strip: null,
    eeg_strip: null
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUserData({ ...userData, [name]: value });
  };

  const handleFileChange = (e) => {
    const { name, files } = e.target;
    if (files.length > 0) {
      setDiagnosticFiles(prev => ({ ...prev, [name]: files[0] }));
    }
  };
  const handleLoginSubmit = () => {
    if (userData.email && userData.password) {
      // In a real app, you'd verify against a /login endpoint
      setStep('upload');
    } else {
      alert("Please enter credentials");
    }
  };
const goToMain = () => setStep('upload');

  // Function to fetch and go to History (Account Page)
  const goToAccount = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history/${userData.email}`);
      const data = await response.json();
      setPatientHistory(data);
      setStep('history');
    } catch (err) {
      alert("Database error: Could not retrieve history.");
    }
  };
  const fetchHistory = async () => {
    try {
      // Assuming you create a @app.get("/history/{email}") in FastAPI
      const response = await fetch(`${API_BASE_URL}/history/${userData.email}`);
      const data = await response.json();
      setPatientHistory(data);
      setStep('history');
    } catch (err) {
      alert("Could not fetch history");
    }
  };
const handleRegister = async () => {
    const formData = new FormData();
    formData.append('full_name', userData.full_name);
    formData.append('email', userData.email);
    formData.append('age', userData.age);
    formData.append('password', userData.password);

    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        alert("Registration Successful! Please login.");
        setIsLogin(true);
      } else {
        alert(data.detail || "Registration failed");
      }
    } catch (err) {
      alert("Database Connection Error");
    }
  };
  const handleUploadAnalysis = async () => {
    if (!diagnosticFiles.ecg_strip && !diagnosticFiles.eeg_strip) {
      alert("⚠️ Mandatory Requirement: Please upload at least an ECG or EEG strip.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('email', userData.email);
    
    if (diagnosticFiles.ecg_strip) formData.append('ecg_strip', diagnosticFiles.ecg_strip);
    if (diagnosticFiles.eeg_strip) formData.append('eeg_strip', diagnosticFiles.eeg_strip);
    if (diagnosticFiles.mri_baseline) formData.append('mri_baseline', diagnosticFiles.mri_baseline);
    if (diagnosticFiles.mri_current) formData.append('mri_current', diagnosticFiles.mri_current);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      setStep('results');
    } catch (err) {
      alert("System Sync Error: Connection to diagnostic engine failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="careon-app">
      {/* --- PERSISTENT NAVIGATION BAR --- */}
      <nav className="nav-blur">
        <div 
          className="logo-group" 
          onClick={userData.email ? goToMain : null} 
          style={{ cursor: userData.email ? 'pointer' : 'default' }}
        >
          <div className="logo-orb"></div>
          <span className="logo-text">CAREON<span className="ai-tag">AI</span></span>
        </div>

        {/* Navigation links only show once authenticated */}
        {step !== 'auth' && (
          <div className="nav-controls animate-fade-in">
            <button 
              className={`nav-link ${step === 'upload' || step === 'results' ? 'active' : ''}`} 
              onClick={goToMain}
            >
              Diagnostic Portal
            </button>
            <button 
              className={`nav-link ${step === 'history' ? 'active' : ''}`} 
              onClick={goToAccount}
            >
              Account
            </button>
            <button 
              className="nav-link logout-btn" 
              onClick={() => { setStep('auth'); setUserData({ email: '', password: '', full_name: '', age: '' }); }}
            >
              Logout
            </button>
          </div>
        )}
      </nav>

      <main className="content-area">
        {/* 1. AUTHENTICATION (Login/Register) */}
        {step === 'auth' && (
          <div className="glass-card auth-box animate-fade-in">
            <div className="auth-header">
              <h2>{isLogin ? "Welcome Back" : "Create Account"}</h2>
              <p>Secure Neural Diagnostic Portal</p>
            </div>
            <div className="input-stack">
              {!isLogin && (
                <>
                  <input name="full_name" placeholder="Full Name" className="luminous-input" onChange={handleInputChange} />
                  <input name="age" type="number" placeholder="Age" className="luminous-input" onChange={handleInputChange} />
                </>
              )}
              <input name="email" type="email" placeholder="Medical Email" className="luminous-input" onChange={handleInputChange} />
              <input name="password" type="password" placeholder="Password" className="luminous-input" onChange={handleInputChange} />
              
              <button className="btn-primary" onClick={isLogin ? handleLoginSubmit : handleRegister}>
                {isLogin ? "Access Portal" : "Sign Up"}
              </button>
              
              <p className="toggle-auth" onClick={() => setIsLogin(!isLogin)}>
                {isLogin ? "Need an account? Register" : "Already have an account? Login"}
              </p>
            </div>
          </div>
        )}

        {/* 2. UPLOAD PAGE (Main Portal) */}
        {step === 'upload' && (
          <div className="upload-layout animate-fade-in">
            <div className="hero-text">
              <span className="badge-modern">Physiological Priority v1.1</span>
              <h1>Neuro-Cardiac <br/><span>Engine</span></h1>
              <p>Primary: Heart & Brain Waves | Secondary: Structural MRI (Optional)</p>
            </div>

            <div className={`glass-card upload-panel ${loading ? 'active-scan' : ''}`}>
              {loading && <div className="laser-beam"></div>}
              
              <div className="upload-grid">
                <div className="upload-column primary-zone">
                  <h3 className="section-title required">Mandatory Signals</h3>
                  <div className={`file-box ${diagnosticFiles.ecg_strip ? 'loaded' : 'unloaded'}`}>
                    <label>ECG Strip (Heart Rhythm)</label>
                    <input type="file" name="ecg_strip" onChange={handleFileChange} />
                  </div>
                  <div className={`file-box ${diagnosticFiles.eeg_strip ? 'loaded' : 'unloaded'}`}>
                    <label>EEG Strip (Neural Activity)</label>
                    <input type="file" name="eeg_strip" onChange={handleFileChange} />
                  </div>
                </div>

                <div className="upload-column secondary-zone">
                  <h3 className="section-title">MRI Scans</h3>
                  <div className="file-box">
                    <label>Baseline MRI</label>
                    <input type="file" name="mri_baseline" onChange={handleFileChange} />
                  </div>
                  <div className="file-box">
                    <label>Current MRI</label>
                    <input type="file" name="mri_current" onChange={handleFileChange} />
                  </div>
                </div>
              </div>

              <div className="action-bar">
                <button className="btn-primary full-width" onClick={handleUploadAnalysis} disabled={loading}>
                  {loading ? "Analyzing..." : "Execute Diagnosis"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 3. HISTORY PAGE (Account) */}
{step === 'history' && (
  <div className="history-container animate-fade-in">
    <div className="account-header">
       <h2>Patient Profile</h2>
       <p>Identity: {userData.full_name || "Verified User"} | {userData.email}</p>
    </div>

    <div className="history-grid">
      {patientHistory.length > 0 ? (
        patientHistory.map((rec) => (
          <div key={rec.id} className="history-card">
            <div className="card-top">
              <span className="date">📅 {new Date(rec.created_at).toLocaleDateString()}</span>
              <span className={`status-badge ${rec.diagnostic_status.toLowerCase()}`}>
                {rec.diagnostic_status}
              </span>
            </div>
            
            <div className="risk-metric">
              <label>Calculated Stroke Risk</label>
              <span className="value">{rec.risk_score}%</span>
            </div>

            <div style={{marginTop: '20px', display: 'flex', gap: '10px'}}>
               <button className="btn-primary-outline" style={{padding: '5px 10px', fontSize: '0.7rem'}}>
                 Review Full Analysis
               </button>
            </div>
          </div>
        ))
      ) : (
        <div className="empty-state">
           <p style={{fontSize: '2rem'}}>📋</p>
           <p>No previous diagnostic records found for this account.</p>
           <button className="btn-primary" onClick={goToMain} style={{marginTop: '15px'}}>
             Start First Analysis
           </button>
        </div>
      )}
    </div>
  </div>
)}

        {/* 4. RESULTS PAGE */}
        {step === 'results' && result && (
          <div className="bento-grid animate-fade-in">
            <div className="glass-card bento-stats">
               <span className="score-label">Aggregate Stroke Risk</span>
               <div className="score-ring">{result.probability}%</div>
               <div className="status-badge" data-status={result.status?.toLowerCase()}>
                 {result.status}
               </div>
            </div>

            <div className="glass-card bento-metrics">
              <h3>Biomarker Analysis</h3>
              <div className="metric-list">
                <div className="metric-item">
                  <label>MRI Volume Change</label>
                  <span className="value">{result.metrics.mri_change}</span>
                </div>
                <div className="metric-item">
                  <label>Z-Score (Structural)</label>
                  <span className="value">{result.metrics.z_score}</span>
                </div>
                <div className="metric-item">
                  <label>Cardiac Signal</label>
                  <span className={`status-pill ${result.metrics.ecg_status.includes('ABNORMAL') ? 'bad' : 'good'}`}>
                    {result.metrics.ecg_status}
                  </span>
                </div>
                <div className="metric-item">
                  <label>Neural Signal</label>
                  <span className={`status-pill ${result.metrics.eeg_status.includes('STABLE') ? 'good' : 'bad'}`}>
                    {result.metrics.eeg_status}
                  </span>
                </div>
              </div>
            </div>

            <div className="glass-card bento-trend">
              <h3>3-Year Risk Forecast</h3>
              <RiskTrendChart data={result.trend} />
            </div>

            <div className="glass-card bento-rec">
              <h3>Recommendations</h3>
              <ul className="rec-list">
                {result.recommendations && result.recommendations.map((text, i) => (
                  <li key={i}><span className="bullet"></span>{text}</li>
                ))}
              </ul>
              <button className="btn-primary-outline" onClick={() => window.print()} style={{width: '100%'}}>
                Export PDF Report
              </button>
            </div>

            <div className="glass-card bento-mri-bottom">
              <h3>Structural Visualization</h3>
              <div className="comparison-container">
                {result.image ? (
                  <div className="image-frame-full">
                    <img src={result.image} alt="MRI Analysis Mapping" />
                    <p className="image-caption">Current Scan Analysis: Voxel-based Morphometry Map</p>
                  </div>
                ) : (
                  <div className="placeholder-signal">
                     <div className="pulse-icon">💓</div>
                     <p>Cardiac/Neural Signal Analysis Only</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;