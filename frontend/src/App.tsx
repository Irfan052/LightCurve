import { useState, useRef, useEffect, type ChangeEvent } from 'react';
import Plot from 'react-plotly.js';
import {
  Download,
  UploadCloud,
  Search,
  CheckCircle,
  Activity,
  Globe,
  Disc,
  Star,
  Zap
} from 'lucide-react';
import html2pdf from 'html2pdf.js';
import './App.css';

import { analyzeTarget, uploadAndAnalyze, type AnalysisResult } from './api';

function App() {
  const [targetId, setTargetId] = useState('TIC 9991');
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');
  const [runtime, setRuntime] = useState<string>('');
  const [startTime, setStartTime] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  const handleAnalyze = async (id: string) => {
    if (!id.trim()) {
      setError("Target ID cannot be empty. Please enter a valid TIC ID.");
      return;
    }
    setTargetId(id);
    setLoading(true);
    setError('');
    const start = Date.now();
    setStartTime(start);
    try {
      const data = await analyzeTarget(id);
      setResult(data);
      setRuntime(((Date.now() - start) / 1000).toFixed(2));
    } catch (err) {
      setResult(null);
      const error = err as { message?: string; response?: { data?: { detail?: string } } };
      if (error.message === "Network Error") {
        setError("Backend API is currently unavailable. Please ensure the server is running.");
      } else {
        setError(error.response?.data?.detail || error.message || 'An unknown error occurred during analysis.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setLoading(true);
    setError('');
    const start = Date.now();
    setStartTime(start);
    try {
      const data = await uploadAndAnalyze(file);
      setResult(data);
      setRuntime(((Date.now() - start) / 1000).toFixed(2));
    } catch (err) {
      setResult(null);
      const error = err as { message?: string; response?: { data?: { detail?: string } } };
      if (error.message === "Network Error") {
        setError("Backend API is currently unavailable. Please ensure the server is running.");
      } else {
        setError(error.response?.data?.detail || error.message || 'File upload failed or data was invalid.');
      }
    } finally {
      setLoading(false);
      // Reset input so the same file can be uploaded again if needed
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleExportPDF = () => {
    if (!reportRef.current) return;
    
    // Add temporary timestamp to report root before snapshot
    const timestampNode = document.createElement('div');
    timestampNode.id = 'pdf-timestamp';
    timestampNode.style.textAlign = 'right';
    timestampNode.style.color = '#94a3b8';
    timestampNode.style.fontSize = '0.85rem';
    timestampNode.style.marginBottom = '1rem';
    timestampNode.innerText = `Report Generated: ${new Date().toLocaleString()}`;
    reportRef.current.insertBefore(timestampNode, reportRef.current.firstChild);

    const opt = {
      margin: 0.5,
      filename: `astroexo-report-${result?.target_name}.pdf`,
      image: { type: 'jpeg' as const, quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: 'in' as const, format: 'letter' as const, orientation: 'landscape' as const }
    };
    
    html2pdf().set(opt).from(reportRef.current).save().then(() => {
      // Remove timestamp after snapshot
      const node = document.getElementById('pdf-timestamp');
      if (node && node.parentNode) node.parentNode.removeChild(node);
    });
  };

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (loading) {
      const statuses = [
        "Downloading TESS data...",
        "Cleaning data...",
        "Running BLS search...",
        "Classifying candidate..."
      ];
      let i = 0;
      setLoadingStatus(statuses[i]);
      interval = setInterval(() => {
        i = (i + 1) % statuses.length;
        if (i === statuses.length - 1) clearInterval(interval);
        setLoadingStatus(statuses[i]);
      }, 2000);
    } else {
      setLoadingStatus('');
    }
    return () => clearInterval(interval);
  }, [loading]);

  const depth = result?.parameters?.transit_depth_percent ?? 0;
  const duration = result?.parameters?.transit_duration_hours ?? 0;
  const radius = result?.parameters?.planet_radius_earth ?? 0;

  const getExplanation = () => {
    if (!result) return [];
    
    const imp = result.feature_importance_summary.top_random_forest || [];
    const pred = result.prediction;
    
    // Human readable mappings based on class
    return imp.map((f: { feature: string; importance: number }) => {
      let desc = f.feature;
      if (f.feature === "depth") desc = "High transit depth";
      if (f.feature === "binned_transit_depth") desc = "Significant binned transit depth";
      if (f.feature === "secondary_ratio") desc = pred === 'eclipsing_binary' ? "Strong secondary eclipse detected" : "No significant secondary eclipse";
      if (f.feature === "odd_even_diff") desc = pred === 'exoplanet_transit' ? "Low odd-even depth difference" : "High odd-even depth difference (typical of background EBs)";
      if (f.feature === "shape_score") desc = "Symmetric transit shape";
      if (f.feature === "period") desc = `Periodic signal at ${result.parameters?.period?.toFixed(2) || '?'} days`;
      if (f.feature === "folded_point_count") desc = "High data coverage in phase fold";
      
      return desc;
    });
  };

  const getConfidenceLevel = (conf: number) => {
    if (conf >= 0.9) return { level: 'High', color: 'var(--accent-green)' };
    if (conf >= 0.7) return { level: 'Medium', color: 'var(--accent-orange)' };
    return { level: 'Low', color: 'var(--accent-red)' };
  };

  return (
    <div className="dashboard-container">
      <header className="app-header">
        <h1><Globe style={{ display: 'inline', marginRight: '8px' }} /> AstroExo AI Dashboard (Demo Mode)</h1>
        <button className="btn btn-secondary" onClick={handleExportPDF} disabled={!result} style={{width: 'auto'}}>
          <Download size={18} /> Export PDF Report
        </button>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <div className="card">
            <h2>Data Input</h2>
            <div className="input-group">
              <input 
                type="text" 
                value={targetId} 
                onChange={(e) => setTargetId(e.target.value)}
                placeholder="TIC ID or 9991"
              />
              <button className="btn" onClick={() => handleAnalyze(targetId)} style={{width: 'auto'}}>
                <Search size={18} />
              </button>
            </div>
            
              <div className="file-drop" onClick={() => fileInputRef.current?.click()} style={{marginTop: '1rem', padding: '1.5rem 1rem'}}>
              <UploadCloud size={32} style={{opacity: 0.5}} />
              <p>Upload CSV Light Curve</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload}
                style={{display: 'none'}} 
                accept=".csv" 
              />
            </div>
            <div style={{textAlign: 'center', marginTop: '0.5rem'}}>
              <a href="data:text/csv;charset=utf-8,time,flux,flux_err%0A1325.2345,1.0001,0.0005%0A1325.2483,0.9998,0.0005%0A1325.2621,0.9850,0.0006%0A" download="sample_lightcurve.csv" style={{fontSize: '0.85rem', color: 'var(--accent-blue)', textDecoration: 'none'}}>Download Sample CSV</a>
            </div>
          </div>

          <div className="card">
            <h2>Demo Mode</h2>
            <div className="demo-grid">
              <div className="demo-btn" onClick={() => handleAnalyze('9991')}>
                <Globe size={24} color="var(--accent-blue)" />
                Exoplanet (9991)
              </div>
              <div className="demo-btn" onClick={() => handleAnalyze('9992')}>
                <Disc size={24} color="var(--accent-purple)" />
                Binary Star (9992)
              </div>
              <div className="demo-btn" onClick={() => handleAnalyze('9993')}>
                <Star size={24} color="var(--accent-orange)" />
                Stellar Var (9993)
              </div>
              <div className="demo-btn" onClick={() => handleAnalyze('9994')}>
                <Zap size={24} color="var(--accent-red)" />
                Artifact (9994)
              </div>
            </div>
          </div>

          <div className="card">
            <h2>Processing Status</h2>
            <div className="stepper">
              <div className={`step ${result || loading ? 'completed' : ''}`}>
                <div className="step-icon">{result || loading ? <CheckCircle size={20} /> : <Disc size={20} />}</div>
                <span>Data Loaded</span>
              </div>
              <div className={`step ${result || loading ? 'completed' : ''}`}>
                <div className="step-icon">{result || loading ? <CheckCircle size={20} /> : <Disc size={20} />}</div>
                <span>Quality Filtering</span>
              </div>
              <div className={`step ${result || loading ? 'completed' : ''}`}>
                <div className="step-icon">{result || loading ? <CheckCircle size={20} /> : <Disc size={20} />}</div>
                <span>Detrending</span>
              </div>
              <div className={`step ${result || loading ? 'completed' : ''}`}>
                <div className="step-icon">{result || loading ? <CheckCircle size={20} /> : <Disc size={20} />}</div>
                <span>Transit Search (BLS)</span>
              </div>
              <div className={`step ${result ? 'completed' : loading ? 'active' : ''}`}>
                <div className="step-icon">{result ? <CheckCircle size={20} /> : loading ? <Activity size={20} /> : <Disc size={20} />}</div>
                <span>Feature Extraction</span>
              </div>
              <div className={`step ${result ? 'completed' : ''}`}>
                <div className="step-icon">{result ? <CheckCircle size={20} /> : <Disc size={20} />}</div>
                <span>AI Classification</span>
              </div>
            </div>
            
            {result && (
              <div className="runtime-stats" style={{marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.9rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Raw Observations:</span>
                  <strong>{result.data.time.length}</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Observation Time:</span>
                  <strong>{(result.data.time[result.data.time.length-1] - result.data.time[0]).toFixed(1)} days</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Cleaned Points:</span>
                  <strong>{result.data.time_clean.length}</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Data Quality:</span>
                  <strong style={{color: (result.data.time_clean.length / result.data.time.length) > 0.8 ? 'var(--accent-green)' : 'var(--accent-orange)'}}>
                    {((result.data.time_clean.length / result.data.time.length) * 100).toFixed(1)}%
                  </strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px dashed var(--border-color)'}}>
                  <span style={{color: 'var(--text-muted)'}}>Pipeline Runtime:</span>
                  <strong>{runtime}s</strong>
                </div>
              </div>
            )}
          </div>
          <div className="card">
            <h2>Architecture Workflow</h2>
            <div className="workflow-diagram">
              <div className="workflow-node">Raw Light Curve</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node">Quality Filter</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node">Detrending</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node">BLS Transit Search</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node">Feature Engineering</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node highlight">Random Forest</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node">Parameter Estimation</div>
              <div className="workflow-arrow">↓</div>
              <div className="workflow-node end">Scientific Report</div>
            </div>
          </div>
        </aside>

        <main className="results-area" ref={reportRef}>
          {error && (
            <div className="card" style={{borderColor: 'var(--accent-red)', backgroundColor: 'rgba(239, 68, 68, 0.1)', display: 'flex', gap: '1rem', alignItems: 'center'}}>
              <Zap size={32} color="var(--accent-red)" />
              <div>
                <h2 style={{color: 'var(--accent-red)', margin: '0 0 0.5rem 0'}}>Analysis Error</h2>
                <p style={{margin: 0}}>{error}</p>
              </div>
            </div>
          )}
          
          {loading && (
            <div className="loader-container">
              <div className="spinner"></div>
              <h3>{loadingStatus || "Running Pipeline..."}</h3>
              <p style={{color: 'var(--text-muted)'}}>Processing light curve, detrending, and applying AI models.</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="empty-state">
              <Activity />
              <h2>No Data Loaded</h2>
              <p>Select a Demo Mode target or upload a CSV to begin analysis.</p>
            </div>
          )}

          {!loading && result && (
            <>
              <div className="classification-banner">
                <div>
                  <span style={{color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '1px'}}>Target Classification</span>
                  <h2 className="classification-label">
                    {result.prediction.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </h2>
                  <p style={{margin: '0.5rem 0 0 0', opacity: 0.8}}>{result.target_name} {result.is_mock && '(Mock Data)'}</p>
                </div>
                <div style={{textAlign: 'right'}}>
                  <div className="confidence-badge" style={{
                    backgroundColor: `${getConfidenceLevel(result.confidence).color}33`, 
                    color: getConfidenceLevel(result.confidence).color, 
                    border: `1px solid ${getConfidenceLevel(result.confidence).color}`
                  }}>
                    {getConfidenceLevel(result.confidence).level} Confidence ({(result.confidence * 100).toFixed(1)}%)
                  </div>
                </div>
              </div>

              <div className="summary-card" style={{backgroundColor: 'rgba(255,255,255,0.05)', padding: '1.25rem', borderRadius: '8px', borderLeft: '4px solid var(--accent-blue)', display: 'flex', gap: '1rem', alignItems: 'center'}}>
                <div><Activity size={32} color="var(--accent-blue)" /></div>
                <div>
                  <h3 style={{margin: '0 0 0.25rem 0', fontSize: '1.1rem'}}>Scientific Summary</h3>
                  <p style={{margin: 0, lineHeight: 1.5, color: 'var(--text-main)'}}>
                    {result.prediction === 'exoplanet_transit' ? 'Transit signal' : 'Periodic signal'} detected with period <strong>{Number(result.parameters.period || 0).toFixed(2)} days</strong> and depth <strong>{Number(depth || 0).toFixed(2)}%</strong>. 
                    Random Forest classified the candidate as <strong>{result.prediction.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong> with <strong>{(result.confidence * 100).toFixed(1)}%</strong> confidence.
                    {result.prediction === 'exoplanet_transit' && radius > 0 && ` Estimated planetary radius is ${Number(radius || 0).toFixed(2)} Earth radii.`}
                  </p>
                </div>
              </div>

              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Orbital Period</div>
                  <div className="metric-value">{Number(result.parameters.period || 0).toFixed(2)} d</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Transit Depth</div>
                  <div className="metric-value">{Number(depth || 0).toFixed(2)}%</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Duration</div>
                  <div className="metric-value">{Number(duration || 0).toFixed(2)} hrs</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Signal-to-Noise</div>
                  <div className="metric-value">{Number(result.parameters.snr || 0).toFixed(2)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Est. Planet Radius</div>
                  <div className="metric-value">{Number(radius || 0).toFixed(2)} R⊕</div>
                </div>
              </div>

              <div className="charts-grid">
                <div className="chart-container">
                  <Plot
                    data={[{ x: result.data.time, y: result.data.flux, type: 'scatter', mode: 'markers', marker: {size: 2, color: '#94a3b8'}, name: 'Raw' }]}
                    layout={{ title: 'Raw Light Curve', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#fff'}, margin: {t: 40, l: 40, r: 20, b: 40}, xaxis: {title: 'Time (days)'}, yaxis: {title: 'Relative Flux'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot
                    data={[
                      { x: result.data.time_clean, y: result.data.flux_clean, type: 'scatter', mode: 'markers', marker: {size: 2, color: '#3b82f6'}, name: 'Clean' },
                      { x: result.data.time_clean, y: result.data.trend_flux, type: 'scatter', mode: 'lines', line: {color: '#ef4444', width: 2}, name: 'Trend' }
                    ]}
                    layout={{ title: 'Cleaned & Detrended', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#fff'}, margin: {t: 40, l: 40, r: 20, b: 40}, xaxis: {title: 'Time (days)'}, yaxis: {title: 'Relative Flux'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot
                    data={[{ x: result.data.bls_periods, y: result.data.bls_powers, type: 'scatter', mode: 'lines', line: {color: '#8b5cf6'}, name: 'Power' }]}
                    layout={{ title: 'BLS Periodogram', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#fff'}, margin: {t: 40, l: 40, r: 20, b: 40}, xaxis: {title: 'Period (days)'}, yaxis: {title: 'Power'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot
                    data={[
                      { x: result.data.folded_phase, y: result.data.folded_flux, type: 'scatter', mode: 'markers', marker: {size: 2, color: '#94a3b8'}, name: 'Folded' },
                      { x: result.data.bin_centers, y: result.data.bin_flux, type: 'scatter', mode: 'lines', line: {color: '#10b981', width: 3}, name: 'Binned' }
                    ]}
                    layout={{ title: 'Phase Folded Transit', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#fff'}, margin: {t: 40, l: 40, r: 20, b: 40}, xaxis: {title: 'Phase'}, yaxis: {title: 'Relative Flux'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
              </div>

              <div className="card explainable-ai">
                <h2>Explainable AI (XAI)</h2>
                <p style={{color: 'var(--text-muted)'}}>
                  Detected as <strong>{result.prediction.replace('_', ' ')}</strong> because of the following top contributing features:
                </p>
                <ul className="explainable-list">
                  {getExplanation().map((exp: string, idx: number) => (
                    <li key={idx}>
                      <span style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                        <CheckCircle size={16} color="var(--accent-green)" /> {exp}
                      </span>
                      <span style={{color: 'var(--text-muted)', fontSize: '0.85rem'}}>
                        Importance: {((result.feature_importance_summary.top_random_forest?.[idx]?.importance ?? 0) * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
