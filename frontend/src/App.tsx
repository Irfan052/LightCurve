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

import { analyzeTarget, uploadAndAnalyze, generateReport, type AnalysisResult, type CatalogResult } from './api';

function App() {
  const [targetId, setTargetId] = useState('TIC 9991');
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [catalogResult, setCatalogResult] = useState<CatalogResult | null>(null);
  const [autoAnalyzeProgress, setAutoAnalyzeProgress] = useState<{active: boolean, currentIdx: number, total: number, currentId: string} | null>(null);
  const [exhaustedStats, setExhaustedStats] = useState<{found: number, checked: number, downloaded: number, skipped: number} | null>(null);
  const [error, setError] = useState('');
  const [runtime, setRuntime] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  const handleAnalyze = async (id: string) => {
    if (!id.trim()) {
      setError("Target ID cannot be empty. Please enter a valid TIC ID.");
      return;
    }
    setTargetId(id);
    setLoading(true);
    setExhaustedStats(null);
    setError('');
    const start = Date.now();
    try {
      const data = await analyzeTarget(id);
      setResult(data);
      setCatalogResult(null);
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

  const handleAnalyzeCatalog = async () => {
    if (!catalogResult || catalogResult.objects.length === 0) {
        return;
    }
    
    setExhaustedStats(null);
    
    const validObjects = catalogResult.objects.filter(obj => obj.id && !obj.id.toString().startsWith("GAIA"));
    const hasGaia = catalogResult.objects.some(obj => obj.id && obj.id.toString().startsWith("GAIA"));
    
    if (validObjects.length === 0) {
      if (hasGaia) {
        setError("This catalog contains Gaia identifiers only. AstroExo currently performs automatic analysis using TIC identifiers because TESS light curves are indexed by TIC. Please upload a TIC catalog or convert Gaia IDs to TIC IDs.");
      } else {
        setError("No valid TIC identifiers found in this catalog.");
      }
      return;
    }
    
    setAutoAnalyzeProgress({ active: true, currentIdx: 0, total: validObjects.length, currentId: validObjects[0].id.toString() });
    setLoading(true);
    setError('');
    const start = Date.now();
    
    let success = false;
    
    for (let i = 0; i < validObjects.length; i++) {
      const targetId = validObjects[i].id.toString();
      
      setAutoAnalyzeProgress({ active: true, currentIdx: i, total: validObjects.length, currentId: targetId });
      setTargetId(targetId);
      
      try {
        const data = await analyzeTarget(targetId);
        setResult(data);
        setCatalogResult(null);
        setRuntime(((Date.now() - start) / 1000).toFixed(2));
        success = true;
        break; 
      } catch (err: any) {
        if (err.code === 'ECONNABORTED' || err.message?.toLowerCase().includes('timeout')) {
          console.warn(`Target ${targetId} exceeded 90s timeout, skipping...`);
        } else {
          console.warn(`Failed to analyze ${targetId}`, err);
        }
      }
    }
    
    setAutoAnalyzeProgress(null);
    setLoading(false);
    
    if (!success) {
      setExhaustedStats({
        found: catalogResult.objects.length,
        checked: validObjects.length,
        downloaded: 0,
        skipped: catalogResult.objects.length
      });
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setLoading(true);
    setExhaustedStats(null);
    setError('');
    const start = Date.now();
    try {
      const data = await uploadAndAnalyze(file);
      if (data.type === "catalog") {
        setCatalogResult(data);
        setResult(null);
      } else {
        setResult(data as AnalysisResult);
        setCatalogResult(null);
      }
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

  const handleExportPDF = async () => {
    if (!result) return;
    
    try {
      const payload = {
        target_id: result.target_name,
        prediction: result.prediction,
        confidence: result.confidence,
        period: result.parameters?.period || 0,
        depth: result.parameters?.transit_depth_percent || 0,
        duration: result.parameters?.transit_duration_hours || 0,
        planet_radius: result.parameters?.planet_radius_earth || 0,
        snr: result.parameters?.snr || 0,
        probabilities: result.probabilities,
        features: result.feature_importance_summary?.top_random_forest?.map(f => f.feature) || []
      };
      
      const blob = await generateReport(payload as any);
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `astroexo-report-${result.target_name}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export PDF", err);
      alert("Failed to export PDF report from backend.");
    }
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
              <a href="/sample_lightcurve.csv" download="sample_lightcurve.csv" style={{fontSize: '0.85rem', color: 'var(--accent-blue)', textDecoration: 'none'}}>Download Sample CSV</a>
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
              <h3>
                {autoAnalyzeProgress 
                  ? `Testing Target ${autoAnalyzeProgress.currentIdx + 1} of ${autoAnalyzeProgress.total} (TIC ${autoAnalyzeProgress.currentId})` 
                  : (loadingStatus || "Running Pipeline...")}
              </h3>
              <p style={{color: 'var(--text-muted)'}}>
                {autoAnalyzeProgress 
                  ? "Searching MAST archive. Automatically skipping targets without TESS data..." 
                  : "Processing light curve, detrending, and applying AI models."}
              </p>
            </div>
          )}

          {!loading && !result && !catalogResult && !error && (
            <div className="empty-state">
              <Activity />
              <h2>No Data Loaded</h2>
              <p>Select a Demo Mode target or upload a CSV to begin analysis.</p>
            </div>
          )}

          {!loading && catalogResult && (
            <div className="catalog-import-view">
              <div className="classification-banner">
                <div>
                  <span style={{color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '1px'}}>Catalog Import</span>
                  <h2 className="classification-label">Select Target to Analyze</h2>
                  <p style={{margin: '0.5rem 0 0 0', opacity: 0.8}}>Found {catalogResult.objects.length} objects in uploaded catalog.</p>
                </div>
                <div style={{textAlign: 'right'}}>
                  {!exhaustedStats && (
                    <button className="btn btn-primary" onClick={handleAnalyzeCatalog}>
                      <Activity size={18} style={{marginRight: '8px'}} /> Analyze Entire Catalog
                    </button>
                  )}
                </div>
              </div>
              
              {exhaustedStats && (
                <div className="card" style={{borderColor: 'var(--accent-orange)', backgroundColor: 'rgba(245, 158, 11, 0.1)', marginTop: '2rem'}}>
                  <h2 style={{color: 'var(--accent-orange)', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0 0 1rem 0'}}>
                    <Zap size={24} /> No downloadable TESS light curves were found for any targets in this catalog.
                  </h2>
                  <div style={{display: 'flex', gap: '4rem', flexWrap: 'wrap'}}>
                    <div style={{flex: 1, minWidth: '250px'}}>
                      <h3 style={{margin: '0 0 0.5rem 0', color: 'var(--text-color)'}}>Analysis Summary</h3>
                      <ul style={{listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-muted)'}}>
                        <li><strong>Total targets found:</strong> {exhaustedStats.found}</li>
                        <li><strong>Targets checked:</strong> {exhaustedStats.checked}</li>
                        <li><strong>Targets with downloadable TESS data:</strong> {exhaustedStats.downloaded}</li>
                        <li><strong>Targets skipped (invalid/no data):</strong> {exhaustedStats.skipped}</li>
                      </ul>
                    </div>
                    <div style={{flex: 1, minWidth: '250px'}}>
                      <h3 style={{margin: '0 0 0.5rem 0', color: 'var(--text-color)'}}>Next Actions</h3>
                      <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
                        <button className="btn btn-secondary" style={{justifyContent: 'flex-start', width: '100%'}} onClick={() => fileInputRef.current?.click()}>
                          <UploadCloud size={16} style={{marginRight: '8px'}} /> Upload another catalog
                        </button>
                        <button className="btn btn-secondary" style={{justifyContent: 'flex-start', width: '100%'}} onClick={() => fileInputRef.current?.click()}>
                          <UploadCloud size={16} style={{marginRight: '8px'}} /> Upload a Light Curve CSV
                        </button>
                        <button className="btn btn-secondary" style={{justifyContent: 'flex-start', width: '100%'}} onClick={() => handleAnalyze('9991')}>
                          <Globe size={16} style={{marginRight: '8px'}} /> Run Demo Analysis
                        </button>
                        <a className="btn btn-secondary" style={{justifyContent: 'flex-start', width: '100%', textDecoration: 'none', display: 'flex', alignItems: 'center'}} href="/sample_lightcurve.csv" download="sample_lightcurve.csv">
                          <Download size={16} style={{marginRight: '8px'}} /> Download the sample dataset
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="card" style={{marginTop: '2rem', overflowX: 'auto'}}>
                <table className="catalog-table">
                  <thead>
                    <tr>
                      {(() => {
                        const preferred = ['tic_id', 'tic', 'toi', 'gaia_id', 'gaia', 'source_id', 'object', 'target', 'ra', 'dec', 'magnitude', 'sector'];
                        const norm = (c: string) => c.toLowerCase().replace(/[^a-z0-9_]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
                        
                        const preferredMap = preferred.reduce((acc, curr, idx) => { acc[curr] = idx; return acc; }, {} as Record<string, number>);
                        
                        // Deduplicate columns just in case
                        const uniqueCols = Array.from(new Set(catalogResult.columns));
                        
                        const orderedCols = uniqueCols
                          .filter((c: unknown) => {
                            const n = norm(String(c));
                            return !n.includes('unnamed') && !n.includes('index') && !n.match(/^column_\d+$/) && !n.match(/^\d+$/);
                          })
                          .sort((a: unknown, b: unknown) => {
                            const na = norm(String(a));
                            const nb = norm(String(b));
                            const valA = preferredMap[na] !== undefined ? preferredMap[na] : 999;
                            const valB = preferredMap[nb] !== undefined ? preferredMap[nb] : 999;
                            if (valA === valB) return na.localeCompare(nb);
                            return valA - valB;
                          });
                          
                        return orderedCols.map((col: unknown) => (
                          <th key={String(col)}>{String(col).toUpperCase()}</th>
                        ));
                      })()}
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {catalogResult.objects.map((obj: Record<string, any>, idx: number) => {
                      const preferred = ['tic_id', 'tic', 'toi', 'gaia_id', 'gaia', 'source_id', 'object', 'target', 'ra', 'dec', 'magnitude', 'sector'];
                      const norm = (c: string) => c.toLowerCase().replace(/[^a-z0-9_]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
                      const preferredMap = preferred.reduce((acc, curr, i) => { acc[curr] = i; return acc; }, {} as Record<string, number>);
                      
                      const uniqueCols = Array.from(new Set(catalogResult.columns));
                      
                      const orderedCols = uniqueCols
                        .filter((c: unknown) => {
                          const n = norm(String(c));
                          return !n.includes('unnamed') && !n.includes('index') && !n.match(/^column_\d+$/) && !n.match(/^\d+$/);
                        })
                        .sort((a: unknown, b: unknown) => {
                          const na = norm(String(a));
                          const nb = norm(String(b));
                          const valA = preferredMap[na] !== undefined ? preferredMap[na] : 999;
                          const valB = preferredMap[nb] !== undefined ? preferredMap[nb] : 999;
                          if (valA === valB) return na.localeCompare(nb);
                          return valA - valB;
                        });

                      return (
                        <tr key={idx}>
                          {orderedCols.map((col: string) => (
                            <td key={col}>{obj[col] !== null && obj[col] !== undefined ? obj[col] : 'N/A'}</td>
                          ))}
                          <td>
                            <button className="btn btn-sm btn-primary" onClick={() => {
                              if (!obj.id) {
                                alert("No valid identifier found for this row.");
                                return;
                              }
                              
                              if (obj.id.toString().startsWith("GAIA")) {
                                alert("This catalog contains Gaia identifiers only. Automatic Gaia→TIC cross-matching is not yet implemented.");
                                return;
                              }

                              handleAnalyze(obj.id);
                            }}>
                              Analyze
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
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
