import { useState, useRef, useEffect, type ChangeEvent, Component, type ReactNode, type ErrorInfo } from 'react';
import Plot from 'react-plotly.js';
import {
  Download,
  UploadCloud,
  Search,
  CheckCircle,
  AlertTriangle,
  Activity,
  Globe,
  Disc,
  Star,
  Zap
} from 'lucide-react';
import './App.css';

import { analyzeTarget, uploadAndAnalyze, generateReport, type AnalysisResult, type CatalogResult } from './api';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Frontend Crash Caught by ErrorBoundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '2rem', textAlign: 'center', fontFamily: 'sans-serif' }}>
          <AlertTriangle size={64} color="var(--accent-red)" style={{ marginBottom: '1rem' }} />
          <h1 style={{ color: 'var(--accent-red)' }}>Something went wrong.</h1>
          <p style={{ color: 'var(--text-color)' }}>The application encountered an unexpected error while rendering.</p>
          <pre style={{ textAlign: 'left', background: 'var(--bg-card)', padding: '1rem', overflowX: 'auto', borderRadius: '8px', color: 'var(--accent-orange)' }}>
            {this.state.error?.toString()}
          </pre>
          <button 
            className="btn btn-primary" 
            style={{ marginTop: '1rem' }}
            onClick={() => window.location.reload()}
          >
            Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [targetId, setTargetId] = useState('TIC 9991');
  const [loading, setLoading] = useState(false);
  const [loadingType, setLoadingType] = useState<'upload' | 'analyze' | 'pdf' | ''>('');
  const [loadingStatus, setLoadingStatus] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [catalogResult, setCatalogResult] = useState<CatalogResult | null>(null);
  const [error, setError] = useState('');
  const [runtime, setRuntime] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  const handleAnalyze = async (id: string) => {
    console.log("\n--- FRONTEND ANALYZE INSTRUMENTATION ---");
    console.log("selectedTarget:", id);
    console.log("payload:", { target_id: String(id) });
    console.log("----------------------------------------\n");
    if (!id.trim()) {
      setError("Target ID cannot be empty. Please enter a valid TIC ID.");
      return;
    }
    
    // Immediate UI Reset
    setResult(null);
    setCatalogResult(null);
    setLoadingStatus('');
    setError('');
    setRuntime('');

    setTargetId(id);
    setLoadingType('analyze');
    setLoading(true);
    const start = Date.now();
    try {
      const data = await analyzeTarget(id);
      setResult(data);
      setCatalogResult(null);
      setRuntime(((Date.now() - start) / 1000).toFixed(2));
    } catch (err) {
      setResult(null);
      const error = err as { message?: string; response?: { data?: { detail?: string } } };
      const detail = error.response?.data?.detail || '';
      if (error.message === "Network Error") {
        setError("Backend API is currently unavailable. Please ensure the server is running.");
      } else {
        setError(detail || error.message || 'An unknown error occurred during analysis.');
      }
    } finally {
      setLoadingType('');
      setLoading(false);
    }
  };


  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Immediate UI Reset
    setResult(null);
    setCatalogResult(null);
    setTargetId('');
    setLoadingStatus('');
    setError('');
    setRuntime('');
    
    setLoadingType('upload');
    setLoading(true);
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
      setLoadingType('');
      setLoading(false);
      // Reset input so the same file can be uploaded again if needed
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const createPlaceholderImage = (text: string): string => {
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 400;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    }
    return canvas.toDataURL('image/jpeg', 0.8);
  };

  const handleExportPDF = async () => {
    if (!result) return;
    
    const isSuccessful = !result.status || result.status === "SUCCESS";
    const hasData = result.data && result.parameters;
    
    if (!isSuccessful || !hasData) {
      alert(
        "Unable to Generate PDF Report\n\n" +
        "A PDF report cannot be generated because the selected target does not have sufficient analysis results.\n\n" +
        "Possible reasons include:\n" +
        "• No compatible TESS light curve was found.\n" +
        "• The TIC ID is invalid or unavailable.\n" +
        "• The analysis could not complete successfully.\n" +
        "• Required plots, graphs, and scientific results were not generated.\n\n" +
        "Please analyze a valid target with successful results before exporting a PDF report."
      );
      return;
    }
    
    try {
      setLoadingType('pdf');
      setLoading(true);
      
      const capturePlot = async (id: string) => {
        await new Promise(r => setTimeout(r, 100)); // Yield to UI thread
        const el = document.getElementById(id);
        if (!el) return "";
        try {
          const dataUrl = await (window as any).Plotly.toImage(el, {format: 'jpeg', height: 400, width: 800});
          return dataUrl;
        } catch (e) {
          console.warn(`Failed to capture ${id}:`, e);
          return "";
        }
      };

      const captureCombinedPlot = async (id1: string, id2: string) => {
        const img1 = await capturePlot(id1);
        const img2 = await capturePlot(id2);
        
        if (!img1 && !img2) return createPlaceholderImage("Visualization unavailable");
        if (!img1) return img2;
        if (!img2) return img1;
        
        return new Promise<string>((resolve) => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          const image1 = new Image();
          const image2 = new Image();
          
          let loaded = 0;
          const onload = () => {
            loaded++;
            if (loaded === 2) {
              canvas.width = Math.max(image1.width, image2.width);
              canvas.height = image1.height + image2.height;
              if (ctx) {
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(image1, 0, 0);
                ctx.drawImage(image2, 0, image1.height);
              }
              resolve(canvas.toDataURL('image/jpeg', 0.9));
            }
          };
          image1.onload = onload;
          image2.onload = onload;
          image1.src = img1;
          image2.src = img2;
        });
      };

      const plot_raw = await captureCombinedPlot('plot-raw', 'plot-clean');
      const plot_detrended = await captureCombinedPlot('plot-detrended', 'plot-transit');
      const plot_bls = await capturePlot('plot-bls') || createPlaceholderImage("Visualization unavailable");
      const plot_folded = await capturePlot('plot-folded') || createPlaceholderImage("Visualization unavailable");
      const plot_model = await capturePlot('plot-model') || createPlaceholderImage("Visualization unavailable");

      const payload = {
        target_id: result.target_name,
        prediction: result.prediction,
        confidence: result.confidence,
        period: result.parameters?.period || 0,
        depth: result.parameters?.transit_depth_percent || 0,
        duration: result.parameters?.transit_duration_hours || 0,
        planet_radius: result.parameters?.planet_radius_earth || 0,
        semi_major_axis: result.parameters?.semi_major_axis_au || 0,
        snr: result.parameters?.snr || 0,
        probabilities: result.probabilities,
        is_mock: result.is_mock,
        
        xai_evidence: getPhase4Evidence(),
        xai_risks: getPhase4Risks(),
        xai_recommendation: getPhase4Recommendation(result.confidence, result.prediction),
        xai_interpretation: getPhase4ConfidenceInterpretation(result.confidence),
        feature_importance: result.feature_importance_summary?.top_random_forest || [],
        
        plot_raw,
        plot_detrended,
        plot_bls,
        plot_folded,
        plot_model
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
    } finally {
      setLoading(false);
      setLoadingType('');
    }
  };

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (loading) {
      let statuses: string[] = [];
      if (loadingType === 'upload') {
        statuses = [
          "Uploading catalog...",
          "Reading catalog...",
          "Detecting catalog type...",
          "Preparing preview..."
        ];
      } else if (loadingType === 'pdf') {
        statuses = [
          "Generating scientific report...",
          "Rendering charts...",
          "Preparing PDF...",
          "Downloading report..."
        ];
      } else {
        statuses = [
          "Searching MAST...",
          "Downloading light curve...",
          "Running detrending...",
          "Searching transit...",
          "Classifying candidate..."
        ];
      }
      
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
  }, [loading, loadingType]);

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

  const getPhase4ConfidenceInterpretation = (conf: number) => {
    if (conf >= 0.95) return "Very High Confidence";
    if (conf >= 0.85) return "High Confidence";
    if (conf >= 0.70) return "Moderate Confidence";
    return "Low Confidence";
  };

  const getPhase4Recommendation = (conf: number, pred: string) => {
    if (pred === 'stellar_variability' || pred === 'instrumental_artifact') {
       return "Signal insufficient for confirmation. Likely artifact or variability.";
    }
    if (conf >= 0.85) return "Candidate suitable for priority follow-up observations.";
    if (conf >= 0.70) return "Additional observations recommended to rule out false positives.";
    return "Signal insufficient for confirmation.";
  };

  const getPhase4Evidence = () => {
    if (!result) return [];
    const ev = [];
    if ((result.parameters?.snr || 0) > 15) ev.push("High Signal-to-Noise Ratio (Robust Detection)");
    if ((result.features?.u_shape_score || 0) > 0.6) ev.push("U-shaped transit profile consistent with planetary transit");
    if ((result.features?.secondary_ratio || 1) < 0.1) ev.push("Lack of significant secondary eclipse (supports planetary hypothesis)");
    if ((result.features?.odd_even_diff || 1) < 0.1) ev.push("Consistent depths between odd and even transits");
    if ((result.features?.out_of_transit_rms || 1) < 0.005) ev.push("Low photometric noise outside of transit");
    return ev;
  };

  const getPhase4Risks = () => {
    if (!result) return [];
    const risks = [];
    if ((result.parameters?.snr || 0) < 7) risks.push("Low Signal-to-Noise Ratio (Marginal Detection)");
    if ((result.features?.v_shape_score || 0) > 0.7 || (result.features?.u_shape_score || 1) < 0.3) risks.push("V-shaped transit profile (Potential grazing eclipsing binary)");
    if ((result.features?.odd_even_diff || 0) > 0.1) risks.push("High odd-even depth difference (Potential background eclipsing binary)");
    if ((result.features?.secondary_ratio || 0) > 0.1) risks.push("Significant secondary eclipse detected (Suggests stellar companion)");
    return risks;
  };

  const getFailureMessage = (status: string, defaultMsg: string) => {
    switch (status) {
      case 'NO_PUBLIC_DATA':
        return "No compatible public TESS observations were found for this target. The MAST archive does not currently provide downloadable light-curve products suitable for scientific analysis.";
      case 'UNSUPPORTED_PRODUCT':
        return "Public observations exist for this target. However, the available products use an unsupported format (for example TARS HLSP) that does not contain the scientific columns required by the AstroExo pipeline.";
      case 'DOWNLOAD_FAILED':
        return "The observation could not be downloaded successfully. Please try again later.";
      case 'DOWNLOAD_TIMEOUT':
        return "The MAST archive did not respond before the timeout period expired. Please retry later.";
      case 'CORRUPTED_FITS':
        return "The downloaded observation file could not be read successfully. The local cache has been cleared automatically. Please retry.";
      case 'INVALID_TIC':
        return "The supplied TIC identifier could not be validated. Please verify the target identifier.";
      default:
        return defaultMsg || "Data could not be processed.";
    }
  };

  return (
    <ErrorBoundary>
      <div className="dashboard-container">
      <header className="app-header">
        <h1><Globe style={{ display: 'inline', marginRight: '8px' }} /> AstroExo AI Detection Dashboard</h1>
        <button className="btn btn-secondary" onClick={handleExportPDF} disabled={!result || (loading && loadingType === 'pdf')} style={{width: 'auto'}}>
          <Download size={18} /> {loading && loadingType === 'pdf' ? 'Generating...' : 'Export PDF Report'}
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
            <h2>Sample Targets</h2>
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
            
            {result && result.status === "SUCCESS" && result.data && (
              <div className="runtime-stats" style={{marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.9rem'}}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Raw Observations:</span>
                  <strong>{result.data?.time?.length ?? 0}</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Observation Time:</span>
                  <strong>{((result.data?.time?.[(result.data?.time?.length ?? 1) - 1] ?? 0) - (result.data?.time?.[0] ?? 0)).toFixed(1)} days</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Cleaned Points:</span>
                  <strong>{result.data?.time_clean?.length ?? 0}</strong>
                </div>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                  <span style={{color: 'var(--text-muted)'}}>Data Quality:</span>
                  <strong style={{color: ((result.data?.time_clean?.length ?? 0) / (result.data?.time?.length ?? 1)) > 0.8 ? 'var(--accent-green)' : 'var(--accent-orange)'}}>
                    {(((result.data?.time_clean?.length ?? 0) / (result.data?.time?.length ?? 1)) * 100).toFixed(1)}%
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
          
          {result && result.status && result.status !== "SUCCESS" && (
            <div className="failure-dashboard" style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem'}}>
              
              <div className="card" style={{borderColor: 'var(--accent-orange)'}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
                  <AlertTriangle size={24} color="var(--accent-orange)" />
                  <h2 style={{color: 'var(--accent-orange)', margin: 0}}>Analysis Summary</h2>
                </div>
                
                <ul style={{listStyle: 'none', padding: 0, margin: '0 0 1.5rem 0'}}>
                  <li style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-green)'}}>
                    <CheckCircle size={16} /> TIC Validated
                  </li>
                  {result.status !== 'INVALID_TIC' && (
                    <li style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-green)'}}>
                      <CheckCircle size={16} /> MAST Search Completed
                    </li>
                  )}
                  <li style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-green)'}}>
                    <CheckCircle size={16} /> Products Checked
                  </li>
                  <li style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--accent-red)'}}>
                    <Zap size={16} /> No Compatible Light Curve
                  </li>
                </ul>

                <h3 style={{fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.5rem'}}>Skipped Pipeline Stages</h3>
                <ul style={{listStyle: 'none', padding: 0, margin: 0, color: 'var(--text-muted)', opacity: 0.8}}>
                  <li style={{marginBottom: '0.25rem'}}>• Quality Filtering</li>
                  <li style={{marginBottom: '0.25rem'}}>• Detrending</li>
                  <li style={{marginBottom: '0.25rem'}}>• Transit Detection</li>
                  <li style={{marginBottom: '0.25rem'}}>• Feature Extraction</li>
                  <li style={{marginBottom: '0.25rem'}}>• AI Classification</li>
                  <li>• Report Generation</li>
                </ul>
              </div>

              <div className="card">
                <h2 style={{margin: '0 0 1rem 0'}}>Metadata</h2>
                <ul className="info-list">
                  <li><span className="label">Target TIC</span><span className="value">{result.target_name?.replace('TIC ', '') || result.target?.replace('TIC ', '') || '—'}</span></li>
                  <li style={{flexDirection: 'column', alignItems: 'flex-start'}}><span className="label" style={{marginBottom: '0.25rem'}}>Reason</span>
                    <span className="value" style={{color: 'var(--accent-orange)', lineHeight: 1.4, textAlign: 'left'}}>
                      {getFailureMessage(result.status, result.message || '')}
                    </span>
                  </li>
                  <li><span className="label">Products Found</span><span className="value">{(result.metadata as any)?.products_found ?? '—'}</span></li>
                  <li><span className="label">Mission</span><span className="value">{result.metadata?.mission || '—'}</span></li>
                  <li><span className="label">Author</span><span className="value">{result.metadata?.author || '—'}</span></li>
                  <li><span className="label">Collection</span><span className="value">{result.metadata?.collection || '—'}</span></li>
                  <li><span className="label">Pipeline Runtime</span><span className="value">{(result.metadata as any)?.pipeline_runtime ? `${(result.metadata as any).pipeline_runtime}s` : '—'}</span></li>
                </ul>
                <div style={{marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem', display: 'flex', justifyContent: 'flex-end'}}>
                  <button className="btn btn-secondary" onClick={() => handleAnalyze('9991')}>
                    Analyze Synthetic Demo Data
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {loading && loadingType !== 'pdf' && (
            <div className="loader-container" style={{
              position: 'fixed',
              top: 'calc(50% + 45px)',
              left: 'calc(50% + 100px)',
              transform: 'translate(-50%, -50%)',
              zIndex: 9999,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center'
            }}>
              <div className="spinner" style={{ marginBottom: '1.5rem' }}></div>
              <h3 style={{ margin: '0 0 0.5rem 0' }}>Processing Status</h3>
              {(() => {
                const currentStatuses = loadingType === 'upload' ? [
                  "Uploading catalog...",
                  "Parsing CSV...",
                  "Extracting target metadata...",
                  "Detecting identifiers..."
                ] : [
                  "Searching MAST...",
                  "Downloading light curve...",
                  "Running detrending...",
                  "Searching transit...",
                  "Classifying candidate..."
                ];
                const currentIndex = currentStatuses.indexOf(loadingStatus);
                return (
                  <div style={{ textAlign: 'left', marginTop: '1rem' }}>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      {currentStatuses.map((step, idx) => {
                        let icon = "○";
                        let color = "var(--text-muted)";
                        let fontWeight = "normal";
                        
                        if (idx < currentIndex || (currentIndex === -1 && loadingStatus !== '')) {
                          icon = "✓";
                          color = "var(--accent-green)";
                        } else if (idx === currentIndex) {
                          icon = "⟳";
                          color = "var(--accent-blue)";
                          fontWeight = "bold";
                        }
                        
                        return (
                          <li key={step} style={{ color, fontWeight, display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.95rem' }}>
                            <span style={{ display: 'inline-block', width: '16px', textAlign: 'center' }}>{icon}</span> {step}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                );
              })()}
            </div>
          )}

          {!loading && !result && !catalogResult && !error && (
            <div className="empty-state" style={{
              position: 'fixed',
              top: 'calc(50% + 45px)',
              left: 'calc(50% + 100px)',
              transform: 'translate(-50%, -50%)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center'
            }}>
              <Activity />
              <h2>No Data Loaded</h2>
              <p>Select a Sample Target or upload a CSV to begin analysis.</p>
            </div>
          )}

          {!loading && catalogResult && (
            <div className="catalog-import-view">
              <div className="card" style={{ marginBottom: '2rem', backgroundColor: 'var(--surface-color)', border: '1px solid var(--border-color)' }}>
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0 0 1rem 0' }}>
                  <CheckCircle size={24} color="var(--accent-green)" /> Catalog Uploaded Successfully
                </h2>
                <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                  <div style={{ flex: 1, minWidth: '200px' }}>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-muted)' }}>
                      <li><strong>Catalog Type:</strong> {catalogResult.catalog_type ? catalogResult.catalog_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Unknown'}</li>
                      <li><strong>Total Rows:</strong> {catalogResult.total_rows || catalogResult.objects.length}</li>
                      <li><strong>Preview Rows:</strong> {catalogResult.preview_rows || catalogResult.objects.length}</li>
                    </ul>
                  </div>
                  <div style={{ flex: 1, minWidth: '200px' }}>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-muted)' }}>
                      <li><strong>Valid Targets:</strong> {catalogResult.valid_targets !== undefined ? catalogResult.valid_targets : catalogResult.objects.length}</li>
                      <li><strong>Skipped Rows:</strong> {catalogResult.skipped_rows !== undefined ? catalogResult.skipped_rows : 0}</li>
                      <li><strong style={{color: 'var(--accent-green)'}}>Ready for Analysis</strong></li>
                    </ul>
                  </div>
                </div>
              </div>
              
              <div className="card" style={{marginTop: '2rem', overflowX: 'auto'}}>
                {(() => {
                  const norm = (c: string) => c.toLowerCase().replace(/[^a-z0-9_]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
                  const uniqueCols = Array.from(new Set(catalogResult.columns)).map(String);
                  
                  const idKeywords = ['tic_id', 'tic', 'toi', 'gaia_id', 'gaia', 'source_id', 'object', 'target', 'id'];

                  const idCol = (() => {
                    for (const kw of idKeywords) {
                      const found = uniqueCols.find(c => norm(c) === kw);
                      if (found) return found;
                    }
                    for (const kw of idKeywords) {
                      const found = uniqueCols.find(c => norm(c).includes(kw));
                      if (found) return found;
                    }
                    return uniqueCols.length > 0 ? uniqueCols[0] : null;
                  })();

                  const previewCols = [idCol].filter(Boolean) as string[];

                  const getDisplayName = (colKey: string) => {
                    if (colKey === idCol) {
                      if (catalogResult.catalog_type === 'tic_catalog') return 'TIC ID';
                      if (catalogResult.catalog_type === 'gaia_catalog') return 'Gaia ID';
                      return 'Target ID';
                    }
                    return colKey.toUpperCase();
                  };

                  return (
                    <table className="catalog-table">
                      <thead>
                        <tr>
                          {previewCols.map((col: string) => (
                            <th key={col}>{getDisplayName(col)}</th>
                          ))}
                          <th>ACTION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {catalogResult.objects.map((obj: Record<string, any>, idx: number) => (
                          <tr key={idx}>
                            {previewCols.map((col: string) => (
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
                        ))}
                      </tbody>
                    </table>
                  );
                })()}
              </div>
            </div>
          )}

          {!(loading && loadingType !== 'pdf') && result && (!result.status || result.status === "SUCCESS") && (
            <>
              <div className="dashboard-grid">
                
                {/* Card 1: Classification & Summary */}
                <div className="dashboard-card" style={{gridColumn: '1 / -1', borderLeft: `4px solid ${getConfidenceLevel(result.confidence).color}`}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem'}}>
                    <div>
                      <span style={{color: 'var(--text-muted)', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '1px'}}>AI Classification</span>
                      <h2 className="classification-label" style={{marginTop: '0.25rem'}}>
                        {result.prediction.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h2>
                      <p style={{margin: '0.5rem 0 0 0', opacity: 0.8}}>{result.target_name} {result.is_mock && <span style={{color: 'var(--accent-orange)'}}>(Mock Data)</span>}</p>
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
                </div>

                {/* Card 2: Observation Metadata */}
                <div className="dashboard-card">
                  <div className="dashboard-card-header">
                    <Globe size={20} color="var(--accent-blue)" />
                    <h3>Observation Metadata</h3>
                  </div>
                  <ul className="info-list">
                    <li><span className="label">Target TIC</span><span className="value">{result.target_name?.replace('TIC ', '') || '—'}</span></li>
                    <li><span className="label">Mission</span><span className="value">{result.metadata?.mission || (result.is_mock ? 'Synthetic' : 'TESS')}</span></li>
                    <li><span className="label">Data Source</span><span className="value">{result.is_mock ? 'Synthetic Demo' : 'MAST Archive'}</span></li>
                    <li><span className="label">Author</span><span className="value">{result.metadata?.author || (result.is_mock ? 'Synthetic' : '—')}</span></li>
                    <li><span className="label">Sector</span><span className="value">{result.metadata?.sector || '—'}</span></li>
                    <li><span className="label">Cadence</span><span className="value">{result.metadata?.cadence ? `${result.metadata.cadence}s` : '—'}</span></li>
                    <li><span className="label">Pipeline Runtime</span><span className="value">{runtime}s</span></li>
                  </ul>
                </div>

                {/* Card 3: Scientific Parameters */}
                <div className="dashboard-card">
                  <div className="dashboard-card-header">
                    <Activity size={20} color="var(--accent-green)" />
                    <h3>Scientific Parameters</h3>
                  </div>
                  <ul className="info-list">
                    <li><span className="label">Orbital Period</span><span className="value">{Number(result.parameters?.period || 0).toFixed(4)} days</span></li>
                    <li><span className="label">Transit Depth</span><span className="value">{Number(depth || 0).toFixed(3)} %</span></li>
                    <li><span className="label">Transit Duration</span><span className="value">{Number(duration || 0).toFixed(2)} hours</span></li>
                    <li><span className="label">Planet Radius</span><span className="value">{radius > 0 ? `${Number(radius).toFixed(2)} Earth Radii` : '—'}</span></li>
                    <li><span className="label">Signal-to-Noise (SNR)</span><span className="value">{Number(result.parameters?.snr || 0).toFixed(2)}</span></li>
                    <li><span className="label">SDE (Detection Efficiency)</span><span className="value">—</span></li>
                    <li><span className="label">Semi-major Axis</span><span className="value">{(result.parameters?.semi_major_axis_au || 0) > 0 ? `${Number(result.parameters?.semi_major_axis_au).toFixed(4)} AU` : '—'}</span></li>
                  </ul>
                </div>

                {/* Card 4: Processing Timeline */}
                <div className="dashboard-card" style={{gridColumn: '1 / -1'}}>
                  <div className="dashboard-card-header">
                    <Zap size={20} color="var(--accent-purple)" />
                    <h3>Processing Timeline</h3>
                  </div>
                  <ul className="timeline-list" style={{display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'center'}}>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Upload</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Parse</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> MAST Search</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Download</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Detrending</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Transit Detection</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Feature Extraction</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> AI Classification</li>
                    <li className="completed"><CheckCircle size={16} color="var(--accent-green)" /> Report Generation</li>
                  </ul>
                </div>

              </div>

              <div className="charts-grid">
                <div className="chart-container">
                  <Plot divId="plot-raw"
                    data={[{ x: result.data?.time ?? [], y: result.data?.flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#64748b', opacity: 0.8}, name: 'Raw Observations' }]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '1. Raw Light Curve', font: {color: '#fff'}}, xaxis: {title: 'Time (BJD)', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Relative Flux', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot divId="plot-clean"
                    data={[
                      { x: result.data?.time ?? [], y: result.data?.flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#ef4444', opacity: 0.5}, name: 'Removed/Outliers' },
                      { x: result.data?.time_clean ?? [], y: result.data?.flux_clean ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#3b82f6'}, name: 'Retained Quality Data' }
                    ]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '2. Quality Filtered Light Curve', font: {color: '#fff'}}, xaxis: {title: 'Time (BJD)', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Relative Flux', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot divId="plot-detrended"
                    data={[
                      { x: result.data?.time_clean ?? [], y: result.data?.flat_flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#10b981'}, name: 'Detrended Flux' },
                      { x: result.data?.time_clean ?? [], y: result.data?.trend_flux ?? [], type: 'scatter', mode: 'lines', line: {color: '#f59e0b', width: 2}, name: 'Removed Trend' }
                    ]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '3. Detrended Light Curve', font: {color: '#fff'}}, xaxis: {title: 'Time (BJD)', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Normalized Flux', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot divId="plot-transit"
                    data={[
                      { x: result.data?.time_clean ?? [], y: result.data?.flat_flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#94a3b8', opacity: 0.7}, name: 'Detrended' }
                    ]}
                    layout={{ 
                      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 },
                      title: {text: '4. Transit Detection', font: {color: '#fff'}}, 
                      xaxis: {title: 'Time (BJD)', gridcolor: 'rgba(255,255,255,0.1)'}, 
                      yaxis: {title: 'Normalized Flux', gridcolor: 'rgba(255,255,255,0.1)'},
                      shapes: (result.parameters?.period || 0) > 0 ? Array.from({length: Math.ceil(((result.data?.time_clean?.[(result.data?.time_clean?.length ?? 1)-1] ?? 1) - (result.data?.time_clean?.[0] ?? 0))/(result.parameters?.period || 1)) + 2}).map((_, i) => {
                        const center = (result.parameters?.epoch || 0) + ((i - 1) * (result.parameters?.period || 0));
                        const half_dur = ((result.parameters?.transit_duration_hours || 0) / 24.0) / 2.0;
                        return {
                          type: 'rect',
                          xref: 'x', yref: 'paper',
                          x0: center - half_dur, y0: 0,
                          x1: center + half_dur, y1: 1,
                          fillcolor: 'rgba(139, 92, 246, 0.2)',
                          line: { width: 0 }
                        };
                      }).filter((s: any) => s.x0 >= (result.data?.time_clean?.[0] ?? 0) - (result.parameters?.period || 0) && s.x1 <= (result.data?.time_clean?.[(result.data?.time_clean?.length ?? 1)-1] ?? 0) + (result.parameters?.period || 0)) : []
                    }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot divId="plot-bls"
                    data={[
                      { x: result.data?.bls_periods ?? [], y: result.data?.bls_powers ?? [], type: 'scatter', mode: 'lines', line: {color: '#8b5cf6'}, name: 'BLS Power' },
                      { x: [result.parameters?.period ?? null], y: [Math.max(...(result.data?.bls_powers ?? [0]))], type: 'scatter', mode: 'markers', marker: {symbol: 'star', size: 12, color: '#f59e0b'}, name: 'Best Candidate' }
                    ]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '5. BLS Periodogram', font: {color: '#fff'}}, xaxis: {title: 'Period (days)', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Power', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container">
                  <Plot divId="plot-folded"
                    data={[
                      { x: result.data?.folded_phase ?? [], y: result.data?.folded_flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#94a3b8', opacity: 0.8}, name: 'Folded Observations' }
                    ]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '6. Phase Folded Light Curve', font: {color: '#fff'}}, xaxis: {title: 'Phase', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Relative Flux', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
                <div className="chart-container" style={{gridColumn: '1 / -1'}}>
                  <Plot divId="plot-model"
                    data={[
                      { x: result.data?.folded_phase ?? [], y: result.data?.folded_flux ?? [], type: 'scatter', mode: 'markers', marker: {size: 3, color: '#64748b', opacity: 0.3}, name: 'Data' },
                      { x: result.data?.bin_centers ?? [], y: result.data?.bin_flux ?? [], type: 'scatter', mode: 'lines+markers', line: {color: '#10b981', width: 3}, marker: {size: 6, color: '#10b981'}, name: 'Binned Model' }
                    ]}
                    layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#94a3b8'}, margin: {t: 40, l: 50, r: 20, b: 40}, showlegend: true, legend: { x: 1, xanchor: 'right', y: 1 }, title: {text: '7. Transit Model Overlay', font: {color: '#fff'}}, xaxis: {title: 'Phase', gridcolor: 'rgba(255,255,255,0.1)'}, yaxis: {title: 'Relative Flux', gridcolor: 'rgba(255,255,255,0.1)'} }}
                    useResizeHandler={true}
                    style={{width: '100%', height: '100%'}}
                  />
                </div>
              </div>

              <div className="card explainable-ai" style={{marginTop: '2rem', background: 'transparent', border: 'none', padding: 0}}>
                <h2><Globe size={24} style={{display: 'inline', verticalAlign: 'middle', marginRight: '8px', color: 'var(--accent-blue)'}}/> Phase 4: Scientific Interpretation & XAI</h2>
                
                <div className="xai-grid">
                  {/* Left Column: AI Decision & Evidence */}
                  <div style={{display: 'flex', flexDirection: 'column', gap: '1.5rem'}}>
                    <div className="xai-card">
                      <h3>AI Decision Summary</h3>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center'}}>
                        <span style={{fontSize: '1.1rem', fontWeight: 600}}>{result.prediction?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}</span>
                        <span className="confidence-badge" style={{background: getConfidenceLevel(result.confidence || 0).color, color: '#111'}}>
                          {getPhase4ConfidenceInterpretation(result.confidence || 0)} {((result.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="xai-summary-text">
                        The detected periodic signal exhibits an orbital period of {Number(result.parameters?.period || 0).toFixed(2)} days. 
                        The AI classified this as {result.prediction?.replace('_', ' ') || 'Unknown'} with {(result.confidence * 100).toFixed(1)}% confidence. 
                        The transit depth of {Number(result.parameters?.transit_depth_percent || 0).toFixed(2)}% and duration of {Number(result.parameters?.transit_duration_hours || 0).toFixed(1)} hours yield an SNR of {Number(result.parameters?.snr || 0).toFixed(1)}.
                      </div>
                      <div className="xai-recommendation">
                        {getPhase4Recommendation(result.confidence, result.prediction)}
                      </div>
                    </div>

                    <div className="xai-card">
                      <h3><CheckCircle size={20} color="var(--accent-green)"/> Supporting Scientific Evidence</h3>
                      {getPhase4Evidence().length > 0 ? (
                        <ul className="evidence-list">
                          {getPhase4Evidence().map((ev, idx) => (
                            <li key={idx}><CheckCircle size={16} color="var(--accent-green)" style={{flexShrink: 0, marginTop: '3px'}} /> {ev}</li>
                          ))}
                        </ul>
                      ) : (
                        <p style={{color: 'var(--text-muted)', fontStyle: 'italic', margin: 0}}>No strong supporting evidence found in derived parameters.</p>
                      )}
                    </div>
                  </div>

                  {/* Right Column: Risks & Feature Importance */}
                  <div style={{display: 'flex', flexDirection: 'column', gap: '1.5rem'}}>
                    <div className="xai-card" style={{borderColor: getPhase4Risks().length > 0 ? 'rgba(239, 68, 68, 0.3)' : 'var(--border-color)'}}>
                      <h3 style={{color: getPhase4Risks().length > 0 ? 'var(--accent-red)' : '#fff'}}>
                        <AlertTriangle size={20} color={getPhase4Risks().length > 0 ? 'var(--accent-red)' : 'var(--text-muted)'}/> Risk Indicators
                      </h3>
                      {getPhase4Risks().length > 0 ? (
                        <ul className="risk-list">
                          {getPhase4Risks().map((risk, idx) => (
                            <li key={idx}><AlertTriangle size={16} color="var(--accent-red)" style={{flexShrink: 0, marginTop: '3px'}} /> {risk}</li>
                          ))}
                        </ul>
                      ) : (
                        <p style={{color: 'var(--accent-green)', margin: 0}}>No significant scientific risk indicators detected.</p>
                      )}
                    </div>

                    <div className="xai-card">
                      <h3><Search size={20} color="var(--accent-purple)"/> Feature Importance (Random Forest)</h3>
                      {result.feature_importance_summary?.top_random_forest ? (
                        <ul className="explainable-list" style={{margin: 0}}>
                          {getExplanation().map((exp: string, idx: number) => (
                            <li key={idx}>
                              <span style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                                <CheckCircle size={16} color="var(--accent-purple)" /> {exp}
                              </span>
                              <span style={{color: 'var(--text-muted)', fontSize: '0.85rem'}}>
                                Weight: {((result.feature_importance_summary?.top_random_forest?.[idx]?.importance ?? 0) * 100).toFixed(1)}%
                              </span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p style={{color: 'var(--text-muted)', fontStyle: 'italic', margin: 0}}>Feature importance is unavailable for the current serialized model.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
    </ErrorBoundary>
  );
}

export default App;
