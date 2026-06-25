# Phase 5 - Final Hackathon Review & Pitch Guide

## 1. Project Audit Results

### 🔴 Critical Issues
*None detected.* The architecture has been systematically hardened. Real data network errors, invalid TIC IDs, empty CSV files, and NaN values are properly caught and displayed gracefully on the frontend. The Random Forest model correctly trains on startup if missing.

### 🟡 Medium Issues
- **First Boot Latency:** If `random_forest_model.pkl` is missing on a fresh clone, the backend will synchronously train the model during startup. This takes a few seconds but delays the server's readiness. (Ensure you run the backend once *before* presenting).
- **PDF Page Clipping:** Since we are using `html2pdf.js` to perfectly capture the Plotly charts, users on very narrow screens might see slight clipping if the layout wraps aggressively. (Present the demo fullscreen).

### 🟢 Nice-to-Have Improvements (Post-Hackathon)
- **Local Data Caching:** Cache MAST Lightkurve downloads locally in a `data/raw/` directory to prevent network latency on repeated TIC queries.
- **WebSocket Streaming:** Upgrade the frontend `loadingStatus` polling to WebSockets or SSE for true, real-time pipeline progress updates.
- **Global Light Curve Processing:** Add a batch-processing endpoint to run against thousands of CSVs overnight.

---

## 2. Deployment Readiness
**Verified clean startup procedure:**
- **Backend**: `pip install -r requirements.txt` followed by `uvicorn backend.app:app --reload`.
- **Frontend**: `cd frontend`, `npm install`, then `npm run dev`.
- **Dependencies**: Verified `package.json` contains `html2pdf.js`, `lucide-react`, and `react-plotly.js`. `requirements.txt` correctly contains `reportlab`, `fastapi`, `lightkurve`, and `scikit-learn`.

---

## 3. Presentation Readiness

### ⏱️ The 2-Minute Pitch
**Hook:** "NASA’s TESS mission generates millions of light curves every month, but confirming a true exoplanet among eclipsing binaries and stellar noise is a massive bottleneck that currently requires hundreds of hours of human vetting."
**Solution:** "Enter AstroExo. We’ve built a fully automated, end-to-end scientific pipeline that ingests raw photometric data, runs a physics-informed Box Least Squares transit search, and applies a Random Forest classifier to instantly validate candidates."
**Differentiator:** "Unlike black-box neural networks, AstroExo utilizes Explainable AI. Our dashboard doesn't just give a prediction—it tells astronomers *why* it made that prediction by ranking the exact physical features, like odd-even depth differences, that drove the decision."
**Impact:** "With AstroExo, researchers can instantly triage millions of targets, export presentation-ready PDF reports in a single click, and focus their telescope time entirely on high-confidence Earth-like planets."

### 🎤 Demo Talking Points (Guided Walkthrough)
1. **The Input:** "Here is our dashboard. We can upload a CSV, but today let's query the live MAST archive. I'll enter a real TESS ID for TOI-700."
2. **The Processing:** "As we hit search, the backend is fetching the raw flux data, removing instrumental noise, and executing a BLS search to find periodic dips."
3. **The Reveal:** "And there it is. The AI correctly identified this as an Exoplanet Transit with 95% confidence. Notice how we calculate the estimated planetary radius right here in the summary."
4. **The Explainability:** "If you scroll down to our XAI panel, you can see the Random Forest didn't just guess—it detected a symmetric transit shape with no secondary eclipse, proving it's not a binary star system."
5. **The Export:** "Finally, with one click, we can generate a perfectly formatted scientific PDF report for our research logs."

### ⚖️ Likely Judge Questions & Answers

**Q: How do you handle false positives like Eclipsing Binaries?**
*A: Our feature engineering layer specifically extracts the "odd-even depth difference" and "secondary eclipse ratio" from the phase-folded light curve. Eclipsing binaries typically have alternating transit depths, which our Random Forest heavily weighs to reject them.*

**Q: Why Random Forest over Deep Learning/CNNs?**
*A: Two reasons: Speed and Explainability. In astronomical vetting, astronomers need to trust the model. Random Forests provide Gini feature importance (XAI), allowing us to output human-readable explanations. CNNs operate on raw arrays and act as black boxes.*

**Q: What happens if the MAST API goes down?**
*A: The platform is built to be resilient. We gracefully catch network timeouts and allow researchers to drag-and-drop local CSV files directly into the dashboard for uninterrupted offline processing.*
