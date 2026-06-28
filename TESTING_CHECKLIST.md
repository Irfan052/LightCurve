# AstroExo: Final Pre-Submission Testing Checklist

## 1. Core Data Ingestion & Pipeline 
- [x] **Large File Uploads:** Uploading a large headerless CSV successfully parses in chunks without crashing the browser.
- [x] **MAST Fallback:** Uploading an invalid/empty TIC correctly halts and displays "Light curve not available" instead of infinite polling.
- [x] **Detrending Algorithm:** Validated that Savitzky-Golay accurately removes stellar variation without clipping deep transits.

## 2. Machine Learning & XAI
- [x] **Prediction Accuracy:** The Random Forest correctly separates known Exoplanet targets from Eclipsing Binaries.
- [x] **XAI Evidence Mappings:** "U-shaped" vs "V-shaped" thresholds correctly flag targets based on the morphological scores.
- [x] **No Fabrication:** Confirmed that fields missing from the backend (like Sector/Cadence) correctly report "Not Available" instead of hallucinated data.

## 3. Dashboard UI & Visualization
- [x] **Responsive Layout:** The CSS grid fluidly reflows from 2 columns to 1 column on mobile/small screens.
- [x] **Plotly Canvas Rendering:** All 7 interactive plots render cleanly without overlapping. Transit rectangles align perfectly with the BLS epoch predictions.
- [x] **Metadata Alignment:** The Scientific Parameters card correctly displays the Semi-major axis in AU.

## 4. PDF Reporting
- [x] **DOM Capture:** `html2canvas` successfully extracts Base64 PNGs of the 5 main charts.
- [x] **ReportLab Generation:** The FastAPI `/api/report` endpoint compiles the multipage PDF with the dark theme without raising encoding errors.
- [x] **Data Integrity:** The exported PDF perfectly mirrors the XAI summaries generated on the frontend.

## 5. Repository Cleanup
- [x] **Removed Dead Code:** Unused legacy scripts like `visualize.py` have been deleted.
- [x] **Clean Logging:** The `startup_event` in `app.py` uses structured `logger.info()` instead of noisy `print()` statements.
- [x] **Unused Endpoints:** The temporary `/api/debug` endpoint has been removed.
