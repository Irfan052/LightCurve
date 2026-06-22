const API_BASE_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
    // Cache DOM Elements
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const backendStatus = document.getElementById("backend-status");
    const presetButtons = document.querySelectorAll(".preset-btn");
    const targetInput = document.getElementById("target_id");
    const analyzeForm = document.getElementById("analyze-form");
    
    // Result views
    const loadingOverlay = document.getElementById("loading-overlay");
    const introCard = document.getElementById("intro-card");
    const resultsPanel = document.getElementById("results-panel");
    const mockBadge = document.getElementById("mock-badge");
    
    const resultTargetName = document.getElementById("result-target-name");
    const resultPrediction = document.getElementById("result-prediction");
    const resultConfidenceBar = document.getElementById("result-confidence-bar");
    const resultConfidenceText = document.getElementById("result-confidence-text");
    
    // Parameter values
    const paramPeriod = document.getElementById("param-period");
    const paramDepth = document.getElementById("param-depth");
    const paramDuration = document.getElementById("param-duration");
    const paramRadius = document.getElementById("param-radius");
    const paramAxis = document.getElementById("param-axis");
    const paramSnr = document.getElementById("param-snr");
    
    // Images
    const imgRaw = document.getElementById("img-raw");
    const imgDetrended = document.getElementById("img-detrended");
    const imgFolded = document.getElementById("img-folded");
    const imgBls = document.getElementById("img-bls");
    
    // File Upload drop zone
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");

    // 1. Check Backend Health
    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/health`);
            const data = await response.json();
            if (data.status === "healthy") {
                backendStatus.className = "status-indicator online";
                backendStatus.innerHTML = `<i class="fa-solid fa-circle"></i> Pipeline Server Online`;
            } else {
                throw new Error("unhealthy");
            }
        } catch (e) {
            backendStatus.className = "status-indicator offline";
            backendStatus.innerHTML = `<i class="fa-solid fa-circle"></i> Pipeline Server Offline`;
        }
    }
    
    checkHealth();
    // Periodically poll backend status
    setInterval(checkHealth, 10000);

    // 2. Tab Navigation
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(t => t.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // 3. Preset Clicks
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            targetInput.value = targetId;
            submitAnalysis(targetId);
        });
    });

    // 4. Form Submit
    analyzeForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const targetId = targetInput.value.trim();
        if (targetId) {
            submitAnalysis(targetId);
        }
    });

    // 5. Submit Search Target ID
    async function submitAnalysis(targetId) {
        // Toggle view states
        introCard.classList.add("hidden");
        resultsPanel.classList.add("hidden");
        loadingOverlay.classList.remove("hidden");
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/analyze`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ target_id: targetId })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server analysis error.");
            }
            
            const result = await response.json();
            renderResults(result);
            
        } catch (err) {
            alert(`Error: ${err.message}`);
            loadingOverlay.classList.add("hidden");
            introCard.classList.remove("hidden");
        }
    }

    // 6. Handle File Upload
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });
    
    async function handleUploadedFile(file) {
        if (!file.name.endsWith(".csv")) {
            alert("Please upload a CSV file containing time and flux columns.");
            return;
        }
        
        fileInfo.innerText = `Selected: ${file.name}`;
        
        // Toggle UI
        introCard.classList.add("hidden");
        resultsPanel.classList.add("hidden");
        loadingOverlay.classList.remove("hidden");
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/upload`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "File processing failed.");
            }
            
            const result = await response.json();
            renderResults(result);
            
        } catch (err) {
            alert(`Upload Error: ${err.message}`);
            loadingOverlay.classList.add("hidden");
            introCard.classList.remove("hidden");
            fileInfo.innerText = "CSV format (columns: time, flux)";
        }
    }

    // 7. Render JSON Response to UI
    function renderResults(data) {
        // Target Header
        resultTargetName.innerText = data.target_name;
        if (data.is_mock) {
            mockBadge.classList.remove("hidden");
        } else {
            mockBadge.classList.add("hidden");
        }
        
        // AI Classification text and class color mapping
        let displayLabel = data.prediction.replace(/_/g, " ");
        resultPrediction.innerText = displayLabel;
        
        // Clear old classes
        resultPrediction.className = "classification-label";
        
        // Apply class
        if (data.prediction === "exoplanet_transit") {
            resultPrediction.classList.add("planet");
        } else if (data.prediction === "eclipsing_binary") {
            resultPrediction.classList.add("binary");
        } else if (data.prediction === "stellar_variability") {
            resultPrediction.classList.add("stellar");
        } else {
            resultPrediction.classList.add("artifact");
        }
        
        // Confidence bar & text
        const confPercent = (data.confidence * 100).toFixed(1);
        resultConfidenceBar.style.width = `${confPercent}%`;
        resultConfidenceText.innerText = `${confPercent}%`;
        
        // Set transit parameter values
        paramPeriod.innerHTML = `${data.parameters.period.toFixed(4)} <span class="unit">days</span>`;
        paramDepth.innerHTML = `${data.parameters.transit_depth_percent.toFixed(3)} <span class="unit">%</span>`;
        paramDuration.innerHTML = `${data.parameters.transit_duration_hours.toFixed(1)} <span class="unit">hours</span>`;
        
        if (data.parameters.planet_radius_earth > 0) {
            paramRadius.innerHTML = `${data.parameters.planet_radius_earth.toFixed(2)} <span class="unit">R<sub>⊕</sub></span>`;
        } else {
            paramRadius.innerHTML = `N/A`;
        }
        
        if (data.parameters.semi_major_axis_au > 0) {
            paramAxis.innerHTML = `${data.parameters.semi_major_axis_au.toFixed(4)} <span class="unit">AU</span>`;
        } else {
            paramAxis.innerHTML = `N/A`;
        }
        
        paramSnr.innerHTML = `${data.parameters.snr.toFixed(1)}`;
        
        // Load Base64 Plot Images
        imgRaw.src = `data:image/png;base64,${data.plots.raw}`;
        imgDetrended.src = `data:image/png;base64,${data.plots.detrended}`;
        imgFolded.src = `data:image/png;base64,${data.plots.folded}`;
        
        if (data.plots.bls) {
            imgBls.src = `data:image/png;base64,${data.plots.bls}`;
            imgBls.parentElement.parentElement.classList.remove("hidden");
        } else {
            imgBls.src = "";
            imgBls.parentElement.parentElement.classList.add("hidden");
        }
        
        // Reveal results container
        loadingOverlay.classList.add("hidden");
        resultsPanel.classList.remove("hidden");
        
        // Switch to Dashboard Tab just in case
        document.querySelector('[data-tab="dashboard"]').click();
    }
});
