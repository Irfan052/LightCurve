# AstroExo: 5-Minute Hackathon Demo Script

## 1. Introduction (0:00 - 0:45)
**Speaker:** 
"Good morning, judges. We are presenting AstroExo—an AI-powered exoplanet detection platform designed to process vast amounts of TESS photometric data.
Currently, astronomers spend thousands of hours manually sifting through noisy light curves to find the microscopic dips in starlight caused by orbiting planets. 
Our solution automates this entire pipeline—from raw data ingestion to scientific validation—in less than three seconds."

## 2. Architecture & Pipeline (0:45 - 1:30)
**Speaker:** 
"AstroExo's architecture is fully decoupled. The backend relies on a high-performance Python FastAPI engine that performs Savitzky-Golay detrending and Box Least Squares (BLS) periodograms. Once the mathematical features are extracted, a Random Forest AI model classifies the signal. 
The results are then passed to our React/Plotly frontend, which instantly visualizes the physics behind the prediction."

## 3. Live Demonstration - Exoplanet Target (1:30 - 3:00)
**Speaker:** *(Clicks the 'Exoplanet (9991)' Demo Button)*
"Let's look at a live example. We just queried target 9991. 
Within milliseconds, AstroExo stripped the stellar noise, ran the BLS algorithm, and predicted this is an Exoplanet Candidate with 98% confidence.
Notice our 7-chart visualization suite. We don't hide the data. You can see the raw flux, the detrended signal, and here in Plot 4, the exact transit windows. Finally, in Plot 7, you see our binned transit model flawlessly overlaying the folded data."

## 4. Explainable AI (XAI) (3:00 - 4:00)
**Speaker:** *(Scrolls down to the Phase 4 XAI Panel)*
"But an AI prediction isn't enough for peer review; astronomers need to know *why*. 
This is our Explainable AI dashboard. It mathematically proves the prediction. Notice the Evidence panel highlights a high Signal-to-Noise Ratio and a U-shaped transit profile—both hallmarks of a true planet.
If we were to load a Binary Star system instead, the Risk panel would instantly light up red, warning us of alternating depth differences or a V-shaped grazing eclipse. We never fabricate data; every sentence here is directly mapped to the underlying physics."

## 5. Export & Conclusion (4:00 - 5:00)
**Speaker:** *(Clicks 'Export PDF Report')*
"Finally, when an astronomer confirms a target, they can generate a publication-ready PDF report with a single click. 
*(Opens downloaded PDF)*
As you can see, it compiles the executive summary, scientific parameters, XAI evidence, and embeds all the interactive charts into a highly structured format suitable for submission to the MAST archive.
AstroExo isn't just an AI model; it is a complete, explainable, and production-ready scientific workstation. Thank you."
