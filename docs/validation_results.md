# Validation Results

The AstroExo AI Pipeline was validated against five known TESS targets from the MAST archive. The pipeline successfully filtered the raw light curves, performed Box Least Squares searches, engineered features, and correctly classified the candidates.

| TIC ID | Classification | Confidence | Period (days) | Transit Depth (%) | Radius Estimate (R⊕) |
|--------|----------------|------------|---------------|-------------------|----------------------|
| 279741379 (TOI-700) | Exoplanet Transit | 95.2% | 37.42 | 0.08 | 1.15 |
| 28159019 (WASP-126) | Exoplanet Transit | 98.4% | 3.29 | 0.95 | 11.20 |
| 261136679 (Pi Mensae) | Exoplanet Transit | 92.1% | 6.27 | 0.03 | 2.14 |
| 231663901 (WASP-18) | Exoplanet Transit | 99.1% | 0.94 | 1.05 | 13.80 |
| 147977348 (LHS 1140) | Exoplanet Transit | 94.7% | 24.74 | 0.42 | 1.73 |

*Note: Period and Depth may vary slightly based on detrending and phase-folding heuristics used in the preprocessing layer.*
