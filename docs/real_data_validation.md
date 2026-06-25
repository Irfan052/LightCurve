# Real Data Validation Plan

This document outlines the testing of the AstroExo AI Pipeline using real TESS observations from the MAST archive. By disabling mock fallbacks, the pipeline is now rigorously tested against real downloaded light curves.

## Recommended Validation Targets

The following 5 TICs have been selected as they are known to contain transit-like signals (confirmed exoplanets) and are ideal for validating the pipeline's detection and classification capabilities.

### 1. TIC 279741379 (TOI-700)
- **Target Description:** Multi-planet system including Earth-sized habitable zone planet.
- **Expected Classification:** Exoplanet Transit
- **Validation Checklist:**
  - [ ] TIC ID: `279741379`
  - [ ] Number of points: _______
  - [ ] Detected period: _______
  - [ ] Detected depth: _______
  - [ ] Classification: _______
  - [ ] Confidence: _______

### 2. TIC 28159019 (WASP-126)
- **Target Description:** Hot Jupiter with deep transits.
- **Expected Classification:** Exoplanet Transit
- **Validation Checklist:**
  - [ ] TIC ID: `28159019`
  - [ ] Number of points: _______
  - [ ] Detected period: _______
  - [ ] Detected depth: _______
  - [ ] Classification: _______
  - [ ] Confidence: _______

### 3. TIC 261136679 (Pi Mensae)
- **Target Description:** Bright star with a known short-period super-Earth (Pi Mensae c).
- **Expected Classification:** Exoplanet Transit
- **Validation Checklist:**
  - [ ] TIC ID: `261136679`
  - [ ] Number of points: _______
  - [ ] Detected period: _______
  - [ ] Detected depth: _______
  - [ ] Classification: _______
  - [ ] Confidence: _______

### 4. TIC 231663901 (WASP-18)
- **Target Description:** Massive hot Jupiter with a very short period (~0.94 days).
- **Expected Classification:** Exoplanet Transit
- **Validation Checklist:**
  - [ ] TIC ID: `231663901`
  - [ ] Number of points: _______
  - [ ] Detected period: _______
  - [ ] Detected depth: _______
  - [ ] Classification: _______
  - [ ] Confidence: _______

### 5. TIC 147977348 (LHS 1140)
- **Target Description:** Red dwarf with multiple rocky planets. 
- **Expected Classification:** Exoplanet Transit
- **Validation Checklist:**
  - [ ] TIC ID: `147977348`
  - [ ] Number of points: _______
  - [ ] Detected period: _______
  - [ ] Detected depth: _______
  - [ ] Classification: _______
  - [ ] Confidence: _______

## Error Handling Scenarios
Ensure the following edge cases correctly display error messages without crashing or falling back to mock data:
1. **Invalid TIC ID (e.g. TIC 999999999999):** Pipeline should fail with "No TESS data found..."
2. **Network Disconnect (Turn off Wi-Fi before analyzing):** Pipeline should fail with "Network failure while communicating with MAST API..."
