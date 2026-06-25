import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from backend.app import run_analysis_pipeline
import traceback

def test_pipeline(name, time, flux, flux_err):
    print(f"\n--- Testing: {name} ---")
    try:
        res = run_analysis_pipeline(time, flux, flux_err, name, True)
        print(f"SUCCESS: {res['prediction']} (Conf: {res['confidence']:.2f})")
    except Exception as e:
        print(f"FAILED: {type(e).__name__} - {str(e)}")

# Edge cases
# 1. Flat signal
time = np.linspace(0, 27, 1000)
flux = np.ones(1000)
flux_err = np.full(1000, 0.001)
test_pipeline("Flat signal", time, flux, flux_err)

# 2. Empty signal
test_pipeline("Empty signal", np.array([]), np.array([]), np.array([]))

# 3. NaN signal
flux_nan = np.ones(1000)
flux_nan[::2] = np.nan
test_pipeline("NaN signal", time, flux_nan, flux_err)

# 4. Random Gaussian noise
flux_noise = 1.0 + np.random.normal(0, 0.5, 1000)
test_pipeline("High noise", time, flux_noise, flux_err)

# 5. Perfect transit (short period)
flux_transit = np.ones(1000)
flux_transit[::50] = 0.9  # Deep transits every 50 points
test_pipeline("Perfect short transit", time, flux_transit, flux_err)

# 6. Invalid durations/depths (all zeros)
test_pipeline("Zero flux", time, np.zeros(1000), flux_err)

# 7. Extremely large values (Infs)
flux_inf = np.ones(1000)
flux_inf[10] = np.inf
test_pipeline("Inf values", time, flux_inf, flux_err)
