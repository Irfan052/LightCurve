import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from backend.config import DETREND_POLYORDER, DETREND_WINDOW_DAYS
from backend.utils import logger


DEFAULT_GAP_THRESHOLD_DAYS = 0.5
DEFAULT_TRANSIT_SIGMA = 3.5
MAX_MASKING_ITERATIONS = 3


class DetrendedLightCurve(tuple):
    """
    Backward-compatible 2-tuple return value with attached detrending metrics.
    """

    def __new__(
        cls,
        flat_flux: np.ndarray,
        trend_flux: np.ndarray,
        metrics: Optional[Dict[str, Any]] = None
    ):
        return super(DetrendedLightCurve, cls).__new__(cls, (flat_flux, trend_flux))

    def __init__(
        self,
        flat_flux: np.ndarray,
        trend_flux: np.ndarray,
        metrics: Optional[Dict[str, Any]] = None
    ):
        self.metrics = metrics if metrics is not None else {}


def _safe_median(values: np.ndarray, fallback: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.median(finite))


def _safe_std(values: np.ndarray, fallback: float = 1e-6) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    std = float(np.std(finite))
    return std if std > 0 else fallback


def _estimate_cadence_days(time: np.ndarray) -> float:
    if len(time) < 2:
        return np.nan

    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan
    return float(np.median(diffs))


def _is_tess_like_cadence(cadence_days: float) -> bool:
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        return False

    cadence_minutes = cadence_days * 24.0 * 60.0
    return any(abs(cadence_minutes - candidate) <= 1.5 for candidate in (2.0, 10.0, 20.0, 30.0))


def _detect_gap_indices(time: np.ndarray, tess_aware: bool = True) -> Tuple[List[int], float]:
    if len(time) < 2:
        return [], DEFAULT_GAP_THRESHOLD_DAYS

    cadence_days = _estimate_cadence_days(time)
    threshold = DEFAULT_GAP_THRESHOLD_DAYS

    if np.isfinite(cadence_days) and cadence_days > 0:
        threshold = max(threshold, 8.0 * cadence_days)
        if tess_aware and _is_tess_like_cadence(cadence_days):
            threshold = max(threshold, 12.0 * cadence_days)

    diffs = np.diff(time)
    gap_indices = np.where(np.isfinite(diffs) & (diffs > threshold))[0]
    return gap_indices.astype(int).tolist(), float(threshold)


def _segment_bounds(length: int, gap_indices: List[int]) -> List[Tuple[int, int]]:
    if length == 0:
        return []

    boundaries = [0]
    boundaries.extend(index + 1 for index in gap_indices)
    boundaries.append(length)

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            segments.append((start, end))
    return segments


def _resolve_window_length(
    segment_time: np.ndarray,
    requested_window_days: Optional[float],
    polyorder: int,
    tess_aware: bool = True
) -> Tuple[int, float]:
    segment_length = len(segment_time)
    if segment_length <= polyorder + 2:
        return segment_length, float("nan")

    cadence_days = _estimate_cadence_days(segment_time)
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        cadence_days = max((segment_time[-1] - segment_time[0]) / max(segment_length - 1, 1), 1e-4)

    span_days = max(float(segment_time[-1] - segment_time[0]), cadence_days)
    base_window_days = requested_window_days if requested_window_days and requested_window_days > 0 else DETREND_WINDOW_DAYS

    if tess_aware and _is_tess_like_cadence(cadence_days):
        cadence_minutes = cadence_days * 24.0 * 60.0
        if cadence_minutes <= 3.0:
            base_window_days = max(base_window_days, 0.30)
        elif cadence_minutes <= 12.0:
            base_window_days = max(base_window_days, 0.40)
        else:
            base_window_days = max(base_window_days, 0.60)

    min_window_days = max(7.0 * cadence_days, 0.08)
    max_window_days = max(min(span_days * 0.45, 3.5), min_window_days)
    window_days_used = float(np.clip(base_window_days, min_window_days, max_window_days))

    window_length = int(round(window_days_used / cadence_days))
    minimum_points = max(polyorder + 3, 5)
    window_length = max(window_length, minimum_points)
    window_length = min(window_length, segment_length if segment_length % 2 == 1 else segment_length - 1)

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1

    if window_length >= segment_length:
        window_length = segment_length - 1 if segment_length % 2 == 0 else segment_length

    if window_length % 2 == 0:
        window_length -= 1

    return max(window_length, polyorder + 3), window_days_used


def _interpolate_masked_flux(flux: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    if np.all(fit_mask):
        return flux.copy()

    valid_idx = np.where(fit_mask & np.isfinite(flux))[0]
    if valid_idx.size < 2:
        median_flux = _safe_median(flux, fallback=1.0)
        return np.full_like(flux, median_flux, dtype=float)

    interp_idx = np.arange(len(flux))
    return np.interp(interp_idx, valid_idx, flux[valid_idx]).astype(float)


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask

    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded[:-shift] |= mask[shift:]
        expanded[shift:] |= mask[:-shift]
    return expanded


def _fit_segment_trend(
    segment_time: np.ndarray,
    segment_flux: np.ndarray,
    requested_window_days: Optional[float],
    polyorder: int,
    tess_aware: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    segment_size = len(segment_flux)
    window_length, window_days_used = _resolve_window_length(
        segment_time,
        requested_window_days,
        polyorder,
        tess_aware=tess_aware
    )

    if segment_size < max(polyorder + 3, 7) or window_length <= polyorder:
        baseline = _safe_median(segment_flux, fallback=1.0)
        trend = np.full_like(segment_flux, baseline, dtype=float)
        return trend, {
            "window_length_points": int(min(segment_size, max(window_length, 1))),
            "window_days_used": window_days_used,
            "transit_masked_points": 0,
            "fit_iterations": 0,
            "fallback_used": True
        }

    fit_mask = np.isfinite(segment_flux)
    transit_mask = np.zeros(segment_size, dtype=bool)
    trend = np.full_like(segment_flux, _safe_median(segment_flux, fallback=1.0), dtype=float)
    fit_iterations = 0

    try:
        for fit_iterations in range(1, MAX_MASKING_ITERATIONS + 1):
            working_flux = _interpolate_masked_flux(segment_flux, fit_mask)
            trend = savgol_filter(
                working_flux,
                window_length=window_length,
                polyorder=polyorder,
                mode="interp"
            )

            safe_trend = np.where(np.abs(trend) > 1e-8, trend, _safe_median(trend, fallback=1.0))
            residual = (segment_flux / safe_trend) - 1.0
            residual[~np.isfinite(residual)] = 0.0

            median_residual = _safe_median(residual, fallback=0.0)
            mad = _safe_median(np.abs(residual - median_residual), fallback=0.0)
            sigma = 1.4826 * mad if mad > 0 else _safe_std(residual, fallback=1e-4)
            threshold = median_residual - (DEFAULT_TRANSIT_SIGMA * sigma)

            new_transit_mask = residual < threshold
            dilation_radius = max(1, min(5, window_length // 25))
            new_transit_mask = _expand_mask(new_transit_mask, dilation_radius)

            candidate_fit_mask = np.isfinite(segment_flux) & ~new_transit_mask
            if candidate_fit_mask.sum() < max(polyorder + 3, 5):
                break

            if np.array_equal(new_transit_mask, transit_mask):
                fit_mask = candidate_fit_mask
                break

            transit_mask = new_transit_mask
            fit_mask = candidate_fit_mask

        working_flux = _interpolate_masked_flux(segment_flux, fit_mask)
        trend = savgol_filter(
            working_flux,
            window_length=window_length,
            polyorder=polyorder,
            mode="interp"
        )
        trend = np.where(np.isfinite(trend) & (np.abs(trend) > 1e-8), trend, _safe_median(working_flux, fallback=1.0))

        return trend, {
            "window_length_points": int(window_length),
            "window_days_used": window_days_used,
            "transit_masked_points": int(np.sum(transit_mask)),
            "fit_iterations": fit_iterations,
            "fallback_used": False
        }
    except Exception as exc:
        logger.warning(f"Segment detrending fallback activated: {exc}")
        baseline = _safe_median(segment_flux, fallback=1.0)
        trend = np.full_like(segment_flux, baseline, dtype=float)
        return trend, {
            "window_length_points": int(window_length),
            "window_days_used": window_days_used,
            "transit_masked_points": int(np.sum(transit_mask)),
            "fit_iterations": fit_iterations,
            "fallback_used": True
        }


def flatten_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = DETREND_WINDOW_DAYS,
    polyorder: int = DETREND_POLYORDER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens a light curve using gap-aware, orbit-segmented Savitzky-Golay detrending.

    The return value remains backward compatible with the legacy 2-tuple
    `(flat_flux, trend_flux)`, while richer run diagnostics are exposed through
    the `.metrics` attribute and `flatten_lightcurve.last_metrics`.
    """
    if len(time) != len(flux):
        raise ValueError(f"Array length mismatch: time ({len(time)}) vs flux ({len(flux)})")

    metrics: Dict[str, Any] = {
        "method": "orbit_segmented_savgol",
        "input_points": int(len(time)),
        "valid_points": int(np.sum(np.isfinite(time) & np.isfinite(flux))),
        "polyorder": int(polyorder),
        "window_days_requested": float(window_days),
        "median_cadence_days": None,
        "median_cadence_minutes": None,
        "tess_like_data": False,
        "gap_count": 0,
        "gap_indices": [],
        "gap_threshold_days": DEFAULT_GAP_THRESHOLD_DAYS,
        "segments_detected": 0,
        "segment_lengths": [],
        "segment_metrics": [],
        "window_points_used": [],
        "window_days_used": [],
        "transit_masked_points": 0,
        "transit_masked_fraction": 0.0,
        "fallback_segments": 0
    }

    if len(time) < 10:
        flat_flux = flux.copy()
        trend_flux = np.ones_like(flux, dtype=float)
        metrics["segments_detected"] = 1 if len(time) > 0 else 0
        metrics["segment_lengths"] = [int(len(time))] if len(time) > 0 else []
        metrics["fallback_segments"] = metrics["segments_detected"]
        metrics["reason"] = "insufficient_points"
        result = DetrendedLightCurve(flat_flux, trend_flux, metrics)
        flatten_lightcurve.last_metrics = metrics
        return result

    cadence_days = _estimate_cadence_days(time)
    if np.isfinite(cadence_days):
        metrics["median_cadence_days"] = float(cadence_days)
        metrics["median_cadence_minutes"] = float(cadence_days * 24.0 * 60.0)
        metrics["tess_like_data"] = _is_tess_like_cadence(cadence_days)

    gap_indices, gap_threshold_days = _detect_gap_indices(time, tess_aware=True)
    segment_bounds = _segment_bounds(len(time), gap_indices)

    metrics["gap_indices"] = gap_indices
    metrics["gap_count"] = len(gap_indices)
    metrics["gap_threshold_days"] = gap_threshold_days
    metrics["segments_detected"] = len(segment_bounds)
    metrics["segment_lengths"] = [int(end - start) for start, end in segment_bounds]

    trend_flux = np.ones_like(flux, dtype=float)
    for segment_number, (start, end) in enumerate(segment_bounds):
        segment_time = time[start:end]
        segment_flux = flux[start:end]
        trend_segment, segment_metrics = _fit_segment_trend(
            segment_time,
            segment_flux,
            requested_window_days=window_days,
            polyorder=polyorder,
            tess_aware=True
        )
        trend_flux[start:end] = trend_segment

        metrics["segment_metrics"].append({
            "segment_index": int(segment_number),
            "start_index": int(start),
            "end_index": int(end - 1),
            **segment_metrics
        })
        metrics["window_points_used"].append(int(segment_metrics["window_length_points"]))
        metrics["window_days_used"].append(segment_metrics["window_days_used"])
        metrics["transit_masked_points"] += int(segment_metrics["transit_masked_points"])
        metrics["fallback_segments"] += int(segment_metrics["fallback_used"])

    invalid_trend = ~np.isfinite(trend_flux) | (np.abs(trend_flux) <= 1e-8)
    if np.any(invalid_trend):
        fallback_level = _safe_median(flux, fallback=1.0)
        if abs(fallback_level) <= 1e-8:
            fallback_level = 1.0
        trend_flux[invalid_trend] = fallback_level

    flat_flux = flux / trend_flux
    invalid_flat = ~np.isfinite(flat_flux)
    if np.any(invalid_flat):
        flat_flux[invalid_flat] = 1.0

    metrics["transit_masked_fraction"] = float(metrics["transit_masked_points"] / max(len(flux), 1))

    logger.info(
        "Detrending complete: segments=%s gaps=%s cadence=%.2f min transit_masked=%s",
        metrics["segments_detected"],
        metrics["gap_count"],
        metrics["median_cadence_minutes"] if metrics["median_cadence_minutes"] is not None else -1.0,
        metrics["transit_masked_points"]
    )

    result = DetrendedLightCurve(flat_flux, trend_flux, metrics)
    flatten_lightcurve.last_metrics = metrics
    return result


flatten_lightcurve.last_metrics = {}


class TestDetrendModule(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.time = np.arange(0.0, 12.0, 30.0 / (24.0 * 60.0))
        slow_trend = 1.0 + 0.004 * np.sin(2.0 * np.pi * self.time / 6.0)
        noise = rng.normal(0.0, 2.5e-4, len(self.time))
        self.flux = slow_trend + noise

    def test_backward_compatible_tuple_and_metrics(self):
        result = flatten_lightcurve(self.time, self.flux, window_days=0.5)
        flat_flux, trend_flux = result

        self.assertEqual(len(result), 2)
        self.assertEqual(len(flat_flux), len(self.time))
        self.assertEqual(len(trend_flux), len(self.time))
        self.assertTrue(hasattr(result, "metrics"))
        self.assertIn("segments_detected", result.metrics)
        self.assertAlmostEqual(np.nanmedian(flat_flux), 1.0, places=2)

    def test_gap_aware_segmentation_reports_gap_indices(self):
        time = self.time.copy()
        time[250:] += 0.9
        flux = self.flux.copy()
        flux[250:] += 0.003

        result = flatten_lightcurve(time, flux, window_days=0.5)

        self.assertEqual(result.metrics["gap_count"], 1)
        self.assertEqual(result.metrics["gap_indices"], [249])
        self.assertEqual(result.metrics["segments_detected"], 2)
        self.assertEqual(result.metrics["segment_lengths"], [250, len(time) - 250])

    def test_transit_preserving_flattening_keeps_dip_depth(self):
        flux = self.flux.copy()
        transit_slice = slice(180, 186)
        flux[transit_slice] *= 0.985

        flat_flux, _ = flatten_lightcurve(self.time, flux, window_days=0.5)

        self.assertLess(np.nanmedian(flat_flux[transit_slice]), 0.992)
        self.assertGreater(np.nanmedian(flat_flux), 0.998)

    def test_short_input_uses_safe_fallback(self):
        short_time = np.array([0.0, 0.1, 0.2, 0.3])
        short_flux = np.array([1.0, 1.01, 0.99, 1.0])

        result = flatten_lightcurve(short_time, short_flux)
        flat_flux, trend_flux = result

        np.testing.assert_allclose(flat_flux, short_flux)
        np.testing.assert_allclose(trend_flux, np.ones_like(short_flux))
        self.assertEqual(result.metrics["reason"], "insufficient_points")

    def test_tess_aware_window_selection_produces_valid_windows(self):
        result = flatten_lightcurve(self.time, self.flux, window_days=0.2)

        self.assertTrue(result.metrics["tess_like_data"])
        self.assertTrue(all(window > DETREND_POLYORDER for window in result.metrics["window_points_used"]))
        self.assertTrue(all(window % 2 == 1 for window in result.metrics["window_points_used"]))


if __name__ == "__main__":
    unittest.main()
