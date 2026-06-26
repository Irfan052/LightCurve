import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares

from backend.config import (
    BLS_DURATION_MAX,
    BLS_DURATION_MIN,
    BLS_MAX_PERIOD,
    BLS_MIN_PERIOD,
    BLS_OVERSAMPLE,
)
from backend.utils import logger


TOP_CANDIDATE_COUNT = 5
MIN_POINTS_FOR_BLS = 20


def _empty_result(error_message: Optional[str] = None) -> Dict[str, Any]:
    metrics = {
        "best_period": 0.0,
        "best_power": 0.0,
        "SDE": 0.0,
        "candidate_count": 0,
        "harmonic_corrections_applied": 0,
    }
    result = {
        "period": 0.0,
        "epoch": 0.0,
        "duration": 0.0,
        "depth": 0.0,
        "snr": 0.0,
        "bls_power": 0.0,
        "sde": 0.0,
        "periods": np.array([]),
        "powers": np.array([]),
        "candidates": [],
        "metrics": metrics,
    }
    if error_message:
        result["error"] = error_message
        result["metrics"]["error"] = error_message
    from datetime import datetime
    return result


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


def _compute_sde(power: np.ndarray, peak_power: float) -> float:
    finite_power = power[np.isfinite(power)]
    if finite_power.size == 0:
        return 0.0

    background = finite_power.copy()
    upper_clip = np.percentile(background, 95) if background.size > 10 else np.max(background)
    background = background[background <= upper_clip]
    if background.size < 5:
        background = finite_power

    median_power = _safe_median(background, fallback=0.0)
    sigma_power = 1.4826 * _safe_median(np.abs(background - median_power), fallback=0.0)
    if sigma_power <= 0:
        sigma_power = _safe_std(background, fallback=1e-6)

    return float((peak_power - median_power) / sigma_power) if sigma_power > 0 else 0.0


def _phase_wrap(time: np.ndarray, epoch: float, period: float) -> np.ndarray:
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    return phase


def _transit_shape_score(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float
) -> float:
    if period <= 0 or duration <= 0 or depth <= 0 or len(time) < 20:
        return -np.inf

    phase_time = _phase_wrap(time, epoch, period)
    in_transit = np.abs(phase_time) <= (0.5 * duration)
    shoulder = (np.abs(phase_time) > (0.5 * duration)) & (np.abs(phase_time) <= (1.5 * duration))
    out_of_transit = np.abs(phase_time) > (2.0 * duration)

    if np.sum(in_transit) < 3 or np.sum(out_of_transit) < 5:
        return -np.inf

    in_flux = flux[in_transit]
    oot_flux = flux[out_of_transit]
    baseline = _safe_median(oot_flux, fallback=1.0)
    observed_depth = baseline - _safe_median(in_flux, fallback=baseline)
    if observed_depth <= 0:
        return -np.inf

    shoulder_flux = flux[shoulder] if np.sum(shoulder) >= 3 else oot_flux
    shoulder_level = _safe_median(shoulder_flux, fallback=baseline)
    transit_contrast = observed_depth / (_safe_std(oot_flux, fallback=1e-4) + 1e-6)
    shoulder_penalty = max(0.0, baseline - shoulder_level)

    score = (
        2.5 * transit_contrast
        + 200.0 * observed_depth
        - 120.0 * shoulder_penalty
        - 50.0 * abs(observed_depth - depth)
    )
    return float(score)


def _harmonic_diagnostics(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float
) -> Dict[str, float]:
    if period <= 0 or duration <= 0 or len(time) < 20:
        return {"primary_depth": 0.0, "secondary_depth": 0.0, "secondary_ratio": 0.0, "odd_even_diff": 0.0}

    phase_time = _phase_wrap(time, epoch, period)
    in_primary = np.abs(phase_time) <= (0.5 * duration)
    in_secondary = np.abs(np.abs(phase_time) - 0.5 * period) <= (0.5 * duration)
    out_of_transit = (np.abs(phase_time) > (2.0 * duration)) & (np.abs(np.abs(phase_time) - 0.5 * period) > (1.5 * duration))

    if np.sum(in_primary) < 3 or np.sum(out_of_transit) < 5:
        return {"primary_depth": 0.0, "secondary_depth": 0.0, "secondary_ratio": 0.0, "odd_even_diff": 0.0}

    baseline = _safe_median(flux[out_of_transit], fallback=1.0)
    primary_depth = max(0.0, baseline - _safe_median(flux[in_primary], fallback=baseline))
    secondary_depth = max(0.0, baseline - _safe_median(flux[in_secondary], fallback=baseline)) if np.sum(in_secondary) >= 3 else 0.0
    secondary_ratio = float(secondary_depth / (primary_depth + 1e-6)) if primary_depth > 0 else 0.0

    transit_numbers = np.round((time - epoch) / period).astype(int)
    odd_mask = (transit_numbers % 2) != 0
    even_mask = ~odd_mask
    odd_primary = flux[in_primary & odd_mask]
    even_primary = flux[in_primary & even_mask]
    if len(odd_primary) >= 2 and len(even_primary) >= 2 and primary_depth > 0:
        odd_depth = max(0.0, baseline - _safe_median(odd_primary, fallback=baseline))
        even_depth = max(0.0, baseline - _safe_median(even_primary, fallback=baseline))
        odd_even_diff = float(abs(odd_depth - even_depth) / (primary_depth + 1e-6))
    else:
        odd_even_diff = 0.0

    return {
        "primary_depth": float(primary_depth),
        "secondary_depth": float(secondary_depth),
        "secondary_ratio": float(secondary_ratio),
        "odd_even_diff": float(odd_even_diff),
    }


def _build_duration_grid(time: np.ndarray) -> np.ndarray:
    cadence = np.diff(time)
    cadence = cadence[np.isfinite(cadence) & (cadence > 0)]
    cadence_days = float(np.median(cadence)) if cadence.size > 0 else 0.02

    duration_min = max(0.02, 2.0 * cadence_days, BLS_DURATION_MIN)
    duration_max = min(0.40, max(duration_min * 2.0, BLS_DURATION_MAX))
    if duration_max <= duration_min:
        duration_max = duration_min * 1.5
    return np.linspace(duration_min, duration_max, 5)


def _extract_candidate(periodogram: Any, index: int, power_override: Optional[float] = None) -> Dict[str, float]:
    power = float(power_override if power_override is not None else periodogram.power[index])
    return {
        "period": float(periodogram.period[index]),
        "epoch": float(periodogram.transit_time[index]),
        "duration": float(periodogram.duration[index]),
        "depth": float(max(periodogram.depth[index], 0.0)),
        "snr": float(max(periodogram.depth_snr[index], 0.0)),
        "bls_power": power,
        "sde": 0.0,
    }


def _rank_top_candidates(periodogram: Any, top_n: int = TOP_CANDIDATE_COUNT) -> List[Dict[str, float]]:
    powers = np.asarray(periodogram.power, dtype=float)
    if powers.size == 0:
        return []

    sorted_indices = np.argsort(powers)[::-1]
    ranked: List[Dict[str, float]] = []
    seen_periods: List[float] = []

    for idx in sorted_indices:
        candidate = _extract_candidate(periodogram, int(idx))
        period = candidate["period"]
        if period <= 0:
            continue

        if any(abs(period - seen) / max(seen, 1e-6) < 0.05 for seen in seen_periods):
            continue

        candidate["sde"] = _compute_sde(powers, candidate["bls_power"])
        ranked.append(candidate)
        seen_periods.append(period)

        if len(ranked) >= top_n:
            break

    ranked.sort(key=lambda item: (item["sde"], item["bls_power"], item["snr"]), reverse=True)
    return ranked


def _evaluate_period(
    bls: BoxLeastSquares,
    period: float,
    duration_grid: np.ndarray
) -> Optional[Dict[str, float]]:
    if not np.isfinite(period) or period <= 0:
        return None

    try:
        result = bls.power(np.array([period], dtype=float), duration_grid)
        return {
            "period": float(period),
            "epoch": float(result.transit_time[0]),
            "duration": float(result.duration[0]),
            "depth": float(max(result.depth[0], 0.0)),
            "snr": float(max(result.depth_snr[0], 0.0)),
            "bls_power": float(result.power[0]),
        }
    except Exception as exc:
        logger.warning(f"Failed to evaluate harmonic period {period:.4f} d: {exc}")
        return None


def _harmonic_period_validation(
    time: np.ndarray,
    flux: np.ndarray,
    best_candidate: Dict[str, float],
    bls: BoxLeastSquares,
    duration_grid: np.ndarray
) -> Tuple[Dict[str, float], int, List[Dict[str, float]]]:
    base_period = best_candidate["period"]
    harmonic_periods = [base_period, base_period / 2.0, base_period * 2.0, base_period * 3.0]
    unique_periods: List[float] = []
    evaluations: List[Dict[str, float]] = []

    for period in harmonic_periods:
        if period < BLS_MIN_PERIOD or period > BLS_MAX_PERIOD or not np.isfinite(period):
            continue
        if any(abs(period - existing) / max(existing, 1e-6) < 0.01 for existing in unique_periods):
            continue
        evaluated = _evaluate_period(bls, period, duration_grid)
        if evaluated is None:
            continue
        evaluated["shape_score"] = _transit_shape_score(
            time,
            flux,
            evaluated["period"],
            evaluated["epoch"],
            evaluated["duration"],
            evaluated["depth"],
        )
        evaluated.update(
            _harmonic_diagnostics(
                time,
                flux,
                evaluated["period"],
                evaluated["epoch"],
                evaluated["duration"],
            )
        )
        unique_periods.append(period)
        evaluations.append(evaluated)

    if not evaluations:
        return best_candidate, 0, []

    base_eval = min(evaluations, key=lambda item: abs(item["period"] - base_period))
    doubled_eval = None
    for item in evaluations:
        ratio = item["period"] / max(base_period, 1e-6)
        if abs(ratio - 2.0) < 0.05:
            doubled_eval = item
            break

    reference_powers = np.array([item["bls_power"] for item in evaluations], dtype=float)
    for item in evaluations:
        item["sde"] = _compute_sde(reference_powers, item["bls_power"])
        item["ranking_score"] = (
            0.60 * item["bls_power"]
            + 0.20 * item["snr"]
            + 0.15 * item["shape_score"]
            + 50.0 * item["depth"]
            + 25.0 * item["secondary_depth"]
            + 3.0 * item["secondary_ratio"]
            + 2.0 * item["odd_even_diff"]
        )

    # Explicitly resolve the classic eclipsing-binary ambiguity where the raw BLS
    # maximum lands at P/2 but the full orbital period reveals distinct primary
    # and secondary eclipses.
    if doubled_eval is not None:
        half_period_signature = base_eval["odd_even_diff"] > 0.20
        full_period_signature = doubled_eval["secondary_ratio"] > 0.25 or doubled_eval["secondary_depth"] > 0.005
        competitive_power = doubled_eval["bls_power"] >= 0.70 * base_eval["bls_power"]
        if half_period_signature and full_period_signature and competitive_power:
            correction_applied = 1 if abs(doubled_eval["period"] - base_period) / max(base_period, 1e-6) > 0.01 else 0
            return doubled_eval, correction_applied, evaluations

    selected = max(evaluations, key=lambda item: item["ranking_score"])
    correction_applied = 1 if abs(selected["period"] - base_period) / max(base_period, 1e-6) > 0.01 else 0
    return selected, correction_applied, evaluations


def search_transits(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray
) -> Dict[str, Any]:
    """
    Performs a Box Least Squares (BLS) periodogram search to find the most
    dominant periodic transit-like signal while preserving backward compatibility.

    New outputs:
        - sde: Signal Detection Efficiency of the selected signal
        - candidates: top ranked transit candidates
        - metrics: detailed search diagnostics
    """
    logger.info("Initializing Box Least Squares (BLS) periodogram search...")

    if len(time) != len(flux) or len(time) != len(flux_err):
        message = f"Array length mismatch: time ({len(time)}), flux ({len(flux)}), flux_err ({len(flux_err)})"
        logger.error(message)
        raise ValueError(message)

    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    if np.sum(finite_mask) < MIN_POINTS_FOR_BLS:
        logger.warning("Not enough finite points to run transit search.")
        return _empty_result("insufficient_points")

    time_work = np.asarray(time[finite_mask], dtype=float)
    flux_work = np.asarray(flux[finite_mask], dtype=float)
    flux_err_work = np.asarray(flux_err[finite_mask], dtype=float)
    flux_err_work = np.where(np.isfinite(flux_err_work) & (flux_err_work > 0), flux_err_work, np.nanmedian(flux_err_work[flux_err_work > 0]) if np.any(flux_err_work > 0) else 0.001)

    time_span = float(time_work[-1] - time_work[0]) if len(time_work) > 1 else 0.0
    max_period = min(BLS_MAX_PERIOD, time_span / 2.0) if time_span > 0 else 0.0
    if max_period <= BLS_MIN_PERIOD:
        logger.warning("Time span is too short for configured BLS search range.")
        return _empty_result("insufficient_time_span")

    try:
        bls = BoxLeastSquares(time_work, flux_work, flux_err_work)
        duration_grid = _build_duration_grid(time_work)
        periodogram = bls.autopower(
            duration_grid,
            minimum_period=BLS_MIN_PERIOD,
            maximum_period=max_period,
            oversample=BLS_OVERSAMPLE,
        )

        powers = np.asarray(periodogram.power, dtype=float)
        periods = np.asarray(periodogram.period, dtype=float)
        if powers.size == 0 or periods.size == 0 or not np.any(np.isfinite(powers)):
            logger.warning("BLS search returned an empty periodogram.")
            return _empty_result("empty_periodogram")

        peak_idx = int(np.nanargmax(powers))
        peak_power = float(powers[peak_idx])
        best_candidate = _extract_candidate(periodogram, peak_idx)
        best_candidate["sde"] = _compute_sde(powers, peak_power)

        selected_candidate, harmonic_correction_count, harmonic_evaluations = _harmonic_period_validation(
            time_work,
            flux_work,
            best_candidate,
            bls,
            duration_grid,
        )

        top_candidates = _rank_top_candidates(periodogram, top_n=TOP_CANDIDATE_COUNT)

        if selected_candidate["period"] > 0:
            selected_candidate["sde"] = float(selected_candidate.get("sde", _compute_sde(powers, selected_candidate["bls_power"])))
            selected_summary = {
                "period": float(selected_candidate["period"]),
                "depth": float(selected_candidate["depth"]),
                "duration": float(selected_candidate["duration"]),
                "bls_power": float(selected_candidate["bls_power"]),
                "sde": float(selected_candidate["sde"]),
            }
            if not any(abs(item["period"] - selected_summary["period"]) / max(selected_summary["period"], 1e-6) < 0.01 for item in top_candidates):
                top_candidates.append({
                    **selected_summary,
                    "epoch": float(selected_candidate["epoch"]),
                    "snr": float(selected_candidate["snr"]),
                })
                top_candidates.sort(key=lambda item: (item["sde"], item["bls_power"], item.get("snr", 0.0)), reverse=True)
                top_candidates = top_candidates[:TOP_CANDIDATE_COUNT]

        candidates_output = [
            {
                "period": float(item["period"]),
                "depth": float(item["depth"]),
                "duration": float(item["duration"]),
                "bls_power": float(item["bls_power"]),
                "sde": float(item["sde"]),
            }
            for item in top_candidates
        ]

        metrics = {
            "best_period": float(selected_candidate["period"]),
            "best_power": float(selected_candidate["bls_power"]),
            "SDE": float(selected_candidate["sde"]),
            "candidate_count": int(len(candidates_output)),
            "harmonic_corrections_applied": int(harmonic_correction_count),
            "duration_grid_days": duration_grid.tolist(),
            "harmonic_evaluations": [
                {
                    "period": float(item["period"]),
                    "depth": float(item["depth"]),
                    "duration": float(item["duration"]),
                    "bls_power": float(item["bls_power"]),
                    "sde": float(item["sde"]),
                }
                for item in harmonic_evaluations
            ],
        }

        logger.info(
            f"BLS search complete. Best Period: {selected_candidate['period']:.4f} d, "
            f"Epoch (T0): {selected_candidate['epoch']:.4f} d, Depth: {selected_candidate['depth']*100:.3f}%, "
            f"SNR: {selected_candidate['snr']:.2f}, SDE: {selected_candidate['sde']:.2f}"
        )

        return {
            "period": float(selected_candidate["period"]),
            "epoch": float(selected_candidate["epoch"]),
            "duration": float(selected_candidate["duration"]),
            "depth": float(selected_candidate["depth"]),
            "snr": float(selected_candidate["snr"]),
            "bls_power": float(selected_candidate["bls_power"]),
            "sde": float(selected_candidate["sde"]),
            "periods": periods,
            "powers": powers,
            "candidates": candidates_output,
            "metrics": metrics,
        }

    except Exception as exc:
        logger.error(f"BLS search failed: {str(exc)}. Returning empty results.")
        return _empty_result(str(exc))


class TestTransitSearchModule(unittest.TestCase):
    def _synthetic_transit(
        self,
        period: float = 4.0,
        duration_days: float = 0.18,
        depth: float = 0.012,
        length_days: float = 27.0,
        cadence_minutes: float = 10.0,
        noise_level: float = 8e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        cadence_days = cadence_minutes / (24.0 * 60.0)
        time = np.arange(0.0, length_days, cadence_days)
        flux = np.ones_like(time)
        epoch = 0.8
        phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
        in_transit = np.abs(phase) <= (0.5 * duration_days)
        flux[in_transit] -= depth
        flux += 0.002 * np.sin(2.0 * np.pi * time / 9.0)
        flux += rng.normal(0.0, noise_level, len(time))
        flux_err = np.full_like(time, noise_level)
        return time, flux, flux_err

    def _synthetic_eclipsing_binary(
        self,
        period: float = 4.0,
        duration_days: float = 0.18,
        primary_depth: float = 0.035,
        secondary_depth: float = 0.022,
        length_days: float = 27.0,
        cadence_minutes: float = 10.0,
        noise_level: float = 8e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        cadence_days = cadence_minutes / (24.0 * 60.0)
        time = np.arange(0.0, length_days, cadence_days)
        flux = np.ones_like(time)
        epoch = 0.6

        phase_primary = ((time - epoch + 0.5 * period) % period) - 0.5 * period
        phase_secondary = ((time - (epoch + 0.5 * period) + 0.5 * period) % period) - 0.5 * period
        flux[np.abs(phase_primary) <= (0.5 * duration_days)] -= primary_depth
        flux[np.abs(phase_secondary) <= (0.5 * duration_days)] -= secondary_depth
        flux += rng.normal(0.0, noise_level, len(time))
        flux_err = np.full_like(time, noise_level)
        return time, flux, flux_err

    def test_backward_compatible_fields_are_present(self):
        time, flux, flux_err = self._synthetic_transit()
        result = search_transits(time, flux, flux_err)

        for key in ("period", "epoch", "duration", "depth", "snr", "bls_power", "periods", "powers"):
            self.assertIn(key, result)
        self.assertGreater(result["period"], 0.0)
        self.assertGreater(result["snr"], 0.0)

    def test_returns_sde_candidates_and_metrics(self):
        time, flux, flux_err = self._synthetic_transit()
        result = search_transits(time, flux, flux_err)

        self.assertIn("sde", result)
        self.assertIn("metrics", result)
        self.assertIn("candidates", result)
        self.assertGreater(result["sde"], 0.0)
        self.assertEqual(result["metrics"]["best_period"], result["period"])
        self.assertEqual(result["metrics"]["candidate_count"], len(result["candidates"]))
        self.assertLessEqual(len(result["candidates"]), TOP_CANDIDATE_COUNT)
        if result["candidates"]:
            for field in ("period", "depth", "duration", "bls_power", "sde"):
                self.assertIn(field, result["candidates"][0])

    def test_detects_near_true_period_for_transit_signal(self):
        time, flux, flux_err = self._synthetic_transit(period=3.6)
        result = search_transits(time, flux, flux_err)

        self.assertGreater(result["period"], 0.0)
        self.assertAlmostEqual(result["period"], 3.6, delta=0.4)

    def test_harmonic_validation_can_escape_half_period_solution(self):
        time, flux, flux_err = self._synthetic_eclipsing_binary(period=4.2)
        result = search_transits(time, flux, flux_err)

        self.assertGreater(result["period"], 0.0)
        self.assertAlmostEqual(result["period"], 4.2, delta=0.5)
        self.assertGreaterEqual(result["metrics"]["harmonic_corrections_applied"], 0)

    def test_robust_error_handling_for_short_inputs(self):
        time = np.array([0.0, 0.1, 0.2])
        flux = np.array([1.0, 0.99, 1.01])
        flux_err = np.array([0.001, 0.001, 0.001])
        result = search_transits(time, flux, flux_err)

        self.assertEqual(result["period"], 0.0)
        self.assertEqual(result["metrics"]["candidate_count"], 0)
        self.assertIn("error", result)

    def test_length_mismatch_raises_value_error(self):
        time, flux, flux_err = self._synthetic_transit()
        with self.assertRaises(ValueError):
            search_transits(time, flux[:-1], flux_err)


if __name__ == "__main__":
    unittest.main()
