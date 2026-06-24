import unittest
from typing import Any, Dict, Tuple

import numpy as np

from backend.phase_fold import bin_folded_lightcurve, fold_lightcurve


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _safe_median(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def _safe_std(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    std = float(np.std(finite))
    return std if np.isfinite(std) else default


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def _phase_from_time(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    return phase / period


def _width_from_mask(bin_centers: np.ndarray, mask: np.ndarray) -> float:
    if np.sum(mask) < 2:
        return 0.0
    selected = bin_centers[mask]
    return float(np.max(selected) - np.min(selected))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2:
        return 0.0
    x_fit = x[finite]
    y_fit = y[finite]
    if np.allclose(x_fit, x_fit[0]):
        return 0.0
    slope, _ = np.polyfit(x_fit, y_fit, 1)
    return float(slope)


def _initialize_features() -> Dict[str, Any]:
    return {
        # Legacy ML features kept stable for backward compatibility.
        "period": 0.0,
        "depth": 0.0,
        "duration": 0.0,
        "snr": 0.0,
        "shape_score": 0.0,
        "odd_even_diff": 0.0,
        "secondary_depth": 0.0,
        "secondary_ratio": 0.0,
        "out_transit_var": 0.0,
        "symmetry": 0.0,
        # New explicit morphology features.
        "odd_depth": 0.0,
        "even_depth": 0.0,
        "secondary_phase_flux": 1.0,
        "ingress_slope": 0.0,
        "egress_slope": 0.0,
        "transit_symmetry": 0.0,
        "u_shape_score": 0.0,
        "v_shape_score": 1.0,
        "out_of_transit_rms": 0.0,
        "depth_to_rms": 0.0,
        "transit_bottom_width": 0.0,
        "transit_full_width": 0.0,
        "in_transit_points": 0.0,
        "out_of_transit_points": 0.0,
        "folded_point_count": 0.0,
        "binned_transit_min_flux": 1.0,
        "binned_transit_depth": 0.0,
        "transit_centroid_offset": 0.0,
        "feature_metrics": {
            "version": "phase_2d",
            "description": {
                "odd_even_diff": "Normalized difference between odd and even transit depths.",
                "secondary_ratio": "Secondary-eclipse depth relative to the primary transit depth.",
                "ingress_slope": "Linear slope of the flux drop entering transit.",
                "egress_slope": "Linear slope of the flux rise exiting transit.",
                "transit_symmetry": "Similarity of ingress and egress morphology; closer to 1 is more symmetric.",
                "u_shape_score": "Bottom-width to top-width score; higher values are more U-shaped.",
                "out_of_transit_rms": "RMS scatter outside transit in folded space.",
            }
        }
    }


def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    bls_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extracts folded-light-curve morphology features for machine learning.

    Backward compatibility:
    - The legacy flat feature keys remain present and numeric.
    - New morphology diagnostics are added as extra numeric keys plus a nested
      `feature_metrics` dictionary for documentation and traceability.
    """
    features = _initialize_features()

    period = _safe_float(bls_results.get("period", 0.0))
    epoch = _safe_float(bls_results.get("epoch", 0.0))
    duration = _safe_float(bls_results.get("duration", 0.0))
    depth = max(_safe_float(bls_results.get("depth", 0.0)), 0.0)
    snr = max(_safe_float(bls_results.get("snr", 0.0)), 0.0)

    if period <= 0 or duration <= 0 or len(time) < 10 or depth <= 0:
        return features

    features["period"] = period
    features["depth"] = depth
    features["duration"] = duration
    features["snr"] = snr

    try:
        folded = fold_lightcurve(time, flux, flux_err, period, epoch)
        phase = np.asarray(folded["phase"], dtype=float)
        fflux = np.asarray(folded["flux"], dtype=float)
    except Exception:
        return features

    finite_mask = np.isfinite(phase) & np.isfinite(fflux)
    phase = phase[finite_mask]
    fflux = fflux[finite_mask]

    if len(phase) < 10:
        return features

    transit_half_width = max((duration / period) / 2.0, 1e-4)
    transit_core_width = 0.60 * transit_half_width
    shoulder_width = 1.50 * transit_half_width

    in_transit_mask = np.abs(phase) <= transit_half_width
    out_transit_mask = np.abs(phase) > shoulder_width
    secondary_mask = np.abs(np.abs(phase) - 0.5) <= transit_half_width

    features["in_transit_points"] = float(np.sum(in_transit_mask))
    features["out_of_transit_points"] = float(np.sum(out_transit_mask))
    features["folded_point_count"] = float(len(phase))

    baseline = _safe_median(fflux[out_transit_mask], default=1.0) if np.sum(out_transit_mask) >= 5 else 1.0

    if np.sum(out_transit_mask) >= 5:
        out_rms = _safe_std(fflux[out_transit_mask], default=0.001)
    else:
        out_rms = 0.001
    features["out_transit_var"] = out_rms
    features["out_of_transit_rms"] = out_rms
    features["depth_to_rms"] = float(depth / (out_rms + 1e-6))

    try:
        bin_centers, bin_flux, _ = bin_folded_lightcurve(phase, fflux, num_bins=100)
        bin_centers = np.asarray(bin_centers, dtype=float)
        bin_flux = np.asarray(bin_flux, dtype=float)
    except Exception:
        return features

    finite_bins = np.isfinite(bin_centers) & np.isfinite(bin_flux)
    bin_centers = bin_centers[finite_bins]
    bin_flux = bin_flux[finite_bins]
    if len(bin_centers) < 10:
        return features

    dip_flux = baseline - bin_flux
    in_transit_bins = np.abs(bin_centers) <= transit_half_width

    if np.any(in_transit_bins):
        binned_min_flux = float(np.min(bin_flux[in_transit_bins]))
        features["binned_transit_min_flux"] = binned_min_flux
        features["binned_transit_depth"] = float(max(0.0, baseline - binned_min_flux))

    # U-shape vs V-shape morphology.
    thresh_10 = 0.10 * depth
    thresh_50 = 0.50 * depth
    thresh_80 = 0.80 * depth
    bins_10_mask = in_transit_bins & (dip_flux >= thresh_10)
    bins_50_mask = in_transit_bins & (dip_flux >= thresh_50)
    bins_80_mask = in_transit_bins & (dip_flux >= thresh_80)

    transit_full_width = _width_from_mask(bin_centers, bins_10_mask)
    transit_mid_width = _width_from_mask(bin_centers, bins_50_mask)
    transit_bottom_width = _width_from_mask(bin_centers, bins_80_mask)

    features["transit_full_width"] = transit_full_width
    features["transit_bottom_width"] = transit_bottom_width

    if np.sum(bins_10_mask) > 0:
        u_shape_score = float(np.sum(bins_80_mask) / max(np.sum(bins_10_mask), 1))
    else:
        u_shape_score = 0.0
    features["shape_score"] = u_shape_score
    features["u_shape_score"] = u_shape_score
    features["v_shape_score"] = float(max(0.0, 1.0 - u_shape_score))

    # Secondary eclipse around phase 0.5.
    if np.sum(secondary_mask) >= 3:
        secondary_flux = fflux[secondary_mask]
        secondary_phase_flux = _safe_median(secondary_flux, default=baseline)
        secondary_depth = max(0.0, baseline - secondary_phase_flux)
        features["secondary_phase_flux"] = float(secondary_phase_flux)
        features["secondary_depth"] = float(secondary_depth)
        features["secondary_ratio"] = float(secondary_depth / (depth + 1e-6))

    # Odd-even depth comparison using unfolded timing to preserve transit numbering.
    try:
        transit_numbers = np.round((time - epoch) / period).astype(int)
        time_phase = _phase_from_time(np.asarray(time, dtype=float), period, epoch)
        in_transit_time_mask = np.abs(time_phase) <= transit_half_width
        odd_mask = (transit_numbers % 2) != 0
        even_mask = ~odd_mask

        odd_flux = np.asarray(flux, dtype=float)[in_transit_time_mask & odd_mask]
        even_flux = np.asarray(flux, dtype=float)[in_transit_time_mask & even_mask]

        if len(odd_flux) >= 2:
            features["odd_depth"] = float(max(0.0, baseline - _safe_median(odd_flux, default=baseline)))
        if len(even_flux) >= 2:
            features["even_depth"] = float(max(0.0, baseline - _safe_median(even_flux, default=baseline)))

        if features["odd_depth"] > 0 or features["even_depth"] > 0:
            features["odd_even_diff"] = float(
                abs(features["odd_depth"] - features["even_depth"]) / (depth + 1e-6)
            )
    except Exception:
        pass

    # Ingress and egress slopes from the binned folded profile.
    ingress_mask = (bin_centers >= -shoulder_width) & (bin_centers <= -transit_core_width)
    egress_mask = (bin_centers >= transit_core_width) & (bin_centers <= shoulder_width)

    ingress_flux = baseline - bin_flux[ingress_mask]
    egress_flux = baseline - bin_flux[egress_mask]
    ingress_slope = _linear_slope(bin_centers[ingress_mask], ingress_flux)
    egress_slope = _linear_slope(bin_centers[egress_mask], egress_flux)
    features["ingress_slope"] = float(abs(ingress_slope))
    features["egress_slope"] = float(abs(egress_slope))

    # Transit symmetry from mirrored left/right profiles.
    left_mask = (bin_centers >= -transit_half_width) & (bin_centers < 0.0)
    right_mask = (bin_centers > 0.0) & (bin_centers <= transit_half_width)
    left_profile = baseline - bin_flux[left_mask]
    right_profile = (baseline - bin_flux[right_mask])[::-1]
    left_phase = np.abs(bin_centers[left_mask])
    right_phase = np.abs(bin_centers[right_mask])[::-1]

    min_len = min(len(left_profile), len(right_profile))
    if min_len >= 3:
        left_profile = left_profile[:min_len]
        right_profile = right_profile[:min_len]
        morphology_scale = max(depth, out_rms, 1e-6)
        mean_abs_diff = _safe_mean(np.abs(left_profile - right_profile), default=0.0)
        transit_symmetry = float(max(0.0, 1.0 - (mean_abs_diff / morphology_scale)))
        features["symmetry"] = transit_symmetry
        features["transit_symmetry"] = transit_symmetry

        centroid_weights = np.maximum(baseline - bin_flux[in_transit_bins], 0.0)
        if np.sum(centroid_weights) > 0:
            features["transit_centroid_offset"] = float(
                np.sum(bin_centers[in_transit_bins] * centroid_weights) / np.sum(centroid_weights)
            )

    # Folded morphology statistics suitable for ML extensions.
    features["feature_metrics"].update({
        "baseline_flux": float(baseline),
        "transit_half_width_phase": float(transit_half_width),
        "transit_mid_width": float(transit_mid_width),
        "shape_thresholds": {
            "depth_10_percent": float(thresh_10),
            "depth_50_percent": float(thresh_50),
            "depth_80_percent": float(thresh_80),
        },
        "morphology_summary": {
            "odd_depth": float(features["odd_depth"]),
            "even_depth": float(features["even_depth"]),
            "secondary_depth": float(features["secondary_depth"]),
            "ingress_slope": float(features["ingress_slope"]),
            "egress_slope": float(features["egress_slope"]),
            "u_shape_score": float(features["u_shape_score"]),
            "transit_symmetry": float(features["transit_symmetry"]),
            "out_of_transit_rms": float(features["out_of_transit_rms"]),
        },
    })

    # Final numeric cleanup for flat features while leaving nested metrics structured.
    for key, value in list(features.items()):
        if key == "feature_metrics":
            continue
        features[key] = _safe_float(value, default=0.0)

    return features


class TestFeatureEngineeringModule(unittest.TestCase):
    def _synthetic_transit(
        self,
        period: float = 4.0,
        duration_days: float = 0.20,
        depth: float = 0.012,
        cadence_minutes: float = 10.0,
        length_days: float = 24.0,
        noise_level: float = 8e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        cadence_days = cadence_minutes / (24.0 * 60.0)
        time = np.arange(0.0, length_days, cadence_days)
        flux = np.ones_like(time)
        epoch = 0.8
        phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
        flux[np.abs(phase) <= (0.5 * duration_days)] -= depth
        flux += rng.normal(0.0, noise_level, len(time))
        flux_err = np.full_like(time, noise_level)
        bls = {"period": period, "epoch": epoch, "duration": duration_days, "depth": depth, "snr": 25.0}
        return time, flux, flux_err, bls

    def _synthetic_eclipsing_binary(
        self,
        period: float = 4.0,
        duration_days: float = 0.20,
        primary_depth: float = 0.030,
        secondary_depth: float = 0.015,
        cadence_minutes: float = 10.0,
        length_days: float = 24.0,
        noise_level: float = 8e-4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(7)
        cadence_days = cadence_minutes / (24.0 * 60.0)
        time = np.arange(0.0, length_days, cadence_days)
        flux = np.ones_like(time)
        epoch = 0.5
        phase_primary = ((time - epoch + 0.5 * period) % period) - 0.5 * period
        phase_secondary = ((time - (epoch + 0.5 * period) + 0.5 * period) % period) - 0.5 * period
        flux[np.abs(phase_primary) <= (0.5 * duration_days)] -= primary_depth
        flux[np.abs(phase_secondary) <= (0.5 * duration_days)] -= secondary_depth
        flux += rng.normal(0.0, noise_level, len(time))
        flux_err = np.full_like(time, noise_level)
        bls = {"period": period, "epoch": epoch, "duration": duration_days, "depth": primary_depth, "snr": 35.0}
        return time, flux, flux_err, bls

    def test_backward_compatible_feature_keys_exist(self):
        time, flux, flux_err, bls = self._synthetic_transit()
        feats = extract_features(time, flux, flux_err, bls)

        for key in (
            "period", "depth", "duration", "snr", "shape_score",
            "odd_even_diff", "secondary_depth", "secondary_ratio",
            "out_transit_var", "symmetry"
        ):
            self.assertIn(key, feats)
            self.assertTrue(np.isfinite(feats[key]))

    def test_secondary_eclipse_and_ratio_are_detected(self):
        time, flux, flux_err, bls = self._synthetic_eclipsing_binary()
        feats = extract_features(time, flux, flux_err, bls)

        self.assertGreater(feats["secondary_depth"], 0.005)
        self.assertGreater(feats["secondary_ratio"], 0.20)

    def test_odd_even_difference_highlights_alternating_depths(self):
        time, flux, flux_err, bls = self._synthetic_transit()
        period = bls["period"]
        epoch = bls["epoch"]
        duration = bls["duration"]
        phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
        transit_numbers = np.round((time - epoch) / period).astype(int)
        odd_mask = (transit_numbers % 2) != 0
        even_transit = np.abs(phase) <= (0.5 * duration)
        flux[even_transit & odd_mask] -= 0.006

        feats = extract_features(time, flux, flux_err, bls)

        self.assertGreater(feats["odd_even_diff"], 0.20)
        self.assertGreater(feats["odd_depth"], feats["even_depth"])

    def test_ingress_egress_and_symmetry_metrics_are_numeric(self):
        time, flux, flux_err, bls = self._synthetic_transit()
        feats = extract_features(time, flux, flux_err, bls)

        self.assertGreaterEqual(feats["ingress_slope"], 0.0)
        self.assertGreaterEqual(feats["egress_slope"], 0.0)
        self.assertGreaterEqual(feats["transit_symmetry"], 0.0)
        self.assertLessEqual(feats["transit_symmetry"], 1.0)

    def test_u_shape_score_favors_boxier_transits(self):
        time_u, flux_u, flux_err_u, bls_u = self._synthetic_transit()
        feats_u = extract_features(time_u, flux_u, flux_err_u, bls_u)

        time_v, flux_v, flux_err_v, bls_v = self._synthetic_transit(depth=0.0)
        period = 4.0
        epoch = 0.8
        duration_days = 0.20
        phase = ((time_v - epoch + 0.5 * period) % period) - 0.5 * period
        width = duration_days / 2.0
        triangular = np.clip(1.0 - np.abs(phase) / width, 0.0, 1.0)
        flux_v -= 0.012 * triangular
        bls_v = {"period": period, "epoch": epoch, "duration": duration_days, "depth": 0.012, "snr": 25.0}
        feats_v = extract_features(time_v, flux_v, flux_err_v, bls_v)

        self.assertGreater(feats_u["u_shape_score"], feats_v["u_shape_score"])

    def test_feature_metrics_structure_is_present(self):
        time, flux, flux_err, bls = self._synthetic_transit()
        feats = extract_features(time, flux, flux_err, bls)

        self.assertIn("feature_metrics", feats)
        self.assertIn("morphology_summary", feats["feature_metrics"])
        self.assertEqual(feats["feature_metrics"]["version"], "phase_2d")


if __name__ == "__main__":
    unittest.main()
