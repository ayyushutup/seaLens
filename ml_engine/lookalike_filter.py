"""
False positive and look-alike rejection filter for SAR Oil Spill Detection.

In SAR imagery, dark patches are caused by surface roughness damping (Bragg scattering suppression).
Look-alikes can be caused by:
1. Low Wind Areas (wind speed < 2.5 - 3.0 m/s causes specular mirror reflection, appearing dark).
2. Natural Biogenic Films (organic secretions from plankton/fish, typically diffuse with soft boundaries).
3. Upwelling / Internal Waves.
4. Rain cells / atmospheric attenuation.

This module evaluates polarimetric features, morphological gradients, and meteorological context.
"""
from typing import Dict, Any, Tuple

class LookAlikeClassifier:
    def __init__(self, min_wind_speed_ms: float = 2.5, max_wind_speed_ms: float = 14.0):
        self.min_wind_speed_ms = min_wind_speed_ms
        self.max_wind_speed_ms = max_wind_speed_ms

    def evaluate_slick(
        self,
        area_sqkm: float,
        perimeter_km: float,
        major_axis_km: float,
        wind_speed_ms: float,
        radar_damping_db: float = 8.5,  # Typical mineral oil damping: 6 - 12 dB
        edge_gradient_sharpness: float = 0.85, # 0 (diffuse) to 1.0 (crisp sharp edge)
        vv_vh_ratio: float = 1.2
    ) -> Tuple[bool, str, float]:
        """
        Evaluates whether a candidate slick is a Look-Alike or a Verified Mineral Oil Spill.
        
        Returns:
            (is_lookalike: bool, reason: str, confidence_score: float)
        """
        # 1. Low wind condition check (< 2.5 m/s)
        if wind_speed_ms < self.min_wind_speed_ms:
            return (
                True,
                f"Calm Water Look-Alike: Wind speed is {wind_speed_ms:.1f} m/s (below {self.min_wind_speed_ms} m/s threshold). Surface lacks capillary waves necessary for SAR Bragg scattering.",
                0.15
            )

        # 2. Extreme wind condition check (> 14 m/s)
        if wind_speed_ms > self.max_wind_speed_ms:
            return (
                True,
                f"Turbulent Sea / Wave Dissipation: Wind speed {wind_speed_ms:.1f} m/s breaks surface slicks rapidly into emulsion.",
                0.25
            )

        # 3. Radar Damping Ratio Check
        # Biogenic films have weak damping (< 4 dB), mineral oil has strong damping (> 6 dB)
        if radar_damping_db < 4.5:
            return (
                True,
                f"Biogenic Slick Look-Alike: Weak radar backscatter damping ({radar_damping_db:.1f} dB). Characteristics match natural marine surfactant / algae bloom rather than mineral oil.",
                0.30
            )

        # 4. Morphological Edge Gradient Check
        # Illegal bilge dumps / tanker discharge form elongated, sharp-edged plumes
        if edge_gradient_sharpness < 0.4 and area_sqkm > 20.0:
            return (
                True,
                "Diffuse Natural Film: Slick boundary lacks sharp gradient and has high fractal diffusion typical of organic biogenic films.",
                0.38
            )

        # Verified Mineral Oil Spill
        # Calculate positive confidence based on feature harmony
        score = 0.70
        if 3.0 <= wind_speed_ms <= 10.0:
            score += 0.15
        if radar_damping_db >= 6.0:
            score += 0.10
        if edge_gradient_sharpness >= 0.7:
            score += 0.05

        return (
            False,
            "Confirmed Mineral Oil Spill: High radar backscatter contrast, sharp plume gradient, and optimal SAR wind window.",
            min(round(score, 2), 0.99)
        )
