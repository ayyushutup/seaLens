"""
Lagrangian Ocean Drift & Reverse Backtrack Physics Engine.

Models the transport and dispersion of surface oil slicks under the combined influence
of ocean surface currents and surface atmospheric wind (Leeway model).

Physics Formulation:
  V_slick = V_current + alpha * V_wind
  alpha ≈ 0.030 to 0.035 (3.0% - 3.5% wind leeway factor)
  Deflection angle (Coriolis leeway deflection): ~0-5 degrees to the right in Northern Hemisphere.
"""
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from ml_engine.metrics import haversine_distance_km

class DriftEngine:
    def __init__(self, leeway_factor: float = 0.032, coriolis_deg: float = 2.0):
        self.leeway_factor = leeway_factor
        self.coriolis_deg = coriolis_deg

    def calculate_drift_velocity(
        self,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float
    ) -> Tuple[float, float, float]:
        """
        Computes net drift velocity vector.
        
        Returns:
            (v_east_ms, v_north_ms, net_speed_kmh, net_bearing_deg)
        """
        # Wind pushes in the direction opposite to where it originates
        wind_push_bearing = (wind_direction_from_deg + 180.0 + self.coriolis_deg) % 360.0
        wind_push_rad = math.radians(wind_push_bearing)
        
        wind_u = self.leeway_factor * wind_speed_ms * math.sin(wind_push_rad)
        wind_v = self.leeway_factor * wind_speed_ms * math.cos(wind_push_rad)

        # Current velocity components (towards current_direction_to_deg)
        current_rad = math.radians(current_direction_to_deg)
        current_u = current_speed_ms * math.sin(current_rad)
        current_v = current_speed_ms * math.cos(current_rad)

        # Net velocity vector (m/s)
        net_u = wind_u + current_u
        net_v = wind_v + current_v

        net_speed_ms = math.sqrt(net_u**2 + net_v**2)
        net_speed_kmh = net_speed_ms * 3.6
        
        net_bearing_rad = math.atan2(net_u, net_v)
        net_bearing_deg = (math.degrees(net_bearing_rad) + 360.0) % 360.0

        return (net_u, net_v, net_speed_kmh, net_bearing_deg)

    def backtrack_origin(
        self,
        detect_lat: float,
        detect_lng: float,
        elapsed_hours: float,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float,
        dispersion_sigma_km: float = 0.5
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Reverse-simulates the spill movement back in time by elapsed_hours to find the
        initial discharge location and uncertainty bounding cone.
        
        Returns:
            (origin_lat, origin_lng, origin_cone_geojson)
        """
        net_u, net_v, net_speed_kmh, net_bearing = self.calculate_drift_velocity(
            wind_speed_ms, wind_direction_from_deg,
            current_speed_ms, current_direction_to_deg
        )

        # Reverse drift: move in opposite direction
        # 1 degree lat ≈ 111.139 km
        # 1 degree lng ≈ 111.139 * cos(lat) km
        total_drift_km = net_speed_kmh * elapsed_hours
        reverse_bearing_rad = math.radians((net_bearing + 180.0) % 360.0)

        delta_north_km = total_drift_km * math.cos(reverse_bearing_rad)
        delta_east_km = total_drift_km * math.sin(reverse_bearing_rad)

        origin_lat = detect_lat + (delta_north_km / 111.139)
        origin_lng = detect_lng + (delta_east_km / (111.139 * math.cos(math.radians(detect_lat))))

        # Generate expanding uncertainty cone / ellipse
        # Spread increases linearly with time due to turbulent eddy diffusivity
        uncertainty_radius_km = max(0.8, dispersion_sigma_km * math.sqrt(elapsed_hours))
        
        # Generate polygon coordinates for the uncertainty envelope
        cone_coords = []
        for angle_deg in range(0, 360, 20):
            rad = math.radians(angle_deg)
            lat_offset = (uncertainty_radius_km * math.cos(rad)) / 111.139
            lng_offset = (uncertainty_radius_km * math.sin(rad)) / (111.139 * math.cos(math.radians(origin_lat)))
            cone_coords.append([round(origin_lng + lng_offset, 6), round(origin_lat + lat_offset, 6)])
        
        # Close polygon
        cone_coords.append(cone_coords[0])

        origin_cone_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [cone_coords]
            },
            "properties": {
                "origin_lat": round(origin_lat, 6),
                "origin_lng": round(origin_lng, 6),
                "elapsed_hours": elapsed_hours,
                "uncertainty_radius_km": round(uncertainty_radius_km, 2),
                "drift_speed_knots": round(net_speed_kmh / 1.852, 2)
            }
        }

        return origin_lat, origin_lng, origin_cone_geojson

    def generate_drift_trajectory_points(
        self,
        origin_lat: float,
        origin_lng: float,
        elapsed_hours: float,
        steps: int,
        wind_speed_ms: float,
        wind_direction_from_deg: float,
        current_speed_ms: float,
        current_direction_to_deg: float
    ) -> List[Dict[str, Any]]:
        """
        Generates step-by-step intermediate points from origin to detection site for map animation.
        """
        net_u, net_v, net_speed_kmh, net_bearing = self.calculate_drift_velocity(
            wind_speed_ms, wind_direction_from_deg,
            current_speed_ms, current_direction_to_deg
        )
        
        points = []
        for step in range(steps + 1):
            t = (step / steps) * elapsed_hours
            dist_km = net_speed_kmh * t
            rad = math.radians(net_bearing)
            d_lat = (dist_km * math.cos(rad)) / 111.139
            d_lng = (dist_km * math.sin(rad)) / (111.139 * math.cos(math.radians(origin_lat)))
            
            points.append({
                "hour_offset": round(t, 2),
                "lat": round(origin_lat + d_lat, 6),
                "lng": round(origin_lng + d_lng, 6)
            })
            
        return points
