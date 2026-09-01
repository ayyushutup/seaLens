"""
Forward Drift Simulation, Coastal Landfall ETA & Environmental Risk Predictor.

Projects the future trajectory of detected oil slicks over a 72-hour horizon,
determines intersection with coastline / Marine Protected Areas (MPAs), and computes
Shoreline Landfall Impact ETA and containment boom deployment plans.
"""
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from ml_engine.metrics import haversine_distance_km
from backend.services.ocean_grid_engine import DynamicOceanGridEngine
from backend.services.weathering_engine import OilWeatheringEngine

# Sensitive Coastal Infrastructure & Ecological Zones
COASTAL_TARGETS = [
    # Arabian Sea / Mumbai Region
    {"name": "Manori & Gorai Coastal Wetlands & Marine Sanctuary", "lat": 19.240, "lng": 72.780, "type": "Coastal Mangrove Wetland", "vulnerability": "CRITICAL", "region": "Arabian Sea"},
    {"name": "Vasai Creek Fishery & Mangrove Ecosystem", "lat": 19.330, "lng": 72.810, "type": "Estuarine Fishery", "vulnerability": "CRITICAL", "region": "Arabian Sea"},
    {"name": "Alibag Coastal Reefs & Tourism Beaches", "lat": 18.640, "lng": 72.870, "type": "Tourist Beach & Coral Habitat", "vulnerability": "HIGH", "region": "Arabian Sea"},
    {"name": "JNPT Port International Navigation Channel", "lat": 18.940, "lng": 72.880, "type": "Major Port Channel", "vulnerability": "CRITICAL", "region": "Arabian Sea"},
    {"name": "Elephanta Island UNESCO Protected Marine Sanctuary", "lat": 18.960, "lng": 72.930, "type": "Marine Protected Area", "vulnerability": "CRITICAL", "region": "Arabian Sea"},
    {"name": "Thane Creek Flamingo & Mangrove Sanctuary", "lat": 19.040, "lng": 72.980, "type": "Mangrove Wetland Ecosystem", "vulnerability": "CRITICAL", "region": "Arabian Sea"},
    
    # Singapore Strait
    {"name": "Sisters' Islands Marine Park", "lat": 1.215, "lng": 103.835, "type": "Coral Reef Sanctuary", "vulnerability": "CRITICAL", "region": "Singapore Strait"},
    {"name": "Sentosa Island Recreational Beaches", "lat": 1.250, "lng": 103.820, "type": "Tourism Beach", "vulnerability": "HIGH", "region": "Singapore Strait"},
    
    # Bay of Bengal
    {"name": "Pulicat Lagoon Estuary & Mangroves", "lat": 13.420, "lng": 80.320, "type": "Estuarine Biosphere", "vulnerability": "CRITICAL", "region": "Bay of Bengal"},
    {"name": "Marina Coastal Biosphere", "lat": 13.040, "lng": 80.280, "type": "Public Coastline", "vulnerability": "HIGH", "region": "Bay of Bengal"},
]

class CoastalLandfallPredictor:
    def __init__(self):
        self.grid_engine = DynamicOceanGridEngine()
        self.weathering_engine = OilWeatheringEngine()

    def simulate_forward_drift_72h(
        self,
        start_lat: float,
        start_lng: float,
        initial_volume_m3: float,
        base_wind_speed: float,
        base_wind_deg: float,
        base_current_speed: float,
        base_current_deg: float,
        start_time_iso: str = "2026-09-01T06:00:00Z"
    ) -> Dict[str, Any]:
        """
        Runs forward Lagrangian simulation in 1-hour increments up to +72 hours.
        """
        dt_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        
        trajectory_points = []
        cur_lat, cur_lng = start_lat, start_lng
        cone_polygons = []

        landfall_hit = None
        min_shore_dist_km = float("inf")
        closest_target = None

        for h in range(1, 73):
            # Dynamic wind/current vectors at current location and time
            dyn = self.grid_engine.get_interpolated_vectors(
                lat=cur_lat,
                lng=cur_lng,
                hour_offset=float(h),
                base_wind_speed=base_wind_speed,
                base_wind_deg=base_wind_deg,
                base_current_speed=base_current_speed,
                base_current_deg=base_current_deg
            )

            # Move forward 1 hour
            # Net velocity in km/h
            net_u_kmh = dyn["net_u_ms"] * 3.6
            net_v_kmh = dyn["net_v_ms"] * 3.6

            d_north_km = net_v_kmh * 1.0
            d_east_km = net_u_kmh * 1.0

            cur_lat += (d_north_km / 111.139)
            cur_lng += (d_east_km / (111.139 * math.cos(math.radians(cur_lat))))

            pt_time = (dt_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Dispersion uncertainty cone radius (widens over time)
            cone_radius_km = round(0.4 + 0.15 * math.sqrt(h), 2)

            trajectory_points.append({
                "hour": h,
                "timestamp": pt_time,
                "lat": round(cur_lat, 6),
                "lng": round(cur_lng, 6),
                "uncertainty_radius_km": cone_radius_km,
                "wind_speed_ms": dyn["wind_speed_ms"],
                "current_speed_ms": dyn["current_speed_ms"]
            })

            # Check proximity to coastal targets
            for target in COASTAL_TARGETS:
                dist = haversine_distance_km(cur_lat, cur_lng, target["lat"], target["lng"])
                if dist < min_shore_dist_km:
                    min_shore_dist_km = round(dist, 2)
                    closest_target = target

                # If slick gets within 3.5 km of target, mark landfall impact!
                if dist <= 3.5 and landfall_hit is None:
                    landfall_hit = {
                        "target_name": target["name"],
                        "target_type": target["type"],
                        "vulnerability": target["vulnerability"],
                        "eta_hours": h,
                        "impact_timestamp": pt_time,
                        "impact_lat": round(cur_lat, 6),
                        "impact_lng": round(cur_lng, 6),
                        "distance_km": round(dist, 2)
                    }

        # If no direct hit within 3.5km, calculate closest approach ETA
        if not landfall_hit and closest_target and min_shore_dist_km < 15.0:
            est_hours = round(min_shore_dist_km / 1.2, 1)
            landfall_hit = {
                "target_name": closest_target["name"],
                "target_type": closest_target["type"],
                "vulnerability": closest_target["vulnerability"],
                "eta_hours": est_hours,
                "impact_timestamp": (dt_start + timedelta(hours=est_hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "impact_lat": closest_target["lat"],
                "impact_lng": closest_target["lng"],
                "distance_km": min_shore_dist_km,
                "is_closest_approach": True
            }

        # Threat classification
        if landfall_hit:
            threat_level = "CRITICAL_SHORELINE_THREAT" if landfall_hit["eta_hours"] <= 24 else "HIGH_RISK_APPROACH"
        else:
            threat_level = "OFFSHORE_SAFE_CORRIDOR"

        # Containment & Response Recommendations
        containment_plan = {
            "containment_booms_recommended_m": 1200 if initial_volume_m3 > 10 else 600,
            "skimmer_vessels_needed": 2 if initial_volume_m3 > 10 else 1,
            "chemical_dispersant_status": "PROHIBITED (Within 10km Coastal Buffer)" if min_shore_dist_km < 10.0 else "AUTHORIZED FOR DEEP SEA APPLICATION",
            "priority_defense_site": closest_target["name"] if closest_target else "Open Sea Corridor",
            "suggested_barrier_coords": [round(start_lat + 0.04, 4), round(start_lng + 0.04, 4)]
        }

        # 72h Weathering projection
        weathering_curve = self.weathering_engine.generate_72h_weathering_curve(
            initial_volume_m3=initial_volume_m3,
            wind_speed_ms=base_wind_speed
        )

        return {
            "threat_level": threat_level,
            "landfall_impact": landfall_hit,
            "closest_coastal_target": closest_target,
            "min_distance_to_shore_km": min_shore_dist_km,
            "trajectory_72h": trajectory_points,
            "containment_plan": containment_plan,
            "weathering_timeline": weathering_curve
        }
