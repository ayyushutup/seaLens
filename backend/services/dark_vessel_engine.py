"""
Dark Vessel Detection & AIS Cross-Matching Intelligence Service.

Correlates physical ships detected on SAR radar with real-time AIS broadcasts
to flag non-broadcasting 'Dark Vessels' evading maritime surveillance.
"""
from typing import List, Dict, Any, Tuple
from ml_engine.metrics import haversine_distance_km
from backend.models import VesselTrack

class DarkVesselEngine:
    def __init__(self, match_tolerance_km: float = 1.8):
        self.match_tolerance_km = match_tolerance_km

    def cross_match_radar_and_ais(
        self,
        radar_ships: List[Dict[str, Any]],
        ais_vessels: List[VesselTrack],
        origin_lat: float,
        origin_lng: float
    ) -> List[Dict[str, Any]]:
        """
        Cross-matches radar-detected ships with AIS telemetry.
        
        Returns a list of categorized intelligence targets:
        - AIS-Compliant Vessels
        - 🚨 DARK VESSELS (No AIS broadcast)
        """
        results = []

        for r_ship in radar_ships:
            r_lat = r_ship["lat"]
            r_lng = r_ship["lng"]

            # Search for closest AIS vessel
            matched_ais = None
            min_dist_km = float("inf")

            for vessel in ais_vessels:
                if not vessel.positions:
                    continue
                # Compare against latest or closest position
                for pos in vessel.positions:
                    dist = haversine_distance_km(r_lat, r_lng, pos.lat, pos.lng)
                    if dist < min_dist_km:
                        min_dist_km = dist
                        matched_ais = {
                            "mmsi": vessel.metadata.mmsi,
                            "name": vessel.metadata.name,
                            "ship_type": vessel.metadata.ship_type,
                            "flag": vessel.metadata.flag,
                            "distance_km": round(dist, 3)
                        }

            # Proximity to spill origin
            dist_to_spill_origin = round(haversine_distance_km(r_lat, r_lng, origin_lat, origin_lng), 2)

            if matched_ais and min_dist_km <= self.match_tolerance_km:
                # AIS MATCH CONFIRMED
                results.append({
                    "id": r_ship["id"],
                    "is_dark_vessel": False,
                    "status": "AIS-TRANSMITTING",
                    "lat": r_lat,
                    "lng": r_lng,
                    "radar_length_m": r_ship["estimated_length_m"],
                    "radar_beam_m": r_ship["estimated_beam_m"],
                    "matched_ais_name": matched_ais["name"],
                    "matched_mmsi": matched_ais["mmsi"],
                    "matched_type": matched_ais["ship_type"],
                    "matched_flag": matched_ais["flag"],
                    "ais_offset_m": int(min_dist_km * 1000),
                    "dist_to_spill_origin_km": dist_to_spill_origin,
                    "threat_level": "NORMAL"
                })
            else:
                # 🚨 DARK VESSEL DETECTED
                threat = "CRITICAL_SUSPECT" if dist_to_spill_origin < 5.0 else "SURVEILLANCE_TARGET"
                results.append({
                    "id": r_ship["id"],
                    "is_dark_vessel": True,
                    "status": "🚨 DARK VESSEL (AIS OFFLINE / SPOOFED)",
                    "lat": r_lat,
                    "lng": r_lng,
                    "radar_length_m": r_ship["estimated_length_m"],
                    "radar_beam_m": r_ship["estimated_beam_m"],
                    "matched_ais_name": "UNKNOWN_DARK_TARGET",
                    "matched_mmsi": None,
                    "matched_type": f"Unregistered Vessel (~{int(r_ship['estimated_length_m'])}m Hull)",
                    "matched_flag": "UNIDENTIFIED",
                    "ais_offset_m": None,
                    "dist_to_spill_origin_km": dist_to_spill_origin,
                    "threat_level": threat,
                    "intel_notes": [
                        f"Physical radar echo detected: {r_ship['estimated_length_m']}m length, SNR {r_ship['cfar_snr_db']} dB.",
                        "Zero AIS broadcast received in sector (Transponder deliberately disabled or faulty).",
                        f"Distance to back-tracked oil discharge cone: {dist_to_spill_origin} km."
                    ]
                })

        return results
