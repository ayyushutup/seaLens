# Maritime Sentinel: Oil Spill Detection & AIS Vessel Correlation System
**SIH Problem #143 (NTRO)**

## System Overview
Maritime Sentinel is an AI-powered maritime environmental defense and forensic intelligence system. It pairs Synthetic Aperture Radar (SAR) satellite Earth Observation with global Automatic Identification System (AIS) vessel telemetry to detect illegal oil discharges at sea, reverse-simulate ocean drift, and correlate culprit vessels with forensic certainty.

### Key Capabilities
1. **SAR Oil Slick Segmentation Engine**: Uses polarimetric SAR features (VV/VH ratio, speckle suppression, entropy) + deep convolutional segmentation (U-Net / SegFormer) to delineate slick contours and calculate spill volume/area.
2. **False Positive & Look-Alike Rejection**: Employs oceanographic context (wind speed from ECMWF/ERA5, biogenic film characteristics) to weed out natural low-wind calm water and algae blooms.
3. **Reverse Lagrangian Drift Physics**: Back-projects the slick's movement using vector fields of ocean surface currents and wind leeway ($3\%$ rule) to establish the spatio-temporal origin cone $(X_0, Y_0, T_0)$.
4. **PostGIS Spatio-Temporal Trajectory Correlation**: Intersects the origin cone against AIS vessel tracks, evaluating Closest Point of Approach (CPA), speed anomalies, and vessel risk priors.
5. **Interactive C2 Dashboard & Evidence Dossier**: Real-time map with timeline scrubber, vessel telemetry drilldowns, and 1-click legal PDF evidence reports.
