# Incident report: battery range misreporting, 2 October 2025

**Severity:** SEV-2
**Incident owner:** Mira Fernandes (Firmware)

## Impact
After the over-the-air release of battery management firmware v2.3.1 on 29 September,
37 customers with the Battery Pack 48V reported the range estimate dropping to 0 km while
the battery was still at about 20% charge. Bikes kept working, but pedal assist switched to
its lowest level. No safety issue was found.

## Root cause
A recalibrated voltage-to-charge lookup table in v2.3.1 had one row shifted by a single
entry, so readings between 3.45 V and 3.50 V per cell were mapped to 0% instead of 20%.

## Fix
- Rolled back affected bikes to v2.2.8 over the air on 2 October (within 3 days of release).
- Released v2.3.2 with the corrected table on 10 October.
- Added a unit test that checks the lookup table is monotonic.
