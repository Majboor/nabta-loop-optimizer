"""solar.py — sun position for the Al Safa Park 2 site (25.1635 N, 55.2308 E).

Standard textbook formulas (Cooper declination, Spencer equation of time,
hour-angle geometry). Accuracy is a fraction of a degree — plenty for a
shade-placement proof of concept.

The design condition of the nabta entry: 15:00 local (UTC+4) on 21 June,
the hottest walkable hour of the solstice afternoon.
"""

from __future__ import annotations

import numpy as np

LAT = 25.1635        # deg N
LON = 55.2308        # deg E
TZ_OFFSET = 4.0      # Gulf Standard Time, hours ahead of UTC
DAY_OF_YEAR = 172    # 21 June (non-leap year)
LOCAL_HOUR = 15.0    # 15:00 local


def solar_position(lat: float = LAT, lon: float = LON,
                   day_of_year: int = DAY_OF_YEAR,
                   local_hour: float = LOCAL_HOUR,
                   tz_offset: float = TZ_OFFSET) -> tuple[float, float]:
    """Return (elevation_deg, azimuth_deg). Azimuth measured from north,
    clockwise (E = 90, S = 180, W = 270)."""
    n = day_of_year

    # Cooper (1969) declination, deg
    dec = 23.45 * np.sin(np.radians(360.0 * (284 + n) / 365.0))

    # Spencer (1971) equation of time, minutes
    b = 2.0 * np.pi * (n - 1) / 365.0
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                    - 0.014615 * np.cos(2 * b) - 0.040849 * np.sin(2 * b))

    # Local clock time -> apparent solar time (standard meridian = 15 * tz)
    solar_time = local_hour + (4.0 * (lon - 15.0 * tz_offset) + eot) / 60.0
    hour_angle = 15.0 * (solar_time - 12.0)  # deg, afternoon positive

    lat_r, dec_r, ha_r = map(np.radians, (lat, dec, hour_angle))

    sin_el = (np.sin(lat_r) * np.sin(dec_r)
              + np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
    el = np.degrees(np.arcsin(sin_el))

    cos_az = ((np.sin(dec_r) - np.sin(lat_r) * sin_el)
              / (np.cos(lat_r) * np.cos(np.radians(el))))
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    if hour_angle > 0:          # afternoon -> sun in the western sky
        az = 360.0 - az
    return float(el), float(az)


def shadow_offset(height: float,
                  elevation_deg: float | None = None,
                  azimuth_deg: float | None = None) -> np.ndarray:
    """Horizontal displacement (dx_east, dy_north) of the shadow cast by a
    canopy at `height` metres: length = h / tan(el), direction opposite the
    sun's azimuth."""
    if elevation_deg is None or azimuth_deg is None:
        elevation_deg, azimuth_deg = solar_position()
    length = height / np.tan(np.radians(elevation_deg))
    away = np.radians(azimuth_deg + 180.0)          # shadow points away from sun
    return np.array([length * np.sin(away), length * np.cos(away)])


if __name__ == "__main__":
    el, az = solar_position()
    off = shadow_offset(3.2, el, az)
    print(f"21 Jun, 15:00 GST @ {LAT:.4f} N {LON:.4f} E")
    print(f"  elevation : {el:6.2f} deg")
    print(f"  azimuth   : {az:6.2f} deg (from N, clockwise)")
    print(f"  a 3.2 m pergola canopy casts its shadow "
          f"{np.linalg.norm(off):.2f} m toward ({off[0]:+.2f} E, {off[1]:+.2f} N)")
