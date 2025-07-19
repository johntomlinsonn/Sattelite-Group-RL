"""
Utility functions for Earth-related calculations in the 3D satellite environment.
"""

import numpy as np

def spherical_to_cartesian(lat, lon, radius):
    """Convert spherical coordinates to cartesian."""
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def calculate_coverage(satellite_positions, earth_points, coverage_radius):
    """Calculate coverage of Earth points by satellites."""
    pass
