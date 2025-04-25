"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from scipy.interpolate import CubicSpline
import numpy as np

# Acceleration profile for maximum allowed acceleration
MAX_ACCEL_PROFILES = {
  "eco":    [2.00, 1.90, 1.85, 1.52, 0.90, .59,  .48, .39, .32, .12],
  "normal": [2.00, 1.95, 1.85, 1.70, 1.10, .75,  .61, .50, .38, .2],
  "sport":  [2.00, 2.00, 1.98, 1.90, 1.30, 1.00, .72, .60, .48, .3],
}

# Acceleration profile for minimum (braking) acceleration
MIN_ACCEL_PROFILES = {
  "eco":    [-0.006, -0.005, -0.009, -1.2, -1.2],
  "normal": [-0.007, -0.006, -0.010, -1.2, -1.2],
  "sport":  [-0.008, -0.007, -0.011, -1.3, -1.3],
  "stock":  [-1.2,   -1.2,   -1.2,  -1.2, -1.2],
}

# Speed breakpoints for interpolation
MAX_ACCEL_BREAKPOINTS = [0., 1., 6., 8., 11., 16., 20., 25., 30., 55.]
MIN_ACCEL_BREAKPOINTS = [0.,  3.,  14.,  20., 40.]

# Create cubic splines for all acceleration profiles
MAX_ACCEL_SPLINES = {
  mode: CubicSpline(MAX_ACCEL_BREAKPOINTS, values, bc_type='clamped')
  for mode, values in MAX_ACCEL_PROFILES.items()
}

MIN_ACCEL_SPLINES = {
  mode: CubicSpline(MIN_ACCEL_BREAKPOINTS, values, bc_type='clamped')
  for mode, values in MIN_ACCEL_PROFILES.items()
}

def get_max_accel_spline(v_ego: float, mode: str = "normal") -> float:
  v_ego_clipped = np.clip(v_ego, MAX_ACCEL_BREAKPOINTS[0], MAX_ACCEL_BREAKPOINTS[-1])
  spline = MAX_ACCEL_SPLINES.get(mode, MAX_ACCEL_SPLINES["normal"])
  return float(spline(v_ego_clipped))

def get_min_accel_spline(v_ego: float, mode: str = "normal") -> float:
  v_ego_clipped = np.clip(v_ego, MIN_ACCEL_BREAKPOINTS[0], MIN_ACCEL_BREAKPOINTS[-1])
  spline = MIN_ACCEL_SPLINES.get(mode, MIN_ACCEL_SPLINES["normal"])
  return float(spline(v_ego_clipped))