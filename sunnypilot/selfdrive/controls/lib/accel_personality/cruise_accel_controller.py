"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom
from numpy import interp
from openpilot.common.params import Params

AccelPersonality = custom.LongitudinalPlanSP.AccelerationPersonality

class CruiseAccelController:
  """
  Controls dynamic cruise minimum acceleration based on vehicle speed and driving personality.
  """

  def __init__(self):
    # Speed breakpoints (m/s)
    self.speed_breakpoints = [0.,  .01,  .02,   .3,    1.]

    # Minimum allowed accelerations (m/s²) corresponding to each speed breakpoint for different personalities
    self.min_accel_vals_stock = [-1.0, -1.1, -1.2, -1.3, -1.4]
    self.min_accel_vals_normal = [-1.1, -1.2, -1.3, -1.4, -1.5]  # New values for normal
    self.min_accel_vals_eco = [-0.8, -0.9, -1.0, -1.1, -1.2]
    self.min_accel_vals_sport = [-1.2, -1.3, -1.4, -1.5, -1.6]

    # Default personality is stock
    self._personality = AccelPersonality.stock
    self._params = Params()

  def _read_params(self):
    """
    Reads the acceleration personality from Params.
    """
    # Read the personality from params (could be stock, eco, normal, or sport)
    personality_str = self._params.get("AccelPersonality", encoding='utf-8')
    if personality_str is not None:
      personality_int = int(personality_str)
      if personality_int in [AccelPersonality.stock, AccelPersonality.normal, AccelPersonality.eco, AccelPersonality.sport]:
        self._personality = personality_int

  def get_min_accel(self, v_ego):
    """
    Interpolates and returns the minimum allowed cruise acceleration based on vehicle speed and personality.

    :param v_ego: Vehicle speed in m/s
    :return: Minimum acceleration (m/s²)
    """
    # Read the parameters to ensure the personality is up to date
    self._read_params()

    # Debug print to show current personality and speed
    print(f"Personality: {self._personality}")
    print(f"Vehicle speed (v_ego): {v_ego}")

    # Select appropriate min acceleration values based on personality
    if self._personality == AccelPersonality.eco:
      min_accel_vals = self.min_accel_vals_eco
      print("Using eco personality")
    elif self._personality == AccelPersonality.sport:
      min_accel_vals = self.min_accel_vals_sport
      print("Using sport personality")
    elif self._personality == AccelPersonality.normal:
      min_accel_vals = self.min_accel_vals_normal  # Now using normal personality values
      print("Using normal personality")
    else:
      min_accel_vals = self.min_accel_vals_stock
      print("Using stock personality")

    # Debug print to show selected min_accel_vals for the current personality
    print(f"Selected min_accel_vals: {min_accel_vals}")

    # Calculate the minimum acceleration by interpolation
    min_accel = interp(v_ego, self.speed_breakpoints, min_accel_vals)

    # Debug print to show the result of interpolation
    print(f"Interpolated minimum acceleration: {min_accel}")

    return min_accel
