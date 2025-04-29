#!/usr/bin/env python3
from scipy.interpolate import CubicSpline
from cereal import log

class DynamicPersonalityController:
  """
  Controller for managing dynamic personality-based following distances
  that adapt based on vehicle speed and selected driving personality.
  """

  def __init__(self):
    pass

  def get_dynamic_follow_distance(self, v_ego, personality=log.LongitudinalPersonality.standard):
    """
    Calculate the dynamic follow distance based on current speed and personality.

    Args:
        v_ego: Current vehicle speed
        personality: Selected longitudinal personality mode

    Returns:
        float: The calculated follow distance factor
    """
    if personality == log.LongitudinalPersonality.relaxed:
      x_vel =  [0., 8,  40]
      y_dist = [1.25, 1.25, 1.75]
    elif personality == log.LongitudinalPersonality.standard:
      x_vel =  [0.,  8.,  40]
      y_dist = [1.20, 1.20, 1.50]
    elif personality == log.LongitudinalPersonality.aggressive:
      x_vel =  [0.,  5.,  13.,  40]
      y_dist = [1.18, 1.18, 1.15, 1.25]
    else:
      raise NotImplementedError("Dynamic personality not supported")

    # Ensure we don't exceed the maximum of our defined range
    v_ego = min(v_ego, max(x_vel))

    # Use cubic spline interpolation for smooth transitions
    cs = CubicSpline(x_vel, y_dist)
    result = float(cs(v_ego))

    # Add print statement
    #print(f"DynamicPersonalityController: v_ego: {v_ego:.2f}, personality: {personality_name}, follow distance: {result:.2f}")

    return result