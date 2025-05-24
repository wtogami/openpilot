# The MIT License
#
# Copyright (c) 2019-, Rick Lan, dragonpilot community, and a number of other of contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Version = 2025-1-18

import numpy as np

from cereal import messaging
from opendbc.car import structs
from numpy import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import WMACConstants

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33


class KalmanFilter:
  """
  Simple one-dimensional Kalman filter implementation with forgetting factor.
  - x: state estimate
  - P: error covariance
  - R: measurement error (uncertainty in measurements)
  - Q: process noise (uncertainty in system dynamics)
  - alpha: forgetting factor (1.0 = no forgetting, >1.0 = forget old data faster)
  - window_size_equivalent: approx window size this filter behavior matches
  """
  def __init__(self, initial_value=0, initial_estimate_error=1.0, measurement_noise=0.1,
               process_noise=0.01, alpha=1.0, window_size_equivalent=None):
    # State estimate
    self.x = initial_value
    # Error covariance
    self.P = initial_estimate_error
    # Measurement noise
    self.R = measurement_noise
    # Process noise
    self.Q = process_noise
    # Forgetting factor (>1.0 means forget old data faster)
    self.alpha = alpha
    # Whether the filter has been initialized with data
    self.initialized = False
    # History of most recent measurements for debugging and adaptive behavior
    self.history = []
    self.max_history = 10
    # For diagnostic purposes
    self.window_size_equivalent = window_size_equivalent

  def add_data(self, measurement):
    # Keep track of some history for debugging
    if len(self.history) >= self.max_history:
      self.history.pop(0)
    self.history.append(measurement)

    if not self.initialized:
      self.x = measurement
      self.initialized = True
      return

    # Prediction step (state extrapolation)
    # x = x (no system dynamics in this simple implementation)
    # Error covariance extrapolation with forgetting factor
    self.P = self.alpha * self.P + self.Q

    # Update step
    # Compute Kalman gain
    K = self.P / (self.P + self.R)
    # Update state estimate with measurement
    self.x = self.x + K * (measurement - self.x)
    # Update error covariance
    self.P = (1 - K) * self.P

  def get_value(self):
    return self.x if self.initialized else None

  def get_recent_trend(self):
    """Return the trend from recent data, -1 (decreasing), 0 (stable), 1 (increasing)"""
    if len(self.history) < 3:
      return 0

    # Simple trend detector based on recent history
    recent = self.history[-3:]
    if recent[2] > recent[0] + 0.05:  # Increasing with threshold
      return 1
    elif recent[0] > recent[2] + 0.05:  # Decreasing with threshold
      return -1
    return 0

  def reset_data(self):
    self.initialized = False
    self.history = []


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._mode: str = 'acc'
    self._frame: int = 0
    self._urgency = 0.0

    # Simple anti-oscillation variables
    self._mode_stability_counter = 0  # Counter to prevent rapid mode switching
    self._min_mode_duration = 15     # Minimum frames to stay in a mode (about 0.3 seconds)

    # Using Kalman filters for improved filtering - with more stable settings

    # Lead vehicle tracking
    self._lead_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=1.0,
      measurement_noise=0.2,   # Stable lead detection
      process_noise=0.05,
      alpha=1.02,
      window_size_equivalent=WMACConstants.LEAD_WINDOW_SIZE
    )
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False

    # Slow down detection - more stable
    self._slow_down_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=1.0,
      measurement_noise=0.25,  # More stable slow down detection
      process_noise=0.08,
      alpha=1.03,
      window_size_equivalent=WMACConstants.SLOW_DOWN_WINDOW_SIZE
    )
    self._has_slow_down: bool = False

    # Slowness detection - more stable
    self._slowness_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=1.0,
      measurement_noise=0.12,  # More stable slowness detection
      process_noise=0.04,
      alpha=1.01,              # Very stable
      window_size_equivalent=WMACConstants.SLOWNESS_WINDOW_SIZE
    )
    self._has_slowness: bool = False

    # For tracking lead vehicle distance
    self._lead_dist_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=5.0,
      measurement_noise=0.2,
      process_noise=1.0,
      alpha=1.05
    )
    self._lead_dist = 0.0

    # For tracking lead vehicle relative velocity
    self._lead_vel_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=2.0,
      measurement_noise=0.15,
      process_noise=0.5,
      alpha=1.1
    )
    self._lead_rel_vel = 0.0

    # Acceleration filter to detect rapid deceleration of lead
    self._lead_accel_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=1.0,
      measurement_noise=0.2,
      process_noise=1.0,
      alpha=1.2
    )
    self._lead_accel = 0.0
    self._prev_lead_vel = 0.0

    # Trajectory curvature detection for curves
    self._curvature_filter = KalmanFilter(
      initial_value=0,
      initial_estimate_error=1.0,
      measurement_noise=0.3,
      process_noise=0.06,
      alpha=1.02
    )
    self._curvature = 0.0
    self._high_curvature = False

    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.
    self._has_standstill = False

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return str(self._mode)

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = car_state.vCruise
    self._has_standstill = car_state.standstill

    # Lead detection with Kalman filtering
    self._lead_filter.add_data(float(lead_one.status))
    lead_filtered_value = self._lead_filter.get_value() or 0.0
    self._has_lead_filtered = lead_filtered_value > WMACConstants.LEAD_PROB

    # Track lead vehicle parameters if present
    if lead_one.status:
      # Track lead distance
      self._lead_dist_filter.add_data(lead_one.dRel)
      self._lead_dist = self._lead_dist_filter.get_value() or 0.0

      # Track lead relative velocity
      self._lead_vel_filter.add_data(lead_one.vRel)
      current_vel = self._lead_vel_filter.get_value() or 0.0
      self._lead_rel_vel = current_vel

      # Calculate lead acceleration from velocity changes
      if self._prev_lead_vel != 0:
        accel = (current_vel - self._prev_lead_vel) / DT_MDL
        self._lead_accel_filter.add_data(accel)
        self._lead_accel = self._lead_accel_filter.get_value() or 0.0
      self._prev_lead_vel = current_vel

    # Calculate path curvature
    if len(md.position.x) == len(md.position.y) == TRAJECTORY_SIZE:
      idx1, idx2, idx3 = 5, 15, 25

      if idx3 < TRAJECTORY_SIZE:
        try:
          # Calculate vectors between points
          v1x = md.position.x[idx2] - md.position.x[idx1]
          v1y = md.position.y[idx2] - md.position.y[idx1]
          v2x = md.position.x[idx3] - md.position.x[idx2]
          v2y = md.position.y[idx3] - md.position.y[idx2]

          # Calculate vector magnitudes
          mag1 = (v1x**2 + v1y**2)**0.5
          mag2 = (v2x**2 + v2y**2)**0.5

          if mag1 > 0.1 and mag2 > 0.1:
            # Calculate the dot product
            dot_product = v1x * v2x + v1y * v2y
            # Calculate the cosine of the angle
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            # Calculate the angle
            angle = np.arccos(cos_angle)
            # Normalize by the distance to get a curvature measure
            curvature = angle / ((mag1 + mag2) / 2)

            self._curvature_filter.add_data(curvature)
            self._curvature = self._curvature_filter.get_value() or 0.0
            self._high_curvature = self._curvature > 0.045
        except Exception:
          pass

    # Slow down detection with hysteresis
    slow_down_threshold = float(
      interp(self._v_ego_kph, WMACConstants.SLOW_DOWN_BP, WMACConstants.SLOW_DOWN_DIST)
    )

    curv_score = np.clip(self._curvature / 0.08, 0.0, 1.0)
    endpt_score = 0.0
    if len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE:
      endpoint_x = md.position.x[TRAJECTORY_SIZE - 1]
      endpt_score = np.clip((slow_down_threshold - endpoint_x) / 10.0, 0.0, 1.0)

    urgency = max(curv_score, endpt_score)
    self._slow_down_filter.add_data(urgency)
    urgency_filtered = self._slow_down_filter.get_value() or 0.0

    # Apply hysteresis to slow down detection
    if self._has_slow_down:
      # Currently detecting slow down, use lower threshold to stay stable
      self._has_slow_down = urgency_filtered > (WMACConstants.SLOW_DOWN_PROB - 0.08)
    else:
      # Not currently detecting, use normal threshold
      self._has_slow_down = urgency_filtered > WMACConstants.SLOW_DOWN_PROB

    self._urgency = urgency_filtered

    # Slowness detection with hysteresis
    if not self._has_standstill:
      slowness_raw = float(self._v_ego_kph <= (self._v_cruise_kph * WMACConstants.SLOWNESS_CRUISE_OFFSET))
      self._slowness_filter.add_data(slowness_raw)
      slowness_filtered_value = self._slowness_filter.get_value() or 0.0

      # Apply hysteresis to slowness detection
      if self._has_slowness:
        # Currently slow, use lower threshold to stay stable
        self._has_slowness = slowness_filtered_value > (WMACConstants.SLOWNESS_PROB - 0.08)
      else:
        # Not currently slow, use normal threshold
        self._has_slowness = slowness_filtered_value > WMACConstants.SLOWNESS_PROB

    self._has_lead_filtered_prev = self._has_lead_filtered

  def _should_use_blended(self) -> bool:
    """Simple logic to determine if blended mode should be used"""

    # Always use blended when stopped
    if self._has_standstill:
      return True

    # Use blended when detecting slow down scenarios
    if self._has_slow_down:
      return True

    # Use blended for high curvature at speed
    if self._high_curvature and self._v_ego_kph > 45.0:
      return True

    # For radar mode: don't switch to blended just because we have a lead
    # Keep it simple - lead following is handled well by ACC

    # Default to ACC mode
    return False

  def _set_mode_with_stability(self, target_mode: str) -> None:
    """Set mode with anti-oscillation logic"""

    # If we want to change modes
    if target_mode != self._mode:
      # Reset stability counter when we want to change
      if self._mode_stability_counter <= 0:
        self._mode_stability_counter = self._min_mode_duration
        self._mode = target_mode
      else:
        # Still in stability period, don't change yet
        self._mode_stability_counter -= 1
    else:
      # Same mode as current, reset counter
      self._mode_stability_counter = max(0, self._mode_stability_counter - 1)

  def _radarless_mode(self) -> None:
    """Simple radarless mode logic"""

    # Determine target mode
    if self._should_use_blended():
      target_mode = 'blended'
    else:
      target_mode = 'acc'

    # Set mode with stability check
    self._set_mode_with_stability(target_mode)

  def _radar_mode(self) -> None:
    """Simple radar mode logic"""

    # Same logic as radarless for now - keep it simple
    # Lead vehicle handling is already good in ACC mode

    if self._should_use_blended():
      target_mode = 'blended'
    else:
      target_mode = 'acc'

    # Set mode with stability check
    self._set_mode_with_stability(target_mode)

  def update(self, sm: messaging.SubMaster) -> None:
    self._read_params()
    self._update_calculations(sm)

    if self._CP.radarUnavailable:
      self._radarless_mode()
    else:
      self._radar_mode()

    self._active = sm['selfdriveState'].experimentalMode and self._enabled
    self._frame += 1