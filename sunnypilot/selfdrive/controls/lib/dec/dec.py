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
from typing import Optional, List, Tuple

from cereal import messaging
from opendbc.car import structs
from numpy import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import WMACConstants, SNG_State

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33

STOP_AND_GO_FRAME = 60

SET_MODE_TIMEOUT = 10

# Added for hysteresis
MODE_HYSTERESIS = {
  'acc_to_blended': 0.70,    # Threshold to switch to blended mode
  'blended_to_acc': 0.45      # Threshold to switch back to acc mode
}


class GenericMovingAverageCalculator:
  def __init__(self, window_size):
    self.window_size = window_size
    self.data = []
    self.total = 0

  def add_data(self, value: float) -> None:
    if len(self.data) == self.window_size:
      self.total -= self.data.pop(0)
    self.data.append(value)
    self.total += value

  def get_moving_average(self) -> Optional[float]:
    return None if len(self.data) == 0 else self.total / len(self.data)

  def reset_data(self) -> None:
    self.data = []
    self.total = 0


class WeightedMovingAverageCalculator:
  def __init__(self, window_size, weight_profile="linear"):
    self.window_size = window_size
    self.data = []

    # Enhanced weight profiles for different use cases
    if weight_profile == "linear":
      self.weights = np.linspace(1, 3, window_size)
    elif weight_profile == "responsive":
      # More weight on recent values for quicker response to changes
      self.weights = np.power(1.3, np.linspace(0, window_size-1, window_size))
    elif weight_profile == "stable":
      # More balanced weights for stable readings
      self.weights = np.linspace(0.8, 1.2, window_size)
    else:
      self.weights = np.linspace(1, 3, window_size)

  def add_data(self, value: float) -> None:
    if len(self.data) == self.window_size:
      self.data.pop(0)
    self.data.append(value)

  def get_weighted_average(self) -> Optional[float]:
    if len(self.data) == 0:
      return None
    weighted_sum: float = float(np.dot(self.data, self.weights[-len(self.data):]))
    weight_total: float = float(np.sum(self.weights[-len(self.data):]))
    return weighted_sum / weight_total

  def reset_data(self) -> None:
    self.data = []


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._mode: str = 'acc'
    self._frame: int = 0
    self._prev_mode: str = 'acc'  # Track previous mode for transition smoothness
    self._mode_stable_frames: int = 0  # Count frames in current mode

    # Use specialized weight profiles for different detectors
    self._lead_gmac = WeightedMovingAverageCalculator(
      window_size=WMACConstants.LEAD_WINDOW_SIZE,
      weight_profile="stable"
    )
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False

    # Use responsive profile for quicker slow-down detection
    self._slow_down_gmac = WeightedMovingAverageCalculator(
      window_size=WMACConstants.SLOW_DOWN_WINDOW_SIZE,
      weight_profile="responsive"
    )
    self._has_slow_down: bool = False
    self._slow_down_confidence: float = 0.0

    # Store previous distance readings to detect trends
    self._prev_distances: List[float] = []
    self._max_distance_history = 5

    self._has_blinkers = False

    self._slowness_gmac = WeightedMovingAverageCalculator(
      window_size=WMACConstants.SLOWNESS_WINDOW_SIZE,
      weight_profile="stable"
    )
    self._has_slowness: bool = False

    self._has_nav_instruction = False

    self._dangerous_ttc_gmac = WeightedMovingAverageCalculator(
      window_size=WMACConstants.DANGEROUS_TTC_WINDOW_SIZE,
      weight_profile="responsive"
    )
    self._has_dangerous_ttc: bool = False

    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.

    self._has_lead = False

    self._has_standstill = False
    self._has_standstill_prev = False

    self._sng_transit_frame = 0
    self._sng_state = SNG_State.off

    self._mpc_fcw_gmac = WeightedMovingAverageCalculator(
      window_size=WMACConstants.MPC_FCW_WINDOW_SIZE,
      weight_profile="responsive"
    )
    self._has_mpc_fcw: bool = False
    self._mpc_fcw_crash_cnt = 0

    self._set_mode_timeout = 0

    # Road curvature detection
    self._curve_detected = False
    self._curve_severity: float = 0.0

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return str(self._mode)

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active

  def _filter_false_positives(self, trigger_value: bool, previous_values: List[float],
                              required_confirmations: int = 2) -> bool:
    """
    Filter potential false positives by requiring multiple confirmations.
    """
    if not trigger_value:
      return False

    # Check if we have enough history
    if len(previous_values) < required_confirmations:
      return False

    # Count recent triggers
    recent_triggers = sum(previous_values[-required_confirmations:])

    # Only confirm if we have enough past triggers
    return recent_triggers >= required_confirmations - 1

  def _detect_curvature(self, md) -> Tuple[bool, float]:
    """
    Detect upcoming road curvature from model predictions.
    Returns (curve_detected, curve_severity)
    """
    if len(md.orientation.x) != TRAJECTORY_SIZE:
      return False, 0.0

    # Calculate path curvature from orientation
    orientation_deltas = np.diff([o for o in md.orientation.y])
    max_curve = np.max(np.abs(orientation_deltas))

    # Detect if we're approaching a curve and estimate severity
    curve_detected = max_curve > 0.07  # Lower threshold for earlier detection
    curve_severity = min(1.0, max_curve / 0.2)

    return curve_detected, curve_severity

  def _adaptive_slowdown_threshold(self) -> float:
    """
    Adapts the slow-down threshold based on vehicle speed and context.
    """
    # Base threshold from lookup table
    base_threshold = interp(self._v_ego_kph, WMACConstants.SLOW_DOWN_BP, WMACConstants.SLOW_DOWN_DIST)

    # Apply context-based adjustments
    context_factor = 1.0

    # Adjust for curve detection
    if self._curve_detected:
      # Reduce threshold (increase sensitivity) when curve detected
      context_factor *= (1.0 - self._curve_severity * 0.2)

    # Apply small reduction for early detection but maintain false positive protection
    return float(base_threshold * context_factor * 0.95)

  def _detect_distance_trend(self, current_distance: float) -> bool:
    """
    Detect consistent decreasing trend in distance readings.
    """
    # Add current distance to history
    self._prev_distances.append(current_distance)
    if len(self._prev_distances) > self._max_distance_history:
      self._prev_distances.pop(0)

    # Need at least 3 readings to detect trend
    if len(self._prev_distances) < 3:
      return False

    # Check if distance is consistently decreasing
    window_size = min(3, len(self._prev_distances))
    window = self._prev_distances[-window_size:]

    # Must be decreasing pattern with significant change
    is_decreasing = all(window[i] > window[i+1] for i in range(len(window)-1))
    significant_change = window[0] - window[-1] > 1.0

    return is_decreasing and significant_change

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = car_state.vCruise
    self._has_lead = lead_one.status
    self._has_standstill = car_state.standstill

    # Detect curvature
    self._curve_detected, self._curve_severity = self._detect_curvature(md)

    # fcw detection
    self._mpc_fcw_gmac.add_data(self._mpc_fcw_crash_cnt > 0)
    if _mpc_fcw_weighted_average := self._mpc_fcw_gmac.get_weighted_average():
      self._has_mpc_fcw = _mpc_fcw_weighted_average > WMACConstants.MPC_FCW_PROB
    else:
      self._has_mpc_fcw = False

    # lead detection
    self._lead_gmac.add_data(float(lead_one.status))
    if _lead_weighted_average := self._lead_gmac.get_weighted_average():
      self._has_lead_filtered = _lead_weighted_average > WMACConstants.LEAD_PROB
    else:
      self._has_lead_filtered = False

    # Enhanced adaptive slow down detection with false positive protection
    adaptive_threshold = self._adaptive_slowdown_threshold()
    current_distance = (md.position.x[TRAJECTORY_SIZE - 1]
                        if (len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE)
                        else float('inf'))

    # Basic trigger
    slow_down_trigger = current_distance < adaptive_threshold

    # Add trend detection for earlier warning with false positive protection
    if self._detect_distance_trend(current_distance):
      slow_down_trigger = True

    # Add filtered data
    self._slow_down_gmac.add_data(slow_down_trigger)

    if _slow_down_weighted_average := self._slow_down_gmac.get_weighted_average():
      # Apply hysteresis for mode transitions
      threshold = (MODE_HYSTERESIS['acc_to_blended'] if self._mode == 'acc'
                   else MODE_HYSTERESIS['blended_to_acc'])

      # Filter for consistency to prevent false positives
      raw_trigger = _slow_down_weighted_average > threshold
      self._has_slow_down = self._filter_false_positives(raw_trigger, self._slow_down_gmac.data, 2)
      self._slow_down_confidence = _slow_down_weighted_average
    else:
      self._has_slow_down = False
      self._slow_down_confidence = 0.0

    # blinker detection
    self._has_blinkers = car_state.leftBlinker or car_state.rightBlinker

    # sng detection with smoother transitions
    if self._has_standstill:
      self._sng_state = SNG_State.stopped
      self._sng_transit_frame = 0
    else:
      if self._sng_transit_frame == 0:
        if self._sng_state == SNG_State.stopped:
          self._sng_state = SNG_State.going
          self._sng_transit_frame = STOP_AND_GO_FRAME
        elif self._sng_state == SNG_State.going:
          self._sng_state = SNG_State.off
      elif self._sng_transit_frame > 0:
        self._sng_transit_frame -= 1

    # slowness detection
    if not self._has_standstill:
      self._slowness_gmac.add_data(self._v_ego_kph <= (self._v_cruise_kph * WMACConstants.SLOWNESS_CRUISE_OFFSET))
      if _slowness_weighted_average := self._slowness_gmac.get_weighted_average():
        self._has_slowness = _slowness_weighted_average > WMACConstants.SLOWNESS_PROB
      else:
        self._has_slowness = False

    # dangerous TTC detection
    if not self._has_lead_filtered and self._has_lead_filtered_prev:
      self._dangerous_ttc_gmac.reset_data()
      self._has_dangerous_ttc = False

    if self._has_lead and car_state.vEgo >= 0.01:
      self._dangerous_ttc_gmac.add_data(lead_one.dRel / car_state.vEgo)

    if _dangerous_ttc_weighted_average := self._dangerous_ttc_gmac.get_weighted_average():
      self._has_dangerous_ttc = _dangerous_ttc_weighted_average <= WMACConstants.DANGEROUS_TTC
    else:
      self._has_dangerous_ttc = False

    # keep prev values
    self._has_standstill_prev = self._has_standstill
    self._has_lead_filtered_prev = self._has_lead_filtered

  def _radarless_mode(self) -> None:
    # Priority 1: Safety-critical conditions
    if self._has_mpc_fcw:
      self._set_mode('blended', high_priority=True)
      return

    # Priority 2: Stop and go handling
    if self._has_standstill:
      self._set_mode('blended')
      return

    # Priority 3: Slow down detection - now with better early detection
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # Priority 4: Dangerous TTC detection
    if self._has_dangerous_ttc:
      self._set_mode('blended')
      return

    # Priority 5: During SNG state transitions
    if self._sng_state == SNG_State.going:
      self._set_mode('blended')
      return

    # Priority 6: Driving slower than cruise setpoint
    if self._has_slowness:
      self._set_mode('acc')
      return

    # Default to ACC for steady-state cruising
    self._set_mode('acc')

  def _radar_mode(self) -> None:
    # Priority 1: Safety-critical conditions
    if self._has_mpc_fcw:
      self._set_mode('blended', high_priority=True)
      return

    # Priority 2: Lead vehicle handling
    if self._has_lead_filtered and not self._has_standstill:
      self._set_mode('acc')
      return

    # Priority 3: Standstill handling
    if self._has_standstill:
      self._set_mode('blended')
      return

    # Priority 4: Enhanced slow down detection with false positive protection
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # Priority 5: Driving slower than cruise setpoint
    if self._has_slowness:
      self._set_mode('acc')
      return

    # Default to ACC for steady-state cruising
    self._set_mode('acc')

  def set_mpc_fcw_crash_cnt(self) -> None:
    self._mpc_fcw_crash_cnt = self._mpc.crash_cnt

  def _set_mode(self, mode: str, high_priority: bool = False) -> None:
    """
    Enhanced mode setting with priority levels and smoother transitions.
    """
    # Track mode stability
    if mode == self._mode:
      self._mode_stable_frames += 1
    else:
      self._mode_stable_frames = 0
      self._prev_mode = self._mode

    # High priority can override timeout
    if high_priority:
      self._mode = mode
      if mode == 'blended':
        self._set_mode_timeout = max(1, SET_MODE_TIMEOUT // 2)
      return

    # Standard timeout logic
    if self._set_mode_timeout == 0:
      self._mode = mode
      if mode == 'blended':
        self._set_mode_timeout = SET_MODE_TIMEOUT

    if self._set_mode_timeout > 0:
      self._set_mode_timeout -= 1

  def update(self, sm: messaging.SubMaster) -> None:
    self._read_params()

    self.set_mpc_fcw_crash_cnt()

    self._update_calculations(sm)

    if self._CP.radarUnavailable:
      self._radarless_mode()
    else:
      self._radar_mode()

    self._active = sm['selfdriveState'].experimentalMode and self._enabled

    self._frame += 1