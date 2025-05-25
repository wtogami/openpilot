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
from enum import IntEnum
from typing import Optional, List, Tuple

from cereal import messaging
from opendbc.car import structs
from numpy import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33
HIGHWAY_CRUISE_KPH = 70
STOP_AND_GO_FRAME = 60
SET_MODE_TIMEOUT = 10
V_ACC_MIN = 9.72


class WMACConstants:
  # Lead detection - increased window for smoother filtering
  LEAD_WINDOW_SIZE = 8
  LEAD_PROB = 0.45  # Slightly lower threshold for better responsiveness

  # Slow down detection - larger window for stability
  SLOW_DOWN_WINDOW_SIZE = 8
  SLOW_DOWN_PROB = 0.35  # FIXED: Reduced from 0.55 for better triggering

  # Enhanced breakpoints with more granular control
  SLOW_DOWN_BP = [0., 5., 15., 25., 35., 45., 55., 65., 75.]
  SLOW_DOWN_DIST = [20., 32., 48., 68., 88., 108., 128., 148., 168.]

  # Slowness detection - larger window for stability
  SLOWNESS_WINDOW_SIZE = 15
  SLOWNESS_PROB = 0.6
  SLOWNESS_CRUISE_OFFSET = 1.08  # Slightly more tolerance

  # Dangerous TTC - increased window for stability
  DANGEROUS_TTC_WINDOW_SIZE = 5
  DANGEROUS_TTC = 2.5  # Slightly more conservative

  # MPC FCW - larger window for stability
  MPC_FCW_WINDOW_SIZE = 12
  MPC_FCW_PROB = 0.4  # More sensitive to FCW events

  MODE_CHANGE_HYSTERESIS = 0.15  # Prevents rapid mode switching
  CONFIDENCE_DECAY_RATE = 0.92  # How fast confidence decays
  MIN_CONFIDENCE_THRESHOLD = 0.3
  SPEED_SMOOTHING_FACTOR = 0.8


class SNG_State(IntEnum):
  OFF = 0
  STOPPED = 1
  GOING = 2


class ModeTransitionManager:
  """Manages smooth transitions between modes with hysteresis and confidence tracking"""

  def __init__(self):
    self.current_mode = 'acc'
    self.target_mode = 'acc'
    self.mode_confidence = {'acc': 1.0, 'blended': 0.0}
    self.transition_cooldown = 0
    self.mode_history = ['acc'] * 5

  def update_mode_confidence(self, mode: str, confidence: float):
    """Update confidence for a specific mode"""
    # Smooth confidence updates
    alpha = 0.3
    self.mode_confidence[mode] = alpha * confidence + (1 - alpha) * self.mode_confidence.get(mode, 0.0)

    # Decay other mode's confidence
    other_mode = 'blended' if mode == 'acc' else 'acc'
    self.mode_confidence[other_mode] *= WMACConstants.CONFIDENCE_DECAY_RATE

  def should_switch_mode(self, target_mode: str) -> bool:
    """Determine if mode should switch based on confidence and hysteresis"""
    if self.transition_cooldown > 0:
      self.transition_cooldown -= 1
      return False

    current_conf = self.mode_confidence[self.current_mode]
    target_conf = self.mode_confidence[target_mode]

    # Require significant confidence difference to switch
    confidence_diff = target_conf - current_conf
    hysteresis_threshold = WMACConstants.MODE_CHANGE_HYSTERESIS

    if target_mode != self.current_mode and confidence_diff > hysteresis_threshold:
      self.transition_cooldown = 3  # Brief cooldown after switching
      return True

    return False

  def set_mode(self, mode: str) -> str:
    """Set mode with history tracking"""
    if self.should_switch_mode(mode):
      self.current_mode = mode
      self.mode_history.append(mode)
      self.mode_history = self.mode_history[-5:]  # Keep last 5 modes

    return self.current_mode


class EnhancedMovingAverageCalculator:
  """Enhanced moving average with outlier detection and adaptive weighting"""

  def __init__(self, window_size: int, use_exponential: bool = True):
    self.window_size = window_size
    self.data = []
    self.use_exponential = use_exponential

    if use_exponential:
      # Exponential weights favor recent data more
      alpha = 0.3
      self.weights = np.array([alpha * (1 - alpha) ** i for i in range(window_size)][::-1])
      self.weights /= self.weights.sum()
    else:
      # Linear weights
      self.weights = np.linspace(1, 3, window_size)

  def add_data(self, value: float) -> None:
    if len(self.data) == self.window_size:
      self.data.pop(0)
    self.data.append(value)

  def get_weighted_average(self) -> Optional[float]:
    if len(self.data) == 0:
      return None

    data_array = np.array(self.data)
    weights_subset = self.weights[-len(self.data):]

    # Simple outlier detection
    if len(self.data) >= 3:
      median = np.median(data_array)
      mad = np.median(np.abs(data_array - median))
      if mad > 0:
        # Reduce weight of outliers
        outlier_mask = np.abs(data_array - median) > 2 * mad
        weights_subset = weights_subset.copy()
        weights_subset[outlier_mask] *= 0.3

    weighted_sum = float(np.dot(data_array, weights_subset))
    weight_total = float(np.sum(weights_subset))

    return weighted_sum / weight_total if weight_total > 0 else None

  def get_confidence(self) -> float:
    """Get confidence based on data consistency"""
    if len(self.data) < 3:
      return 0.5

    variance = float(np.var(self.data))
    # Lower variance = higher confidence
    confidence = 1.0 / (1.0 + variance)
    return min(max(confidence, 0.1), 1.0)

  def reset_data(self) -> None:
    self.data = []


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._frame: int = 0

    # Enhanced mode management
    self._mode_manager = ModeTransitionManager()

    # Enhanced moving averages with exponential weighting
    self._lead_gmac = EnhancedMovingAverageCalculator(WMACConstants.LEAD_WINDOW_SIZE)
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False

    self._slow_down_gmac = EnhancedMovingAverageCalculator(WMACConstants.SLOW_DOWN_WINDOW_SIZE)
    self._has_slow_down: bool = False
    self._slow_down_confidence: float = 0.0

    self._slowness_gmac = EnhancedMovingAverageCalculator(WMACConstants.SLOWNESS_WINDOW_SIZE)
    self._has_slowness: bool = False

    self._dangerous_ttc_gmac = EnhancedMovingAverageCalculator(WMACConstants.DANGEROUS_TTC_WINDOW_SIZE)
    self._has_dangerous_ttc: bool = False

    self._mpc_fcw_gmac = EnhancedMovingAverageCalculator(WMACConstants.MPC_FCW_WINDOW_SIZE)
    self._has_mpc_fcw: bool = False
    self._mpc_fcw_crash_cnt = 0

    # Smoothed state variables
    self._v_ego_kph = 0.
    self._v_ego_kph_smoothed = 0.
    self._v_cruise_kph = 0.
    self._has_lead = False
    self._has_blinkers = False
    self._has_standstill = False
    self._has_standstill_prev = False
    self._has_nav_instruction = False

    # SNG state management
    self._sng_transit_frame = 0
    self._sng_state = SNG_State.OFF

    # Mode timeout
    self._set_mode_timeout = 0

    # Driving context awareness
    self._driving_context = {
      'is_highway': False,
      'is_city': False,
      'traffic_density': 0.0,
    }

    # Debug variables - can be toggled
    self._debug_enabled = False

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return self._mode_manager.current_mode

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active

  def _update_driving_context(self) -> None:
    """Update driving context for better decision making"""
    # Highway detection based on speed and cruise setting
    self._driving_context['is_highway'] = (
          self._v_cruise_kph >= HIGHWAY_CRUISE_KPH and
          self._v_ego_kph_smoothed >= 50
    )

    # City driving detection
    self._driving_context['is_city'] = (
          self._v_cruise_kph < 50 and
          self._has_lead_filtered
    )

    # Traffic density estimation
    lead_changes = len([i for i in range(1, len(self._lead_gmac.data))
                        if self._lead_gmac.data[i] != self._lead_gmac.data[i-1]])
    self._driving_context['traffic_density'] = min(lead_changes / max(len(self._lead_gmac.data), 1), 1.0)

  def _adaptive_slowdown_threshold(self) -> float:
    """Simplified adaptive threshold with less aggressive multipliers"""
    base_threshold = float(interp(self._v_ego_kph_smoothed, WMACConstants.SLOW_DOWN_BP, WMACConstants.SLOW_DOWN_DIST))

    # FIXED: Apply only one context adjustment, not stacking them
    if self._driving_context['is_highway']:
      return base_threshold * 1.1  # Slight increase for highway
    elif self._driving_context['is_city']:
      return base_threshold * 0.95  # Slight decrease for city

    # Default case - no stacking of multiple factors
    return base_threshold

  def _debug_slow_down_detection(self, md, slow_down_trigger: bool, adaptive_threshold: float) -> None:
    """Debug helper for slow down detection"""
    if not self._debug_enabled:
      return

    # Check trajectory data validity
    has_valid_trajectory = (
          hasattr(md, 'position') and hasattr(md.position, 'x') and
          hasattr(md, 'orientation') and hasattr(md.orientation, 'x') and
          len(md.position.x) >= TRAJECTORY_SIZE and
          len(md.orientation.x) >= TRAJECTORY_SIZE
    )

    print(f"=== SLOW DOWN DEBUG ===")
    print(f"Valid trajectory: {has_valid_trajectory}")
    if hasattr(md, 'position') and hasattr(md.position, 'x'):
      print(f"Position.x length: {len(md.position.x)}")
    if hasattr(md, 'orientation') and hasattr(md.orientation, 'x'):
      print(f"Orientation.x length: {len(md.orientation.x)}")
    print(f"Expected TRAJECTORY_SIZE: {TRAJECTORY_SIZE}")

    if has_valid_trajectory:
      final_position = md.position.x[TRAJECTORY_SIZE - 1]
      print(f"Final trajectory position: {final_position:.2f}m")
      print(f"Adaptive threshold: {adaptive_threshold:.2f}m")
      print(f"Slow down trigger: {slow_down_trigger}")

      # Debug threshold calculation
      base_threshold = float(interp(self._v_ego_kph_smoothed,
                                    WMACConstants.SLOW_DOWN_BP,
                                    WMACConstants.SLOW_DOWN_DIST))
      print(f"Base threshold: {base_threshold:.2f}m")
      print(f"Current speed: {self._v_ego_kph_smoothed:.1f} kph")
      print(f"Highway mode: {self._driving_context['is_highway']}")
      print(f"City mode: {self._driving_context['is_city']}")

    # Debug moving average
    slow_down_avg = self._slow_down_gmac.get_weighted_average()
    print(f"Slow down moving average: {slow_down_avg:.3f}")
    print(f"Slow down threshold: {WMACConstants.SLOW_DOWN_PROB}")
    print(f"Has slow down: {self._has_slow_down}")
    print(f"GMAC data: {self._slow_down_gmac.data}")
    print("========================")

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    # Smooth velocity updates
    raw_v_ego_kph = car_state.vEgo * 3.6
    alpha = WMACConstants.SPEED_SMOOTHING_FACTOR
    self._v_ego_kph_smoothed = alpha * self._v_ego_kph_smoothed + (1 - alpha) * raw_v_ego_kph
    self._v_ego_kph = raw_v_ego_kph
    self._v_cruise_kph = car_state.vCruise

    self._has_lead = lead_one.status
    self._has_standstill = car_state.standstill
    self._has_blinkers = car_state.leftBlinker or car_state.rightBlinker

    # Update driving context
    self._update_driving_context()

    # Enhanced FCW detection
    self._mpc_fcw_gmac.add_data(float(self._mpc_fcw_crash_cnt > 0))
    fcw_avg = self._mpc_fcw_gmac.get_weighted_average()
    self._has_mpc_fcw = (fcw_avg or 0.) > WMACConstants.MPC_FCW_PROB
    if self._has_mpc_fcw:
      self._mode_manager.update_mode_confidence('blended', 0.9)

    # Enhanced lead detection
    self._lead_gmac.add_data(float(lead_one.status))
    lead_avg = self._lead_gmac.get_weighted_average()
    self._has_lead_filtered = (lead_avg or 0.) > WMACConstants.LEAD_PROB

    # Update mode confidence based on lead detection
    if self._has_lead_filtered:
      confidence = self._lead_gmac.get_confidence()
      self._mode_manager.update_mode_confidence('acc', confidence)

    # Enhanced slow down detection with better validation
    slow_down_trigger = False
    adaptive_threshold = self._adaptive_slowdown_threshold()

    # Better data validation - check if data exists and has minimum required length
    if (hasattr(md, 'position') and hasattr(md.position, 'x') and
          hasattr(md, 'orientation') and hasattr(md.orientation, 'x')):

      pos_len = len(md.position.x) if md.position.x else 0
      ori_len = len(md.orientation.x) if md.orientation.x else 0

      # Use >= instead of == for more flexible validation
      if pos_len >= TRAJECTORY_SIZE and ori_len >= TRAJECTORY_SIZE:
        slow_down_trigger = md.position.x[TRAJECTORY_SIZE - 1] < adaptive_threshold

    # Debug output
    self._debug_slow_down_detection(md, slow_down_trigger, adaptive_threshold)

    self._slow_down_gmac.add_data(float(slow_down_trigger))
    slow_down_avg = self._slow_down_gmac.get_weighted_average()
    self._has_slow_down = (slow_down_avg or 0.) > WMACConstants.SLOW_DOWN_PROB
    self._slow_down_confidence = slow_down_avg or 0.

    if self._has_slow_down:
      confidence = self._slow_down_gmac.get_confidence()
      self._mode_manager.update_mode_confidence('blended', confidence)

    # Enhanced SNG detection with smoother transitions
    if self._has_standstill:
      self._sng_state = SNG_State.STOPPED
      self._sng_transit_frame = 0
    else:
      if self._sng_transit_frame == 0:
        if self._sng_state == SNG_State.STOPPED:
          self._sng_state = SNG_State.GOING
          self._sng_transit_frame = STOP_AND_GO_FRAME
        elif self._sng_state == SNG_State.GOING:
          self._sng_state = SNG_State.OFF
      elif self._sng_transit_frame > 0:
        self._sng_transit_frame -= 1

    # Enhanced slowness detection
    if not self._has_standstill:
      slowness_trigger = self._v_ego_kph_smoothed <= (self._v_cruise_kph * WMACConstants.SLOWNESS_CRUISE_OFFSET)
      self._slowness_gmac.add_data(float(slowness_trigger))
      slowness_avg = self._slowness_gmac.get_weighted_average()
      self._has_slowness = (slowness_avg or 0.) > WMACConstants.SLOWNESS_PROB

      if self._has_slowness:
        confidence = self._slowness_gmac.get_confidence()
        self._mode_manager.update_mode_confidence('acc', confidence)

    # Enhanced dangerous TTC detection
    if not self._has_lead_filtered and self._has_lead_filtered_prev:
      self._dangerous_ttc_gmac.reset_data()
      self._has_dangerous_ttc = False

    if self._has_lead and car_state.vEgo >= 0.01:
      ttc = lead_one.dRel / max(car_state.vEgo, 0.01)
      self._dangerous_ttc_gmac.add_data(ttc)

    ttc_avg = self._dangerous_ttc_gmac.get_weighted_average()
    self._has_dangerous_ttc = (ttc_avg or float('inf')) <= WMACConstants.DANGEROUS_TTC

    if self._has_dangerous_ttc:
      self._mode_manager.update_mode_confidence('blended', 0.8)

    # Store previous values
    self._has_standstill_prev = self._has_standstill
    self._has_lead_filtered_prev = self._has_lead_filtered

  def _determine_optimal_mode(self) -> str:
    """Determine optimal mode based on current conditions and confidence"""

    # PRIORITY 1: If there's a reliable lead, prefer ACC (it's designed for this)
    if self._has_lead_filtered and not self._has_standstill:
      # Only override ACC for critical safety situations
      if self._has_mpc_fcw:
        return 'blended'  # FCW overrides everything
      if self._has_dangerous_ttc:
        return 'blended'  # Dangerous TTC overrides
      return 'acc'  # Default to ACC with lead

    # PRIORITY 2: Most critical safety conditions
    if self._has_mpc_fcw:
      return 'blended'

    if self._has_dangerous_ttc:
      return 'blended'

    # PRIORITY 3: Active driving scenarios requiring immediate response
    if self._has_slow_down and self._slow_down_confidence > 0.5:
      return 'blended'

    # PRIORITY 4: Standstill scenarios (less urgent than active slow-down)
    if self._has_standstill:
      return 'blended'

    # PRIORITY 5: No lead scenarios - decide based on driving conditions
    if self._has_slowness:
      return 'acc'

    return 'acc'

  def set_mpc_fcw_crash_cnt(self) -> None:
    self._mpc_fcw_crash_cnt = self._mpc.crash_cnt

  def enable_debug(self, enabled: bool = True) -> None:
    """Enable/disable debug output"""
    self._debug_enabled = enabled

  def update(self, sm: messaging.SubMaster) -> None:
    self._read_params()
    self.set_mpc_fcw_crash_cnt()
    self._update_calculations(sm)

    # Determine optimal mode
    target_mode = self._determine_optimal_mode()

    # Use mode manager for smooth transitions
    actual_mode = self._mode_manager.set_mode(target_mode)

    self._active = sm['selfdriveState'].experimentalMode and self._enabled
    self._frame += 1