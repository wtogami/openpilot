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
from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import FilterConstants, SNG_State

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33

HIGHWAY_CRUISE_KPH = 70

STOP_AND_GO_FRAME = 60

SET_MODE_TIMEOUT = 10

V_ACC_MIN = 9.72


class FirstOrderFilter:
  def __init__(self, alpha=0.2, initial_value=None):
    """
    First-order low-pass filter optimized for automotive control

    Args:
        alpha: Filter coefficient (0 < alpha <= 1)
              - Higher alpha = more responsive (less filtering)
              - Lower alpha = more filtering (less responsive)
        initial_value: Optional initial value to reduce startup transients
    """
    self.alpha = alpha
    self.filtered_value = initial_value
    self.initialized = initial_value is not None

  def update(self, new_value: float) -> float:
    """Update filter with new value and return filtered result"""
    if not self.initialized:
      self.filtered_value = new_value
      self.initialized = True
    else:
      self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value

    return self.filtered_value

  def get_value(self) -> float | None:
    """Get current filtered value"""
    return self.filtered_value

  def reset(self, initial_value=None) -> None:
    """Reset filter state with optional initial value"""
    self.filtered_value = initial_value
    self.initialized = initial_value is not None

  def set_alpha(self, new_alpha: float) -> None:
    """Dynamically adjust filter responsiveness"""
    self.alpha = max(0.01, min(1.0, new_alpha))  # Clamp between 0.01 and 1.0


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._mode: str = 'acc'
    self._frame: int = 0

    # Initialize first-order filters with tuned parameters and initial values
    self._lead_filter = FirstOrderFilter(FilterConstants.LEAD_ALPHA, initial_value=0.0)
    self._slow_down_filter = FirstOrderFilter(FilterConstants.SLOW_DOWN_ALPHA, initial_value=0.0)
    self._slowness_filter = FirstOrderFilter(FilterConstants.SLOWNESS_ALPHA, initial_value=0.0)
    self._dangerous_ttc_filter = FirstOrderFilter(FilterConstants.DANGEROUS_TTC_ALPHA, initial_value=10.0)  # Safe initial TTC
    self._mpc_fcw_filter = FirstOrderFilter(FilterConstants.MPC_FCW_ALPHA, initial_value=0.0)

    # State variables
    self._has_lead_filtered = False
    self._has_slow_down = False
    self._has_slowness = False
    self._has_dangerous_ttc = False
    self._has_mpc_fcw = False
    self._has_lead_filtered_prev = False

    # Additional variables from original code
    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.
    self._has_lead = False
    self._has_standstill = False
    self._has_standstill_prev = False
    self._sng_transit_frame = 0
    self._sng_state = SNG_State.off
    self._mpc_fcw_crash_cnt = 0
    self._set_mode_timeout = 0
    self._has_blinkers = False
    self._has_nav_instruction = False

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return str(self._mode)

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active

  def _adaptive_slowdown_threshold(self) -> float:
    """
    Adapts the slow-down threshold based on vehicle speed.
    Uses interpolation from your original constants.
    """
    return float(interp(self._v_ego_kph, FilterConstants.SLOW_DOWN_BP, FilterConstants.SLOW_DOWN_DIST))

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = car_state.vCruise
    self._has_lead = lead_one.status
    self._has_standstill = car_state.standstill
    self._has_blinkers = car_state.leftBlinker or car_state.rightBlinker

    # Lead detection with adaptive filtering
    # Use higher alpha when speed is high for faster response
    adaptive_lead_alpha = min(0.4, FilterConstants.LEAD_ALPHA + 0.002 * self._v_ego_kph)
    self._lead_filter.set_alpha(adaptive_lead_alpha)
    lead_filtered = self._lead_filter.update(float(lead_one.status))
    self._has_lead_filtered = lead_filtered > FilterConstants.LEAD_PROB

    # Slow down detection with speed-adaptive filtering
    adaptive_threshold = self._adaptive_slowdown_threshold()
    slow_down_trigger = (len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE and
                         md.position.x[TRAJECTORY_SIZE - 1] < adaptive_threshold)

    # More responsive filtering at higher speeds for safety
    adaptive_slowdown_alpha = FilterConstants.SLOW_DOWN_ALPHA
    if self._v_ego_kph > 50:  # Highway speeds
      adaptive_slowdown_alpha = min(0.5, FilterConstants.SLOW_DOWN_ALPHA + 0.15)

    self._slow_down_filter.set_alpha(adaptive_slowdown_alpha)
    slow_down_filtered = self._slow_down_filter.update(float(slow_down_trigger))
    self._has_slow_down = slow_down_filtered > FilterConstants.SLOW_DOWN_PROB

    # Slowness detection - only when not at standstill
    if not self._has_standstill:
      slowness_trigger = self._v_ego_kph <= (self._v_cruise_kph * FilterConstants.SLOWNESS_CRUISE_OFFSET)
      slowness_filtered = self._slowness_filter.update(float(slowness_trigger))
      self._has_slowness = slowness_filtered > FilterConstants.SLOWNESS_PROB
    else:
      # Reset slowness filter when at standstill
      self._slowness_filter.reset(initial_value=0.0)
      self._has_slowness = False

    # Dangerous TTC detection with reset logic
    if not self._has_lead_filtered and self._has_lead_filtered_prev:
      self._dangerous_ttc_filter.reset(initial_value=10.0)  # Safe TTC when no lead
      self._has_dangerous_ttc = False
    elif self._has_lead and car_state.vEgo >= 0.01:
      ttc = lead_one.dRel / car_state.vEgo
      # Clamp TTC to reasonable range
      ttc = max(0.1, min(15.0, ttc))
      ttc_filtered = self._dangerous_ttc_filter.update(ttc)
      self._has_dangerous_ttc = ttc_filtered <= FilterConstants.DANGEROUS_TTC
    else:
      self._has_dangerous_ttc = False

    # MPC FCW detection with crash count
    fcw_filtered = self._mpc_fcw_filter.update(float(self._mpc_fcw_crash_cnt > 0))
    self._has_mpc_fcw = fcw_filtered > FilterConstants.MPC_FCW_PROB

    # SNG (Stop and Go) state machine - unchanged from original
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

    # Update previous values
    self._has_standstill_prev = self._has_standstill
    self._has_lead_filtered_prev = self._has_lead_filtered

  def _radarless_mode(self) -> None:
    # when mpc fcw crash prob is high
    # use blended to slow down quickly
    if self._has_mpc_fcw:
      self._set_mode('blended')
      return

    # Nav enabled and distance to upcoming turning is 300 or below
    # if self._has_nav_instruction:
    #  self._set_mode('blended')
    #  return

    # when blinker is on and speed is driving below V_ACC_MIN: blended
    # we don't want it to switch mode at higher speed, blended may trigger hard brake
    # if self._has_blinkers and self._v_ego_kph < V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when at highway cruise and SNG: blended
    # ensuring blended mode is used because acc is bad at catching SNG lead car
    # especially those who accel very fast and then brake very hard.
    # if self._sng_state == SNG_State.going and self._v_cruise_kph >= V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when standstill: blended
    # in case of lead car suddenly move away under traffic light, acc mode won't brake at traffic light.
    if self._has_standstill:
      self._set_mode('blended')
      return

    # when detecting slow down scenario: blended
    # e.g. traffic light, curve, stop sign etc.
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # when detecting lead slow down: blended
    # use blended for higher braking capability
    if self._has_dangerous_ttc:
      self._set_mode('blended')
      return

    # car driving at speed lower than set speed: acc
    if self._has_slowness:
      self._set_mode('acc')
      return

    self._set_mode('acc')

  def _radar_mode(self) -> None:
    # when mpc fcw crash prob is high
    # use blended to slow down quickly
    if self._has_mpc_fcw:
      self._set_mode('blended')
      return

    # If there is a filtered lead, the vehicle is not in standstill, and the lead vehicle's yRel meets the condition,
    if self._has_lead_filtered and not self._has_standstill:
      self._set_mode('acc')
      return

    # when blinker is on and speed is driving below V_ACC_MIN: blended
    # we don't want it to switch mode at higher speed, blended may trigger hard brake
    # if self._has_blinkers and self._v_ego_kph < V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when standstill: blended
    # in case of lead car suddenly move away under traffic light, acc mode won't brake at traffic light.
    if self._has_standstill:
      self._set_mode('blended')
      return

    # when detecting slow down scenario: blended
    # e.g. traffic light, curve, stop sign etc.
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # car driving at speed lower than set speed: acc
    if self._has_slowness:
      self._set_mode('acc')
      return

    # Nav enabled and distance to upcoming turning is 300 or below
    # if self._has_nav_instruction:
    #  self._set_mode('blended')
    #  return

    self._set_mode('acc')

  def set_mpc_fcw_crash_cnt(self) -> None:
    self._mpc_fcw_crash_cnt = self._mpc.crash_cnt

  def _set_mode(self, mode: str) -> None:
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