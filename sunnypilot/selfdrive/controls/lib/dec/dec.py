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
SET_MODE_TIMEOUT = 10


class AdaptiveWeightedMovingAverageCalculator:
  """Enhanced WMAC with adaptive weights and confidence building"""
  def __init__(self, window_size, adaptation_rate=0.1):
    self.window_size = window_size
    self.data = []
    self.adaptation_rate = adaptation_rate
    self.confidence_history = []

    # Dynamic weights that adapt based on recent consistency
    self.base_weights = np.linspace(0.5, 2.0, window_size)
    self.current_weights = self.base_weights.copy()

  def add_data(self, value: float, confidence: float = 1.0) -> None:
    if len(self.data) == self.window_size:
      self.data.pop(0)
      self.confidence_history.pop(0)

    self.data.append(value)
    self.confidence_history.append(confidence)
    self._adapt_weights()

  def _adapt_weights(self) -> None:
    """Adapt weights based on recent data consistency"""
    if len(self.data) < 3:
      return

    # Calculate consistency of recent data
    recent_variance = np.var(self.data[-5:]) if len(self.data) >= 5 else np.var(self.data)
    consistency_factor = 1.0 / (1.0 + recent_variance)

    # Adapt weights - more consistent data gets higher recent weights
    adaptation = consistency_factor * self.adaptation_rate
    for i in range(len(self.current_weights)):
      if i >= len(self.current_weights) - 3:  # Last 3 weights
        self.current_weights[i] = min(3.0, self.current_weights[i] + adaptation)
      else:
        self.current_weights[i] = max(0.3, self.current_weights[i] - adaptation * 0.5)

  def get_weighted_average(self) -> float | None:
    if len(self.data) == 0:
      return None

    # Apply confidence weighting
    confidence_weights = np.array(self.confidence_history[-len(self.data):])
    effective_weights = self.current_weights[-len(self.data):] * confidence_weights

    weighted_sum = float(np.dot(self.data, effective_weights))
    weight_total = float(np.sum(effective_weights))

    return weighted_sum / weight_total if weight_total > 0 else None

  def get_confidence(self) -> float:
    """Return confidence level based on data consistency"""
    if len(self.data) < 3:
      return 0.5

    variance = np.var(self.data)
    consistency = 1.0 / (1.0 + variance)
    avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.5

    return min(0.95, consistency * avg_confidence)

  def reset_data(self) -> None:
    self.data = []
    self.confidence_history = []
    self.current_weights = self.base_weights.copy()


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._mode: str = 'acc'
    self._frame: int = 0

    # Enhanced calculators with adaptive behavior
    self._lead_gmac = AdaptiveWeightedMovingAverageCalculator(
      window_size=WMACConstants.LEAD_WINDOW_SIZE, adaptation_rate=0.15
    )
    self._slow_down_gmac = AdaptiveWeightedMovingAverageCalculator(
      window_size=WMACConstants.SLOW_DOWN_WINDOW_SIZE, adaptation_rate=0.12
    )
    self._slowness_gmac = AdaptiveWeightedMovingAverageCalculator(
      window_size=WMACConstants.SLOWNESS_WINDOW_SIZE, adaptation_rate=0.08
    )

    # New anticipation calculator for human-like prediction
    self._anticipation_gmac = AdaptiveWeightedMovingAverageCalculator(
      window_size=WMACConstants.ANTICIPATION_WINDOW_SIZE, adaptation_rate=0.05
    )

    # State variables
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False
    self._has_slow_down: bool = False
    self._has_slowness: bool = False
    self._has_anticipation: bool = False

    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.
    self._has_standstill = False

    # Enhanced mode switching with hysteresis
    self._mode_confidence = 0.5
    self._mode_switch_cooldown = 0
    self._mode_history = ['acc'] * 10

    # Human-like learning parameters
    self._driving_style_aggressiveness = 0.5  # 0 = very conservative, 1 = aggressive
    self._adaptation_learning_rate = 0.01

  def _calculate_driving_context(self, sm: messaging.SubMaster) -> dict:
    """Analyze current driving context like a human would"""
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    context = {
      'traffic_density': 0.0,
      'road_curvature': 0.0,
      'relative_speed_comfort': 0.0,
      'following_distance_comfort': 0.0,
      'anticipation_level': 0.0
    }

    # Traffic density estimation
    if lead_one.status and lead_one.dRel > 0:
      expected_distance = max(20.0, self._v_ego_kph * 0.6)  # Rule of thumb: 0.6s following
      context['traffic_density'] = min(1.0, expected_distance / lead_one.dRel)

    # Road curvature analysis - handle Cap'n Proto lists properly
    try:
      if len(md.orientation.x) >= 10:
        # Convert to numpy array first to handle Cap'n Proto list
        orientation_x = np.array(list(md.orientation.x))
        if len(orientation_x) >= 10:
          curvature_changes = np.diff(orientation_x[:10])
          context['road_curvature'] = min(1.0, np.std(curvature_changes) * 10)
    except (AttributeError, IndexError, TypeError):
      # Fallback if orientation data is not available or malformed
      context['road_curvature'] = 0.0

    # Relative speed comfort (how comfortable is current speed relative to cruise)
    if self._v_cruise_kph > 0:
      speed_ratio = self._v_ego_kph / self._v_cruise_kph
      context['relative_speed_comfort'] = max(0.0, min(1.0, speed_ratio))

    return context

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = car_state.vCruise
    self._has_standstill = car_state.standstill

    # Get driving context for more nuanced decisions
    context = self._calculate_driving_context(sm)

    # Enhanced lead detection with context awareness
    lead_confidence = 1.0
    if lead_one.status and lead_one.dRel > 0:
      # Adjust confidence based on distance and relative speed
      distance_confidence = min(1.0, lead_one.dRel / 50.0)  # More confident with closer leads
      speed_confidence = 1.0 - min(0.5, abs(lead_one.vRel) / 10.0)  # Less confident with high relative speeds
      lead_confidence = (distance_confidence + speed_confidence) / 2.0

    self._lead_gmac.add_data(lead_one.status, confidence=lead_confidence)
    lead_prob_threshold = WMACConstants.LEAD_PROB * (1.0 + context['traffic_density'] * 0.3)
    self._has_lead_filtered = (self._lead_gmac.get_weighted_average() or -1.) > lead_prob_threshold

    # Enhanced slow down detection with anticipation
    slow_down_threshold = float(
      interp(self._v_ego_kph, WMACConstants.SLOW_DOWN_BP, WMACConstants.SLOW_DOWN_DIST)
    )

    # Adjust threshold based on road curvature and traffic
    adjusted_threshold = slow_down_threshold * (1.0 + context['road_curvature'] * 0.4)
    adjusted_threshold *= (1.0 + context['traffic_density'] * 0.2)

    slow_down_trigger = False
    try:
      if len(md.orientation.x) >= TRAJECTORY_SIZE and len(md.position.x) >= TRAJECTORY_SIZE:
        # Convert to list first to access by index
        position_x = list(md.position.x)
        if len(position_x) >= TRAJECTORY_SIZE:
          slow_down_trigger = position_x[TRAJECTORY_SIZE - 1] < adjusted_threshold
    except (AttributeError, IndexError, TypeError):
      # Fallback to original behavior if trajectory data unavailable
      slow_down_trigger = False

    slow_down_confidence = 1.0 - context['road_curvature'] * 0.3  # Less confident on curvy roads
    self._slow_down_gmac.add_data(slow_down_trigger, confidence=slow_down_confidence)
    self._has_slow_down = (self._slow_down_gmac.get_weighted_average() or 0) > WMACConstants.SLOW_DOWN_PROB

    # Enhanced slowness detection with patience
    if not self._has_standstill:
      # More tolerant slowness threshold that adapts to conditions
      adaptive_offset = WMACConstants.SLOWNESS_CRUISE_OFFSET * (1.0 + context['traffic_density'] * 0.2)
      slowness_trigger = self._v_ego_kph <= (self._v_cruise_kph / adaptive_offset)

      slowness_confidence = context['relative_speed_comfort']
      self._slowness_gmac.add_data(slowness_trigger, confidence=slowness_confidence)
      self._has_slowness = (self._slowness_gmac.get_weighted_average() or 0) > WMACConstants.SLOWNESS_PROB

    # Anticipation system - predict future needs
    anticipation_factors = [
      context['traffic_density'] * 0.4,
      context['road_curvature'] * 0.3,
      (1.0 - context['relative_speed_comfort']) * 0.3
    ]
    anticipation_level = sum(anticipation_factors)
    self._anticipation_gmac.add_data(anticipation_level, confidence=0.8)
    self._has_anticipation = (self._anticipation_gmac.get_weighted_average() or 0) > WMACConstants.ANTICIPATION_PROB

    self._has_lead_filtered_prev = self._has_lead_filtered

  def _calculate_mode_confidence(self, proposed_mode: str) -> float:
    """Calculate confidence in proposed mode switch"""
    confidence_factors = []

    # Historical consistency
    recent_modes = self._mode_history[-5:]
    mode_consistency = recent_modes.count(proposed_mode) / len(recent_modes)
    confidence_factors.append(mode_consistency * 0.3)

    # Sensor confidence
    lead_confidence = self._lead_gmac.get_confidence()
    slow_down_confidence = self._slow_down_gmac.get_confidence()
    confidence_factors.append((lead_confidence + slow_down_confidence) / 2 * 0.4)

    # Situational confidence
    if proposed_mode == 'blended':
      situational_confidence = 0.8 if (self._has_standstill or self._has_slow_down) else 0.4
    else:
      situational_confidence = 0.8 if self._has_lead_filtered else 0.6
    confidence_factors.append(situational_confidence * 0.3)

    return sum(confidence_factors)

  def _set_mode_with_hysteresis(self, proposed_mode: str) -> None:
    """Set mode with hysteresis to prevent oscillation"""
    if self._mode_switch_cooldown > 0:
      self._mode_switch_cooldown -= 1
      return

    mode_confidence = self._calculate_mode_confidence(proposed_mode)

    # Only switch if confidence is high enough and different from current mode
    if proposed_mode != self._mode:
      confidence_threshold = 0.65 if self._mode == 'blended' else 0.6  # Slightly harder to switch out of blended

      if mode_confidence > confidence_threshold:
        self._mode = proposed_mode
        self._mode_confidence = mode_confidence
        self._mode_switch_cooldown = 15 if proposed_mode == 'blended' else 10  # Longer cooldown for blended

        # Update mode history
        self._mode_history.append(proposed_mode)
        if len(self._mode_history) > 20:
          self._mode_history.pop(0)

  def _radar_mode(self) -> None:
    """Enhanced radar mode with human-like decision making"""

    # Priority 1: Safety first - standstill situations
    if self._has_standstill:
      self._set_mode_with_hysteresis('blended')
      return

    # Priority 2: Anticipation - if we see challenging conditions ahead
    if self._has_anticipation and (self._has_slow_down or not self._has_lead_filtered):
      self._set_mode_with_hysteresis('blended')
      return

    # Priority 3: Lead vehicle handling - stay in ACC when following
    if self._has_lead_filtered and not self._has_slow_down:
      self._set_mode_with_hysteresis('acc')
      return

    # Priority 4: Slow down scenarios - use blended for smooth handling
    if self._has_slow_down:
      self._set_mode_with_hysteresis('blended')
      return

    # Priority 5: Slowness - be patient but eventually switch to ACC
    if self._has_slowness:
      # Only switch to ACC if we've been slow for a while and no other factors
      if (self._slowness_gmac.get_confidence() > 0.7 and
            not self._has_anticipation and
            not self._has_slow_down):
        self._set_mode_with_hysteresis('acc')
        return

    # Default: Stay in current mode or prefer ACC
    if self._mode != 'acc':
      self._set_mode_with_hysteresis('acc')

  def _radarless_mode(self) -> None:
    """Enhanced radarless mode with more conservative but smooth behavior"""

    # Be more conservative without radar
    if self._has_standstill:
      self._set_mode_with_hysteresis('blended')
      return

    # Use blended mode more often without radar for safety
    if self._has_slow_down or self._has_anticipation:
      self._set_mode_with_hysteresis('blended')
      return

    # Only use ACC when confident about clear conditions
    if (not self._has_slowness and
          not self._has_anticipation and
          self._anticipation_gmac.get_confidence() > 0.6):
      self._set_mode_with_hysteresis('acc')
      return

    # Default to blended for safety
    self._set_mode_with_hysteresis('blended')

  def update(self, sm: messaging.SubMaster) -> None:
    self._read_params()
    self._update_calculations(sm)

    if self._CP.radarUnavailable:
      self._radarless_mode()
    else:
      self._radar_mode()

    self._active = sm['selfdriveState'].experimentalMode and self._enabled
    self._frame += 1

  # Keep existing methods for compatibility
  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return str(self._mode)

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active