class FilterConstants:
  LEAD_ALPHA = 0.2          # Lead detection - moderate filtering to reduce false positives
  SLOW_DOWN_ALPHA = 0.35    # Slow down detection - responsive for traffic scenarios
  SLOWNESS_ALPHA = 0.12     # Slowness detection - heavy filtering, less critical
  DANGEROUS_TTC_ALPHA = 0.5 # TTC detection - highly responsive for safety
  MPC_FCW_ALPHA = 0.45      # FCW detection - responsive for collision avoidance

  # Thresholds remain the same
  LEAD_PROB = 0.5
  SLOW_DOWN_PROB = 0.6
  SLOW_DOWN_BP = [0., 10., 20., 30., 40., 50., 55., 60.]
  SLOW_DOWN_DIST = [25., 38., 55., 75., 95., 115., 130., 150.]
  SLOWNESS_PROB = 0.5
  SLOWNESS_CRUISE_OFFSET = 1.05
  DANGEROUS_TTC = 2.3
  MPC_FCW_PROB = 0.5


class SNG_State:
  off = 0
  stopped = 1
  going = 2