class WMACConstants:
  # Lead detection - longer window for stability
  LEAD_WINDOW_SIZE = 15  # Increased from 5
  LEAD_PROB = 0.4  # Slightly lower threshold for earlier detection

  # Slow down detection - more gradual response
  SLOW_DOWN_WINDOW_SIZE = 12  # Increased from 5
  SLOW_DOWN_PROB = 0.4  # Lower threshold for smoother transitions

  # More granular speed-distance mapping for smoother behavior
  SLOW_DOWN_BP = [0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 70., 80.]
  SLOW_DOWN_DIST = [15., 25., 35., 45., 55., 65., 75., 85., 95., 105., 115., 125., 135., 155., 175.]

  # Slowness detection - more patient like humans
  SLOWNESS_WINDOW_SIZE = 20  # Increased from 12
  SLOWNESS_PROB = 0.3  # Lower threshold
  SLOWNESS_CRUISE_OFFSET = 1.15  # More tolerant (allow 87% of cruise speed)

  # Additional human-like parameters
  ANTICIPATION_WINDOW_SIZE = 25
  ANTICIPATION_PROB = 0.35

  # Confidence building parameters
  CONFIDENCE_WINDOW_SIZE = 30
  MIN_CONFIDENCE_THRESHOLD = 0.25
  MAX_CONFIDENCE_THRESHOLD = 0.85

  DANGEROUS_TTC_WINDOW_SIZE = 8  # Increased from 3
  DANGEROUS_TTC = 3.0  # Increased from 2.3 for more safety margin

  MPC_FCW_WINDOW_SIZE = 15  # Increased from 10
  MPC_FCW_PROB = 0.35  # Lower threshold
