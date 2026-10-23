POSITIVE_CLASSES = ["Box_broken", "Open_package"]   # "compromised": set aside at intake
NEGATIVE_CLASSES = ["Box", "Package"]
SCORE_RULE = "max confidence over detections of any POSITIVE_CLASSES in the image; 0.0 if none"

# Costs in DOP, per unit. Source: owner interview + July 2025 inventory snapshot.
COST_FN_DOP = 7000           # damaged unit passes: replacement at cost 4400 + two-way local freight 2500; claim window closed
COST_FP_DOP = 40             # intact unit sent to full unpack: 10 min x RD$3/min loaded wage
COST_REVIEW_DOP = 40         # human review = full 10-min unpack, catches everything
COST_FAIL_CONFIRM_DOP = 10   # model FAIL: photo is the claim evidence, 3-min confirmation glance
MIN_PER_REVIEW = 10
MIN_PER_FAIL_CONFIRM = 3
WAGE_DOP_PER_MIN = 3         # ~RD$32,000/month loaded / ~190 h

BASE_RATE = 0.01             # ~1 damaged per 100 received, per owner
BASE_RATE_SWEEP = [0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
CONTAINER_SIZES = {"20ft": 200, "40ft_low": 400, "40ft_high": 600}   # per owner; no daily flow, receiving is per container
STAFF = 2
HOURS_PER_DAY = 7            # 8 h minus 1 h lunch
CLAIM_WINDOW_DAYS = 7
DOP_PER_EUR = 68.67          # poundsterlinglive.com, 2026-10-23
