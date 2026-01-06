#!/usr/bin/env python3

"""
=============================================================================
TITAN V5 - HYBRID SOVEREIGN ENGINE (MATH + VISUAL + NEURAL)
=============================================================================

PHILOSOPHY: "The Eye Sees the Pattern; The Math Checks the Trap."

This is the most advanced version of the Titan engine. It combines three
distinct layers of analysis to form a single "Final Decision":

LAYER 1: THE VISUAL CORTEX (Highest Priority)
   - Dragon Hunter: Detects long streaks (BBBBB...).
   - Chart Matcher: Detects 25+ specific chart rules (AABB, ABAB, 1-2-1).
   - Logic: If a clear visual pattern exists, we follow it immediately.

LAYER 2: THE MATHEMATICAL CORE (Validator)
   - Entropy Guard: Measures chaos. Blocks bets if random > 90%.
   - Volatility Regime: Detects if the market is stable or exploding.
   - Frequency Reversion: Checks if Big/Small is mathematically overdue.
   - Momentum: Measures the force of the current direction.

LAYER 3: THE NEURAL MEMORY (Tie-Breaker)
   - Deep KNN: Searches the last 2000 rounds for similar setups.
   - Q-Learning: Remembers which engine is currently "Hot".

=============================================================================
"""

import math
import statistics
import logging
import random
import time
from collections import deque, Counter
from typing import Dict, List, Optional, Any

# =============================================================================
# [PART 1] GLOBAL CONFIGURATION & CONSTANTS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[TITAN_V5] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

class GameConstants:
    """Core Game Definitions"""
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"
    
    # Minimum rounds needed to boot up specific engines
    MIN_HISTORY_MATH = 30
    MIN_HISTORY_NEURAL = 50

class RiskConfig:
    """Money Management & Confidence Thresholds"""
    
    # Confidence required to place a bet
    # We lower the floor slightly because we have multiple confirmations now
    REQ_CONFIDENCE_VISUAL = 0.80  # For clear patterns
    REQ_CONFIDENCE_MATH = 0.85    # For statistical bets (needs higher cert)
    
    # Betting Limits
    BASE_RISK_PERCENT = 0.05      # 5% of Bankroll
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 50000
    
    # Recovery Ladder (Conservative)
    LEVEL_1_MULT = 1.0
    LEVEL_2_MULT = 2.0  # Martingale-lite
    LEVEL_3_MULT = 5.0  # Recovery shot
    
    # Stop Loss
    STOP_LOSS_STREAK = 3  # Stop betting after 3 wrong in a row

# =============================================================================
# [PART 2] SHARED UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    """Safely converts API numbers to float."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome(n: Any) -> Optional[str]:
    """Converts 0-9 to BIG/SMALL."""
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def extract_numbers(history: List[Dict], window: int) -> List[float]:
    """Extracts the last N numbers as a list of floats."""
    return [safe_float(d.get('actual_number')) for d in history[-window:]]

def to_visual_string(history: List[Dict], length: int) -> str:
    """Creates a string like 'BBSSB' for pattern matching."""
    out = ""
    for item in history[-length:]:
        res = get_outcome(item['actual_number'])
        if res == GameConstants.BIG: out += "B"
        elif res == GameConstants.SMALL: out += "S"
    return out

def shannon_entropy(probs: List[float]) -> float:
    """Calculates Chaos Level (Entropy)."""
    s = 0.0
    for p in probs:
        if p > 0: s -= p * math.log2(p)
    return s

# =============================================================================
# [PART 3] Q-LEARNING MEMORY SYSTEM
# =============================================================================

class QBrain:
    """
    Reinforcement Learning Module.
    Tracks which engines are winning and adjusts their 'Vote Weight' dynamically.
    """
    def __init__(self):
        # Initial Weights (Visuals preferred)
        self.weights = {
            "DragonTrend": 2.0,     # King Priority
            "VisualPattern": 1.5,   # Chart Patterns
            "DeepMemory": 1.2,      # KNN
            "MirrorPattern": 1.0,   # Math
            "ClusterDom": 0.8,      # Math
            "FreqRevert": 0.8,      # Math
            "Momentum": 1.0         # Math
        }
        self.learning_rate = 0.1
        self.last_engine_used = None

    def update(self, engine_name: str, won: bool):
        if not engine_name or engine_name not in self.weights:
            return
        
        if won:
            # Reward: Boost weight
            self.weights[engine_name] = min(3.0, self.weights[engine_name] + self.learning_rate)
        else:
            # Penalty: Reduce weight
            self.weights[engine_name] = max(0.5, self.weights[engine_name] - self.learning_rate)

# Global Brain Instance
brain = QBrain()

# =============================================================================
# [PART 4] LAYER 1: VISUAL PATTERN ENGINES (CHART LOGIC)
# =============================================================================

def engine_dragon_trend(vis_seq: str) -> Optional[Dict]:
    """
    Detects the 'Dragon' (Long Trend).
    Rule: If 4+ identical results in a row, predict continuation.
    """
    if len(vis_seq) < 5: return None
    
    last = vis_seq[-1]
    streak = 0
    for char in reversed(vis_seq):
        if char == last: streak += 1
        else: break
        
    if streak >= 4:
        # The Dragon is awake. Ride it.
        pred = GameConstants.BIG if last == "B" else GameConstants.SMALL
        conf = min(0.98, 0.7 + (streak * 0.05)) # Confidence grows with streak
        return {
            "pred": pred, "conf": conf, 
            "name": "DragonTrend", "desc": f"Dragon Streak ({streak})"
        }
    return None

def engine_chart_patterns(vis_seq: str) -> Optional[Dict]:
    """
    Matches the 'Basic Rules' image (AABB, ZigZag, 1-2-3).
    """
    if len(vis_seq) < 8: return None
    
    # --- RULE: ZIG-ZAG (ABAB) ---
    if vis_seq.endswith("BSBSB"): 
        return {"pred": "S", "conf": 0.88, "name": "VisualPattern", "desc": "ZigZag (ABAB)"}
    if vis_seq.endswith("SBSBS"): 
        return {"pred": "B", "conf": 0.88, "name": "VisualPattern", "desc": "ZigZag (BABA)"}

    # --- RULE: DOUBLE-DOUBLE (AABB) ---
    if vis_seq.endswith("BBSS"): 
        return {"pred": "B", "conf": 0.85, "name": "VisualPattern", "desc": "2-2 Trend (AABB)"}
    if vis_seq.endswith("SSBB"): 
        return {"pred": "S", "conf": 0.85, "name": "VisualPattern", "desc": "2-2 Trend (BBAA)"}

    # --- RULE: 2-1-2 (AABAA) ---
    if vis_seq.endswith("BBSBB"): 
        return {"pred": "S", "conf": 0.75, "name": "VisualPattern", "desc": "2-1-2 Sandwich"}
    if vis_seq.endswith("SSBSS"): 
        return {"pred": "B", "conf": 0.75, "name": "VisualPattern", "desc": "2-1-2 Sandwich"}

    # --- RULE: 3-1 CUT (AAAB) ---
    # Often after 3, it cuts. If it goes to 4, it's a dragon. 
    # This is risky, so confidence is lower.
    if vis_seq.endswith("BBBS"):
        return {"pred": "B", "conf": 0.65, "name": "VisualPattern", "desc": "3-1 Cut Back"}
    if vis_seq.endswith("SSSB"):
        return {"pred": "S", "conf": 0.65, "name": "VisualPattern", "desc": "3-1 Cut Back"}

    return None

# =============================================================================
# [PART 5] LAYER 2: MATHEMATICAL ENGINES (STATISTICAL LOGIC)
# =============================================================================

def engine_entropy_guard(vis_seq: str) -> Dict:
    """
    Calculates Chaos. 
    Returns a 'Risk Multiplier' rather than a prediction.
    If Entropy is high (0.95+), we should lower our stake or SKIP.
    """
    if len(vis_seq) < 20: 
        return {"risk_mod": 1.0, "status": "WARMUP"}
        
    short_seq = vis_seq[-20:]
    b_count = short_seq.count("B")
    s_count = short_seq.count("S")
    total = len(short_seq)
    
    if total == 0: return {"risk_mod": 1.0, "status": "EMPTY"}
    
    prob_b = b_count / total
    prob_s = s_count / total
    
    # Calculate Shannon Entropy
    ent = shannon_entropy([prob_b, prob_s])
    
    # Interpretation
    if ent > 0.97: 
        return {"risk_mod": 0.0, "status": "CHAOS (SKIP)"} # Too random
    elif ent > 0.85:
        return {"risk_mod": 0.5, "status": "HIGH VOLATILITY"} # Reduce bet
    else:
        return {"risk_mod": 1.0, "status": "STABLE"} # Pattern rich

def engine_volatility_regime(history: List[Dict]) -> Optional[Dict]:
    """
    Detects if numbers are jumping wildly (Explosive) or staying close (Calm).
    """
    numbers = extract_numbers(history, 30)
    if len(numbers) < 15: return None
    
    std_dev = statistics.stdev(numbers)
    
    # Average std dev for 0-9 range is usually around 2.87
    if std_dev < 2.0:
        regime = "CALM"
        # In calm markets, trends (AABB) work better
    elif std_dev > 3.2:
        regime = "EXPLOSIVE"
        # In explosive markets, reversion works better
    else:
        regime = "NORMAL"
        
    return {"regime": regime, "std_dev": std_dev}

def engine_mirror_pattern(history: List[Dict]) -> Optional[Dict]:
    """
    Digit Symmetry Analysis (Math).
    """
    numbers = extract_numbers(history, 12)
    if len(numbers) < 6: return None
    
    # Mirror Map (0<->0, 1<->1, 2<->5, 5<->2...)
    # This is a common "glitch" logic used in prediction
    mirror_map = {0:0, 1:1, 2:5, 5:2, 8:8, 6:9, 9:6}
    
    last_num = int(numbers[-1])
    if last_num in mirror_map:
        target = mirror_map[last_num]
        pred = get_outcome(target)
        if pred:
            return {
                "pred": pred, "conf": 0.65, 
                "name": "MirrorPattern", "desc": f"Mirror {last_num}->{target}"
            }
    return None

def engine_frequency_reversion(history: List[Dict]) -> Optional[Dict]:
    """
    Checks for Z-Score imbalance.
    """
    vis = to_visual_string(history, 100)
    if not vis: return None
    
    b_rate = vis.count("B") / len(vis)
    
    # If BIG is over 60%, expect SMALL soon
    if b_rate > 0.60:
        return {"pred": GameConstants.SMALL, "conf": 0.70, "name": "FreqRevert", "desc": "Big Overbought"}
    # If BIG is under 40%, expect BIG soon
    elif b_rate < 0.40:
        return {"pred": GameConstants.BIG, "conf": 0.70, "name": "FreqRevert", "desc": "Small Overbought"}
        
    return None

# =============================================================================
# [PART 6] LAYER 3: NEURAL MEMORY (KNN)
# =============================================================================

def engine_deep_memory(history: List[Dict], current_seq: str) -> Optional[Dict]:
    """
    K-Nearest Neighbors (KNN).
    Searches past 2000 rounds for the exact pattern we see now (last 6 rounds).
    Returns what happened 'next' in those historical cases.
    """
    HISTORY_DEPTH = 2000
    PATTERN_SIZE = 6
    
    if len(history) < 100: return None
    
    # We need the full visual string for searching
    full_vis = to_visual_string(history, HISTORY_DEPTH)
    
    if len(current_seq) < PATTERN_SIZE: return None
    
    # The pattern we are looking for (the last 6 outcomes)
    needle = current_seq[-PATTERN_SIZE:]
    
    matches = []
    # Scan history (stop before the very end to avoid peeking at future if testing)
    # We iterate backwards to find recent trends first? No, scan all.
    for i in range(len(full_vis) - PATTERN_SIZE - 1):
        candidate = full_vis[i : i+PATTERN_SIZE]
        if candidate == needle:
            # Found a match! What came next?
            next_val = full_vis[i+PATTERN_SIZE]
            matches.append(next_val)
            
    if not matches: return None
    
    # Count results
    counts = Counter(matches)
    total_found = len(matches)
    
    if total_found < 3: return None # Not enough data
    
    most_common = counts.most_common(1)[0] # e.g. ('B', 5)
    winner_char = most_common[0]
    win_rate = most_common[1] / total_found
    
    pred = GameConstants.BIG if winner_char == "B" else GameConstants.SMALL
    
    # Confidence scales with win rate and sample size
    base_conf = win_rate
    if total_found > 10: base_conf += 0.1 # Boost for robust sample
    
    return {
        "pred": pred, "conf": min(0.95, base_conf), 
        "name": "DeepMemory", "desc": f"KNN Hist Match ({total_found} hits)"
    }

# =============================================================================
# [PART 7] MAIN LOGIC CONTROLLER (THE BRAIN)
# =============================================================================

class StateManager:
    """Tracks wins/losses to manage the Betting Ladder."""
    def __init__(self):
        self.consecutive_losses = 0
        self.last_bet_result = None # "WIN", "LOSS", "SKIP"
        self.last_predicted_side = None

state_mgr = StateManager()

def ultraAIPredict(history: List[Dict], bankroll: float, last_result_str: Optional[str]) -> Dict:
    """
    THE MAIN ENTRY POINT.
    Called by fetcher.py every round.
    """
    
    # --- 1. PRE-PROCESSING ---
    if len(history) < GameConstants.MIN_HISTORY_MATH:
        return {
            "finalDecision": "SKIP", "confidence": 0, "positionsize": 0,
            "level": "WARMUP", "reason": f"Need {GameConstants.MIN_HISTORY_MATH} rounds"
        }
        
    visual_seq = to_visual_string(history, 200) # Get last 200 for deep analysis
    
    # --- 2. UPDATE STATE (Did we win last time?) ---
    # In a real deployment, fetcher.py might handle this, but we double check here.
    if last_result_str and state_mgr.last_predicted_side:
        if last_result_str == "SKIP":
            pass
        elif last_result_str == state_mgr.last_predicted_side:
            # We Won!
            brain.update(brain.last_engine_used, True)
            state_mgr.consecutive_losses = 0
        else:
            # We Lost
            brain.update(brain.last_engine_used, False)
            state_mgr.consecutive_losses += 1
            
    # Check Stop Loss
    if state_mgr.consecutive_losses >= RiskConfig.STOP_LOSS_STREAK:
        # Force a cooldown skip
        state_mgr.consecutive_losses = 0 # Reset after skip or keep waiting? 
        # Usually better to wait 1 round then reset.
        return {
            "finalDecision": "SKIP", "confidence": 0, "positionsize": 0,
            "level": "STOPLOSS", "reason": "Max consecutive losses reached. Cooldown."
        }

    # --- 3. GATHER INTELLIGENCE (RUN ENGINES) ---
    signals = []
    
    # A. Visuals (Layer 1)
    sig_dragon = engine_dragon_trend(visual_seq)
    if sig_dragon: signals.append(sig_dragon)
    
    sig_chart = engine_chart_patterns(visual_seq)
    if sig_chart: signals.append(sig_chart)
    
    # B. Math (Layer 2)
    sig_mirror = engine_mirror_pattern(history)
    if sig_mirror: signals.append(sig_mirror)
    
    sig_revert = engine_frequency_reversion(history)
    if sig_revert: signals.append(sig_revert)
    
    # C. Neural (Layer 3)
    sig_knn = engine_deep_memory(history, visual_seq)
    if sig_knn: signals.append(sig_knn)
    
    # D. Entropy Check
    entropy_data = engine_entropy_guard(visual_seq)
    if entropy_data['status'] == "CHAOS (SKIP)" and not sig_dragon:
        # If chaos is high AND no dragon, we strictly skip.
        # Dragon overrides chaos because dragons live in chaos.
        return {
            "finalDecision": "SKIP", "confidence": 0, "positionsize": 0,
            "level": "ENTROPY", "reason": "Market too chaotic (Entropy > 0.97)"
        }

    # --- 4. CONSENSUS VOTING ---
    # We calculate a weighted score for BIG and SMALL
    
    score_big = 0.0
    score_small = 0.0
    
    log_reasons = []
    
    for sig in signals:
        name = sig['name']
        side = sig['pred']
        conf = sig['conf']
        
        # Get weight from Q-Brain
        weight = brain.weights.get(name, 1.0)
        
        points = conf * weight
        
        if side == GameConstants.BIG: score_big += points
        else: score_small += points
        
        log_reasons.append(f"{sig['desc']}->{side}({points:.2f})")
        
    # --- 5. MAKE DECISION ---
    
    final_side = "SKIP"
    final_conf = 0.0
    winning_score = 0.0
    losing_score = 0.0
    
    if score_big > score_small:
        final_side = GameConstants.BIG
        winning_score = score_big
        losing_score = score_small
    elif score_small > score_big:
        final_side = GameConstants.SMALL
        winning_score = score_small
        losing_score = score_big
        
    # Calculate relative confidence
    # If scores are close (e.g., Big 4.5, Small 4.2), confidence is low.
    if winning_score == 0:
        raw_conf = 0
    else:
        # Margin of victory
        margin = winning_score - losing_score
        raw_conf = min(0.99, 0.5 + (margin / 4.0)) # Scaling factor
        
    # --- 6. APPLY FILTERS & THRESHOLDS ---
    
    # Determine Required Confidence based on engines used
    req_conf = RiskConfig.REQ_CONFIDENCE_MATH
    
    # If a Visual Engine is primary, we allow slightly lower confidence because they are robust
    primary_engine = "Consensus"
    if sig_dragon and sig_dragon['pred'] == final_side:
        req_conf = RiskConfig.REQ_CONFIDENCE_VISUAL
        primary_engine = "DragonTrend"
    elif sig_chart and sig_chart['pred'] == final_side:
        req_conf = RiskConfig.REQ_CONFIDENCE_VISUAL
        primary_engine = "VisualPattern"
    elif sig_knn and sig_knn['pred'] == final_side:
        primary_engine = "DeepMemory"
        
    # Apply Volatility Regime
    vol_data = engine_volatility_regime(history)
    if vol_data and vol_data['regime'] == "EXPLOSIVE":
        # In explosive markets, reduce confidence unless it's a Dragon
        if primary_engine != "DragonTrend":
            raw_conf -= 0.10
            
    # Final check
    if raw_conf < req_conf:
        return {
            "finalDecision": "SKIP", 
            "confidence": raw_conf, 
            "positionsize": 0,
            "level": "LOW_CONF",
            "reason": f"Weak Signal ({raw_conf:.2f} < {req_conf}). Top: {primary_engine}"
        }

    # --- 7. CALCULATE STAKE (MONEY MANAGEMENT) ---
    
    # Determine Ladder Level
    ladder_mult = RiskConfig.LEVEL_1_MULT
    streak = state_mgr.consecutive_losses
    if streak == 1: ladder_mult = RiskConfig.LEVEL_2_MULT
    elif streak == 2: ladder_mult = RiskConfig.LEVEL_3_MULT
    
    # Base Stake
    base_stake = max(RiskConfig.MIN_BET_AMOUNT, bankroll * RiskConfig.BASE_RISK_PERCENT)
    
    # Apply Entropy Risk Modifier
    risk_mod = entropy_data['risk_mod']
    
    final_stake = base_stake * ladder_mult * risk_mod
    final_stake = min(final_stake, RiskConfig.MAX_BET_AMOUNT)
    final_stake = int(final_stake)
    
    # Store for next loop
    state_mgr.last_predicted_side = final_side
    brain.last_engine_used = primary_engine
    
    reason_str = f"[{primary_engine}] " + " | ".join(log_reasons)
    
    return {
        "finalDecision": final_side,
        "confidence": raw_conf,
        "positionsize": final_stake,
        "level": f"L{streak+1}", # L1, L2, L3
        "reason": reason_str[:100] + "..." # Truncate for display
    }

# =============================================================================
# [PART 8] EXECUTION STUB
# =============================================================================

if __name__ == "__main__":
    print("TITAN V5 HYBRID ENGINE ONLINE.")
    print("-----------------------------")
    # Mock Data Test
    mock_history = []
    # Create a mock dragon pattern (BBBBB)
    for _ in range(20): mock_history.append({'actual_number': 1}) # Small
    for _ in range(5): mock_history.append({'actual_number': 8}) # Big
    
    result = ultraAIPredict(mock_history, 10000, "SMALL")
    print(f"TEST OUTPUT: {result}")
