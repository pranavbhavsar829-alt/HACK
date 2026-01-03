import math
import statistics
import random
import traceback
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# SECTION 1: IMMUTABLE GAME CONSTANTS
# =============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Ultra-Fast Start: Only needs 5 past results to start betting
    MIN_HISTORY_FOR_PREDICTION = 5 
    DEBUG_MODE = True

# =============================================================================
# SECTION 2: RISK CONFIGURATION (HYPER-ACTIVE MODE)
# =============================================================================

class RiskConfig:
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # 3% Base Risk
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (REMOVED BARRIERS)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard - NO BARRIER
    # If we have 50.1% confidence, we take the bet.
    LVL1_MIN_CONFIDENCE = 0.50 
    
    # LEVEL 2: Recovery (After 1 Loss)
    # Slight filter to ensure we don't double down on garbage.
    LVL2_MIN_CONFIDENCE = 0.55 
    
    # LEVEL 3: SNIPER (After 2+ Losses)
    # The "Must Win" layer.
    LVL3_MIN_CONFIDENCE = 0.65 

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS (CLEAR WITHIN 3 LEVELS)
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.2   # Aggressive recovery to clear profit fast
    TIER_3_MULT = 5.0   # The Final Shot (High multiplier to cover losses + profit)
    STOP_LOSS_STREAK = 3 # Hard stop after Level 3

# =============================================================================
# SECTION 3: MATHEMATICAL UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_rsi(data: List[float], period: int = 14) -> float:
    if len(data) < period + 1: return 50.0
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = calculate_mean(gains[-period:])
    avg_loss = calculate_mean(losses[-period:])
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# SECTION 4: THE TRIDENT ENGINES (ALWAYS ON)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (HIGH SENSITIVITY)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Reacts to almost ANY market deviation.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 5: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # REMOVED DRAGON TRAP: We bet even during runs.
        
        strength = min(abs(z_score) / 1.5, 1.0) 
        
        # SUPER LOW THRESHOLD: 0.5
        if z_score > 0.5:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
        elif z_score < -0.5:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (RAPID FIRE)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    try:
        if len(history) < 10: return None
        
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # Scan very shallow patterns (immediate trends)
        for depth in range(6, 2, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            
            count_b_next = 0
            count_s_next = 0
            
            start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b_next += 1
                    else: count_s_next += 1
                
                start = idx + 1
            
            total_matches = count_b_next + count_s_next
            
            if total_matches >= 1: # If we've seen this ONCE before, use it.
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                
                imbalance = abs(prob_b - prob_s)
                
                if imbalance > highest_confidence: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': imbalance * 1.2, 'source': f'PatternV3-D{depth}'}
                    
                    if imbalance > 0.6: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (ALWAYS VOTES)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    Forced to vote. No neutral zone.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 10: return None
        
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        fast_sma = calculate_mean(numbers[-5:])
        slow_sma = calculate_mean(numbers[-20:])
        input_mom = (fast_sma - slow_sma) / 10.0
        
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        input_rev = (1.5 - b_count) / 5.0
        
        w_rsi = -2.0 
        w_mom = 1.5
        w_rev = 1.2
        
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z) 
        
        # ALWAYS VOTE LOGIC
        dist_from_neutral = abs(probability - 0.5)
        weight = max(dist_from_neutral * 4.0, 0.2) # Minimum weight ensuring it counts
        
        if probability >= 0.5:
            return {'prediction': GameConstants.BIG, 'weight': weight, 'source': f'Neural({probability:.2f})'}
        else:
            return {'prediction': GameConstants.SMALL, 'weight': weight, 'source': f'Neural({probability:.2f})'}
            
    except: return None

# =============================================================================
# SECTION 5: THE ARCHITECT (MAIN LOGIC)
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.last_outcome = None
        
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    
    # 1. Update Streak
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result == GameConstants.SKIP:
            pass
        elif last_result == actual_outcome:
            state_manager.loss_streak = 0
        else:
            state_manager.loss_streak += 1
            
    streak = state_manager.loss_streak
    
    # 2. Run Engines
    signals = []
    
    s1 = engine_quantum_adaptive(history)
    if s1: signals.append(s1)
    
    s2 = engine_deep_pattern_v3(history)
    if s2: signals.append(s2)
    
    s3 = engine_neural_perceptron(history)
    if s3: signals.append(s3)
    
    # 3. Aggregate Signals
    big_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    total_score = big_score + small_score

    # 4. FORCE BET PROTOCOL (If engines are somehow silent)
    if total_score < 0.01:
        # Default to Trend Following (Follow last result)
        last_num = safe_float(history[-1]['actual_number'])
        forced_pred = get_outcome_from_number(last_num)
        
        active_engine_names = ["FORCE_TREND"]
        final_pred = forced_pred
        confidence = 0.51 # Artificial confidence to force a bet
    else:
        # Normal Logic
        if big_score > small_score:
            final_pred = GameConstants.BIG
            confidence = big_score / total_score 
        else:
            final_pred = GameConstants.SMALL
            confidence = small_score / total_score
        
        active_engine_names = [s['source'] for s in signals]
    
    confidence = min(confidence, 0.99)
    
    # 5. Determine Stake & Level
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- LOGIC GATE ---
    
    if streak >= 2:
        # SNIPER LEVEL (Level 3)
        # We need slightly higher confidence, but still aggressive
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 LEVEL 3"
        else:
             # If we are at Level 3 but confidence is low, we Force Bet anyway 
             # because we need to recover. We switch to Pattern Following.
             stake = base_bet * RiskConfig.TIER_3_MULT
             level = "🔥 LVL3 FORCE"
             reason = "Must Recover"
            
    elif streak == 1:
        # RECOVERY LEVEL (Level 2)
        if confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "LEVEL 2"
        else:
            # Force bet on Level 2 to keep momentum
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "LVL 2 FORCE"
    
    else:
        # STANDARD LEVEL (Level 1)
        # ALWAYS BET if confidence > 0.5
        if confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "LEVEL 1"
        else:
            # This should rarely happen with the new settings
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "LVL 1 FORCE"
            
    if stake > current_bankroll * 0.5: stake = current_bankroll * 0.5
    
    return {
        'finalDecision': final_pred if stake > 0 else GameConstants.SKIP,
        'confidence': confidence,
        'positionsize': int(stake),
        'level': level,
        'reason': reason,
        'topsignals': active_engine_names
    }

if __name__ == "__main__":
    print("TITAN V300 HYPER-ACTIVE LOADED.")
