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
    
    # MODIFIED: Reduced from 40 to 12. 
    # This fixes the "Render" issue where it skips because it's waiting for data.
    MIN_HISTORY_FOR_PREDICTION = 12 
    DEBUG_MODE = True

# =============================================================================
# SECTION 2: RISK & SNIPER CONFIGURATION (SMART & BALANCED)
# =============================================================================

class RiskConfig:
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # 3% Base Risk
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (The "Smart Entry" Gate)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard
    # RESTORED LOGIC: We only bet if we see an edge.
    # Adjusted: 0.60 -> 0.55. (Takes "Good" trends, not just "Perfect" ones)
    LVL1_MIN_CONFIDENCE = 0.55  
    
    # LEVEL 2: Recovery (After 1 Loss)
    LVL2_MIN_CONFIDENCE = 0.65  
    
    # LEVEL 3: SNIPER (After 2+ Losses)
    LVL3_MIN_CONFIDENCE = 0.80  

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.1   # Strong Recovery to clear profit
    TIER_3_MULT = 4.5   # The "Sniper" Shot (Max Win)
    STOP_LOSS_STREAK = 3 

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
    # Adaptive Period: If we don't have enough data, use smaller period
    if len(data) < period + 1: 
        period = len(data) - 1
        if period < 2: return 50.0
        
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = calculate_mean(gains[-period:])
    avg_loss = calculate_mean(losses[-period:])
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# SECTION 4: THE TRIDENT ENGINES (RESTORED SMART LOGIC)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER + DRAGON TRAP)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    The original "Smart" Engine. 
    Detects Reversion but avoids the "Dragon" (Massive Trends).
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        # FIXED: Now works with just 12 numbers (Render fix)
        if len(numbers) < 12: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # DRAGON TRAP (SMART LOGIC): 
        # If Z-Score > 2.6, the trend is too strong. We SKIP to be safe.
        if abs(z_score) > 2.6:
            return None 
        
        strength = min(abs(z_score) / 2.0, 1.0) 
        
        # THRESHOLD: 1.5 (Standard Deviation)
        if z_score > 1.5:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
        elif z_score < -1.5:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (THE MEMORY)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    """
    The "Pattern Hunter". Finds repeating sequences.
    """
    try:
        if len(history) < 20: return None
        
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # Scans for patterns length 3 to 10
        for depth in range(10, 3, -1):
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
            
            # Need at least 2 past matches to trust the pattern
            if total_matches >= 2:
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                
                imbalance = abs(prob_b - prob_s)
                
                # We only want STRONG patterns (>70% probability)
                if imbalance > highest_confidence and imbalance > 0.35: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    # Weight boost for deeper patterns
                    weight = imbalance * (1 + (depth * 0.15))
                    best_signal = {'prediction': pred, 'weight': weight, 'source': f'PatternV3-D{depth}'}
                    
                    if depth > 6 and imbalance > 0.8: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (THE MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    Combines RSI, Momentum, and Reversion Logic.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        # FIXED: Now works with 15 numbers (Render fix)
        if len(numbers) < 15: return None
        
        # --- INPUTS ---
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        # Momentum (Short vs Long Term)
        short_window = min(len(numbers), 5)
        long_window = min(len(numbers), 20)
        
        fast_sma = calculate_mean(numbers[-short_window:])
        slow_sma = calculate_mean(numbers[-long_window:])
        input_mom = (fast_sma - slow_sma) / 10.0
        
        # Reversion Input
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        input_rev = (1.5 - b_count) / 5.0
        
        # --- TUNED WEIGHTS (THE "PERFECT" LOGIC) ---
        w_rsi = -1.6  
        w_mom = 1.3
        w_rev = 0.9
        
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z) 
        
        dist_from_neutral = abs(probability - 0.5)
        
        # Only vote if the Neural Net is confident (>55% or <45%)
        if probability > 0.55:
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 2.5, 'source': f'Neural({probability:.2f})'}
        elif probability < 0.45:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 2.5, 'source': f'Neural({probability:.2f})'}
            
        return None
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
    
    # -------------------------------------------------------------------------
    # RESTORED: "VIOLET GUARD" REMOVED
    # -------------------------------------------------------------------------
    # We removed the block on numbers 0 & 5. It will now analyze them properly 
    # using the engines instead of blindly skipping.
    
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

    # SMART CHECK: If NO engine sees a pattern, we SKIP.
    # We do NOT force a bet if the engines are silent. This keeps it "Smart".
    if total_score < 0.15:
         return {
             'finalDecision': GameConstants.SKIP, 
             'confidence': 0, 
             'positionsize': 0, 
             'level': 'NO_SIG', 
             'reason': 'No Pattern Found', 
             'topsignals': []
         }
         
    # 4. Calculate Confidence
    if big_score > small_score:
        final_pred = GameConstants.BIG
        confidence = big_score / total_score 
    else:
        final_pred = GameConstants.SMALL
        confidence = small_score / total_score
    
    confidence = min(confidence, 0.99)
    
    # 5. Determine Stake & Level
    active_engine_names = [s['source'] for s in signals]
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- SMART LOGIC GATE ---
    
    # SCENARIO: SNIPER (2+ Losses)
    if streak >= 2:
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
            reason = "High Probability"
        else:
            # Fallback for Recovery: If confidence is decent (70%), take a defensive shot
            if confidence >= 0.70:
                 stake = base_bet * RiskConfig.TIER_2_MULT
                 level = "DEFENSIVE"
                 reason = "Soft Recovery"
            else:
                level = "SKIP (Recov)"
                reason = "Wait for Clarity"
            
    # SCENARIO: RECOVERY (1 Loss)
    elif streak == 1:
        if confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "RECOVERY"
        else:
            level = "SKIP (Recov)"
    
    # SCENARIO: STANDARD (0 Losses)
    else:
        # HERE IS THE BALANCE:
        # We take the bet if Confidence > 55%.
        # This filters out "Coin Flips" (50/50) but takes "Good Trends".
        if confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            reason = "Low Edge (<55%)"
            
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
    print("TITAN V202 BALANCED MODE LOADED.")
