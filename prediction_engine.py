#!/usr/bin/env python3
"""
================================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V312 - DEEP ANALYST EDITION (MODIFIED V313)
  ==============================================================================
  THE PATIENT LOGIC:
  1. WINNING PHASE: Standard active betting (Level 1/2/3).
  2. RECOVERY PHASE (1 Loss): 
     - STRICT: Skips all "Solo" signals.
     - REQUIREMENT: Must have 2+ Engines agreeing (Level 2).
  3. DEEP RECOVERY (2+ Losses):
     - MODIFIED: Accepts "Strong Consensus" (Level 2) or "Perfect" (Level 3).
     - PREVENTS: The "300+ round wait" by allowing 2 engines to trigger recovery.
     - SAFETY: Still ignores weak "Solo" signals.
  4. FILTERS: Tightened to 0.25 to ignore market noise.
================================================================================
"""

import math
import statistics
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: CONSTANTS
# ==============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    MIN_HISTORY_FOR_PREDICTION = 30 # Increased for better data analysis
    DEBUG_MODE = True

# ==============================================================================
# SECTION 2: RISK CONFIGURATION
# ==============================================================================

class RiskConfig:
    # --------------------------------------------------------------------------
    # BANKROLL
    # --------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # --------------------------------------------------------------------------
    # RECOVERY STAKING
    # --------------------------------------------------------------------------
    TIER_1_MULT = 1.0   # Standard
    TIER_2_MULT = 2.0   # Recovery (Only on Strong Signal)
    TIER_3_MULT = 5.0   # Sniper (On Strong OR Perfect Signal)
    
    STOP_LOSS_STREAK = 5 

# ==============================================================================
# SECTION 3: UTILITIES
# ==============================================================================

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

def sigmoid(x):
    try: return 1 / (1 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

# ==============================================================================
# SECTION 4: ENGINES (TIGHTER FILTERS)
# ==============================================================================

def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """Detects Trends using Z-Score."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        mean = calculate_mean(numbers); std = calculate_stddev(numbers)
        if std == 0: return None
        z_score = (numbers[-1] - mean) / std
        
        # TIGHTER FILTER: 0.25 (Ignores small noise)
        if abs(z_score) < 0.18: return None
        if abs(z_score) > 2.8: return None 
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.2: return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.2: return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """Scans 500 rounds. Depth 20."""
    try:
        data_len = len(history)
        if data_len < 30: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        max_search_depth = 20 if data_len >= 400 else 12
        best_signal = None; highest_confidence = 0.0
        
        for depth in range(max_search_depth, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            count_b = 0; count_s = 0; start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                if idx + depth < len(search_area):
                    if search_area[idx + depth] == 'B': count_b += 1
                    else: count_s += 1
                start = idx + 1
            total = count_b + count_s
            if total >= 3:
                imbalance = abs((count_b/total) - (count_s/total))
                # TIGHTER FILTER: 0.30
                if imbalance > highest_confidence and imbalance > 0.30: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if count_b > count_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    if depth > 6 and imbalance > 0.65: break
        return best_signal
    except: return None

def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """Recognizes Visual Patterns."""
    try:
        if len(history) < 15: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-15:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None
        last = s[-1]; opp = 'S' if last == 'B' else 'B'
        
        if len(s)>=4 and s[-4:]==last*4: return {'prediction': last, 'weight': 0.95, 'source': 'Chart:Dragon'}
        if len(s)>=4 and s[-4:]==(last+opp+last+opp)[-4:]: return {'prediction': opp, 'weight': 0.85, 'source': 'Chart:1A1B'}
        if len(s)>=3 and s[-3:]==opp+opp+last: return {'prediction': last, 'weight': 0.80, 'source': 'Chart:2A2B_Finish'}
        if len(s)>=5 and s[-5:]==(last+last+opp+last+last): return {'prediction': opp, 'weight': 0.85, 'source': 'Chart:2A1B'}
        if len(s)>=5 and s[-5:]==(opp+opp+last+opp+opp): return {'prediction': last, 'weight': 0.82, 'source': 'Chart:AAB_Rhythm'}
        return None
    except: return None

def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """Detects momentum shifts."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-50:]]
        if len(numbers) < 25: return None
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        fast = calculate_mean(numbers[-5:]); slow = calculate_mean(numbers[-20:])
        mom = (fast - slow) / 10.0
        z = (input_rsi * -1.5) + (mom * 1.2)
        prob = sigmoid(z) 
        # TIGHTER FILTER: 0.60/0.40
        if prob > 0.60: return {'prediction': GameConstants.BIG, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        elif prob < 0.40: return {'prediction': GameConstants.SMALL, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        return None
    except: return None

# ==============================================================================
# SECTION 5: DEEP ANALYST STATE MANAGER
# ==============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.engine_scores = {'Quantum': 12, 'DeepPattern': 12, 'Chart': 12, 'Neural': 12}
        self.last_round_predictions = {} 
        self.skip_streak = 0 

state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    TITAN V312 - DEEP ANALYST LOGIC
    """
    # 1. SCORING
    if len(history) > 1:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if state_manager.last_round_predictions:
            for engine_name, pred_val in state_manager.last_round_predictions.items():
                if pred_val == actual_outcome:
                    state_manager.engine_scores[engine_name] = min(state_manager.engine_scores[engine_name] + 1, 25)
                elif pred_val is not None:
                    state_manager.engine_scores[engine_name] = max(state_manager.engine_scores[engine_name] - 1, 5)
            state_manager.last_round_predictions = {}
    
    # 2. UPDATE STREAK
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result != GameConstants.SKIP:
            if last_result == actual_outcome:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
    
    streak = state_manager.loss_streak

    # 3. VIOLET GUARD
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in [0, 5]:
            return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'VIOLET_GUARD', 
                    'reason': 'Violet Reset', 'topsignals': [], 'positionsize': 0}
    except: pass

    # 4. RUN ALL ENGINES
    signals = []
    s_quant = engine_quantum_adaptive(history); 
    if s_quant: signals.append(s_quant)
    s_deep = engine_deep_memory_v4(history); 
    if s_deep: signals.append(s_deep)
    s_chart = engine_chart_patterns(history); 
    if s_chart: signals.append(s_chart)
    s_neur = engine_neural_perceptron(history); 
    if s_neur: signals.append(s_neur)

    current_preds = {
        'Quantum': s_quant['prediction'] if s_quant else None,
        'DeepPattern': s_deep['prediction'] if s_deep else None,
        'Chart': s_chart['prediction'] if s_chart else None,
        'Neural': s_neur['prediction'] if s_neur else None
    }
    state_manager.last_round_predictions = current_preds

    # 5. DECISION LOGIC
    final_decision = GameConstants.SKIP
    level_name = "SKIP"
    reason_log = "Analysing Data..."
    
    sorted_engines = sorted(state_manager.engine_scores.items(), key=lambda x: x[1], reverse=True)
    top_engine_name = sorted_engines[0][0]
    
    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    most_common, count = vote_counts.most_common(1)[0] if vote_counts else (None, 0)
    
    # Levels
    is_level_3 = (count >= 3)
    is_level_2 = (count == 2)
    is_level_1 = (count < 2 and any(current_preds.values()))

    if is_level_3:
        final_decision = most_common
        level_name = "🔥 LEVEL 3 (PERFECT)"
        reason_log = f"Full Consensus ({count} Engines)"
    elif is_level_2:
        final_decision = most_common
        level_name = "⚡ LEVEL 2 (STRONG)"
        reason_log = f"2 Engines Agree"
    elif is_level_1:
        if current_preds.get(top_engine_name):
            final_decision = current_preds[top_engine_name]
            level_name = f"🟢 LEVEL 1 ({top_engine_name.upper()})"
            reason_log = f"Trusting {top_engine_name}"
        else:
            for eng, pred in current_preds.items():
                if pred:
                    final_decision = pred
                    level_name = f"🟢 LEVEL 1 ({eng.upper()})"
                    reason_log = f"Trusting {eng}"
                    break

    # 6. STAKING LOGIC (PATIENT SNIPER)
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    stake = 0
    
    # Force Skip logic for Boredom Breaker:
    # Only allowed if NOT losing (Streak 0). 
    # If we are losing, we DISABLE boredom breaker to ensure quality.
    if streak == 0 and state_manager.skip_streak >= 8 and final_decision != GameConstants.SKIP:
        reason_log += " [FORCE ACTION]"
    
    if final_decision != GameConstants.SKIP:
        
        # --- PHASE 1: WINNING (Active) ---
        if streak == 0:
            if "LEVEL 3" in level_name: stake = base_bet * 2.5
            elif "LEVEL 2" in level_name: stake = base_bet * 1.5
            elif "LEVEL 1" in level_name: stake = base_bet * 1.0 
        
        # --- PHASE 2: FIRST LOSS (Cautious) ---
        elif streak == 1:
            # NO SOLO BETS. Must be Consensus (Level 2) or Level 3.
            if "LEVEL 2" in level_name or "LEVEL 3" in level_name:
                 stake = base_bet * RiskConfig.TIER_2_MULT 
                 level_name = f"⚔️ RECOVERY ({level_name})"
            else:
                 # Signal too weak for recovery.
                 return {
                    'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'WAIT', 
                    'reason': 'Analyzing for Stronger Signal...', 'topsignals': [], 'positionsize': 0
                }

        # --- PHASE 3: DEEP LOSS (Deep Analyst - MODIFIED) ---
        elif streak >= 2:
            # We are down 2 bets. DANGER ZONE.
            # MODIFIED: Now accepts Level 2 (Strong) or Level 3 (Perfect).
            # This prevents waiting 300+ rounds while still avoiding weak "Solo" signals.
            if "LEVEL 3" in level_name or "LEVEL 2" in level_name:
                 stake = base_bet * RiskConfig.TIER_3_MULT 
                 level_name = f"🎯 SNIPER ({level_name})"
            else:
                 # We still skip Level 1 (Solo signals) to remain safe.
                 return {
                    'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'DEEP_WAIT', 
                    'reason': 'Waiting for Strong Consensus (2+ Engines)...', 'topsignals': [], 'positionsize': 0
                }

    # Track Skips
    if stake > 0:
        state_manager.skip_streak = 0
    else:
        state_manager.skip_streak += 1

    # Safety Cap
    if stake > current_bankroll * 0.4: stake = current_bankroll * 0.4

    return {
        'finalDecision': final_decision if stake > 0 else GameConstants.SKIP,
        'confidence': 0.99,
        'positionsize': int(stake),
        'level': level_name,
        'reason': reason_log,
        'topsignals': [s['source'] for s in signals]
    }

if __name__ == "__main__":
    print("TITAN V312 DEEP ANALYST LOADED.")
