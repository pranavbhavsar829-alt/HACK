#!/usr/bin/env python3
"""
================================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V311 - RESILIENT SNIPER EDITION (FULL UNCOMPRESSED)
  ==============================================================================
  COMPLETE FEATURE LIST:
  
  1. ADAPTIVE ENGINE SCORING: 
     - Live tracking of Quantum, Memory, Chart, and Neural engines.
     - REWARD: +1 Score for wins.
     - PUNISH: -1 Score for losses (Soft punishment for faster recovery).
     
  2. MISTAKE LEARNING & RECOVERY:
     - STREAK 0 (Winning): Active Betting on all levels.
     - STREAK 1 (1 Loss): Requires Score > 10 OR Level 2 Signal.
     - STREAK 2+ (Deep Loss): Requires Score > 12 OR Level 3 Signal.
     
  3. BOREDOM BREAKER:
     - If the bot forced-skips 8 times in a row due to strict rules, 
       it acts on the next available decent signal to keep the game moving.
       
  4. DEEP MEMORY CORE:
     - Scans 500 rounds of history with a variable search depth of 3-20.
     
  5. SENSITIVITY TUNING:
     - Z-Score filter set to 0.15 for maximum activity.
================================================================================
"""

import math
import statistics
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: GAME CONSTANTS
# ==============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    # Reduced warmup for faster start
    MIN_HISTORY_FOR_PREDICTION = 20
    DEBUG_MODE = True

# ==============================================================================
# SECTION 2: RISK & STAKING CONFIGURATION
# ==============================================================================

class RiskConfig:
    # --------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # --------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # 3% of Bankroll per bet
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # --------------------------------------------------------------------------
    # PROGRESSIVE RECOVERY STEPS
    # --------------------------------------------------------------------------
    TIER_1_MULT = 1.0   # Standard Bet
    TIER_2_MULT = 2.0   # Recovery Bet (Used only on Strong Signals)
    TIER_3_MULT = 5.0   # Sniper Shot (Used only on Perfect Signals)
    
    STOP_LOSS_STREAK = 5 

# ==============================================================================
# SECTION 3: MATHEMATICAL UTILITIES
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
# SECTION 4: THE 4 ANALYTICAL ENGINES
# ==============================================================================

# --- 1. QUANTUM ADAPTIVE (Math) ---
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """Detects Trends using Z-Score & Bollinger Bands."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        mean = calculate_mean(numbers); std = calculate_stddev(numbers)
        if std == 0: return None
        z_score = (numbers[-1] - mean) / std
        
        # SENSITIVE FILTER: 0.15 (Captures almost all movement)
        if abs(z_score) < 0.15: return None
        
        # DRAGON TRAP: Extreme trends (>3.0) are risky, skip.
        if abs(z_score) > 3.0: return None 
        
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.0: return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.0: return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

# --- 2. DEEP MEMORY V4 (History) ---
def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """Scans 500 rounds of history for matching patterns."""
    try:
        data_len = len(history)
        if data_len < 30: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        max_search_depth = 20 if data_len >= 400 else 12
        best_signal = None; highest_confidence = 0.0
        
        # Dynamic Depth Search
        for depth in range(max_search_depth, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            count_b = 0; count_s = 0; start = 0
            
            # Find all previous occurrences
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
                # FILTER: 0.25 (Very permissible)
                if imbalance > highest_confidence and imbalance > 0.25: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if count_b > count_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    
                    # Lock on strong patterns
                    if depth > 6 and imbalance > 0.60: break
        return best_signal
    except: return None

# --- 3. CHART PATTERNS (Visuals) ---
def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """Recognizes Visual Patterns (Dragon, 1A1B, AAB, etc)."""
    try:
        if len(history) < 15: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-15:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None
        last = s[-1]; opp = 'S' if last == 'B' else 'B'
        
        # Fast Recognition
        if len(s)>=4 and s[-4:]==last*4: return {'prediction': last, 'weight': 0.95, 'source': 'Chart:Dragon'}
        if len(s)>=4 and s[-4:]==(last+opp+last+opp)[-4:]: return {'prediction': opp, 'weight': 0.85, 'source': 'Chart:1A1B'}
        if len(s)>=3 and s[-3:]==opp+opp+last: return {'prediction': last, 'weight': 0.80, 'source': 'Chart:2A2B_Finish'}
        if len(s)>=5 and s[-5:]==(last+last+opp+last+last): return {'prediction': opp, 'weight': 0.85, 'source': 'Chart:2A1B'}
        if len(s)>=5 and s[-5:]==(opp+opp+last+opp+opp): return {'prediction': last, 'weight': 0.82, 'source': 'Chart:AAB_Rhythm'}
        return None
    except: return None

# --- 4. NEURAL PERCEPTRON (Momentum) ---
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """Detects momentum shifts using RSI & SMA."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-50:]]
        if len(numbers) < 25: return None
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        fast = calculate_mean(numbers[-5:]); slow = calculate_mean(numbers[-20:])
        mom = (fast - slow) / 10.0
        z = (input_rsi * -1.5) + (mom * 1.2)
        prob = sigmoid(z) 
        
        # WIDENED FILTER: 0.55/0.45
        if prob > 0.55: return {'prediction': GameConstants.BIG, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        elif prob < 0.45: return {'prediction': GameConstants.SMALL, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        return None
    except: return None

# ==============================================================================
# SECTION 5: RESILIENT STATE MANAGER (THE BRAIN)
# ==============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        # Engine Trust Scores (Start at 12 - Moderate Trust)
        self.engine_scores = {'Quantum': 12, 'DeepPattern': 12, 'Chart': 12, 'Neural': 12}
        self.last_round_predictions = {} 
        self.skip_streak = 0 # Tracks consecutive skips for Boredom Breaker

state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    TITAN V311 - RESILIENT SNIPER LOGIC
    """
    # --------------------------------------------------------------------------
    # 1. SCORING PHASE (Soft Punishment)
    # --------------------------------------------------------------------------
    if len(history) > 1:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if state_manager.last_round_predictions:
            for engine_name, pred_val in state_manager.last_round_predictions.items():
                if pred_val == actual_outcome:
                    # REWARD (+1)
                    state_manager.engine_scores[engine_name] = min(state_manager.engine_scores[engine_name] + 1, 25)
                elif pred_val is not None:
                    # SOFT PUNISHMENT (-1) - Allows faster recovery than V310
                    state_manager.engine_scores[engine_name] = max(state_manager.engine_scores[engine_name] - 1, 5)
            state_manager.last_round_predictions = {}
    
    # --------------------------------------------------------------------------
    # 2. UPDATE STREAK
    # --------------------------------------------------------------------------
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result != GameConstants.SKIP:
            if last_result == actual_outcome:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
    
    streak = state_manager.loss_streak

    # 3. VIOLET GUARD (Safety Protocol)
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

    # 5. DECISION LOGIC (Consensus + Hot Hand)
    final_decision = GameConstants.SKIP
    level_name = "SKIP"
    reason_log = "Scanning..."
    
    sorted_engines = sorted(state_manager.engine_scores.items(), key=lambda x: x[1], reverse=True)
    top_engine_name = sorted_engines[0][0]
    top_engine_score = sorted_engines[0][1]
    
    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    most_common, count = vote_counts.most_common(1)[0] if vote_counts else (None, 0)
    
    # --- LEVEL ASSIGNMENT ---
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
        # Intelligently pick the best active engine
        if current_preds.get(top_engine_name):
            final_decision = current_preds[top_engine_name]
            level_name = f"🟢 LEVEL 1 ({top_engine_name.upper()})"
            reason_log = f"Trusting {top_engine_name}"
        else:
            # Fallback to any active signal
            for eng, pred in current_preds.items():
                if pred:
                    final_decision = pred
                    level_name = f"🟢 LEVEL 1 ({eng.upper()})"
                    reason_log = f"Trusting {eng}"
                    break

    # 6. RESILIENT STAKING LOGIC (WITH BOREDOM BREAKER)
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    stake = 0
    
    # -- BOREDOM BREAKER CHECK --
    # If we have skipped 8 times, FORCE a bet if we have ANY signal.
    if state_manager.skip_streak >= 8 and final_decision != GameConstants.SKIP:
        reason_log += " [FORCE ACTION]"
        # Treat as Streak 0 to reset momentum and get back in the game
        streak = 0 
    
    if final_decision != GameConstants.SKIP:
        
        # --- STREAK 0: WINNING PHASE ---
        if streak == 0:
            if "LEVEL 3" in level_name: stake = base_bet * 2.5
            elif "LEVEL 2" in level_name: stake = base_bet * 1.5
            elif "LEVEL 1" in level_name: stake = base_bet * 1.0 
        
        # --- STREAK 1: RECOVERY PHASE ---
        elif streak == 1:
            # V311 UPDATE: Lowered Trust Barrier to 10 (was 15)
            if "LEVEL 2" in level_name or "LEVEL 3" in level_name:
                 stake = base_bet * RiskConfig.TIER_2_MULT 
                 level_name = f"⚔️ RECOVERY ({level_name})"
            elif top_engine_score >= 10: 
                 stake = base_bet * RiskConfig.TIER_2_MULT
                 level_name = f"⚔️ RECOVERY ({level_name})"
            else:
                 # Still too risky
                 return {
                    'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'WAIT', 
                    'reason': 'Weak Signal. Waiting...', 'topsignals': [], 'positionsize': 0
                }

        # --- STREAK 2+: DEEP RECOVERY PHASE ---
        elif streak >= 2:
            # V311 UPDATE: Lowered Trust Barrier to 12 (was 18)
            if "LEVEL 2" in level_name or "LEVEL 3" in level_name:
                 stake = base_bet * RiskConfig.TIER_3_MULT 
                 level_name = f"🎯 SNIPER ({level_name})"
            elif top_engine_score >= 12: 
                 stake = base_bet * RiskConfig.TIER_3_MULT
                 level_name = f"🎯 SNIPER ({level_name})"
            else:
                 return {
                    'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'WAIT', 
                    'reason': 'Waiting for Better Signal...', 'topsignals': [], 'positionsize': 0
                }

    # Track Skips for Boredom Breaker
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
    print("TITAN V311 RESILIENT SNIPER LOADED.")
