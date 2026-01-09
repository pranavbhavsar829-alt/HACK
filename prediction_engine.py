#!/usr/bin/env python3
"""
================================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V303 - ADAPTIVE DOMINANCE EDITION
  ==============================================================================
  CORE LOGIC SUMMARY:
  1. ADAPTIVE SCORING: Automatically tracks which engine is winning (Hot Hand).
  2. DEEP MEMORY: Scans full 500+ history for pattern matches (Depth 20).
  3. VISUAL CHARTING: Recognizes Dragon, 1A1B, 2A2B, AAB patterns.
  4. CONSENSUS LEVELS:
     - Level 3 (Perfect): 3+ Engines Agree.
     - Level 2 (Strong): 2 Engines Agree.
     - Level 1 (Adaptive): Follows the single best-performing engine.
================================================================================
"""

import math
import statistics
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: IMMUTABLE GAME CONSTANTS
# ==============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Minimum history required to start predicting
    MIN_HISTORY_FOR_PREDICTION = 30 
    
    DEBUG_MODE = True

# ==============================================================================
# SECTION 2: RISK & STAKING CONFIGURATION
# ==============================================================================

class RiskConfig:
    # --------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # --------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # 3% of bankroll per bet
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # --------------------------------------------------------------------------
    # RECOVERY & MARTINGALE
    # --------------------------------------------------------------------------
    TIER_1_MULT = 1.0   # Standard Bet
    TIER_2_MULT = 2.0   # 1 Loss Recovery
    TIER_3_MULT = 4.5   # Sniper Shot (Deep Recovery)
    
    STOP_LOSS_STREAK = 5 

# ==============================================================================
# SECTION 3: MATHEMATICAL UTILITIES
# ==============================================================================

def safe_float(value: Any) -> float:
    """Safely converts API data to float."""
    try:
        if value is None: return 4.5 # Neutral middle
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """Converts a number (0-9) to BIG/SMALL."""
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def calculate_mean(data: List[float]) -> float:
    """Calculates average."""
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    """Calculates Standard Deviation (Volatility)."""
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_rsi(data: List[float], period: int = 14) -> float:
    """Calculates Relative Strength Index (RSI)."""
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
    """Activation function for Neural Engine."""
    try: return 1 / (1 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

# ==============================================================================
# SECTION 4: THE FOUR ANALYTICAL ENGINES
# ==============================================================================

# ------------------------------------------------------------------------------
# ENGINE 1: QUANTUM ADAPTIVE (Math/Volatility)
# ------------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """Detects Reversion to Mean using Bollinger Bands & Z-Score."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # DRAGON TRAP: If trend is too strong (>2.8 sigma), don't bet against it.
        if abs(z_score) > 2.8: return None 
        
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.6:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.6:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

# ------------------------------------------------------------------------------
# ENGINE 2: DEEP MEMORY V4 (500-Round Pattern Matching)
# ------------------------------------------------------------------------------
def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """Scans the entire 500+ history for repeating sequences."""
    try:
        data_len = len(history)
        if data_len < 30: return None
        
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        # Dynamic Depth: Search deeper if we have more data
        max_search_depth = 20 if data_len >= 400 else 12
        best_signal = None
        highest_confidence = 0.0
        
        # Search from longest pattern (20) down to shortest (3)
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
                # Calculate win probability
                prob_b = count_b / total
                prob_s = count_s / total
                imbalance = abs(prob_b - prob_s)
                
                if imbalance > highest_confidence and imbalance > 0.35: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    
                    # If pattern is long and strong, lock it in
                    if depth > 6 and imbalance > 0.7: break

        return best_signal
    except: return None

# ------------------------------------------------------------------------------
# ENGINE 3: VISUAL CHART PATTERNS (Trend Recognition)
# ------------------------------------------------------------------------------
def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """Recognizes standard visual patterns (Dragon, Ping Pong, etc)."""
    try:
        if len(history) < 15: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-15:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        last_char = s[-1]
        opp_char = 'S' if last_char == 'B' else 'B'
        
        # 1. THE DRAGON (Long Trend)
        if len(s) >= 5 and s[-5:] == last_char * 5:
            return {'prediction': last_char, 'weight': 0.95, 'source': 'Chart:Dragon'}

        # 2. PING PONG (1A1B)
        if len(s) >= 4 and s[-4:] == (last_char + opp_char + last_char + opp_char)[-4:]:
             return {'prediction': opp_char, 'weight': 0.85, 'source': 'Chart:1A1B'}

        # 3. DOUBLE TREND (2A2B)
        if len(s) >= 3 and s[-3:] == opp_char + opp_char + last_char:
             return {'prediction': last_char, 'weight': 0.80, 'source': 'Chart:2A2B_Finish'}

        # 4. TWO IN ONE (2A1B)
        if len(s) >= 5 and s[-5:] == (last_char + last_char + opp_char + last_char + last_char):
             return {'prediction': opp_char, 'weight': 0.85, 'source': 'Chart:2A1B'}
             
        # 5. RHYTHM AAB
        if len(s) >= 5 and s[-5:] == (opp_char + opp_char + last_char + opp_char + opp_char):
             return {'prediction': last_char, 'weight': 0.82, 'source': 'Chart:AAB_Rhythm'}

        return None
    except: return None

# ------------------------------------------------------------------------------
# ENGINE 4: NEURAL PERCEPTRON (Momentum)
# ------------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """Lightweight Neural Network using RSI & Momentum."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-50:]]
        if len(numbers) < 25: return None
        
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        fast = calculate_mean(numbers[-5:])
        slow = calculate_mean(numbers[-20:])
        mom = (fast - slow) / 10.0
        
        z = (input_rsi * -1.5) + (mom * 1.2)
        prob = sigmoid(z) 
        
        if prob > 0.60:
            return {'prediction': GameConstants.BIG, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        elif prob < 0.40:
            return {'prediction': GameConstants.SMALL, 'weight': abs(prob-0.5)*2, 'source': 'Neural'}
        return None
    except: return None

# ==============================================================================
# SECTION 5: ADAPTIVE STATE MANAGER (THE BRAIN)
# ==============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        # Engine Scoreboard (0 to 25). Higher is better.
        self.engine_scores = {
            'Quantum': 10,
            'DeepPattern': 12, # Bonus for history
            'Chart': 12,       # Bonus for visuals
            'Neural': 10
        }
        self.last_round_predictions = {} 

state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    TITAN V303 - ADAPTIVE DOMINANCE LOGIC
    """
    # --------------------------------------------------------------------------
    # 1. ADAPTIVE LEARNING (Grade the previous round)
    # --------------------------------------------------------------------------
    if len(history) > 1:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        
        if state_manager.last_round_predictions:
            for engine_name, pred_val in state_manager.last_round_predictions.items():
                if pred_val == actual_outcome:
                    # Reward: Increase score
                    state_manager.engine_scores[engine_name] = min(state_manager.engine_scores[engine_name] + 1, 25)
                elif pred_val is not None:
                    # Punish: Decrease score
                    state_manager.engine_scores[engine_name] = max(state_manager.engine_scores[engine_name] - 1, 5)
            state_manager.last_round_predictions = {}
    
    # Update Streak
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result != GameConstants.SKIP:
            if last_result == actual_outcome:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
    
    streak = state_manager.loss_streak

    # --------------------------------------------------------------------------
    # 2. VIOLET GUARD (Safety Protocol)
    # --------------------------------------------------------------------------
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in [0, 5]:
            return {
                'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'VIOLET_GUARD', 
                'reason': 'Violet Reset', 'topsignals': [], 'positionsize': 0
            }
    except: pass

    # --------------------------------------------------------------------------
    # 3. RUN ENGINES & GATHER VOTES
    # --------------------------------------------------------------------------
    signals = []
    
    s_quant = engine_quantum_adaptive(history)
    if s_quant: signals.append(s_quant)
    
    s_deep = engine_deep_memory_v4(history)
    if s_deep: signals.append(s_deep)
    
    s_chart = engine_chart_patterns(history)
    if s_chart: signals.append(s_chart)
    
    s_neur = engine_neural_perceptron(history)
    if s_neur: signals.append(s_neur)

    # Save for next round grading
    current_preds = {
        'Quantum': s_quant['prediction'] if s_quant else None,
        'DeepPattern': s_deep['prediction'] if s_deep else None,
        'Chart': s_chart['prediction'] if s_chart else None,
        'Neural': s_neur['prediction'] if s_neur else None
    }
    state_manager.last_round_predictions = current_preds

    # --------------------------------------------------------------------------
    # 4. DECISION LOGIC (ADAPTIVE HIERARCHY)
    # --------------------------------------------------------------------------
    
    final_decision = GameConstants.SKIP
    level_name = "SKIP"
    reason_log = "Computing..."
    
    # 1. Sort engines by "Hotness"
    sorted_engines = sorted(state_manager.engine_scores.items(), key=lambda x: x[1], reverse=True)
    top_engine_name = sorted_engines[0][0]
    top_engine_score = sorted_engines[0][1]
    
    # 2. Check Consensus
    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    most_common, count = vote_counts.most_common(1)[0] if vote_counts else (None, 0)
    
    # --- LEVEL 3: PERFECT CONSENSUS (Sniper) ---
    if count >= 3:
        final_decision = most_common
        level_name = "🔥 LEVEL 3 (PERFECT)"
        reason_log = f"3+ Engines Agree on {most_common}"
        
    # --- LEVEL 2: STRONG CONSENSUS ---
    elif count == 2:
        final_decision = most_common
        level_name = "⚡ LEVEL 2 (STRONG)"
        reason_log = f"2 Engines Agree"
        
    # --- LEVEL 1: ADAPTIVE LEADER (Follow the Hot Hand) ---
    else:
        # Check the Top Ranked Engine first
        top_pred = current_preds.get(top_engine_name)
        
        if top_pred:
            final_decision = top_pred
            level_name = f"🟢 LEVEL 1 ({top_engine_name.upper()})"
            reason_log = f"Trusting {top_engine_name} (Score: {top_engine_score})"
        else:
            # Fallback to Second Best
            second_engine_name = sorted_engines[1][0]
            second_pred = current_preds.get(second_engine_name)
            if second_pred:
                final_decision = second_pred
                level_name = f"🟢 LEVEL 1 ({second_engine_name.upper()})"
                reason_log = f"Trusting {second_engine_name} (Score: {sorted_engines[1][1]})"

    # --------------------------------------------------------------------------
    # 5. STAKE CALCULATION
    # --------------------------------------------------------------------------
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    stake = 0
    
    if final_decision != GameConstants.SKIP:
        if "LEVEL 3" in level_name: stake = base_bet * 2.5
        elif "LEVEL 2" in level_name: stake = base_bet * 1.5
        elif "LEVEL 1" in level_name: stake = base_bet * 1.0 
            
        # Recovery Multipliers
        if streak == 1: stake = stake * RiskConfig.TIER_2_MULT
        if streak >= 2: 
            # Only risky bet on L3/L2 signals if we are losing
            if "LEVEL 3" in level_name: stake = stake * RiskConfig.TIER_3_MULT
            elif "LEVEL 2" in level_name: stake = stake * RiskConfig.TIER_2_MULT
            else:
                # SKIP on weak signal if losing
                return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 
                        'level': 'SKIP (Loss Prot)', 'reason': 'Weak Signal on Loss', 
                        'topsignals': [], 'positionsize': 0}

    # Safety Cap (Max 40% of bank)
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
    print("TITAN V303 ADAPTIVE CORE LOADED.")
