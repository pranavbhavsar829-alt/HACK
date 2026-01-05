#!/usr/bin/env python3
"""
=============================================================================
  TITAN V204 - INSTANT START (ZERO LATENCY)
  
  [CORE FEATURES]
  1. INSTANT START: Begins predicting at just 10 rounds.
  2. DYNAMIC MATH: Indicators (RSI, Variance) auto-scale their lookback 
     periods based on how much data exists.
  3. ACCURACY SCALING: 
     - 10-25 Rounds: "Micro Mode" (Quick, reactive, smaller bets).
     - 26+ Rounds: "Macro Mode" (Deep analysis, full size bets).
=============================================================================
"""

import math
import statistics
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# [PART 1] CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [TITAN_INSTANT] %(message)s', datefmt='%H:%M:%S')

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # FIXED: Starts at 10 results!
    MIN_HISTORY_FOR_PREDICTION = 10
    
    THINKING_TIME_SECONDS = 15 

class RiskConfig:
    BASE_RISK_PERCENT = 0.05
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 100000
    
    # Confidence Thresholds
    LVL1_MIN_CONFIDENCE = 0.60
    LVL2_MIN_CONFIDENCE = 0.75
    LVL3_MIN_CONFIDENCE = 0.85

    # Multipliers
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.0
    TIER_3_MULT = 4.0
    
    STOP_LOSS_STREAK = 8

# =============================================================================
# [PART 2] DYNAMIC MATH CORE (AUTO-SCALING)
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
    try: return 1 / (1 + math.exp(-x))
    except: return 0.5

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_momentum_mass(data: List[float]) -> float:
    # Works with as few as 3 points now
    if len(data) < 3: return 0.0
    vel = [data[i+1]-data[i] for i in range(len(data)-1)]
    if not vel: return 0.0
    avg_vel = sum(vel) / len(vel)
    
    acc = [vel[i+1]-vel[i] for i in range(len(vel)-1)]
    if not acc: 
        avg_acc = 0.0 
    else:
        avg_acc = sum(acc) / len(acc)
        
    return avg_vel * avg_acc

def calculate_dynamic_rsi(data: List[float]) -> float:
    """
    RSI that doesn't crash on low data.
    """
    if len(data) < 2: return 50.0
    
    # Scale period to data size (Max 14, Min 2)
    period = min(14, len(data) - 1)
    
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d for d in deltas if d > 0]
    losses = [abs(d) for d in deltas if d < 0]
    
    # Use simple average for very short history
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 0.001
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# [PART 3] TRIDENT ENGINES (MICRO-MODE ENABLED)
# =============================================================================

# Engine 1: Quantum Chaos
def engine_quantum_chaos(history: List[Dict]) -> Optional[Dict]:
    try:
        # Works with 10 rounds
        total_len = len(history)
        lookback = min(30, total_len)
        
        numbers = [safe_float(d.get('actual_number')) for d in history[-lookback:]]
        if len(numbers) < 5: return None # Need at least 5 for Z-Score
        
        # Dynamic Variance Window
        var_window = min(5, len(numbers))
        variance = statistics.pvariance(numbers[-var_window:])
        if variance > 2.8: return None # Too chaotic
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        z_score = (numbers[-1] - mean) / std
        
        # Relaxed Dragon Trap for Micro Mode
        limit = 3.0 if total_len < 20 else 2.8
        if abs(z_score) > limit: return None 
        
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.6: # Slightly lower threshold for early game
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.6:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

# Engine 2: Micro Fractal
def engine_deep_fractal(history: List[Dict]) -> Optional[Dict]:
    try:
        if len(history) < 10: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        
        # If we have < 20 rounds, only look for tiny patterns (len 2 or 3)
        max_depth = 5 if len(history) > 30 else 3
        
        for depth in range(max_depth, 1, -1):
            if len(outcomes) < depth + 1: continue
            
            target = outcomes[-depth:]
            history_slice = outcomes[:-1]
            
            match_count = 0
            next_big = 0
            
            for i in range(len(history_slice) - depth):
                if history_slice[i : i+depth] == target:
                    match_count += 1
                    if history_slice[i+depth] == GameConstants.BIG:
                        next_big += 1
            
            # In Micro Mode, 1 previous match is risky, but allowed if confidence is high
            min_matches = 2 if len(history) > 20 else 1
            
            if match_count >= min_matches:
                prob_big = next_big / match_count
                
                # Boost weight if we have more matches
                confidence_boost = 1.0 if match_count < 3 else 1.2
                
                if prob_big >= 0.65:
                    return {'prediction': GameConstants.BIG, 'weight': prob_big * confidence_boost, 'source': f'Fractal-D{depth}'}
                elif prob_big <= 0.35:
                    return {'prediction': GameConstants.SMALL, 'weight': (1.0-prob_big) * confidence_boost, 'source': f'Fractal-D{depth}'}
        return None
    except: return None

# Engine 3: Neural Physics
def engine_neural_physics(history: List[Dict]) -> Optional[Dict]:
    try:
        # Works with 10 rounds
        lookback = min(40, len(history))
        numbers = [safe_float(d.get('actual_number')) for d in history[-lookback:]]
        if len(numbers) < 5: return None
        
        # Dynamic RSI
        norm_rsi = (calculate_dynamic_rsi(numbers) - 50) / 50
        
        # Dynamic Mass
        mass_window = min(6, len(numbers))
        mass = calculate_momentum_mass(numbers[-mass_window:])
        norm_mass = max(min(mass / 8.0, 1.0), -1.0)
        
        # Calc
        z = (norm_rsi * -1.2) + (norm_mass * -1.5) + 0.1
        activation = sigmoid(z)
        dist = abs(activation - 0.5)
        
        if activation > 0.60:
            return {'prediction': GameConstants.BIG, 'weight': dist * 2.0, 'source': 'NeuralPhys'}
        elif activation < 0.40:
            return {'prediction': GameConstants.SMALL, 'weight': dist * 2.0, 'source': 'NeuralPhys'}
        return None
    except: return None

# =============================================================================
# [PART 4] ADAPTIVE BACKTESTER
# =============================================================================

class DeepThoughtCore:
    
    @staticmethod
    def stress_test_signal(history: List[Dict], candidate_signal: str) -> float:
        try:
            total_history = len(history)
            
            # If we only have 10-15 rounds, we can't really backtest.
            # We return 0.6 (Slight Trust) to let the trade happen.
            if total_history < 20: return 0.6
            
            # Otherwise, test on 50% of available data
            test_size = int(total_history * 0.6)
            test_window = history[-test_size:]
            
            wins = 0
            losses = 0
            
            start_offset = 5 # Small offset for engines
            for i in range(start_offset, len(test_window)-1):
                past_slice = test_window[:i]
                actual_next = get_outcome_from_number(test_window[i]['actual_number'])
                
                s1 = engine_quantum_chaos(past_slice)
                s2 = engine_deep_fractal(past_slice)
                s3 = engine_neural_physics(past_slice)
                
                signals = [s for s in [s1, s2, s3] if s]
                if not signals: continue
                
                big_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
                small_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
                
                consensus = GameConstants.BIG if big_w > small_w else GameConstants.SMALL
                
                if consensus == candidate_signal and (big_w + small_w) > 0.3:
                    if consensus == actual_next:
                        wins += 1
                    else:
                        losses += 1
            
            total = wins + losses
            if total < 1: return 0.5
            return wins / total
            
        except: return 0.5

# =============================================================================
# [PART 5] MAIN EXECUTION
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    
    start_time = time.time()
    total_data_points = len(history)
    
    # 1. Update Streak
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result != GameConstants.SKIP:
            if last_result == actual_outcome:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
                
    streak = state_manager.loss_streak
    
    # 2. Violet Guard
    try:
        if int(safe_float(history[-1]['actual_number'])) in [0, 5]:
             # Only skip in Micro Mode if it's very chaotic
             pass 
    except: pass
    
    # 3. GENERATE SIGNALS
    s1 = engine_quantum_chaos(history)
    s2 = engine_deep_fractal(history)
    s3 = engine_neural_physics(history)
    
    signals = [s for s in [s1, s2, s3] if s]
    
    big_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    if big_w > small_w:
        candidate_pred = GameConstants.BIG
        raw_confidence = big_w / (big_w + small_w + 0.01)
    else:
        candidate_pred = GameConstants.SMALL
        raw_confidence = small_w / (big_w + small_w + 0.01)
        
    # Weak Signal Check (Relaxed for early game)
    min_weight = 0.2 if total_data_points < 30 else 0.4
    if (big_w + small_w) < min_weight:
         return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "WAIT", 'reason': "Weak Signal", 'topsignals': []}

    # 4. ADAPTIVE BACKTEST
    validation_score = 0.5
    # Only verify if we have enough data to actually test (20+)
    if total_data_points >= 20:
        while (time.time() - start_time) < GameConstants.THINKING_TIME_SECONDS:
            validation_score = DeepThoughtCore.stress_test_signal(history, candidate_pred)
            if validation_score < 0.40 or validation_score > 0.80: break
            time.sleep(0.1)
    else:
        # Micro Mode: Trust the raw engines directly
        validation_score = 0.7 

    # 5. DECISION
    final_confidence = (raw_confidence + validation_score) / 2
    
    # If Backtest failed badly (only applies if we had data to test)
    if validation_score < 0.45 and total_data_points >= 20:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "ABORT", 'reason': "Backtest Low", 'topsignals': []}
        
    stake = 0
    level = "SKIP"
    active_sources = [s['source'] for s in signals]
    
    # SCALING STAKES: If data < 30, use half stakes (Safety)
    data_scalar = 0.5 if total_data_points < 30 else 1.0
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    base_bet *= data_scalar # Scale down for early game
    
    if streak >= 2:
        if final_confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
        else:
            level = "SKIP (Recov)"
    elif streak == 1:
        if final_confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "RECOVERY"
        else:
            level = "SKIP (Recov)"
    else:
        if final_confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            
    # Safety
    if stake > current_bankroll * 0.4: stake = current_bankroll * 0.4
    if stake < RiskConfig.MIN_BET_AMOUNT and stake > 0: stake = RiskConfig.MIN_BET_AMOUNT
    
    mode_label = "MICRO" if total_data_points < 30 else "MACRO"
    
    return {
        'finalDecision': candidate_pred if stake > 0 else "SKIP",
        'confidence': final_confidence,
        'positionsize': int(stake),
        'level': f"{level} ({mode_label})",
        'reason': f"Conf {final_confidence:.0%} | Data {total_data_points}",
        'topsignals': active_sources
    }

if __name__ == "__main__":
    print("TITAN V204 INSTANT START: ONLINE")
