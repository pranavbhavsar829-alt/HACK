#!/usr/bin/env python3
"""
=============================================================================
  _______ _____ _______       _   _     __      _______  ___   ___  
 |__   __|_   _|__   __|     | \ | |    \ \    / /___  |/ _ \ / _ \ 
    | |    | |    | |  ____  |  \| |_____\ \  / /   / /| | | | | | |
    | |    | |    | | |____| | . ` |______\ \/ /   / / | | | | | | |
    | |   _| |_   | |        | |\  |       \  /   / /  | |_| | |_| |
    |_|  |_____|  |_|        |_| \_|        \/   /_/    \___/ \___/ 
                                                                    
  TITAN V202 - ADVANCED TRIDENT (HYBRID CORE)
  
  [SYSTEM UPGRADES]
  1. DEEP FRACTAL MEMORY: Scans 2,000 rounds for exact pattern replays.
  2. NEURAL PHYSICS: Uses Momentum Mass (Velocity*Accel) as a neural input.
  3. QUANTUM CHAOS: Z-Score Reversion protected by Variance Analysis.
  4. SNIPER STAKING: Level 1 (Standard), Level 2 (Recovery), Level 3 (Sniper).
=============================================================================
"""

import math
import statistics
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# [PART 1] CONFIGURATION & CONSTANTS
# =============================================================================

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [TITAN_V202] %(message)s', datefmt='%H:%M:%S')

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Expanded history buffer for Deep Pattern Logic
    MIN_HISTORY_FOR_PREDICTION = 50 

class RiskConfig:
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.05    # 5% Base Risk
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 100000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (THE TRIDENT LEVELS)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: STANDARD (0 Losses)
    LVL1_MIN_CONFIDENCE = 0.60  # 60%
    
    # LEVEL 2: RECOVERY (1 Loss)
    LVL2_MIN_CONFIDENCE = 0.75  # 75% (Stricter)
    
    # LEVEL 3: SNIPER (2+ Losses)
    LVL3_MIN_CONFIDENCE = 0.85  # 85% (Only Perfect Shots)

    # -------------------------------------------------------------------------
    # MARTINGALE MULTIPLIERS
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0   # Standard Bet
    TIER_2_MULT = 2.0   # Recovery Bet
    TIER_3_MULT = 4.0   # Sniper Bet ( Aggressive Recovery)
    
    STOP_LOSS_STREAK = 8

# =============================================================================
# [PART 2] ADVANCED MATH LIBRARY
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
    """Neural Activation Function"""
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

def calculate_momentum_mass(data: List[float]) -> float:
    """
    PHYSICS UPGRADE: Calculates 'Momentum Mass' (Velocity * Acceleration)
    Used to detect if the trend is too heavy to stop.
    """
    if len(data) < 5: return 0.0
    
    # 1st Derivative (Velocity)
    vel = [data[i+1]-data[i] for i in range(len(data)-1)]
    avg_vel = sum(vel) / len(vel)
    
    # 2nd Derivative (Acceleration)
    acc = [vel[i+1]-vel[i] for i in range(len(vel)-1)]
    avg_acc = sum(acc) / len(acc)
    
    return avg_vel * avg_acc

# =============================================================================
# [PART 3] THE TRIDENT ENGINES (ADVANCED)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM CHAOS (Adaptive Reversion + Variance Protection)
# -----------------------------------------------------------------------------
def engine_quantum_chaos(history: List[Dict]) -> Optional[Dict]:
    """
    Detects 'Reversion to Mean' but checks Variance (Chaos) first.
    If Variance is too high, it skips (The 'Chaos Guard').
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        # 1. CHAOS GUARD (Variance Check)
        variance = statistics.pvariance(numbers[-5:])
        if variance > 2.5: 
            return None # Too Chaotic, Skip
            
        # 2. Z-SCORE CALCULATION
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # 3. DRAGON TRAP (Don't bet against massive trends)
        if abs(z_score) > 2.8:
            return None 
        
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.8:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f}|Stable)'}
        elif z_score < -1.8:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f}|Stable)'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP FRACTAL MEMORY (2000-Round Scan)
# -----------------------------------------------------------------------------
def engine_deep_fractal(history: List[Dict]) -> Optional[Dict]:
    """
    Scans the last 2,000 rounds for the current pattern.
    Returns the specific historical probability.
    """
    try:
        if len(history) < 60: return None
        
        # Convert entire history to outcomes
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        
        # We need the raw list of strings for comparison
        raw_outcomes = outcomes
        
        best_signal = None
        best_depth = 0
        
        # Scan lengths 7 down to 3
        for depth in range(7, 2, -1):
            if len(outcomes) < depth + 1: continue
            
            target = raw_outcomes[-depth:]
            history_slice = raw_outcomes[:-1] # Look at the past
            
            match_count = 0
            next_big = 0
            
            # Limited scan (Last 2000)
            scan_limit = min(len(history_slice), 2000)
            start_idx = len(history_slice) - scan_limit
            
            # Linear Scan
            for i in range(len(history_slice) - depth - 1, start_idx, -1):
                if history_slice[i : i+depth] == target:
                    match_count += 1
                    if history_slice[i+depth] == GameConstants.BIG:
                        next_big += 1
                        
            # Analyze Matches
            if match_count >= 3:
                prob_big = next_big / match_count
                prob_small = 1.0 - prob_big
                
                # We want strong probabilities (>60%)
                if prob_big >= 0.65:
                    weight = prob_big
                    best_signal = {
                        'prediction': GameConstants.BIG, 
                        'weight': weight * 1.5, # High priority
                        'source': f'Fractal-D{depth}(Match:{match_count}|{prob_big:.0%})'
                    }
                    best_depth = depth
                    break # Found a long match, stop looking
                    
                elif prob_small >= 0.65:
                    weight = prob_small
                    best_signal = {
                        'prediction': GameConstants.SMALL, 
                        'weight': weight * 1.5,
                        'source': f'Fractal-D{depth}(Match:{match_count}|{prob_small:.0%})'
                    }
                    best_depth = depth
                    break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PHYSICS (Mass + Momentum)
# -----------------------------------------------------------------------------
def engine_neural_physics(history: List[Dict]) -> Optional[Dict]:
    """
    Fused Neural Network: Inputs are RSI, Momentum, and PHYSICS MASS.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 25: return None
        
        # INPUT 1: RSI (Relative Strength)
        deltas = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
        gains = [d for d in deltas if d > 0]
        losses = [abs(d) for d in deltas if d < 0]
        avg_gain = sum(gains[-14:]) / 14 if gains else 0
        avg_loss = sum(losses[-14:]) / 14 if losses else 0.001
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
        norm_rsi = (rsi_val - 50) / 50 # -1 to 1
        
        # INPUT 2: MOMENTUM MASS (Physics)
        mass = calculate_momentum_mass(numbers[-6:])
        norm_mass = max(min(mass / 10.0, 1.0), -1.0) # Clamp -1 to 1
        
        # NEURAL WEIGHTS
        w_rsi = -1.2
        w_mass = -1.5 # High mass means reversal likely (Negative correlation)
        bias = 0.1
        
        # PERCEPTRON CALCULATION
        z = (norm_rsi * w_rsi) + (norm_mass * w_mass) + bias
        activation = sigmoid(z)
        
        dist_from_neutral = abs(activation - 0.5)
        
        if activation > 0.65:
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 2.0, 'source': f'NeuralPhys({activation:.2f})'}
        elif activation < 0.35:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 2.0, 'source': f'NeuralPhys({activation:.2f})'}
            
        return None
    except: return None

# =============================================================================
# [PART 4] THE ARCHITECT (MAIN LOGIC)
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.last_outcome = None
        
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    MAIN ENTRY POINT
    """
    # 1. Update Streak based on result
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
    # VIOLET GUARD (0 & 5 PROTECTION)
    # -------------------------------------------------------------------------
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in [0, 5]:
            return {
                'finalDecision': GameConstants.SKIP,
                'confidence': 0, 'positionsize': 0,
                'level': 'VIOLET_GUARD', 'reason': f'Reset Num ({last_num})', 'topsignals': []
            }
    except: pass
    
    # 2. RUN ADVANCED TRIDENT ENGINES
    signals = []
    
    # Engine 1: Quantum Chaos
    s1 = engine_quantum_chaos(history)
    if s1: signals.append(s1)
    
    # Engine 2: Deep Fractal Memory (The requested Pattern Results)
    s2 = engine_deep_fractal(history)
    if s2: signals.append(s2)
    
    # Engine 3: Neural Physics
    s3 = engine_neural_physics(history)
    if s3: signals.append(s3)
    
    # 3. AGGREGATE SCORES
    big_weight = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_weight = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    total_weight = big_weight + small_weight
    
    # MINIMUM QUORUM (Must have decent signal strength)
    if total_weight < 0.40:
         return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'positionsize': 0, 'level': 'WAITING', 'reason': 'Weak Signal', 'topsignals': []}
         
    # 4. DETERMINE CONFIDENCE
    if big_weight > small_weight:
        final_pred = GameConstants.BIG
        confidence = big_weight / (total_weight + 0.1) 
    else:
        final_pred = GameConstants.SMALL
        confidence = small_weight / (total_weight + 0.1)
    
    confidence = min(confidence, 0.99)
    active_sources = [s['source'] for s in signals]
    
    # 5. SNIPER STAKING LOGIC
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- LEVEL LOGIC ---
    
    # LEVEL 3: SNIPER (2+ Losses) -> Needs 85% Confidence
    if streak >= 2:
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
            reason = "High Confidence Lock"
        else:
            level = "SKIP (Recover)"
            reason = "Waiting for Perfect Shot"
            
    # LEVEL 2: RECOVERY (1 Loss) -> Needs 75% Confidence
    elif streak == 1:
        if confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "RECOVERY"
        else:
            level = "SKIP (Recover)"
    
    # LEVEL 1: STANDARD (0 Losses) -> Needs 60% Confidence
    else:
        if confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            
    # Safety Cap
    if stake > current_bankroll * 0.4: stake = current_bankroll * 0.4
    
    return {
        'finalDecision': final_pred if stake > 0 else GameConstants.SKIP,
        'confidence': confidence,
        'positionsize': int(stake),
        'level': level,
        'reason': reason,
        'topsignals': active_sources
    }

if __name__ == "__main__":
    print("TITAN V202 ADVANCED TRIDENT: ONLINE")
