

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
    MIN_HISTORY_FOR_PREDICTION = 25 
    DEBUG_MODE = True

# ==============================================================================
# SECTION 2: RISK CONFIGURATION (SAFE MODE)
# ==============================================================================

class RiskConfig:
    # --------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # --------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # --------------------------------------------------------------------------
    # RECOVERY STAKING (OPTIMIZED)
    # --------------------------------------------------------------------------
    # Capped at 3.0x to prevent catastrophic loss
    TIER_1_MULT = 1.0   
    TIER_2_MULT = 2.0   
    TIER_3_MULT = 3.0   
    
    STOP_LOSS_STREAK = 5 

# ==============================================================================
# SECTION 3: UTILITIES & GUARDS
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

def is_market_choppy(history: List[Dict]) -> bool:
    """
    CHAOS GUARD: Returns True if the market is switching too fast (Ping-Pong).
    """
    try:
        if len(history) < 15: return False
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        outcomes = [o for o in outcomes if o] # Filter Nones
        
        if len(outcomes) < 10: return False
        
        switches = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                switches += 1
                
        # If it switched 8 or more times in 12 rounds, it is dangerous.
        return switches >= 8
    except: return False

# ==============================================================================
# SECTION 4: MATHEMATICAL ENGINES (REPLACING AI)
# ==============================================================================

def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """Detects Trends using Z-Score (Active Mode)."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        mean = calculate_mean(numbers); std = calculate_stddev(numbers)
        if std == 0: return None
        z_score = (numbers[-1] - mean) / std
        
        if abs(z_score) < 0.13: return None 
        if abs(z_score) > 2.8: return None 
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.0: return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.0: return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """Scans 500 rounds for repeating patterns."""
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
                if imbalance > highest_confidence and imbalance > 0.15: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if count_b > count_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    if depth > 6 and imbalance > 0.65: break
        return best_signal
    except: return None

def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """Matches history against technical patterns (Auto-Mirror)."""
    try:
        if len(history) < 20: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-30:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        base_patterns = {
            'ZigZag': ['BSBSBS'], 'Double': ['SSBBSSBB'], 'Triple': ['BBBSSSBBB'],
            'Dragon': ['BBBBBB', 'BBBBBBBB'], 'Ratio_1A2B': ['BSSBSS'],
            'Mirror': ['BBBBSSBSSBBBB'], 'Stairs': ['BSBBSSBBBSSS']
        }
        best_match = None; max_len = 0

        for p_name, p_list in base_patterns.items():
            for p_str in p_list:
                clean_p = p_str.replace(" ", "")
                inverse_p = clean_p.replace('B', 'X').replace('S', 'B').replace('X', 'S')
                
                for pattern_ver, label_suffix in [(clean_p, ""), (inverse_p, "_Rev")]:
                    required = pattern_ver[:-1]
                    pred_char = pattern_ver[-1]
                    if len(required) < 3: continue 
                    if s.endswith(required):
                        if len(required) > max_len:
                            max_len = len(required)
                            pred = GameConstants.BIG if pred_char == 'B' else GameConstants.SMALL
                            weight = min(0.85 + (len(required) * 0.01), 0.98)
                            best_match = {'prediction': pred, 'weight': weight, 'source': f'Chart:{p_name}{label_suffix}'}
        return best_match
    except: return None

def engine_bayesian_probability(history: List[Dict]) -> Optional[Dict]:
    """
    SYNTHETIC AI: BAYESIAN ENGINE
    Calculates the probability of B/S given the last 4 outcomes based on
    pure statistical frequency in the provided history.
    """
    try:
        if len(history) < 50: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        cleaned = [o[0] for o in outcomes if o] # ['B', 'S', 'B', ...]
        
        # Look at the last 4 outcomes as the "Context"
        context_len = 4
        if len(cleaned) < context_len + 1: return None
        
        last_context = tuple(cleaned[-context_len:]) # e.g., ('B', 'S', 'B', 'B')
        
        b_count = 0
        s_count = 0
        
        # Scan history for this exact context
        for i in range(len(cleaned) - context_len - 1):
            window = tuple(cleaned[i : i+context_len])
            if window == last_context:
                next_val = cleaned[i+context_len]
                if next_val == 'B': b_count += 1
                elif next_val == 'S': s_count += 1
                
        total = b_count + s_count
        if total < 3: return None # Not enough data
        
        prob_b = b_count / total
        prob_s = s_count / total
        
        # Threshold: 70% probability required
        if prob_b > 0.70:
            return {'prediction': GameConstants.BIG, 'weight': prob_b, 'source': f'BayesAI({int(prob_b*100)}%)'}
        elif prob_s > 0.70:
            return {'prediction': GameConstants.SMALL, 'weight': prob_s, 'source': f'BayesAI({int(prob_s*100)}%)'}
            
        return None
    except: return None

def engine_momentum_oscillator(history: List[Dict]) -> Optional[Dict]:
    """
    SYNTHETIC AI: MOMENTUM OSCILLATOR
    Measures the 'velocity' of BIG vs SMALL over the last 12 rounds.
    """
    try:
        if len(history) < 20: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        
        # Scoring: Recent results have higher weight
        score = 0
        weight = 1.0
        for o in reversed(outcomes):
            if o == GameConstants.BIG: score += weight
            elif o == GameConstants.SMALL: score -= weight
            weight *= 0.85 # Decay factor
            
        # Interpretation
        # High Positive Score = Strong BIG Momentum
        # High Negative Score = Strong SMALL Momentum
        
        if score > 1.5:
             return {'prediction': GameConstants.BIG, 'weight': 0.8, 'source': 'Momentum'}
        elif score < -1.5:
             return {'prediction': GameConstants.SMALL, 'weight': 0.8, 'source': 'Momentum'}
             
        return None
    except: return None

# ==============================================================================
# SECTION 5: DEEP ANALYST STATE MANAGER
# ==============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.engine_scores = {
            'Quantum': 12, 
            'DeepPattern': 12, 
            'Chart': 12, 
            'BayesAI': 15, # Replaces Ollama
            'Momentum': 12
        }
        self.last_round_predictions = {} 
        self.skip_streak = 0 

state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    TITAN V600 LOGIC CORE (SYNTHETIC AI)
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
            if last_result == actual_outcome: state_manager.loss_streak = 0
            else: state_manager.loss_streak += 1
    
    streak = state_manager.loss_streak

    # 3. GUARDS
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in [0, 5]:
            return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'VIOLET_GUARD', 
                    'reason': 'Violet Reset', 'topsignals': [], 'positionsize': 0}
        
        if is_market_choppy(history):
            return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'level': 'CHAOS_GUARD', 
                    'reason': 'Market too Choppy', 'topsignals': [], 'positionsize': 0}
    except: pass

    # 4. RUN ALL ENGINES
    signals = []
    s_quant = engine_quantum_adaptive(history)
    s_deep = engine_deep_memory_v4(history)
    s_chart = engine_chart_patterns(history)
    s_bayes = engine_bayesian_probability(history)
    s_mom = engine_momentum_oscillator(history)

    if s_quant: signals.append(s_quant)
    if s_deep: signals.append(s_deep)
    if s_chart: signals.append(s_chart)
    if s_bayes: signals.append(s_bayes)
    if s_mom: signals.append(s_mom)

    current_preds = {
        'Quantum': s_quant['prediction'] if s_quant else None,
        'DeepPattern': s_deep['prediction'] if s_deep else None,
        'Chart': s_chart['prediction'] if s_chart else None,
        'BayesAI': s_bayes['prediction'] if s_bayes else None,
        'Momentum': s_mom['prediction'] if s_mom else None
    }
    state_manager.last_round_predictions = current_preds

    # 5. DECISION LOGIC (STRICT CONSENSUS)
    final_decision = GameConstants.SKIP
    level_name = "SKIP"
    reason_log = "Analysing Data..."
    
    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    most_common, count = vote_counts.most_common(1)[0] if vote_counts else (None, 0)
    
    is_level_3 = (count >= 3)
    is_level_2 = (count == 2)
    
    # PRIORITY 1: PERFECT CONSENSUS
    if is_level_3:
        final_decision = most_common
        level_name = "🔥 LEVEL 3 (PERFECT)"
        reason_log = f"Full Consensus ({count} Engines)"
        
    # PRIORITY 2: STRONG CONSENSUS
    elif is_level_2:
        final_decision = most_common
        # Check if "Synthetic AI" (Bayes) is involved
        if s_bayes and s_bayes['prediction'] == most_common:
            level_name = "⚡ LEVEL 2 (MATH AI)"
            reason_log = "Bayes Probability + 1 Engine"
        else:
            level_name = "⚡ LEVEL 2 (STRONG)"
            reason_log = "2 Standard Engines Agree"

    # PRIORITY 3: LEVEL 1 -> STRICTLY SKIP
    else:
        final_decision = GameConstants.SKIP
        level_name = "⚪ SKIP"
        reason_log = "No Consensus (Safety First)"

    # 6. STAKING LOGIC
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    stake = 0
    
    if final_decision != GameConstants.SKIP:
        if streak == 0:
            if "LEVEL 3" in level_name: stake = base_bet * 2.0
            elif "LEVEL 2" in level_name: stake = base_bet * 1.5
            else: stake = base_bet * 1.0 
        elif streak == 1:
            stake = base_bet * RiskConfig.TIER_2_MULT 
            level_name = f"⚔️ RECOVERY ({level_name})"
        elif streak >= 2:
            stake = base_bet * RiskConfig.TIER_3_MULT 
            level_name = f"🎯 SAFE RECOVERY ({level_name})"

    if stake > current_bankroll * 0.4: stake = current_bankroll * 0.4
    if stake > 0: state_manager.skip_streak = 0
    else: state_manager.skip_streak += 1

    return {
        'finalDecision': final_decision if stake > 0 else GameConstants.SKIP,
        'confidence': 0.99,
        'positionsize': int(stake),
        'level': level_name,
        'reason': reason_log,
        'topsignals': [s['source'] for s in signals]
    }

if __name__ == "__main__":
    print("TITAN V600 SYNTHETIC AI LOADED.")
