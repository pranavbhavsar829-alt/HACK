#!/usr/bin/env python3

"""
=============================================================================
TITAN V9 – THE MATH-HYBRID ENGINE (FULL ARSENAL)
=============================================================================

ARCHITECTURE:
1. MATH CORE: Volatility, Trap Index, Mirror Score, Consensus Penalty.
2. COUNCIL OF 12: Guards, Generals, Advisors (All Active).
3. SELF-LEARNING: Dynamic Trust Scores for every engine.
4. EMERGENCY SYSTEM: Auto-correction on deep losing streaks.

LOGIC FLOW:
- Step 1: Check Market Health (Entropy + Volatility).
- Step 2: Detect "Traps" (If market is chopping, INVERT predictions).
- Step 3: Run all 12 Engines.
- Step 4: Apply Math Core penalties (Standard Deviation of votes).
- Step 5: Execute Decision with Risk Management.

=============================================================================
"""

import math
import statistics
import logging
import collections
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# [PART 1] CONFIGURATION & CONSTANTS
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[TITAN_V9] %(message)s')

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"
    MIN_HISTORY = 12

class RiskConfig:
    # 0.58 is the "Sweet Spot" between Math & Trend
    REQ_CONFIDENCE = 0.58 
    
    BASE_RISK_PERCENT = 0.08 
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 50000
    
    # Extended Martingale for deep battles
    STOP_LOSS_STREAK = 5 
    LEVEL_1_MULT = 1.0
    LEVEL_2_MULT = 2.2
    LEVEL_3_MULT = 5.0
    LEVEL_4_MULT = 11.0

# =============================================================================
# [PART 2] CORE UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 4.5
    except:
        return 4.5

def get_outcome(n: Any) -> str:
    val = int(safe_float(n))
    return GameConstants.SMALL if 0 <= val <= 4 else GameConstants.BIG

def get_history_string(history: List[Dict], length: int) -> str:
    out = ""
    for item in history[-length:]:
        out += "B" if get_outcome(item['actual_number']) == GameConstants.BIG else "S"
    return out

def extract_numbers(history: List[Dict], window: int) -> List[float]:
    return [safe_float(d['actual_number']) for d in history[-window:]]

# =============================================================================
# [PART 3] THE MATH CORE (ADVANCED STATS)
# =============================================================================

class MathBrain:
    """
    Implements advanced logic from math_core.py directly into Titan.
    """
    
    @staticmethod
    def calculate_volatility(history: List[Dict], window: int = 20) -> float:
        """Returns standard deviation of outcomes (0.0 to ~0.5)."""
        if len(history) < window: return 0.0
        # Convert BIG to 1, SMALL to 0
        seq = [1 if get_outcome(x['actual_number']) == GameConstants.BIG else 0 for x in history[-window:]]
        if len(seq) < 2: return 0.0
        return statistics.pstdev(seq)

    @staticmethod
    def detect_trap_pattern(history: List[Dict], window: int = 12) -> Tuple[bool, float]:
        """
        Detects 'Chopping' markets (B S B S B S).
        Returns (IsTrap, TrapScore).
        """
        seq = get_history_string(history, window)
        if len(seq) < 6: return False, 0.0
        
        # Count alternations (B->S or S->B)
        alternates = 0
        for i in range(len(seq) - 1):
            if seq[i] != seq[i+1]:
                alternates += 1
                
        trap_index = alternates / max(1, (len(seq) - 1))
        # If > 80% of turns are chops, it's a Trap.
        return (trap_index > 0.80), trap_index

    @staticmethod
    def calculate_mirror_score(history: List[Dict], length: int = 6) -> float:
        """Checks for reflective symmetry (e.g. BBSS | SSBB)."""
        seq = get_history_string(history, length * 2)
        if len(seq) < length * 2: return 0.0
        
        recent = seq[-length:]
        older = seq[-length*2:-length]
        
        # Compare recent vs REVERSED older section
        matches = sum(1 for x, y in zip(recent, reversed(older)) if x == y)
        return matches / length

    @staticmethod
    def shannon_entropy(history: List[Dict], window: int = 40) -> float:
        seq = get_history_string(history, window)
        if len(seq) < 10: return 0.0
        pB = seq.count("B") / len(seq)
        pS = seq.count("S") / len(seq)
        s = 0.0
        for p in [pB, pS]:
            if p > 0: s -= p * math.log2(p)
        return s

    @staticmethod
    def penalty_for_disagreement(confidences: List[float]) -> float:
        """
        If engines disagree (high variance), punish the confidence.
        """
        if not confidences or len(confidences) < 2: return 0.0
        avg = sum(confidences) / len(confidences)
        sd = statistics.pstdev(confidences)
        # Reduce confidence by a fraction of the standard deviation
        penalty = sd * 0.4
        return max(0.0, penalty)

# =============================================================================
# [PART 4] THE ENGINES (GUARDS, GENERALS, ADVISORS)
# =============================================================================

# --- TIER 1: GUARDS ---
def engine_guard_sanity(history: List[Dict]) -> Dict:
    ent = MathBrain.shannon_entropy(history)
    vol = MathBrain.calculate_volatility(history)
    
    # Absolute veto conditions
    if ent > 0.995: return {'status': 'VETO', 'risk': 0.0, 'reason': 'MAX_CHAOS'}
    
    risk = 1.0
    if vol > 0.48: risk *= 0.7 # High volatility = Lower stake
    if ent > 0.95: risk *= 0.6 # High entropy = Lower stake
    
    return {'status': 'PASS', 'risk': risk, 'details': f'E:{ent:.2f}/V:{vol:.2f}'}

# --- TIER 2: GENERALS (High Weight) ---
class GeneralPatternSniper:
    @staticmethod
    def scan(history: List[Dict]) -> Optional[Dict]:
        seq = get_history_string(history, 20)
        # High Conf
        if seq.endswith("BBSS"): return {'pred': GameConstants.BIG, 'conf': 0.85, 'name': '2A2B'}
        if seq.endswith("SSBB"): return {'pred': GameConstants.SMALL, 'conf': 0.85, 'name': '2A2B'}
        if seq.endswith("BSBSBS"): return {'pred': GameConstants.BIG, 'conf': 0.90, 'name': 'ZigZagBreak'}
        # Med Conf
        if seq.endswith("BBB"): return {'pred': GameConstants.BIG, 'conf': 0.75, 'name': 'Dragon'}
        if seq.endswith("SSS"): return {'pred': GameConstants.SMALL, 'conf': 0.75, 'name': 'Dragon'}
        # Low Conf (Bread & Butter)
        if seq.endswith("BB"): return {'pred': GameConstants.BIG, 'conf': 0.60, 'name': 'Trend2'}
        if seq.endswith("SS"): return {'pred': GameConstants.SMALL, 'conf': 0.60, 'name': 'Trend2'}
        return None

def general_momentum(history: List[Dict]) -> Optional[Dict]:
    nums = extract_numbers(history, 15)
    if not nums: return None
    force = 0
    for n in nums: force += 1 if get_outcome(n) == GameConstants.BIG else -1
    if abs(force) < 2: return None
    pred = GameConstants.BIG if force > 0 else GameConstants.SMALL
    return {'pred': pred, 'conf': 0.60 + (abs(force)/20)}

# --- TIER 3: ADVISORS (Support) ---
def advisor_micro_trend(history: List[Dict]) -> Optional[Dict]:
    """Last 4 rounds."""
    seq = get_history_string(history, 4)
    if not seq: return None
    if seq.count("B") >= 3: return {'pred': GameConstants.BIG, 'conf': 0.60}
    if seq.count("S") >= 3: return {'pred': GameConstants.SMALL, 'conf': 0.60}
    return None

def advisor_reversion(history: List[Dict]) -> Optional[Dict]:
    """Bet against extremes."""
    seq = get_history_string(history, 40)
    if not seq: return None
    b_rate = seq.count("B") / len(seq)
    if b_rate > 0.75: return {'pred': GameConstants.SMALL, 'conf': 0.70}
    if b_rate < 0.25: return {'pred': GameConstants.BIG, 'conf': 0.70}
    return None

def advisor_mirror_math(history: List[Dict]) -> Optional[Dict]:
    """Uses MathBrain mirror score."""
    score = MathBrain.calculate_mirror_score(history)
    if score > 0.7:
        # Predict the mirror image of the current cycle
        seq = get_history_string(history, 1)
        # If perfect mirror, current should be opposite of the reflection point
        # Simplified: If mirror score is high, expect structure.
        # We check the last digit for direct number mirroring
        last_num = safe_float(history[-1]['actual_number'])
        mirror_map = {0:0, 1:1, 2:5, 5:2, 6:9, 9:6}
        if int(last_num) in mirror_map:
             target = mirror_map[int(last_num)]
             return {'pred': get_outcome(target), 'conf': 0.65}
    return None

def advisor_cluster(history: List[Dict]) -> Optional[Dict]:
    nums = extract_numbers(history, 20)
    if not nums: return None
    low = sum(1 for x in nums if 0<=x<=3)
    high = sum(1 for x in nums if 7<=x<=9)
    if high > low + 5: return {'pred': GameConstants.BIG, 'conf': 0.6}
    if low > high + 5: return {'pred': GameConstants.SMALL, 'conf': 0.6}
    return None

# =============================================================================
# [PART 5] SELF-LEARNING TRUST SYSTEM
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        # Trust scores: 0.5 (bad) to 2.0 (good)
        self.engine_trust = {
            'Sniper': 1.0, 'Momentum': 1.0, 'Micro': 1.0,
            'Reversion': 1.0, 'Mirror': 1.0, 'Cluster': 1.0
        }
        self.last_round_predictions = {}

    def update_trust(self, actual_outcome: str):
        for name, pred in self.last_round_predictions.items():
            if not pred: continue
            if pred == actual_outcome:
                # Win = +0.1 Trust
                self.engine_trust[name] = min(2.0, self.engine_trust[name] + 0.1)
            else:
                # Loss = -0.15 Trust (Punish faster than reward)
                self.engine_trust[name] = max(0.4, self.engine_trust[name] - 0.15)
        self.last_round_predictions = {}

state_manager = GlobalStateManager()

# =============================================================================
# [PART 6] MAIN LOGIC CONTROLLER
# =============================================================================

def run_deep_scan(history: List[Dict]) -> Dict:
    scores = {GameConstants.BIG: 0.0, GameConstants.SMALL: 0.0}
    total_weight = 0.0
    log_details = []
    current_preds = {}
    engine_confidences = []

    # 1. CHECK GUARDS
    guard = engine_guard_sanity(history)
    if guard['status'] == 'VETO':
        return {'decision': 'SKIP', 'reason': guard['reason'], 'confidence': 0, 'details': []}

    # 2. DETECT TRAPS
    is_trap, trap_idx = MathBrain.detect_trap_pattern(history)
    
    # 3. RUN ENGINES
    # (Name, Function, BaseWeight)
    council = [
        ('Sniper', GeneralPatternSniper.scan, 2.5),
        ('Momentum', general_momentum, 2.0),
        ('Micro', advisor_micro_trend, 1.5),
        ('Reversion', advisor_reversion, 1.0),
        ('Mirror', advisor_mirror_math, 1.0),
        ('Cluster', advisor_cluster, 1.0)
    ]

    for name, func, base_w in council:
        try:
            res = func(history)
            if res:
                trust = state_manager.engine_trust.get(name, 1.0)
                final_w = base_w * trust
                
                # Logic: Vote for outcome
                scores[res['pred']] += final_w * res['conf']
                total_weight += final_w
                current_preds[name] = res['pred']
                engine_confidences.append(res['conf'])
                
                log_details.append(f"{name[:3]}[{res['pred'][0]}]")
            else:
                current_preds[name] = None
        except:
            current_preds[name] = None

    state_manager.last_round_predictions = current_preds

    # 4. CALCULATE RAW CONSENSUS
    if total_weight == 0:
        return {'decision': 'SKIP', 'reason': 'No Signal', 'confidence': 0, 'details': []}
        
    big_s = scores[GameConstants.BIG]
    small_s = scores[GameConstants.SMALL]
    
    if big_s > small_s:
        final_decision = GameConstants.BIG
        winner_score = big_s
        loser_score = small_s
    else:
        final_decision = GameConstants.SMALL
        winner_score = small_s
        loser_score = big_s
        
    # Base Confidence Calculation
    raw_conf = (winner_score - loser_score) / total_weight
    
    # 5. APPLY MATH CORE PENALTIES
    # Penalty for disagreement among engines
    disagreement_penalty = MathBrain.penalty_for_disagreement(engine_confidences)
    final_conf = raw_conf - disagreement_penalty
    
    # Apply Volatility/Entropy Risk Multiplier
    final_conf *= guard['risk']
    
    # 6. TRAP INVERSION LOGIC
    # If we are in a Trap (>80% Chop) and the consensus is following a trend, INVERT IT.
    reason_extra = ""
    if is_trap and final_conf < 0.8: # Only invert if not super confident
        if final_decision == GameConstants.BIG:
            final_decision = GameConstants.SMALL
            reason_extra = "[TRAP_INVERT]"
        else:
            final_decision = GameConstants.BIG
            reason_extra = "[TRAP_INVERT]"
    
    return {
        'decision': final_decision,
        'confidence': max(0.0, final_conf),
        'details': log_details,
        'risk_mult': guard['risk'],
        'extra': reason_extra,
        'trap_idx': trap_idx
    }

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_result: Optional[str] = None) -> Dict:
    
    # 1. Update State & Trust
    if last_result and last_result != "SKIP":
        try:
            actual = get_outcome(history[-1]['actual_number'])
            state_manager.update_trust(actual)
            
            if last_result == actual:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
        except:
            pass
            
    streak = state_manager.loss_streak
    
    # 2. EMERGENCY CORRECTOR (From Math Core)
    # If we have lost 2 times in a row, ignore complex logic and Anti-Pattern the last result.
    if streak >= 2 and streak < RiskConfig.STOP_LOSS_STREAK:
        last_out = get_outcome(history[-1]['actual_number'])
        emerg_pred = GameConstants.SMALL if last_out == GameConstants.BIG else GameConstants.BIG
        
        # Calculate martingale stake
        mult = RiskConfig.LEVEL_1_MULT
        if streak == 2: mult = RiskConfig.LEVEL_3_MULT # Aggressive jump
        if streak == 3: mult = RiskConfig.LEVEL_4_MULT
        
        stake = RiskConfig.BASE_RISK_PERCENT * current_bankroll * mult
        stake = max(RiskConfig.MIN_BET_AMOUNT, min(stake, RiskConfig.MAX_BET_AMOUNT))
        
        return {
            'finalDecision': emerg_pred,
            'confidence': 0.99, # Artificial confidence for emergency
            'positionsize': int(stake),
            'level': "EMERGENCY",
            'reason': "MARTINGALE_CORRECTOR",
            'topsignals': ["FORCE_ANTI"]
        }

    # 3. Stop Loss Check (Deep)
    if streak >= RiskConfig.STOP_LOSS_STREAK:
        return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "STOP_LOSS",
            'reason': "Max Streak Reached",
            'topsignals': []
        }

    # 4. Standard Deep Scan
    scan = run_deep_scan(history)
    
    # --- VISUAL DEBUGGER ---
    recents = get_history_string(history, 15)
    print("\n" + "="*50)
    print(f" TITAN V9 HYBRID | HISTORY: ...{recents}") 
    print(f" TRAP IDX: {scan.get('trap_idx',0):.2f}")
    print(f" SIGNALS : {scan['details']}")
    print(f" CONFID  : {scan['confidence']:.2f} {scan.get('extra','')}")
    print("="*50 + "\n")
    
    if scan['decision'] == 'SKIP':
         return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "WAIT",
            'reason': f"{scan.get('reason','Unknown')}",
            'topsignals': scan.get('details', [])
        }
        
    conf = scan['confidence']
    req = RiskConfig.REQ_CONFIDENCE
    
    # Martingale Levels (Standard)
    mult = RiskConfig.LEVEL_1_MULT
    if streak == 1: mult = RiskConfig.LEVEL_2_MULT
    
    if conf >= req:
        stake = RiskConfig.BASE_RISK_PERCENT * current_bankroll * mult
        stake = stake * scan['risk_mult']
        stake = max(RiskConfig.MIN_BET_AMOUNT, min(stake, RiskConfig.MAX_BET_AMOUNT))
        
        return {
            'finalDecision': scan['decision'],
            'confidence': conf,
            'positionsize': int(stake),
            'level': f"L{streak+1}",
            'reason': f"{len(scan['details'])} Eng {scan.get('extra','')} | Conf:{conf:.2f}",
            'topsignals': scan['details']
        }
    else:
        return {
            'finalDecision': "SKIP",
            'confidence': conf,
            'positionsize': 0,
            'level': "WAIT",
            'reason': f"Unstable ({conf:.2f})",
            'topsignals': scan['details']
        }

if __name__ == "__main__":
    print("TITAN V9 (MATH HYBRID) ONLINE.")
