#!/usr/bin/env python3
"""
=============================================================================
TITAN V700 - SNIPER EDITION (VISUAL PATTERNS)
=============================================================================
Logic Stack:
1. VISUAL ENGINE (Fractal): Looks for AABB, ABAB, AAB patterns (From your images).
2. TREND ENGINE: Catches long "Dragon" streaks (AAAAAA).
3. REVERSION ENGINE: The safety valve for extreme anomalies.
4. REMOVED: Neuren & Qaum (Too much noise).
=============================================================================
"""

import math
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TITAN_V700")

class TradeDecision(Enum):
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

@dataclass
class RiskProfile:
    base_risk_percent: float = 0.05       
    max_risk_percent: float = 0.15        
    min_bet_amount: float = 10.0
    max_bet_amount: float = 50000.0
    stop_loss_streak: int = 4             # Balanced Safety
    martingale_multiplier: float = 2.0    

@dataclass
class EngineState:
    name: str
    weight: float
    consecutive_losses: int = 0
    is_active: bool = True
    last_vote: Optional[TradeDecision] = None 

class GlobalState:
    def __init__(self):
        self.loss_streak: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.cooling_off_counter: int = 0  
        self.inversion_mode: bool = False
        self.consecutive_fails: int = 0    
        self.session_start_bankroll: float = 0.0
        self.current_profit: float = 0.0
        
        # --- THE SNIPER SQUAD (Only the best) ---
        self.engines: Dict[str, EngineState] = {
            'visual': EngineState('visual', 3.0),   # Highest Weight (Your Patterns)
            'trend': EngineState('trend', 2.0),     # Strong Weight (Dragons)
            'reversion': EngineState('reversion', 1.0) # Lower Weight (Safety)
        }
        
        self.last_prediction: Optional[TradeDecision] = None
        self.last_confidence: float = 0.0

state = GlobalState()
config = RiskProfile()

# =============================================================================
# SECTION 1: THE VISUAL PATTERN LIBRARY (UPGRADED)
# =============================================================================

class VisualLib:
    @staticmethod
    def analyze_patterns(outcomes: List[str]) -> float:
        """
        Scans for the specific patterns in your screenshots.
        Returns: 1.0 (BIG), -1.0 (SMALL), or 0.0 (NO MATCH)
        """
        if len(outcomes) < 6: return 0.0
        
        # Convert last 6 results to a simple string code (e.g. "BSSBSS")
        # We look at the last 5 to predict the 6th
        recent = outcomes[-6:]
        code = "".join([x[0] for x in recent]) # 'B' or 'S'
        
        # --- PATTERN LOGIC FROM SCREENSHOTS ---
        
        # Rule 1: AB AB AB (Ping Pong)
        # If we see B S B S B -> Predict S
        if code[-5:] == "BSBSB": return -1.0
        if code[-5:] == "SBSBS": return 1.0

        # Rule 2: AABB AABB (Double Double)
        # If we see B B S S B -> Predict B (to complete pair)
        if code[-5:] == "BBSSB": return 1.0
        if code[-5:] == "SSBBS": return -1.0
        
        # Rule 3: AAABBB (Triple Triple)
        # If we see B B B S S -> Predict S (to complete triple)
        if code[-5:] == "BBBSS": return -1.0
        if code[-5:] == "SSSBB": return 1.0

        # Rule 5: AAB AAB (2-1 Rhythm)
        # If we see B B S B B -> Predict S
        if code[-5:] == "BBSBB": return -1.0
        if code[-5:] == "SSBSS": return 1.0
        
        # Rule 7: ABB ABB (1-2 Rhythm)
        # If we see B S S B S -> Predict S
        if code[-5:] == "BSSBS": return -1.0
        if code[-5:] == "SBBsb": return 1.0 # S B B S -> B
        
        # 3-2-1 Strategy (AAA BB C)
        if code[-5:] == "BBBSS": return -1.0
        if code[-5:] == "SSSBB": return 1.0
        
        return 0.0

# =============================================================================
# SECTION 2: THE ENGINES
# =============================================================================

class Engines:
    @staticmethod
    def visual_engine(outcomes: List[str]) -> float:
        # Uses the new Visual Library
        return VisualLib.analyze_patterns(outcomes)

    @staticmethod
    def trend_engine(outcomes: List[str]) -> float:
        # Catches the "Long Trend" (Method 11)
        if len(outcomes) < 5: return 0.0
        last_4 = outcomes[-4:]
        if last_4.count("BIG") == 4: return 1.0     
        if last_4.count("SMALL") == 4: return -1.0
        return 0.0

    @staticmethod
    def reversion_engine(numbers: List[float]) -> float:
        # Standard safety check
        if len(numbers) < 15: return 0.0
        try:
            mean = statistics.mean(numbers[-20:])
            stdev = statistics.pstdev(numbers[-20:])
            z = (numbers[-1] - mean) / stdev if stdev != 0 else 0.0
        except: return 0.0
        
        if z > 2.0: return -1.0
        elif z < -2.0: return 1.0
        return 0.0

# =============================================================================
# SECTION 3: SUPERVISORS
# =============================================================================

class MarketMonitor:
    @staticmethod
    def check_volatility(numbers: List[float]) -> Tuple[bool, str]:
        if len(numbers) < 10: return False, "OK"
        recent = numbers[-8:]
        
        # RELAXED CHOP CHECK: Only skip if it's absolute chaos
        binary = [0 if x <= 4 else 1 for x in recent]
        switches = sum(1 for i in range(len(binary)-1) if binary[i] != binary[i+1])
        
        if switches >= 7: return True, "CHAOS_MODE" # Only skip on max chaos
        return False, "SAFE"

class EngineManager:
    @staticmethod
    def update_performance(last_result_str: str):
        actual = None
        if last_result_str == "BIG": actual = TradeDecision.BIG
        elif last_result_str == "SMALL": actual = TradeDecision.SMALL
        else: return 
        
        for name, engine in state.engines.items():
            if engine.last_vote is None or engine.last_vote == TradeDecision.SKIP:
                continue
            
            if engine.last_vote == actual:
                if engine.consecutive_losses > 0: engine.consecutive_losses -= 1
                if not engine.is_active:
                    engine.is_active = True
            else:
                engine.consecutive_losses += 1
                if engine.is_active and engine.consecutive_losses >= config.stop_loss_streak:
                    engine.is_active = False

# =============================================================================
# SECTION 4: VOTING COUNCIL
# =============================================================================

class VotingCouncil:
    def cast_votes(self, numbers: List[float], outcomes: List[str]) -> Tuple[TradeDecision, float, List[str]]:
        score = 0.0
        reasons = []
        
        # 1. Collect Votes (Only from the 3 Snipers)
        raw_votes = {
            'visual': Engines.visual_engine(outcomes),
            'trend': Engines.trend_engine(outcomes),
            'reversion': Engines.reversion_engine(numbers)
        }
        
        # 2. Weigh Votes
        active_weight_sum = 0.0
        
        for name, val in raw_votes.items():
            eng = state.engines[name]
            
            if val > 0: eng.last_vote = TradeDecision.BIG
            elif val < 0: eng.last_vote = TradeDecision.SMALL
            else: eng.last_vote = TradeDecision.SKIP
            
            if not eng.is_active:
                reasons.append(f"{name}(OFF)")
                continue
                
            active_weight_sum += eng.weight
            score += val * eng.weight
            if val != 0: reasons.append(f"{name}({val:+.1f})")

        if active_weight_sum == 0:
            return TradeDecision.SKIP, 0.0, ["SILENCE"]
            
        # Normalize Score
        # Divisor is smaller now because we have fewer engines
        normalized_score = score / 2.0 
        
        decision = TradeDecision.SKIP
        conf = 0.0
        
        # SNIPER THRESHOLD (0.75)
        THRESHOLD = 0.75
        
        if normalized_score >= THRESHOLD:
            decision = TradeDecision.BIG
            conf = min(0.6 + (normalized_score/5), 0.99)
        elif normalized_score <= -THRESHOLD:
            decision = TradeDecision.SMALL
            conf = min(0.6 + (abs(normalized_score)/5), 0.99)
            
        return decision, conf, reasons

# =============================================================================
# MAIN EXPORT
# =============================================================================

def ultraAIPredict(history: List[Dict], currentbankroll: float, lastresult: Optional[str] = None) -> Dict:
    
    if state.session_start_bankroll == 0:
        state.session_start_bankroll = currentbankroll
    state.current_profit = currentbankroll - state.session_start_bankroll

    # Data Cleaning
    clean_nums = []
    clean_outcomes = []
    for item in reversed(history):
        v = MathLib.safe_float(item.get('actual_number'))
        if v is not None:
            clean_nums.append(v)
            if 0 <= int(v) <= 4: clean_outcomes.append("SMALL")
            elif 5 <= int(v) <= 9: clean_outcomes.append("BIG")

    # Feedback Loop
    if lastresult and state.last_prediction and state.last_prediction != TradeDecision.SKIP:
        EngineManager.update_performance(lastresult)
        
        # Ghost Protocol Check
        real_res = None
        if lastresult == "BIG": real_res = TradeDecision.BIG
        elif lastresult == "SMALL": real_res = TradeDecision.SMALL
        
        if real_res:
            did_win = (state.last_prediction == real_res)
            if did_win:
                state.loss_streak = 0
                state.consecutive_fails = 0
            else:
                state.loss_streak += 1
                if not state.inversion_mode:
                    state.consecutive_fails += 1
                    if state.consecutive_fails >= 3:
                        state.inversion_mode = True

    # Safety Checks
    if state.cooling_off_counter > 0:
        state.cooling_off_counter -= 1
        
    is_unsafe, vol_reason = MarketMonitor.check_volatility(clean_nums)
    if is_unsafe:
        state.cooling_off_counter = 1
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "DANGER", 'reason': f"Volatile: {vol_reason}", 'topsignals': ["MARKET_CHAOS"]}

    # Prediction
    council = VotingCouncil()
    decision, confidence, signals = council.cast_votes(clean_nums, clean_outcomes)
    
    # Ghost Protocol (The Flip)
    final_decision_str = decision.value
    meta_status = "SNIPER"
    
    if decision != TradeDecision.SKIP and state.inversion_mode:
        meta_status = "GHOST"
        if decision == TradeDecision.BIG:
            final_decision_str = "SMALL" 
            signals.append("INVERTED")
        elif decision == TradeDecision.SMALL:
            final_decision_str = "BIG"   
            signals.append("INVERTED")

    # Dynamic Stake
    if final_decision_str == "SKIP":
        state.last_prediction = TradeDecision.SKIP
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "WAIT", 'reason': "No Pattern", 'topsignals': signals}

    base_stake = max(currentbankroll * config.base_risk_percent, config.min_bet_amount)
    
    if state.current_profit > 100: base_stake *= 1.5
    elif state.current_profit < -150: base_stake *= 0.5
        
    stake = base_stake * (config.martingale_multiplier ** state.loss_streak)
    if stake > (currentbankroll * 0.20): stake = currentbankroll * 0.20

    state.last_prediction = TradeDecision(final_decision_str)
    state.last_confidence = confidence

    return {
        'finalDecision': final_decision_str,
        'confidence': round(confidence, 4),
        'positionsize': int(stake),
        'level': f"L{state.loss_streak} ({meta_status})",
        'reason': " | ".join(signals),
        'topsignals': signals
    }

if __name__ == "__main__":
    print("TITAN V700 SNIPER EDITION (VISUAL) LOADED.")
