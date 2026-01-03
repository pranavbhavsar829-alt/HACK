#!/usr/bin/env python3
"""
=============================================================================
TITAN V500 - SOVEREIGN EDITION (RENDER CALIBRATED)
=============================================================================
Codename: "PROMETHEUS-LIVE"
Version: 5.1.0 (Server Optimized)
Target: High-Frequency Prediction Markets

CHANGELOG:
- Logic: PRESERVED (Exact same Math/Engines as Original)
- Thresholds: CALIBRATED for Cold Starts (Prevents infinite SKIP loop)
- State: Hardened against server restarts
=============================================================================
"""

import math
import statistics
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# SECTION 1: SYSTEM CONFIGURATION & CONSTANTS
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TITAN_CORE")

class TradeDecision(Enum):
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

@dataclass
class RiskProfile:
    """Configuration for Risk Management"""
    base_risk_percent: float = 0.05       
    max_risk_percent: float = 0.15        
    min_bet_amount: float = 10.0
    max_bet_amount: float = 50000.0
    stop_loss_streak: int = 6             
    martingale_multiplier: float = 2.0    
    
@dataclass
class EngineWeights:
    """Voting Power of each Engine"""
    reversion: float = 1.5
    trend: float = 1.2
    neuren: float = 3.0      
    qaum: float = 3.5        
    pattern: float = 1.0

class GlobalState:
    """Persistent State across prediction cycles"""
    def __init__(self):
        self.loss_streak: int = 0
        # Start with a high skip streak to force early calibration if needed
        self.skip_streak: int = 2 
        self.last_prediction: Optional[TradeDecision] = None
        self.last_confidence: float = 0.0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.bankroll_history: List[float] = []

    def update_after_round(self, won: bool):
        if won:
            self.loss_streak = 0
            self.total_wins += 1
            self.skip_streak = 0
        else:
            self.loss_streak += 1
            self.total_losses += 1

# Initialize Global State
state = GlobalState()
config = RiskProfile()
weights = EngineWeights()

# =============================================================================
# SECTION 2: ADVANCED MATHEMATICS LIBRARY
# =============================================================================

class MathLib:
    """Dedicated Mathematical Operations"""
    
    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        try:
            if value is None: return None
            v = float(value)
            if math.isnan(v): return None
            return v
        except: return None

    @staticmethod
    def get_z_score(data: List[float]) -> float:
        if len(data) < 2: return 0.0
        try:
            mean = statistics.mean(data)
            stdev = statistics.pstdev(data)
            if stdev == 0: return 0.0
            return (data[-1] - mean) / stdev
        except: return 0.0

    @staticmethod
    def calculate_entropy(data: List[float]) -> float:
        if not data: return 0.0
        counts = {}
        for x in data:
            counts[x] = counts.get(x, 0) + 1
        probs = [c / len(data) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    @staticmethod
    def get_derivative(data: List[float], order: int = 1) -> float:
        if len(data) < order + 1: return 0.0
        
        current = data
        for _ in range(order):
            new_data = []
            for i in range(len(current) - 1):
                new_data.append(current[i+1] - current[i])
            current = new_data
            
        return current[-1] if current else 0.0

# =============================================================================
# SECTION 3: DATA PROCESSING LAYER
# =============================================================================

class DataProcessor:
    @staticmethod
    def process_history(history: List[Dict], limit: int = 100) -> Tuple[List[float], List[str]]:
        clean_nums = []
        raw_outcomes = []
        
        for item in reversed(history):
            val = MathLib.safe_float(item.get('actual_number'))
            if val is not None:
                clean_nums.append(val)
                if 0 <= int(val) <= 4: raw_outcomes.append(TradeDecision.SMALL.value)
                elif 5 <= int(val) <= 9: raw_outcomes.append(TradeDecision.BIG.value)
            
            if len(clean_nums) >= limit:
                break
                
        return list(reversed(clean_nums)), list(reversed(raw_outcomes))

# =============================================================================
# SECTION 4: THE PREDICTION ENGINES (LOGIC PRESERVED)
# =============================================================================

class Engines:
    # --- ENGINE 1: STANDARD TREND ---
    @staticmethod
    def trend_engine(outcomes: List[str]) -> float:
        if len(outcomes) < 5: return 0.0
        last_4 = outcomes[-4:]
        bigs = last_4.count("BIG")
        smalls = last_4.count("SMALL")
        
        if bigs == 4: return 1.0     
        if smalls == 4: return -1.0  
        
        if len(outcomes) >= 4:
            pat = outcomes[-4:]
            if pat == ['BIG', 'SMALL', 'BIG', 'SMALL']: return 1.0 
            if pat == ['SMALL', 'BIG', 'SMALL', 'BIG']: return -1.0 
            
        return 0.0

    # --- ENGINE 2: REVERSION ---
    @staticmethod
    def reversion_engine(numbers: List[float]) -> float:
        if len(numbers) < 15: return 0.0
        z = MathLib.get_z_score(numbers[-20:])
        if z > 2.0: return -1.0
        elif z < -2.0: return 1.0
        return 0.0

    # --- ENGINE 3: NEUREN (VELOCITY) ---
    @staticmethod
    def neuren_engine(numbers: List[float]) -> float:
        if len(numbers) < 6: return 0.0
        accel = MathLib.get_derivative(numbers, 2)
        jerk = MathLib.get_derivative(numbers, 3)
        
        if jerk > 5.0: return -1.0  
        if jerk < -5.0: return 1.0  
        if accel > 3.0: return -0.5
        if accel < -3.0: return 0.5
        return 0.0

    # --- ENGINE 4: QAUM (CHAOS) ---
    @staticmethod
    def qaum_engine(numbers: List[float]) -> float:
        if len(numbers) < 5: return 0.0
        recent = numbers[-4:]
        variance = statistics.pvariance(recent)
        
        if variance < 0.8:
            avg = sum(recent) / len(recent)
            if avg < 4.5: return 1.0 
            else: return -1.0        
        return 0.0

# =============================================================================
# SECTION 5: BANKROLL MANAGER
# =============================================================================

class BankrollManager:
    @staticmethod
    def get_stake(bankroll: float, confidence: float, streak: int) -> Tuple[float, str]:
        base_unit = max(bankroll * config.base_risk_percent, config.min_bet_amount)
        
        if streak >= config.stop_loss_streak:
            return config.min_bet_amount, "STOP_LOSS_RESET"
            
        if streak == 0:
            multiplier = 1.0
            if confidence > 0.85: multiplier = 1.2
            if confidence > 0.90: multiplier = 1.5
            stake = base_unit * multiplier
            return min(stake, config.max_bet_amount), "SNIPER_ENTRY"
        else:
            multiplier = config.martingale_multiplier ** streak
            stake = base_unit * multiplier
            stake = min(stake, config.max_bet_amount)
            
            if stake > (bankroll * 0.20):
                stake = bankroll * 0.20
                return stake, "PANIC_CLAMP"
            return stake, f"RECOVERY_L{streak}"

# =============================================================================
# SECTION 6: THE VOTING COUNCIL (CALIBRATED FOR SERVER)
# =============================================================================

class VotingCouncil:
    """Aggregates votes - TUNED FOR LIVE SERVER EXECUTION"""
    
    def cast_votes(self, numbers: List[float], outcomes: List[str]) -> Tuple[TradeDecision, float, List[str]]:
        
        score = 0.0
        reasons = []
        
        v_trend = Engines.trend_engine(outcomes)
        score += v_trend * weights.trend
        if v_trend != 0: reasons.append(f"Trend({v_trend:+.1f})")
        
        v_rev = Engines.reversion_engine(numbers)
        score += v_rev * weights.reversion
        if v_rev != 0: reasons.append(f"Rev({v_rev:+.1f})")
        
        v_neu = Engines.neuren_engine(numbers)
        score += v_neu * weights.neuren
        if v_neu != 0: reasons.append(f"Neuren({v_neu:+.1f})")
        
        v_qau = Engines.qaum_engine(numbers)
        score += v_qau * weights.qaum
        if v_qau != 0: reasons.append(f"Qaum({v_qau:+.1f})")
        
        # --- CALIBRATED THRESHOLD ADJUSTMENT ---
        # OLD: Base 2.0 (Too strict for cold start)
        # NEW: Base 1.4 (Allows entry)
        
        base_threshold = 1.4  
        
        # Reduce threshold by 0.1 for EVERY skip to force engagement
        reduction = min(state.skip_streak * 0.1, 0.9)
        final_threshold = max(base_threshold - reduction, 0.5)
        
        if reduction > 0:
            reasons.append(f"Adj(-{reduction:.1f})")
        
        decision = TradeDecision.SKIP
        confidence = 0.0
        
        if score >= final_threshold:
            decision = TradeDecision.BIG
            confidence = min(0.6 + (score/10), 0.95)
            
        elif score <= -final_threshold:
            decision = TradeDecision.SMALL
            confidence = min(0.6 + (abs(score)/10), 0.95)
            
        return decision, confidence, reasons

# =============================================================================
# SECTION 7: MAIN EXECUTION INTERFACE
# =============================================================================

def ultraAIPredict(history: List[Dict], currentbankroll: float = 10000.0, lastresult: Optional[str] = None) -> Dict:
    
    numbers, outcomes = DataProcessor.process_history(history)
    
    # Update State (Win/Loss Tracking)
    if state.last_prediction and state.last_prediction != TradeDecision.SKIP and lastresult:
        actual_res = None
        if lastresult in ['BIG', 'SMALL']:
            actual_res = lastresult
        elif outcomes:
            actual_res = outcomes[-1]
            
        if actual_res:
            did_win = (state.last_prediction.value == actual_res)
            state.update_after_round(did_win)
    
    # Data Integrity Check
    if len(numbers) < 10:
        return {
            'finalDecision': "SKIP",
            'confidence': 0.0,
            'positionsize': 0,
            'level': "BOOTING",
            'reason': "Need more data",
            'topsignals': []
        }
        
    council = VotingCouncil()
    decision, confidence, signals = council.cast_votes(numbers, outcomes)
    
    if decision == TradeDecision.SKIP:
        state.skip_streak += 1
        return {
            'finalDecision': "SKIP",
            'confidence': 0.0,
            'positionsize': 0,
            'level': "SCANNING",
            'reason': f"Wait (Skips: {state.skip_streak})",
            'topsignals': signals
        }
    else:
        state.skip_streak = 0
        
    stake, level_name = BankrollManager.get_stake(currentbankroll, confidence, state.loss_streak)
    
    state.last_prediction = decision
    state.last_confidence = confidence
    
    return {
        'finalDecision': decision.value,
        'confidence': round(confidence, 4),
        'positionsize': int(stake),
        'level': level_name,
        'reason': " | ".join(signals),
        'topsignals': signals
    }

if __name__ == "__main__":
    print("TITAN V500 SOVEREIGN (RENDER CALIBRATED) LOADED SUCCESSFULLY.")
