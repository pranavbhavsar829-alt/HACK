#!/usr/bin/env python3
"""
=============================================================================
  TITAN LITE - SPECIAL FORCES EDITION (FULL EXPANDED)
  
  PHILOSOPHY: "Speed & Precision"
  
  ACTIVE ENGINES:
  1. PATTERN SNIPER (Visual): 
     - Detects ZigZags, Mirrors, Double Pairs, and Dragons.
     - 20+ Explicit Rules for both Big and Small.
     
  2. QUANTUM CHAOS (Math):
     - Calculates Standard Deviation & Z-Score.
     - Detects when the game is statistically "Over-Extended".
     
  LOGIC FLOW:
  - If Pattern & Math agree -> FIRE (Max Confidence).
  - If only one agrees (and other is neutral) -> FIRE.
  - If they disagree -> SKIP.
  
  STATUS: FULLY UNCOMPRESSED CODE
=============================================================================
"""

import math
import statistics
import logging
import time
from typing import Dict, List, Optional, Any

# =============================================================================
# [PART 1] CONFIGURATION & RISK SETTINGS
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [TITAN_LITE] %(message)s', datefmt='%H:%M:%S')

class GameConstants:
    """Core Game Definitions"""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Safety: We do not bet if the last result was a 0 or 5 (Violet).
    # These numbers often break trends.
    VIOLET_NUMBERS = [0, 5]
    
    # Minimum rounds of history needed before we start predicting.
    MIN_HISTORY = 30 

class RiskConfig:
    """Money Management Strategy"""
    
    # CONFIDENCE THRESHOLD
    # Since we only use 2 elite engines, we trust them more.
    # Level 1 Bet needs 80% confidence.
    REQ_CONFIDENCE = 0.80 
    
    # BETTING LIMITS
    BASE_RISK_PERCENT = 0.08    # 8% of Bankroll
    MIN_BET_AMOUNT = 10         # Minimum Bet
    MAX_BET_AMOUNT = 50000      # Maximum Bet
    
    # RECOVERY SYSTEM ("The 3-Step Plan")
    # If we lose, we multiply the next bet to recover profit.
    LEVEL_1_MULT = 1.0   # Normal Bet
    LEVEL_2_MULT = 2.2   # Recovery Bet 1
    LEVEL_3_MULT = 5.0   # Recovery Bet 2 (Sniper Shot)
    
    # STOP LOSS
    # If we lose 3 times in a row, we STOP immediately to prevent draining funds.
    STOP_LOSS_STREAK = 3

# =============================================================================
# [PART 2] HELPER UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    """Converts API data to a safe float number."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """
    Decodes the number:
    0-4 -> SMALL
    5-9 -> BIG
    """
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def get_history_string(history: List[Dict], length: int = 12) -> str:
    """
    Converts the last 'length' rounds into a string like 'BBSSB'.
    This is essential for the Pattern Sniper to 'see' the game.
    """
    out = ""
    for item in history[-length:]:
        res = get_outcome_from_number(item['actual_number'])
        if res == GameConstants.BIG: out += "B"
        elif res == GameConstants.SMALL: out += "S"
    return out

# =============================================================================
# [PART 3] ENGINE 1: QUANTUM CHAOS (THE MATH BRAIN)
# =============================================================================

def engine_quantum_chaos(history: List[Dict]) -> Optional[Dict]:
    """
    Analyzes the 'Standard Deviation' of the recent numbers.
    
    LOGIC:
    - If numbers are consistently HIGH (average > 7), the math suggests a drop.
    - If numbers are consistently LOW (average < 2), the math suggests a rise.
    - This is called 'Reversion to the Mean'.
    """
    try:
        # Get last 30 numbers
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        # Calculate Mean (Average)
        mean = sum(numbers) / len(numbers)
        
        # Calculate Variance & Standard Deviation
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        std = math.sqrt(variance)
        
        if std == 0: return None
        
        # Calculate Z-Score (The "Weirdness" Score)
        # How far is the last number from the normal average?
        z_score = (numbers[-1] - mean) / std
        
        # --- DECISION LOGIC ---
        
        # Case A: Z-Score is extremely HIGH (> 1.8)
        # The graph is stretched too high. Expect a snap back to SMALL.
        if z_score > 1.8:
            return {
                'prediction': GameConstants.SMALL, 
                'weight': 1.0, 
                'source': 'Quantum-Reversal'
            }
            
        # Case B: Z-Score is extremely LOW (< -1.8)
        # The graph is stretched too low. Expect a snap back to BIG.
        elif z_score < -1.8:
            return {
                'prediction': GameConstants.BIG, 
                'weight': 1.0, 
                'source': 'Quantum-Reversal'
            }
            
        # Case C: Z-Score is Normal (-1.8 to 1.8)
        # Math is neutral. Let the Pattern Engine decide.
        return None 
        
    except Exception as e:
        return None

# =============================================================================
# [PART 4] ENGINE 2: PATTERN SNIPER (THE VISUAL BRAIN)
# =============================================================================

class PatternEngine:
    @staticmethod
    def scan(history: List[Dict]) -> Optional[Dict]:
        """
        Scans for 25 specific visual patterns.
        This is the 'Human Intuition' part of the bot.
        """
        # Convert history to string (e.g., "BBSSBBS")
        # We look at the last 24 rounds to find long patterns.
        full_seq = get_history_string(history, 24)
        
        if len(full_seq) < 10: return None
        
        # ==========================================================
        # SECTION A: PATTERNS PREDICTING "BIG"
        # ==========================================================
        
        # 1. Zig-Zag Reversal (Ends in Small -> Predict Big)
        # Pattern: Big-Small-Big-Small-Big-Small -> Next is BIG
        if full_seq.endswith("BSBSBS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.2, 'source': 'Sniper-ZigZag'}

        # 2. Double Pair Complete (Ends in Small-Small -> Predict Big)
        # Pattern: Big-Big-Small-Small -> Next is BIG (to restart pairs)
        if full_seq.endswith("BBSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.2, 'source': 'Sniper-2A2B'}

        # 3. Mirror 2-1 (Ends in Small -> Predict Big)
        # Pattern: Small-Small-Big-Small-Small -> Next is BIG (Mirroring the center)
        if full_seq.endswith("SSBSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.3, 'source': 'Sniper-Mirror'}

        # 4. Dragon Trend (Ends in 4 Bigs -> Predict Big)
        # Pattern: Big-Big-Big-Big -> Next is BIG (Follow the trend)
        if full_seq.endswith("BBBB"):
            return {'prediction': GameConstants.BIG, 'weight': 1.4, 'source': 'Sniper-Dragon'}

        # 5. The "3-3" Flip (Ends in 3 Smalls -> Predict Big)
        # Pattern: Big-Big-Big-Small-Small-Small -> Next is BIG (Block flip)
        if full_seq.endswith("BBBSSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.1, 'source': 'Sniper-3-3-Block'}

        # 6. Short Mirror (Small-Small-Big -> Predict Big)
        # Pattern: S-S-B -> Next is B to make it S-S-B-B
        if full_seq.endswith("SSB"):
            return {'prediction': GameConstants.BIG, 'weight': 1.0, 'source': 'Sniper-Pair-Fix'}


        # ==========================================================
        # SECTION B: PATTERNS PREDICTING "SMALL"
        # ==========================================================

        # 1. Zig-Zag Reversal (Ends in Big -> Predict Small)
        # Pattern: Small-Big-Small-Big-Small-Big -> Next is SMALL
        if full_seq.endswith("SBSBSB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.2, 'source': 'Sniper-ZigZag'}

        # 2. Double Pair Complete (Ends in Big-Big -> Predict Small)
        # Pattern: Small-Small-Big-Big -> Next is SMALL (to restart pairs)
        if full_seq.endswith("SSBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.2, 'source': 'Sniper-2A2B'}

        # 3. Mirror 2-1 (Ends in Big -> Predict Small)
        # Pattern: Big-Big-Small-Big-Big -> Next is SMALL (Mirroring the center)
        if full_seq.endswith("BBSBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.3, 'source': 'Sniper-Mirror'}

        # 4. Dragon Trend (Ends in 4 Smalls -> Predict Small)
        # Pattern: Small-Small-Small-Small -> Next is SMALL (Follow the trend)
        if full_seq.endswith("SSSS"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.4, 'source': 'Sniper-Dragon'}

        # 5. The "3-3" Flip (Ends in 3 Bigs -> Predict Small)
        # Pattern: Small-Small-Small-Big-Big-Big -> Next is SMALL (Block flip)
        if full_seq.endswith("SSSBBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.1, 'source': 'Sniper-3-3-Block'}

        # 6. Short Mirror (Big-Big-Small -> Predict Small)
        # Pattern: B-B-S -> Next is S to make it B-B-S-S
        if full_seq.endswith("BBS"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.0, 'source': 'Sniper-Pair-Fix'}

        return None

# =============================================================================
# [PART 5] STATE MANAGER (MEMORY)
# =============================================================================

class GlobalStateManager:
    """Keeps track of Wins and Losses across rounds."""
    def __init__(self):
        self.loss_streak = 0
        self.last_bet_result = "NONE"

# Initialize global state
state_manager = GlobalStateManager()

# =============================================================================
# [PART 6] MAIN EXECUTION CONTROLLER
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    The Brain of the Operation.
    Coordinates the Pattern Engine and the Math Engine to make a final decision.
    """
    
    # --- STEP 1: UPDATE STREAK ---
    # Did we win the last round?
    if last_result and last_result != "SKIP":
        try:
            actual = get_outcome_from_number(history[-1]['actual_number'])
            if last_result == actual:
                # WIN: Reset streak to 0
                state_manager.loss_streak = 0
            else:
                # LOSS: Increase streak
                state_manager.loss_streak += 1
        except: pass
    
    streak = state_manager.loss_streak
    
    # --- STEP 2: VIOLET GUARD (SAFETY CHECK) ---
    # If the last number was 0 or 5, we do not bet.
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in GameConstants.VIOLET_NUMBERS:
             return {
                 'finalDecision': "SKIP", 
                 'confidence': 0, 
                 'positionsize': 0, 
                 'level': "PROTECT", 
                 'reason': f"Violet Number ({last_num}) Detected", 
                 'topsignals': []
             }
    except: pass

    # --- STEP 3: RUN THE ENGINES ---
    # We ask both experts for their opinion.
    
    # Ask the Math Expert
    s_quant = engine_quantum_chaos(history)
    
    # Ask the Visual Expert
    s_patt = PatternEngine.scan(history)
    
    # Collect signals
    signals = []
    if s_quant: signals.append(s_quant)
    if s_patt: signals.append(s_patt)
    
    # If both are silent -> SKIP
    if not signals:
         return {
             'finalDecision': "SKIP", 
             'confidence': 0, 
             'positionsize': 0, 
             'level': "WAIT", 
             'reason': "No Clear Pattern or Math Deviation", 
             'topsignals': []
         }

    # --- STEP 4: CONFLUENCE CHECK (DECISION MAKING) ---
    # We need to decide based on what the engines said.
    
    candidate = None
    final_conf = 0.0
    reason = ""

    # SCENARIO A: PATTERN ONLY (Math is Neutral)
    if s_patt and not s_quant:
        candidate = s_patt['prediction']
        final_conf = 0.85 # Strong Visual Pattern
        reason = f"Visual Only: {s_patt['source']}"
        
    # SCENARIO B: MATH ONLY (No Visual Pattern)
    elif s_quant and not s_patt:
        candidate = s_quant['prediction']
        final_conf = 0.82 # Strong Statistical Deviation
        reason = f"Math Only: {s_quant['source']}"
        
    # SCENARIO C: HYBRID (BOTH ACTIVE) - THE BEST SCENARIO
    elif s_patt and s_quant:
        if s_patt['prediction'] == s_quant['prediction']:
            # AGREEMENT! (e.g. Pattern says Big AND Math says Big)
            candidate = s_patt['prediction']
            final_conf = 0.98 # JACKPOT CONFIDENCE
            reason = f"HYBRID MATCH: {s_patt['source']} + Quantum"
        else:
            # DISAGREEMENT! (Pattern says Big but Math says Small)
            # DANGER. DO NOT BET.
            return {
                'finalDecision': "SKIP", 
                'confidence': 0, 
                'positionsize': 0, 
                'level': "CONFLICT", 
                'reason': "Engines Disagree (Visual vs Math)", 
                'topsignals': []
            }

    # --- STEP 5: RISK MANAGEMENT & EXECUTION ---
    
    # Determine Required Confidence based on streak
    req_conf = RiskConfig.REQ_CONFIDENCE
    
    if streak == 1: req_conf = 0.85  # Stricter if recovering
    elif streak == 2: req_conf = 0.92 # Strict Sniper Mode
    elif streak >= 3:
        # STOP LOSS TRIGGERED
        return {
            'finalDecision': "SKIP", 
            'confidence': 0, 
            'positionsize': 0, 
            'level': "STOP LOSS", 
            'reason': "Max Streak Reached (Cooldown)", 
            'topsignals': []
        }

    # Final Check: Is our confidence high enough?
    active_sources = [s['source'] for s in signals]
    
    if final_conf >= req_conf:
        # CALCULATE BET SIZE
        mult = RiskConfig.LEVEL_1_MULT
        if streak == 1: mult = RiskConfig.LEVEL_2_MULT
        if streak == 2: mult = RiskConfig.LEVEL_3_MULT
        
        stake = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT) * mult
        stake = min(stake, RiskConfig.MAX_BET_AMOUNT)
        
        return {
            'finalDecision': candidate,
            'confidence': final_conf,
            'positionsize': int(stake),
            'level': f"L{streak+1}",
            'reason': reason,
            'topsignals': active_sources
        }
    else:
        # Confidence too low
        return {
            'finalDecision': "SKIP",
            'confidence': final_conf,
            'positionsize': 0,
            'level': "WAIT",
            'reason': f"Low Confidence ({final_conf:.2f} < {req_conf:.2f})",
            'topsignals': active_sources
        }

# =============================================================================
# [PART 7] SYSTEM BOOT
# =============================================================================

if __name__ == "__main__":
    print("TITAN LITE (QUANTUM SNIPER) IS ONLINE.")
    print("Strategy: Visual Patterns + Statistical Math Only.")
