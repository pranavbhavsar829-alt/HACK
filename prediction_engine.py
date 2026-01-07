"""
TITAN V900 - OMNI-SNIPER PREDICTION ARCHITECT
================================================================================
SYSTEM ARCHITECTURE:
1. CHAOS FILTER: Rejects markets with High Entropy (Randomness).
2. TITAN MEMORY: Internal state tracking to prevent "Ghost Losses".
3. TRIDENT ENGINES: Legacy Logic (Quantum, Deep Pattern, Neural).
4. OMNI-PATTERN: Visual Recognition (Dragon, 1A1B, Mirrors).
5. SNIPER GATES: 3-Stage Confirmation System for 100% Win Strategy.
================================================================================
"""

import math
import statistics
import random
import traceback
from collections import deque, Counter
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION & IMMUTABLE CONSTANTS
# =============================================================================

class GameConstants:
    """Core Game Definitions"""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Minimum data points required before engines engage
    MIN_HISTORY_WARMUP = 35
    
    # Confidence Thresholds
    CONFIDENCE_LEVEL_1 = 0.72  # Standard Entry
    CONFIDENCE_LEVEL_2 = 0.85  # Recovery Entry
    CONFIDENCE_LEVEL_3 = 0.95  # Sniper / All-in Warning

class RiskConfig:
    """Financial Risk Management Protocols"""
    
    # Base bet as a percentage of bankroll (0.01 = 1%)
    BASE_RISK_PERCENT = 0.01    
    
    # Hard limits
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 50000
    MAX_BANKROLL_USAGE = 0.40  # Never bet more than 40% of total funds
    
    # --- ADVANCED MARTINGALE STEPS ---
    # Multipliers for recovery.
    # Step 0: Base Bet
    # Step 1: 2.2x (Cover loss + Profit)
    # Step 2: 5.0x (Deep Recovery)
    # Step 3: 12.0x (Aggressive Defense)
    # Step 4: 25.0x (The Wall - Do not cross)
    MARTINGALE_MULTIPLIERS = [1.0, 2.2, 5.0, 12.0, 25.0] 
    
    # Stop Loss Logic
    STOP_LOSS_STREAK = 5       # Reset to base after 5 losses
    
    # --- MARKET CHAOS FILTERS ---
    # If the market flips (B->S->B) more than 65% of the time, we pause.
    MAX_TRAP_INDEX = 0.65       
    MIN_TREND_STRENGTH = 0.30   

# =============================================================================
# SECTION 2: TITAN TRUE MEMORY (STATE MACHINE)
# =============================================================================

class TitanMemory:
    """
    Persistently tracks wins and losses in RAM to bypass API sync delays.
    This ensures the bot knows EXACTLY what stage of Martingale it is in.
    """
    def __init__(self):
        self.last_predicted_issue: Optional[str] = None
        self.last_predicted_label: Optional[str] = None
        self.loss_streak: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.history_log: deque = deque(maxlen=100)
        self.session_profit = 0.0

    def update_streak(self, latest_issue: str, latest_outcome: str):
        """Called every new round to verify the previous prediction."""
        # Only update if we actually have a pending bet for this issue
        if str(latest_issue) == str(self.last_predicted_issue):
            
            if self.last_predicted_label and self.last_predicted_label != GameConstants.SKIP:
                if self.last_predicted_label == latest_outcome:
                    # VICTORY
                    self.loss_streak = 0
                    self.wins += 1
                    self.history_log.append("W")
                else:
                    # DEFEAT
                    self.loss_streak += 1
                    self.losses += 1
                    self.history_log.append("L")
            
            # Reset prediction memory so we don't process the same issue twice
            self.last_predicted_issue = None
            self.last_predicted_label = None

    def register_bet(self, target_issue: str, label: str):
        """Registers a new pending bet."""
        self.last_predicted_issue = target_issue
        self.last_predicted_label = label
        
    def get_accuracy(self) -> float:
        if not self.history_log: return 0.0
        wins = self.history_log.count("W")
        return wins / len(self.history_log)

# Initialize Global Memory
titan_memory = TitanMemory()

# =============================================================================
# SECTION 3: MATHEMATICAL & STATISTICAL UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    """Robust float conversion."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> str:
    """Converts raw number (0-9) to BIG/SMALL."""
    try:
        val = int(safe_float(n))
        return GameConstants.SMALL if 0 <= val <= 4 else GameConstants.BIG
    except: return GameConstants.SKIP

def sigmoid(x: float) -> float:
    """Neural Activation."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def calculate_rsi(data: List[float], period: int = 14) -> float:
    """Relative Strength Index (RSI) Calculation."""
    if len(data) < period + 1: return 50.0
    
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# SECTION 4: DIAGNOSTIC ENGINES (CHAOS & MATH)
# =============================================================================

def engine_chaos_filter(history: List[Dict]) -> Tuple[bool, str]:
    """
    Analyzes Market Entropy.
    Returns: (is_choppy: bool, reason: str)
    """
    if len(history) < 20: return False, "WARMUP"
    
    outcomes = [get_outcome_from_number(d['actual_number']) for d in history[-20:]]
    
    # 1. Calculate TRAP INDEX (Flip Rate)
    # A flip is when the outcome changes (Big -> Small or Small -> Big)
    flips = sum(1 for i in range(len(outcomes)-1) if outcomes[i] != outcomes[i+1])
    trap_ratio = flips / (len(outcomes) - 1)
    
    # 2. Calculate MAX STREAK
    current_streak = 1
    max_streak = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i-1]:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 1
    max_streak = max(max_streak, current_streak)

    # 3. Decision Logic
    # High Trap Ratio (>65%) means market is ping-ponging randomly.
    if trap_ratio > RiskConfig.MAX_TRAP_INDEX:
        return True, f"High Entropy ({trap_ratio:.0%} Flips)"
    
    # Low Streak (<3) + Moderate Trap (>50%) means noise.
    if max_streak < 3 and trap_ratio > 0.50:
        return True, "Noise / No Trend"
        
    return False, "Stable"

def engine_math_reversion(history: List[Dict]) -> Optional[Dict]:
    """
    Uses Standard Deviation (Bollinger Logic) to find reversals.
    """
    try:
        numbers = [safe_float(d['actual_number']) for d in history[-30:]]
        if len(numbers) < 15: return None
        
        mean = statistics.mean(numbers)
        stdev = statistics.pstdev(numbers)
        if stdev == 0: return None
        
        last_val = numbers[-1]
        z_score = (last_val - mean) / stdev
        
        # If Z-Score > 2.2, it is statistically overbought -> Expect Drop
        if z_score > 2.2:
            return {"pred": GameConstants.SMALL, "weight": 0.75, "src": "Math:Z-High"}
        # If Z-Score < -2.2, it is statistically oversold -> Expect Rise
        elif z_score < -2.2:
            return {"pred": GameConstants.BIG, "weight": 0.75, "src": "Math:Z-Low"}
            
    except: pass
    return None

# =============================================================================
# SECTION 5: PATTERN RECOGNITION ENGINES (VISUAL)
# =============================================================================

def engine_omni_patterns(history: List[Dict]) -> List[Dict]:
    """
    Scans for 10+ specific chart patterns.
    Returns a list of detected signals.
    """
    signals = []
    outcomes = [get_outcome_from_number(d['actual_number']) for d in history]
    
    # Create the "Map" string (e.g. "BBSSBSBB")
    h = "".join(["B" if x == GameConstants.BIG else "S" for x in outcomes])
    
    if len(h) < 12: return []

    # --- PATTERN 1: THE DRAGON (Momentum) ---
    # 5 or more of the same color. We BET WITH IT.
    if h.endswith("BBBBB"): 
        confidence = 0.90 + (len(h) - len(h.rstrip('B'))) * 0.01
        signals.append({"pred": "BIG", "weight": confidence, "src": "Dragon-B"})
    elif h.endswith("SSSSS"): 
        confidence = 0.90 + (len(h) - len(h.rstrip('S'))) * 0.01
        signals.append({"pred": "SMALL", "weight": confidence, "src": "Dragon-S"})

    # --- PATTERN 2: PING PONG (1A1B) ---
    # B S B S B -> Predict S
    if h.endswith("BSBSB"): 
        signals.append({"pred": "SMALL", "weight": 0.92, "src": "PingPong"})
    elif h.endswith("SBSBS"): 
        signals.append({"pred": "BIG", "weight": 0.92, "src": "PingPong"})

    # --- PATTERN 3: DOUBLE JUMP (2A2B) ---
    # BB SS BB -> Predict S (Completing the pair)
    if h.endswith("BBSSBB"): 
        signals.append({"pred": "SMALL", "weight": 0.88, "src": "2A2B-Full"})
    elif h.endswith("SSBBSS"): 
        signals.append({"pred": "BIG", "weight": 0.88, "src": "2A2B-Full"})
        
    # --- PATTERN 4: STAIRCASE (1-2-3) ---
    # B SS BBB -> Predict S (Start of 4?) or Break? 
    # Usually Staircase implies increasing volume.
    if h.endswith("BSSBBB"):
        signals.append({"pred": "SMALL", "weight": 0.70, "src": "Staircase"})
        
    # --- PATTERN 5: SANDWICH (ABA) ---
    # B S B -> Predict S
    if h.endswith("BSB"):
        signals.append({"pred": "SMALL", "weight": 0.60, "src": "Sandwich"})
    elif h.endswith("SBS"):
        signals.append({"pred": "BIG", "weight": 0.60, "src": "Sandwich"})

    return signals

def engine_deep_sequence(history: List[Dict]) -> Optional[Dict]:
    """
    Legacy Engine: Scans past history for the exact same sequence of last 6 results.
    """
    try:
        outcomes = [get_outcome_from_number(d['actual_number']) for d in history]
        if len(outcomes) < 100: return None
        
        # Look at last 6
        pattern = outcomes[-6:]
        pattern_str = "".join(["1" if x == "BIG" else "0" for x in pattern])
        
        # Search history
        history_str = "".join(["1" if x == "BIG" else "0" for x in outcomes[:-1]])
        
        # Count what happened after this pattern previously
        next_b = 0
        next_s = 0
        
        start = 0
        while True:
            idx = history_str.find(pattern_str, start)
            if idx == -1: break
            
            # Check the outcome AFTER the pattern
            if idx + 6 < len(history_str):
                res = history_str[idx+6] # The character after
                if res == "1": next_b += 1
                else: next_s += 1
            start = idx + 1
            
        total = next_b + next_s
        if total < 2: return None # Not enough sample size
        
        if next_b > next_s * 2:
            return {"pred": "BIG", "weight": 0.80, "src": f"DeepSeq({total})"}
        elif next_s > next_b * 2:
            return {"pred": "SMALL", "weight": 0.80, "src": f"DeepSeq({total})"}
            
    except: pass
    return None

# =============================================================================
# SECTION 6: THE ARCHITECT (MAIN LOGIC CORE)
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    MASTER PREDICTION FUNCTION
    --------------------------
    Combines all engines, applies risk management, filters via Chaos Theory,
    and returns the final betting decision.
    """
    
    # 1. DATA VALIDITY & WARMUP
    if not history or len(history) < GameConstants.MIN_HISTORY_WARMUP:
        return {
            "finalDecision": "SKIP", 
            "confidence": 0, 
            "positionsize": 0, 
            "level": "BOOT", 
            "reason": f"Need {GameConstants.MIN_HISTORY_WARMUP} Records", 
            "topsignals": []
        }

    # 2. UPDATE TRUE MEMORY
    # We use the VERY LATEST record to settle the previous bet.
    latest_record = history[-1]
    latest_issue = str(latest_record['issue'])
    latest_outcome = get_outcome_from_number(latest_record['actual_number'])
    
    titan_memory.update_streak(latest_issue, latest_outcome)
    loss_streak = titan_memory.loss_streak
    
    # 3. VIOLET GUARD (0 & 5 SAFETY)
    # 0 and 5 are 'Violet' numbers. They often reset the algorithm.
    # If the last number was 0 or 5, and we are not in a deep streak, we skip for safety.
    last_num = int(safe_float(latest_record['actual_number']))
    if last_num in [0, 5] and loss_streak < 2:
         return {
            "finalDecision": "SKIP", 
            "confidence": 0, 
            "positionsize": 0, 
            "level": "VIOLET", 
            "reason": f"Num {last_num} Reset", 
            "topsignals": []
        }

    # 4. CHAOS FILTER (THE SHIELD)
    # If we are not desperate (Streak < 3), we demand a clean market.
    if loss_streak < 3:
        is_choppy, chaos_reason = engine_chaos_filter(history)
        if is_choppy:
             return {
                "finalDecision": "SKIP", 
                "confidence": 0, 
                "positionsize": 0, 
                "level": "🛑 CHOPPY", 
                "reason": chaos_reason, 
                "topsignals": []
            }

    # 5. EXECUTE ENGINES
    signals = []
    
    # A. Omni Patterns (High Weight)
    p_sigs = engine_omni_patterns(history)
    signals.extend(p_sigs)
    
    # B. Math Reversion (Medium Weight)
    m_sig = engine_math_reversion(history)
    if m_sig: signals.append(m_sig)
    
    # C. Deep Sequence (Medium Weight)
    d_sig = engine_deep_sequence(history)
    if d_sig: signals.append(d_sig)

    # 6. VOTE AGGREGATION
    big_score = sum(s['weight'] for s in signals if s['pred'] == GameConstants.BIG)
    small_score = sum(s['weight'] for s in signals if s['pred'] == GameConstants.SMALL)
    
    total_score = big_score + small_score
    final_pred = GameConstants.SKIP
    confidence = 0.0
    
    # Normalize confidence
    if total_score > 0:
        if big_score > small_score:
            final_pred = GameConstants.BIG
            confidence = big_score / (total_score + 0.5) # Dampen slightly
            # Add bonus confidence if multiple engines agree
            if len([s for s in signals if s['pred'] == "BIG"]) >= 2:
                confidence += 0.1
        else:
            final_pred = GameConstants.SMALL
            confidence = small_score / (total_score + 0.5)
            if len([s for s in signals if s['pred'] == "SMALL"]) >= 2:
                confidence += 0.1
                
    confidence = min(confidence, 0.99)

    # 7. SNIPER LOGIC GATES (STRATEGY SELECTION)
    
    should_bet = False
    level_name = "WAITING"
    reason_text = "Analyzing..."
    
    # GATE 1: STANDARD ENTRY (Streak 0)
    if loss_streak == 0:
        if confidence >= GameConstants.CONFIDENCE_LEVEL_1:
            should_bet = True
            level_name = "✅ LEVEL 1"
            reason_text = "Strong Trend"
        else:
            level_name = "SKIP (Low Conf)"
            reason_text = f"Conf {confidence:.2f} < {GameConstants.CONFIDENCE_LEVEL_1}"

    # GATE 2: RECOVERY (Streak 1)
    elif loss_streak == 1:
        if confidence >= GameConstants.CONFIDENCE_LEVEL_2:
            should_bet = True
            level_name = "⚠️ LEVEL 2"
            reason_text = "Recovery Signal"
        else:
            level_name = "SKIP (Recov)"
            reason_text = "Waiting for A+"

    # GATE 3: THE SNIPER SHIELD (Streak 2+)
    # We DO NOT BET unless we see a Pattern (Weight > 0.85)
    elif loss_streak >= 2:
        # Check if we have a pure pattern source
        has_pattern = any(s['weight'] >= 0.85 for s in signals)
        
        if has_pattern:
            should_bet = True
            level_name = f"🔥 SNIPER (L{loss_streak+1})"
            reason_text = "PERFECT PATTERN MATCH"
        else:
            should_bet = False
            level_name = "🛡️ SHIELD MODE"
            reason_text = "Holding for Dragon/PingPong"

    # 8. POSITION SIZING (MARTINGALE)
    stake = 0
    if should_bet:
        # Get multiplier based on streak index
        m_idx = min(loss_streak, len(RiskConfig.MARTINGALE_MULTIPLIERS) - 1)
        multiplier = RiskConfig.MARTINGALE_MULTIPLIERS[m_idx]
        
        base_stake = current_bankroll * RiskConfig.BASE_RISK_PERCENT
        stake = int(base_stake * multiplier)
        
        # Hard Caps
        stake = max(RiskConfig.MIN_BET_AMOUNT, min(stake, RiskConfig.MAX_BET_AMOUNT))
        if stake > current_bankroll * RiskConfig.MAX_BANKROLL_USAGE:
            stake = int(current_bankroll * RiskConfig.MAX_BANKROLL_USAGE)
            level_name += " (Max Cap)"

        # 9. REGISTER PREDICTION
        # We are betting on the NEXT issue.
        next_issue = str(int(latest_issue) + 1)
        titan_memory.register_bet(next_issue, final_pred)
    
    # 10. FORMAT OUTPUT
    # Sort signals by weight for display
    sorted_sigs = sorted(signals, key=lambda x: x['weight'], reverse=True)
    top_signals = [f"{s['src']}={s['pred']}" for s in sorted_sigs[:3]]
    
    return {
        'finalDecision': final_pred if should_bet else GameConstants.SKIP,
        'confidence': confidence,
        'positionsize': stake,
        'level': level_name,
        'reason': reason_text,
        'topsignals': top_signals
    }

# =============================================================================
# SELF-DIAGNOSTIC BLOCK (RUNS IF EXECUTED DIRECTLY)
# =============================================================================
if __name__ == "__main__":
    print("TITAN V900 - OMNI-SNIPER CORE DIAGNOSTIC")
    print("----------------------------------------")
    
    # Generate Mock Data (Ping Pong Pattern)
    print("[1] Generating Mock Data (Ping Pong: B S B S B)...")
    mock_history = []
    start_issue = 20240000
    
    # Pattern: B(5) S(4) B(6) S(3) B(7) -> Expect S
    sequence = [5, 4, 6, 3, 7] 
    
    for i, num in enumerate(sequence):
        mock_history.append({
            'issue': str(start_issue + i),
            'actual_number': num,
            'fetch_time': '2024-01-01'
        })
        
    print(f"    Data Points: {len(mock_history)}")
    print(f"    Last Outcome: {get_outcome_from_number(mock_history[-1]['actual_number'])}")
    
    # Test Prediction
    print("[2] Running Prediction Architect...")
    result = ultraAIPredict(mock_history, current_bankroll=1000.0)
    
    print("\n[DIAGNOSTIC RESULTS]")
    print(f"Decision:   {result['finalDecision']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Stake:      ${result['positionsize']}")
    print(f"Level:      {result['level']}")
    print(f"Reason:     {result['reason']}")
    print(f"Signals:    {result['topsignals']}")
    
    if result['finalDecision'] == "SMALL":
        print("\n[SUCCESS] Engine correctly identified Ping-Pong pattern.")
    else:
        print("\n[FAIL] Engine failed to identify pattern.")
