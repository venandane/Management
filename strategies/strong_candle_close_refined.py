"""
Strong Candle Close Refined Strategy
Implements precise acceptance rules with non-look-ahead bad signal filtering.
Designed for post-hoc analysis (backtesting) with indefinite monitoring logic.
"""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from trading_system.strategies.base import StrategyBase
from trading_system.data.processing.indicators import StrongCandleDetector
from trading_system.core.utils import get_pip_size
import logging

logger = logging.getLogger(__name__)


class StrongCandleCloseRefinedStrategy(StrategyBase):
    """
    Post-hoc analysis strategy for strong candle filtering.
    NOT event-driven - designed for backtesting with full historical context.
    
    Acceptance Rules (SELL example):
      1. C1 close = lowest close in last 30 days → ACCEPT
      2. Else find nearest prior C2 with low < C1 close
      3. From C2 backwards, find nearest bad SELL C3 KNOWN before C1 close
      4. If C1 close ≥ (lowest low in [C3→C1]) + 12 pips → ACCEPT
      5. If no valid C3 → ACCEPT
    
    Critical Features:
      - Indefinite monitoring: Stops at FIRST adverse/profit event (not fixed 24h)
      - Non-look-ahead: Uses actual 'known_at' timestamp for bad signals
      - Pip-based tolerances: 0.5 pip tolerance for all comparisons
      - Symmetric BUY/SELL logic
    """
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        params: Optional[dict] = None
    ):
        super().__init__(symbol, timeframes, params)
        
        # Strategy parameters
        self.min_body_pips = params.get('min_body_pips', 5.0)
        self.min_range_pips = params.get('min_range_pips', 3.0)
        self.adverse_move_pips = params.get('adverse_move_pips', 12.0)
        self.profit_target_pips = params.get('profit_target_pips', 12.0)
        self.bad_signal_lookback_days = params.get('bad_signal_lookback_days', 30)
        self.tolerance_pips = params.get('tolerance_pips', 0.5)  # For price comparisons
        
        # Initialize detector
        self.detector = StrongCandleDetector(
            min_body_pips=self.min_body_pips,
            min_range_pips=self.min_range_pips
        )
    
    def analyze(
        self,
        h1_candles: pd.DataFrame,
        m1_candles: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Core analysis method - performs complete post-hoc strategy evaluation.
        
        Args:
            h1_candles: H1 OHLCV data (indexed by candle OPEN time)
            m1_candles: M1 OHLCV data for monitoring
        
        Returns:
            Tuple of (all_signals_df, accepted_signals_df, rejected_signals_df, bad_signals_df)
            where bad_signals_df includes 'known_at' timestamp column
        """
        logger.info(f"[{self.name}] Starting analysis for {self.symbol}...")
        
        # STEP 1: Detect all strong candles on H1
        h1_with_strong = self.detector.detect(h1_candles, self.symbol)
        strong_mask = h1_with_strong['strong_bull'] | h1_with_strong['strong_bear']
        strong_candles = h1_with_strong[strong_mask].copy()
        logger.info(f"  ✓ Detected {len(strong_candles)} strong candles")
        
        # STEP 2: Identify bad signals with indefinite monitoring
        bad_signals = self._identify_bad_signals(strong_candles, m1_candles)
        logger.info(f"  ✓ Identified {len(bad_signals)} bad signals")
        
        # STEP 3: Apply acceptance rules to generate signals
        all_signals, accepted_signals, rejected_signals = self._apply_acceptance_rules(
            strong_candles, bad_signals, h1_candles
        )
        logger.info(f"  ✓ Generated {len(accepted_signals)} accepted signals ({len(rejected_signals)} rejected)")
        
        return all_signals, accepted_signals, rejected_signals, bad_signals
    
    def _identify_bad_signals(
        self,
        strong_candles: pd.DataFrame,
        m1_candles: pd.DataFrame
    ) -> pd.DataFrame:
        """Indefinite monitoring: Classify strong candles as bad when adverse move occurs FIRST."""
        pip_size = get_pip_size(self.symbol)
        adverse_threshold = self.adverse_move_pips * pip_size
        profit_threshold = self.profit_target_pips * pip_size
        bad_records = []
        
        for idx, row in strong_candles.iterrows():
            direction = 1 if row['strong_bull'] else -1
            signal_price = row['close']
            candle_low = row['low']
            candle_high = row['high']
            
            # Monitoring starts AFTER candle CLOSE (H1 period = [open, open+1h))
            signal_close_time = idx + pd.Timedelta(hours=1)
            window_start = signal_close_time + pd.Timedelta(seconds=1)
            window_end = signal_close_time + pd.Timedelta(days=30)  # Large window (stops at first event)
            
            # Get M1 data in monitoring window
            try:
                m1_window = m1_candles.loc[window_start:window_end]
            except (KeyError, TypeError):
                m1_window = pd.DataFrame()
            
            # Monitor chronologically until FIRST decisive event
            adverse_timestamp = None
            profit_timestamp = None
            
            if not m1_window.empty:
                if direction < 0:  # SELL
                    adverse_level = signal_price + adverse_threshold
                    profit_level = signal_price - profit_threshold
                    
                    for m1_idx, m1_row in m1_window.iterrows():
                        if m1_row['high'] >= adverse_level and adverse_timestamp is None:
                            adverse_timestamp = m1_idx
                        if m1_row['low'] <= profit_level and profit_timestamp is None:
                            profit_timestamp = m1_idx
                        if adverse_timestamp or profit_timestamp:
                            break
                
                else:  # BUY (symmetric)
                    adverse_level = signal_price - adverse_threshold
                    profit_level = signal_price + profit_threshold
                    
                    for m1_idx, m1_row in m1_window.iterrows():
                        if m1_row['low'] <= adverse_level and adverse_timestamp is None:
                            adverse_timestamp = m1_idx
                        if m1_row['high'] >= profit_level and profit_timestamp is None:
                            profit_timestamp = m1_idx
                        if adverse_timestamp or profit_timestamp:
                            break
            
            # Classify as bad if adverse move occurred BEFORE profit (or profit never reached)
            is_bad = False
            known_at = None
            if adverse_timestamp:
                if profit_timestamp is None or adverse_timestamp < profit_timestamp:
                    is_bad = True
                    known_at = adverse_timestamp
            
            if is_bad:
                bad_records.append({
                    'timestamp': idx,
                    'candle_low': candle_low,
                    'candle_high': candle_high,
                    'direction': direction,
                    'signal_price': signal_price,
                    'adverse_timestamp': adverse_timestamp,
                    'profit_timestamp': profit_timestamp,
                    'known_at': known_at  # Critical for non-look-ahead filtering
                })
        
        return pd.DataFrame(bad_records).set_index('timestamp') if bad_records else pd.DataFrame()
    
    def _apply_acceptance_rules(
        self,
        strong_candles: pd.DataFrame,
        bad_signals: pd.DataFrame,
        h1_candles: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply precise acceptance rules with non-look-ahead C3 filtering."""
        pip_size = get_pip_size(self.symbol)
        tolerance_price = self.tolerance_pips * pip_size
        all_signals = []
        accepted = []
        rejected = []
        
        for idx, row in strong_candles.iterrows():
            direction = 1 if row['strong_bull'] else -1
            current_close = row['close']
            current_low = row['low']
            current_high = row['high']
            body_pips = row['body_pips']
            accept = False
            
            # ═══════════════════════════════════════════════════════════════
            # SELL SIGNAL RULES (non-look-ahead)
            # ═══════════════════════════════════════════════════════════════
            if direction < 0:
                # PART 1: Lowest close in 30 days?
                window_start = idx - pd.Timedelta(days=self.bad_signal_lookback_days)
                try:
                    closes_30d = h1_candles.loc[window_start:idx, 'close']
                    if abs(current_close - closes_30d.min()) <= tolerance_price:
                        accept = True
                except Exception:
                    pass
                
                # PART 2: Find C2 then C3
                if not accept:
                    # Find C2: nearest prior candle with low < C1 close
                    c2_time = None
                    prior = h1_candles[h1_candles.index < idx].sort_index(ascending=False)
                    for c2_idx, c2_row in prior.iterrows():
                        if c2_row['low'] <= current_close - tolerance_price:
                            c2_time = c2_idx
                            break
                    
                    if c2_time is None:
                        accept = True  # No C2 → accept
                    else:
                        # Find C3: nearest bad SELL at/before C2 KNOWN before C1 close
                        c1_close_time = idx + pd.Timedelta(hours=1)
                        if not bad_signals.empty:
                            valid_bad = bad_signals[
                                (bad_signals.index <= c2_time) &
                                (bad_signals['direction'] == -1) &
                                (bad_signals['known_at'] < c1_close_time)  # Non-look-ahead
                            ]
                            if not valid_bad.empty:
                                c3_time = valid_bad.index.max()
                                # Lowest low between C3 and C1
                                try:
                                    range_candles = h1_candles.loc[c3_time:idx]
                                    lowest_low = range_candles['low'].min()
                                    if (current_close - lowest_low) >= (12.0 * pip_size - tolerance_price):
                                        accept = True
                                except Exception:
                                    pass
                            else:
                                accept = True  # No valid C3 → accept
                        else:
                            accept = True  # No bad signals → accept
            
            # ═══════════════════════════════════════════════════════════════
            # BUY SIGNAL RULES (symmetric)
            # ═══════════════════════════════════════════════════════════════
            else:
                # PART 1: Highest close in 30 days?
                window_start = idx - pd.Timedelta(days=self.bad_signal_lookback_days)
                try:
                    closes_30d = h1_candles.loc[window_start:idx, 'close']
                    if abs(current_close - closes_30d.max()) <= tolerance_price:
                        accept = True
                except Exception:
                    pass
                
                if not accept:
                    # Find C2: nearest prior candle with high > C1 close
                    c2_time = None
                    prior = h1_candles[h1_candles.index < idx].sort_index(ascending=False)
                    for c2_idx, c2_row in prior.iterrows():
                        if c2_row['high'] >= current_close + tolerance_price:
                            c2_time = c2_idx
                            break
                    
                    if c2_time is None:
                        accept = True
                    else:
                        c1_close_time = idx + pd.Timedelta(hours=1)
                        if not bad_signals.empty:
                            valid_bad = bad_signals[
                                (bad_signals.index <= c2_time) &
                                (bad_signals['direction'] == 1) &
                                (bad_signals['known_at'] < c1_close_time)  # Non-look-ahead
                            ]
                            if not valid_bad.empty:
                                c3_time = valid_bad.index.max()
                                try:
                                    range_candles = h1_candles.loc[c3_time:idx]
                                    highest_high = range_candles['high'].max()
                                    if (highest_high - current_close) >= (12.0 * pip_size - tolerance_price):
                                        accept = True
                                except Exception:
                                    pass
                            else:
                                accept = True
                        else:
                            accept = True
            
            # Record result
            signal_record = {
                'timestamp': idx,
                'price': current_close,
                'direction': direction,
                'body_pips': body_pips,
                'candle_low': current_low,
                'candle_high': current_high,
                'was_accepted': accept
            }
            all_signals.append(signal_record)
            if accept:
                accepted.append(signal_record)
            else:
                rejected.append(signal_record)
        
        all_df = pd.DataFrame(all_signals).set_index('timestamp')
        accepted_df = pd.DataFrame(accepted).set_index('timestamp') if accepted else pd.DataFrame()
        rejected_df = pd.DataFrame(rejected).set_index('timestamp') if rejected else pd.DataFrame()
        
        return all_df, accepted_df, rejected_df
    
    def on_candle(self, event):
        """Not used in post-hoc analysis - stub for StrategyBase compliance."""
        raise NotImplementedError("This strategy is designed for post-hoc analysis only")
    
    def on_signal(self, signal):
        """Not used in post-hoc analysis - stub for StrategyBase compliance."""
        pass