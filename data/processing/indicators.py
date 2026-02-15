"""
Technical indicators and candle pattern detectors.
"""
import pandas as pd
import numpy as np
from typing import Dict
from trading_system.core.utils import calculate_pip_distance


class StrongCandleDetector:
    """
    Detects strong candles with DYNAMIC tail tolerance based on body size:
    
    Body Size  | Max Tail Ratio | Rationale
    -----------|----------------|----------------------------------------
    5 pips     | 20%            | Minimal momentum → needs clean structure
    6 pips     | 30%            | Slightly more momentum → tolerates small rejection
    7 pips     | 40%            | Moderate momentum → accepts moderate wick
    8 pips     | 50%            | Strong momentum → tolerates significant wick
    9 pips     | 60%            | Very strong momentum → accepts large wick
    ≥10 pips   | UNLIMITED      | Overwhelming momentum → tail irrelevant
    
    Why this works: Larger bodies represent stronger directional conviction,
    making minor-to-moderate wick rejections less significant to trade outcome.
    """
    
    # Explicit mapping for clarity and auditability
    BODY_TO_TAIL_RATIO: Dict[int, float] = {
        5: 0.20,
        6: 0.30,
        7: 0.40,
        8: 0.50,
        9: 0.60
    }
    OVERRIDE_BODY_PIPS: float = 10.0  # Bodies ≥ this ignore tail completely
    
    def __init__(
        self,
        min_body_pips: float = 5.0,
        min_range_pips: float = 3.0
    ):
        self.min_body_pips = min_body_pips
        self.min_range_pips = min_range_pips
    
    def detect(
        self,
        df: pd.DataFrame,
        instrument: str
    ) -> pd.DataFrame:
        """
        Detect strong candles with dynamic tail tolerance.
        
        Returns DataFrame with columns:
          - strong_bull: bool (meets dynamic criteria)
          - strong_bear: bool (meets dynamic criteria)
          - body_pips: float
          - upper_tail_ratio: float
          - lower_tail_ratio: float
          - range_pips: float
          - max_allowed_tail_ratio: float (dynamic threshold applied)
          - strong_via_override: bool (body ≥10 pips)
        """
        df = df.copy()
        
        # Calculate raw components
        df['body'] = (df['close'] - df['open']).abs()
        df['upper_tail'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_tail'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Convert to pips
        df['body_pips'] = df.apply(
            lambda row: calculate_pip_distance(row['open'], row['close'], instrument),
            axis=1
        )
        df['range_pips'] = df.apply(
            lambda row: calculate_pip_distance(row['low'], row['high'], instrument),
            axis=1
        )
        
        # Calculate tail ratios (avoid division by zero)
        df['upper_tail_ratio'] = np.where(
            df['total_range'] > 0,
            df['upper_tail'] / df['total_range'],
            0.0
        )
        df['lower_tail_ratio'] = np.where(
            df['total_range'] > 0,
            df['lower_tail'] / df['total_range'],
            0.0
        )
        
        # Determine dynamic max tail ratio per candle based on body size
        df['max_allowed_tail_ratio'] = df['body_pips'].apply(
            lambda body: self._get_max_tail_ratio(body)
        )
        
        # Bullish candle conditions
        standard_bull = (
            (df['close'] > df['open']) &  # Bullish body
            (df['body_pips'] >= self.min_body_pips) &
            (df['body_pips'] < self.OVERRIDE_BODY_PIPS) &  # Not override tier
            (df['upper_tail_ratio'] <= df['max_allowed_tail_ratio']) &
            (df['range_pips'] >= self.min_range_pips)
        )
        
        override_bull = (
            (df['close'] > df['open']) &
            (df['body_pips'] >= self.OVERRIDE_BODY_PIPS) &  # Override threshold
            (df['range_pips'] >= self.min_range_pips)
        )
        
        # Bearish candle conditions
        standard_bear = (
            (df['close'] < df['open']) &  # Bearish body
            (df['body_pips'] >= self.min_body_pips) &
            (df['body_pips'] < self.OVERRIDE_BODY_PIPS) &
            (df['lower_tail_ratio'] <= df['max_allowed_tail_ratio']) &
            (df['range_pips'] >= self.min_range_pips)
        )
        
        override_bear = (
            (df['close'] < df['open']) &
            (df['body_pips'] >= self.OVERRIDE_BODY_PIPS) &
            (df['range_pips'] >= self.min_range_pips)
        )
        
        # Final signals (standard OR override)
        df['strong_bull'] = standard_bull | override_bull
        df['strong_bear'] = standard_bear | override_bear
        df['strong_via_override'] = (override_bull | override_bear)
        
        # Cleanup temporary columns
        df.drop(columns=['body', 'upper_tail', 'lower_tail', 'total_range'], inplace=True, errors='ignore')
        
        return df
    
    def _get_max_tail_ratio(self, body_pips: float) -> float:
        """
        Dynamic tail tolerance mapping:
          5 pips → 20%
          6 pips → 30%
          7 pips → 40%
          8 pips → 50%
          9 pips → 60%
          ≥10 pips → 1.0 (handled separately as override)
        """
        # Floor body size to integer for lookup (5.7 pips → 5)
        body_int = int(np.floor(body_pips))
        
        # Bodies <5 pips get 0% tolerance (will fail min_body_pips check anyway)
        if body_int < 5:
            return 0.0
        
        # Bodies 5-9 use dynamic mapping
        if 5 <= body_int <= 9:
            return self.BODY_TO_TAIL_RATIO.get(body_int, 0.0)
        
        # Bodies ≥10 use override logic (tail ignored completely)
        return 1.0  # Unlimited (but override path bypasses this check)
    
    def get_signal_series(self, df: pd.DataFrame) -> pd.Series:
        """Get single series with values: 1 (bull), -1 (bear), 0 (neutral)"""
        signal = pd.Series(0, index=df.index)
        signal[df['strong_bull']] = 1
        signal[df['strong_bear']] = -1
        return signal
    
    @classmethod
    def get_tolerance_table(cls) -> pd.DataFrame:
        """Return human-readable tolerance table for documentation/UI"""
        return pd.DataFrame([
            {"Body Size (pips)": "5", "Max Tail Ratio": "20%", "Description": "Minimal momentum → needs clean structure"},
            {"Body Size (pips)": "6", "Max Tail Ratio": "30%", "Description": "Slightly more momentum → small rejection OK"},
            {"Body Size (pips)": "7", "Max Tail Ratio": "40%", "Description": "Moderate momentum → moderate wick acceptable"},
            {"Body Size (pips)": "8", "Max Tail Ratio": "50%", "Description": "Strong momentum → significant wick tolerated"},
            {"Body Size (pips)": "9", "Max Tail Ratio": "60%", "Description": "Very strong momentum → large wick acceptable"},
            {"Body Size (pips)": "≥10", "Max Tail Ratio": "UNLIMITED", "Description": "Overwhelming momentum → tail irrelevant"}
        ])