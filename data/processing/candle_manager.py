import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timezone
import logging
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.core.event_bus import EventBus
from trading_system.core.event_types import CandleEvent, create_candle_event

logger = logging.getLogger(__name__)


class CandleManager:
    """
    Multi-timeframe candle manager with pickle loading capability.
    Resamples 1-minute base data to any timeframe required by strategies.
    """
    
    def __init__(self, data_root: Union[str, Path] = "", event_bus: Optional[EventBus] = None):
        self.data_root = Path(data_root)
        self.base_timeframe = "M1"  # OANDA 1-minute base data
        self._raw_data: Dict[str, pd.DataFrame] = {}  # {instrument: df}
        self._resampled: Dict[str, Dict[str, pd.DataFrame]] = {}  # {instrument: {timeframe: df}}
        self.event_bus = event_bus  # Optional integration point
    
    def load_from_pickle(self, instruments: List[str], granularity: str = "M1") -> Dict[str, pd.DataFrame]:
        """Load OANDA pickle files for specified instruments"""
        for instrument in instruments:
            filepath = self.data_root / instrument / granularity / f"{instrument}_{granularity}.pkl"
            
            if not filepath.exists():
                logger.warning(f"Pickle file not found: {filepath}")
                continue
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    df = data['df']
                    
                    # Validate structure
                    required_cols = {'open', 'high', 'low', 'close', 'volume'}
                    if not required_cols.issubset(df.columns):
                        raise ValueError(f"Missing required columns in {instrument} data")
                    
                    self._raw_data[instrument] = df
                    logger.info(f"Loaded {len(df):,} candles for {instrument} ({df.index.min()} → {df.index.max()})")
                    
            except Exception as e:
                logger.error(f"Failed to load {instrument}: {e}")
                continue
        
        return self._raw_data
    
    def resample_to_timeframes(
        self,
        instrument: str,
        target_timeframes: List[str],
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample 1-minute data to multiple timeframes.
        OANDA granularity → Pandas offset conversion:
          M1 → '1min', M5 → '5min', M15 → '15min', H1 → '1H', D → '1D'
        """
        if instrument not in self._raw_data:
            raise ValueError(f"No base data loaded for {instrument}")
        
        if instrument not in self._resampled:
            self._resampled[instrument] = {}
        
        base_df = self._raw_data[instrument]
        
        for tf in target_timeframes:
            # Skip if already resampled (unless forced)
            if tf in self._resampled[instrument] and not force_reload:
                continue
            
            # FIX: Base granularity (M1) gets copied, not resampled
            if tf == self.base_timeframe:
                self._resampled[instrument][tf] = base_df.copy()
                logger.info(f"Copied base {tf} data for {instrument}: {len(base_df):,} candles")
                continue

            # Convert OANDA granularity to pandas offset alias
            pd_freq = self._oanda_to_pandas_freq(tf)
            
            # OHLCV resampling with proper aggregation
            resampled = pd.DataFrame()
            resampled['open'] = base_df['open'].resample(pd_freq).first()
            resampled['high'] = base_df['high'].resample(pd_freq).max()
            resampled['low'] = base_df['low'].resample(pd_freq).min()
            resampled['close'] = base_df['close'].resample(pd_freq).last()
            resampled['volume'] = base_df['volume'].resample(pd_freq).sum()
            
            # Drop incomplete periods (e.g., partial first/last candle)
            resampled.dropna(inplace=True)
            
            self._resampled[instrument][tf] = resampled
            logger.info(f"Resampled {instrument} to {tf}: {len(resampled):,} candles")
        
        return self._resampled[instrument]
    
    def get_candles(
        self,
        instrument: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get candles for instrument/timeframe with optional date slicing"""
        if instrument not in self._resampled or timeframe not in self._resampled[instrument]:
            raise ValueError(f"No resampled data for {instrument} {timeframe}. Call resample_to_timeframes() first.")
        
        df = self._resampled[instrument][timeframe].copy()
        
        if start:
            df = df[df.index >= pd.Timestamp(start, tz=timezone.utc)]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz=timezone.utc)]
        
        return df
    
    @staticmethod
    def _oanda_to_pandas_freq(oanda_granularity: str) -> str:
        """Convert OANDA granularity codes to pandas frequency strings (pandas ≥ 2.2.0 compatible)"""
        # mapping = {
        #     'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
        #     'M10': '10min', 'M15': '15min', 'M30': '30min',
        #     'H1': '1H', 'H2': '2H', 'H3': '3H', 'H4': '4H', 'H6': '6H', 'H8': '8H', 'H12': '12H',
        #     'D': '1D', 'W': '1W', 'M': '1M'
        # }

        mapping = {
            # Minutes: use 'min' (NOT 'T')
            'M1': '1min', 'M2': '2min', 'M3': '3min', 'M4': '4min', 'M5': '5min',
            'M10': '10min', 'M15': '15min', 'M30': '30min',
            
            # Hours: MUST USE LOWERCASE 'h' (NOT 'H') ← CRITICAL FIX
            'H1': '1h', 'H2': '2h', 'H3': '3h', 'H4': '4h', 'H6': '6h', 'H8': '8h', 'H12': '12h',
            
            # Days/Weeks/Months: uppercase is still valid
            'D': '1D', 'W': '1W', 'M': '1M'
        }


        if oanda_granularity not in mapping:
            # Fallback: assume numeric minutes (e.g., "5" → "5min")
            try:
                minutes = int(oanda_granularity)
                return f"{minutes}min"
            except ValueError:
                raise ValueError(f"Unknown OANDA granularity: {oanda_granularity}")
        
        return mapping[oanda_granularity]
    
    def available_instruments(self) -> List[str]:
        return list(self._raw_data.keys())
    
    def available_timeframes(self, instrument: str) -> List[str]:
        return list(self._resampled.get(instrument, {}).keys())

    def publish_candles(self, instrument: str, timeframe: str, start: Optional[datetime] = None):
        """
        Publish all candles for instrument/timeframe as CandleEvents.
        Typically called after resampling to bootstrap event-driven system.
        """
        if self.event_bus is None:
            logger.warning("No EventBus configured - skipping candle publication")
            return
        
        try:
            candles = self.get_candles(instrument, timeframe, start=start)
        except ValueError as e:
            logger.error(f"Cannot publish candles: {e}")
            return
        
        logger.info(f"Publishing {len(candles)} {timeframe} candles for {instrument}...")
        published = 0
        
        for idx, row in candles.iterrows():
            # Only publish COMPLETE candles (skip partial current candle)
            event = create_candle_event(
                symbol=instrument,
                timeframe=timeframe,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row.get('volume', 0)),
                timestamp=idx,
                complete=True
            )
            self.event_bus.publish(event)
            published += 1
        
        logger.info(f"Published {published} {timeframe} candle events for {instrument}")