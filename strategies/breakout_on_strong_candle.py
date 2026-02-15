"""
Simple breakout strategy: Buy when price breaks above strong bull candle high
on higher timeframe (e.g., 1H), using 1-minute data for precise entry timing.
"""
from abc import ABC
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from trading_system.strategies.base import StrategyBase
from trading_system.core.event_types import CandleEvent, SignalEvent, create_signal_event
from trading_system.data.processing.indicators import StrongCandleDetector
from trading_system.core.utils import get_pip_size
import logging

logger = logging.getLogger(__name__)

class BreakoutOnStrongCandleStrategy(StrategyBase):
    """
    Strategy Logic:
      1. Monitor higher timeframe (e.g., 1H) for strong bull candles
      2. When detected, record the candle's HIGH price as breakout level
      3. On 1-minute chart, generate BUY signal when price > breakout level
         (after the higher timeframe candle has closed)
    
    Key Design:
      - Higher timeframe = signal generator (1H strong candles)
      - Lower timeframe = execution vehicle (1-min breakout entries)
      - No repainting: Signals only generated AFTER higher TF candle closes
    """
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        params: Optional[dict] = None
    ):
        super().__init__(symbol, timeframes, params)
        
        # Strategy parameters
        self.higher_timeframe = params.get('higher_timeframe', 'H1')  # Signal generator
        self.lower_timeframe = params.get('lower_timeframe', 'M1')    # Execution vehicle
        self.min_body_pips = params.get('min_body_pips', 5.0)
        self.min_range_pips = params.get('min_range_pips', 3.0)
        self.buffer_pips = params.get('buffer_pips', 0.5)  # Extra buffer above high to avoid false breaks
        
        # Initialize detector for higher timeframe
        self.detector = StrongCandleDetector(
            min_body_pips=self.min_body_pips,
            min_range_pips=self.min_range_pips
        )
        
        # State tracking: {higher_tf_candle_close_time: breakout_price}
        self.active_breakout_levels: Dict[pd.Timestamp, float] = {}
        
        # Track which timeframes we're subscribed to
        if self.higher_timeframe not in timeframes:
            raise ValueError(
                f"Higher timeframe '{self.higher_timeframe}' not in subscribed timeframes {timeframes}"
            )
        if self.lower_timeframe not in timeframes:
            raise ValueError(
                f"Lower timeframe '{self.lower_timeframe}' not in subscribed timeframes {timeframes}"
            )
    
    def on_candle(self, event: CandleEvent) -> Optional[SignalEvent]:
        """
        REQUIRED ABSTRACT METHOD IMPLEMENTATION
        Main event handler - routes candles to appropriate processing based on timeframe.
        
        Returns:
            SignalEvent if breakout detected on lower timeframe, else None
        """
        # Validate event belongs to this strategy's symbol
        if event.symbol != self.symbol:
            return None
        
        # Route to appropriate handler based on timeframe
        if event.timeframe == self.higher_timeframe:
            self._process_higher_timeframe(event)
            return None  # Higher TF never generates signals directly
        
        elif event.timeframe == self.lower_timeframe:
            return self._process_lower_timeframe(event)
        
        else:
            # Ignore other timeframes
            return None
    
    def _process_higher_timeframe(self, event: CandleEvent) -> None:
        """
        Process higher timeframe candles to detect strong bulls.
        Stores breakout levels for lower timeframe monitoring.
        """
        # Convert to single-row DataFrame for detector
        candle_df = pd.DataFrame([{
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'volume': event.volume
        }], index=[event.timestamp])
        
        # Detect strong candle
        analyzed = self.detector.detect(candle_df, self.symbol)
        
        if analyzed['strong_bull'].iloc[0]:
            # Calculate breakout level = high + buffer (in price units)
            pip_size = get_pip_size(self.symbol)
            breakout_price = event.high + (self.buffer_pips * pip_size)
            
            # Store for lower timeframe monitoring
            self.active_breakout_levels[event.timestamp] = breakout_price
            
            logger.debug(
                f"[{self.name}] Strong bull on {event.timeframe} at {event.timestamp}: "
                f"Body={analyzed['body_pips'].iloc[0]:.1f}pips, "
                f"Breakout level={breakout_price:.5f}"
            )
    
    def _process_lower_timeframe(self, event: CandleEvent) -> Optional[SignalEvent]:
        """
        Monitor 1-minute candles for breakout above strong candle highs.
        Generates BUY signal on first close above breakout level.
        """
        # Only process if we have active breakout levels
        if not self.active_breakout_levels:
            return None
        
        # Check all active breakout levels
        for candle_time, breakout_price in list(self.active_breakout_levels.items()):
            # Only trigger AFTER higher TF candle has closed
            if event.timestamp <= candle_time:
                continue
            
            # Breakout condition: close > breakout level
            if event.close > breakout_price:
                # Generate signal
                signal = create_signal_event(
                    symbol=self.symbol,
                    direction=1,  # Long/buy
                    price=event.close,
                    strategy_name=self.name,
                    timeframe=event.timeframe,
                    meta={
                        'breakout_level': breakout_price,
                        'strong_candle_time': candle_time.isoformat(),
                        'higher_timeframe': self.higher_timeframe,
                        'buffer_pips': self.buffer_pips
                    },
                    timestamp=event.timestamp
                )
                
                # Remove level (one-time trigger per strong candle)
                del self.active_breakout_levels[candle_time]
                
                logger.info(
                    f"[{self.name}] BUY signal @ {event.timestamp}: "
                    f"Price={event.close:.5f} > Breakout={breakout_price:.5f} "
                    f"(from strong candle at {candle_time})"
                )
                
                return signal  # Return FIRST signal only (per candle)
        
        return None
    
    def on_signal(self, signal: SignalEvent):
        """Handle execution confirmation (placeholder for position management)"""
        if signal.direction == 1:
            self.position_size += 1  # Simplified position tracking
            self.entry_price = signal.price
            logger.debug(f"[{self.name}] Position opened: {self.position_size} units @ {self.entry_price:.5f}")
