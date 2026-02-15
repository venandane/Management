# strategies/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from trading_system.core.event_types import CandleEvent, SignalEvent
from trading_system.core.event_bus import EventBus


logger = logging.getLogger(__name__)

class StrategyBase(ABC):
    def __init__(self, symbol: str, timeframes: List[str], params: dict, event_bus: Optional[EventBus] = None):
        self.symbol = symbol
        self.timeframes = timeframes
        self.params = params or {}
        self.name = self.__class__.__name__
        self.event_bus = event_bus

        # Strategy state
        self.position_size = 0.0
        self.entry_price = 0.0        

        # Auto-subscribe to candle events if EventBus available
        if self.event_bus:
            self._auto_subscribe()

    def _auto_subscribe(self):
        """Automatically subscribe to required candle events."""
        for tf in self.timeframes:
            pattern = f"CANDLE.{tf}.{self.symbol}"
            self.event_bus.subscribe(pattern, self._on_candle_event)
            logger.info(f"[{self.name}] Subscribed to {pattern}")

    def _on_candle_event(self, event: CandleEvent):
        """Event bus callback - routes to strategy logic and publishes signals."""
        if event.symbol != self.symbol:
            return
        
        signal = self.on_candle(event)
        
        # Auto-publish signal to event bus if generated
        if signal and self.event_bus:
            self.event_bus.publish(signal)

    @abstractmethod
    def on_candle(self, event: CandleEvent) -> Optional[SignalEvent]:
        """Called when new candle arrives for subscribed timeframe"""
        pass
    
    @abstractmethod
    def on_signal(self, signal: SignalEvent):
        """Handle portfolio manager's execution confirmation"""
        pass
    
    def calculate_position_size(self, price: float, risk_pct: float = 1.0) -> int:
        """Standardized position sizing (reusable across strategies)"""
        account_value = self.portfolio.get_equity()
        risk_amount = account_value * (risk_pct / 100)
        # ATR-based stop distance or fixed %
        stop_distance = self._calculate_stop_distance()
        return int(risk_amount / (price * stop_distance))