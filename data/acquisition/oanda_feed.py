"""
Real-time OANDA WebSocket feed publishing CandleEvents to EventBus.
"""
import threading
import time
from datetime import datetime, timezone
from typing import List
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream
from trading_system.core.event_bus import EventBus
from trading_system.core.event_types import create_candle_event
import logging

logger = logging.getLogger(__name__)


class OandaRealTimeFeed:
    """
    Real-time candle feed publishing CandleEvents to EventBus.
    Maintains 1-minute candle state and emits completed candles.
    """
    
    def __init__(
        self,
        account_id: str,
        api_token: str,
        instruments: List[str],
        event_bus: EventBus,
        environment: str = "practice"
    ):
        self.account_id = account_id
        self.instruments = instruments
        self.event_bus = event_bus
        self.environment = environment
        
        # Initialize OANDA API
        self.api = API(
            access_token=api_token,
            environment="fxpractice" if environment == "practice" else "fxtrade"
        )
        
        # Candle state tracking: {instrument: {open, high, low, close, volume, start_time}}
        self.candle_state = {instr: None for instr in instruments}
        self.running = False
        self.thread = None
    
    def start(self):
        """Start real-time feed in background thread."""
        if self.running:
            logger.warning("Feed already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()
        logger.info(f"Started real-time feed for {self.instruments}")
    
    def stop(self):
        """Stop real-time feed."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Stopped real-time feed")
    
    def _run_stream(self):
        """Main stream processing loop."""
        params = {"instruments": ",".join(self.instruments)}
        r = PricingStream(accountID=self.account_id, params=params)
        
        try:
            for tick in self.api.request(r):
                if not self.running:
                    break
                
                if "type" in tick and tick["type"] == "PRICE":
                    self._process_tick(tick)
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            if self.running:
                # Auto-reconnect after delay
                time.sleep(5)
                self._run_stream()
    
    def _process_tick(self, tick: dict):
        """Process price tick and manage candle state."""
        instrument = tick["instrument"]
        time_str = tick["time"]
        bid = float(tick["bids"][0]["price"])
        ask = float(tick["asks"][0]["price"])
        mid = (bid + ask) / 2.0
        
        # Determine if minute boundary crossed
        tick_time = datetime.strptime(time_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        tick_time = tick_time.replace(tzinfo=timezone.utc)
        current_minute = tick_time.replace(second=0, microsecond=0)
        
        # Initialize new candle if needed
        state = self.candle_state[instrument]
        if state is None or state['start_time'] < current_minute:
            # Publish completed candle if exists
            if state is not None:
                self._publish_completed_candle(instrument, state)
            
            # Start new candle
            self.candle_state[instrument] = {
                'open': mid,
                'high': mid,
                'low': mid,
                'close': mid,
                'volume': 1,  # OANDA doesn't provide real-time volume - placeholder
                'start_time': current_minute
            }
        else:
            # Update existing candle
            state['high'] = max(state['high'], mid)
            state['low'] = min(state['low'], mid)
            state['close'] = mid
            state['volume'] += 1
    
    def _publish_completed_candle(self, instrument: str, state: dict):
        """Publish completed 1-minute candle as CandleEvent."""
        event = create_candle_event(
            symbol=instrument,
            timeframe="M1",
            open=state['open'],
            high=state['high'],
            low=state['low'],
            close=state['close'],
            volume=state['volume'],
            timestamp=state['start_time'],
            complete=True
        )
        self.event_bus.publish(event)
        logger.debug(f"Published M1 candle for {instrument} @ {state['start_time']}")


# Usage example:
# feed = OandaRealTimeFeed(
#     account_id="101-004-XXXXXX-001",
#     api_token="your_token",
#     instruments=["EUR_USD", "GBP_USD"],
#     event_bus=event_bus
# )
# feed.start()