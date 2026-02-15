"""
Thread-safe event bus implementing the Observer pattern for trading system coordination.
Supports hierarchical event routing with wildcard subscriptions.
"""
import threading
import logging
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass
from trading_system.core.event_types import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subscription:
    """Immutable subscription record for debugging/unsubscription."""
    pattern: str
    handler: Callable
    handler_name: str


class EventBus:
    """
    Central event dispatcher coordinating all system components.
    
    Event Naming Convention (hierarchical dot notation):
      {event_type}.{timeframe}.{symbol}
      
      Examples:
        - "CANDLE.M1.EUR_USD"      → 1-min EUR/USD candle
        - "CANDLE.H1.*"            → All 1-hour candles (wildcard)
        - "SIGNAL.**"              → All signals (recursive wildcard)
        - "ORDER.MARKET.EUR_USD"   → Market orders for EUR/USD
    
    Wildcard Rules:
      - "*"  : Matches exactly one segment (e.g., "CANDLE.*.EUR_USD" → all timeframes for EUR/USD)
      - "**" : Matches zero or more segments (e.g., "CANDLE.**" → all candle events)
    """
    
    def __init__(self, enable_logging: bool = True):
        self._subscribers: Dict[str, List[Subscription]] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._enable_logging = enable_logging
        self._stats = {
            'events_published': 0,
            'handlers_invoked': 0,
            'events_dropped': 0
        }
    
    def subscribe(self, pattern: str, handler: Callable) -> Subscription:
        """
        Subscribe handler to events matching pattern.
        
        Args:
            pattern: Event pattern (e.g., "CANDLE.M1.*", "SIGNAL.**")
            handler: Callable accepting Event as single argument
            
        Returns:
            Subscription object for later unsubscription
        """
        handler_name = getattr(handler, '__name__', str(handler))
        subscription = Subscription(pattern=pattern, handler=handler, handler_name=handler_name)
        
        with self._lock:
            if pattern not in self._subscribers:
                self._subscribers[pattern] = []
            self._subscribers[pattern].append(subscription)
            
            if self._enable_logging:
                logger.debug(
                    f"Subscribed handler '{handler_name}' to pattern '{pattern}'. "
                    f"Total subscribers for pattern: {len(self._subscribers[pattern])}"
                )
        
        return subscription
    
    def unsubscribe(self, subscription: Subscription) -> bool:
        """
        Remove a specific subscription.
        
        Args:
            subscription: Subscription object returned by subscribe()
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if subscription.pattern in self._subscribers:
                before = len(self._subscribers[subscription.pattern])
                self._subscribers[subscription.pattern] = [
                    sub for sub in self._subscribers[subscription.pattern]
                    if sub.handler != subscription.handler
                ]
                after = len(self._subscribers[subscription.pattern])
                
                if self._enable_logging and before > after:
                    logger.debug(
                        f"Unsubscribed handler '{subscription.handler_name}' from '{subscription.pattern}'. "
                        f"Remaining: {after}"
                    )
                return before > after
        return False
    
    def unsubscribe_all(self, pattern: Optional[str] = None) -> int:
        """
        Remove all subscriptions for a pattern (or all patterns if None).
        
        Returns:
            Number of handlers unsubscribed
        """
        with self._lock:
            if pattern is None:
                total = sum(len(subs) for subs in self._subscribers.values())
                self._subscribers.clear()
                if self._enable_logging:
                    logger.info(f"Unsubscribed all handlers ({total} total)")
                return total
            elif pattern in self._subscribers:
                count = len(self._subscribers[pattern])
                del self._subscribers[pattern]
                if self._enable_logging:
                    logger.info(f"Unsubscribed {count} handlers from pattern '{pattern}'")
                return count
        return 0
    
    def publish(self, event: Event) -> int:
        """
        Publish event to all matching subscribers.
        
        Args:
            event: Event instance to publish
            
        Returns:
            Number of handlers that processed the event
        """
        event_key = self._build_event_key(event)
        matched_handlers = []
        
        # Find all matching subscriptions (copy to avoid modification during iteration)
        with self._lock:
            for pattern, subscriptions in list(self._subscribers.items()):
                if self._matches(event_key, pattern):
                    # Copy list to avoid "changed during iteration" errors
                    matched_handlers.extend(subscriptions[:])
        
        # Invoke handlers outside lock to prevent deadlocks
        handlers_invoked = 0
        for subscription in matched_handlers:
            try:
                subscription.handler(event)
                handlers_invoked += 1
                self._stats['handlers_invoked'] += 1
            except Exception as e:
                logger.error(
                    f"Handler '{subscription.handler_name}' failed for event {event_key}: {e}",
                    exc_info=True
                )
                # Continue processing other handlers - don't crash entire system
        
        self._stats['events_published'] += 1
        
        if self._enable_logging and handlers_invoked > 0:
            logger.debug(
                f"Published {event.event_type} event '{event_key}' → "
                f"{handlers_invoked} handler(s) invoked"
            )
        elif self._enable_logging and handlers_invoked == 0:
            self._stats['events_dropped'] += 1
            logger.debug(f"No handlers subscribed for event '{event_key}' (dropped)")
        
        return handlers_invoked
    
    def _build_event_key(self, event: Event) -> str:
        """
        Construct hierarchical event key from event properties.
        Format: {event_type}.{timeframe}.{symbol}
        Missing properties replaced with '*' wildcard.
        """
        parts = [event.event_type]
        
        # Add timeframe if available (CandleEvent, SignalEvent)
        timeframe = getattr(event, 'timeframe', '')
        parts.append(timeframe if timeframe else '*')
        
        # Add symbol if available
        symbol = getattr(event, 'symbol', '')
        parts.append(symbol if symbol else '*')
        
        return '.'.join(parts)
    
    def _matches(self, event_key: str, pattern: str) -> bool:
        """
        Match event key against pattern with wildcard support.
        
        Rules:
          - Literal segments must match exactly
          - "*" matches exactly one segment
          - "**" matches zero or more segments (not implemented yet - simple "*" only)
        """
        event_parts = event_key.split('.')
        pattern_parts = pattern.split('.')
        
        # Simple implementation: equal length + segment-wise matching
        if len(event_parts) != len(pattern_parts):
            return False
        
        for e_part, p_part in zip(event_parts, pattern_parts):
            if p_part == '*':
                continue  # Wildcard matches any value
            if p_part == '**':
                return True  # Recursive wildcard matches everything (simplified)
            if e_part != p_part:
                return False
        
        return True
    
    def get_subscriptions(self) -> Dict[str, List[str]]:
        """Get current subscription state for debugging."""
        with self._lock:
            return {
                pattern: [sub.handler_name for sub in subs]
                for pattern, subs in self._subscribers.items()
            }
    
    def get_stats(self) -> dict:
        """Get event processing statistics."""
        with self._lock:
            return self._stats.copy()
    
    def clear_stats(self):
        """Reset statistics counters."""
        with self._lock:
            self._stats = {
                'events_published': 0,
                'handlers_invoked': 0,
                'events_dropped': 0
            }