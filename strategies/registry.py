"""
Strategy registry for dynamic instantiation from config.
"""
from typing import Dict, Type
from trading_system.strategies.base import StrategyBase
from trading_system.strategies.strong_candle_close_refined import StrongCandleCloseRefinedStrategy
import importlib

class StrategyRegistry:
    _strategies: Dict[str, Type[StrategyBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class."""
        def decorator(strategy_cls: Type[StrategyBase]):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> StrategyBase:
        """Create strategy instance by name."""
        if name not in cls._strategies:
            # Auto-discover from strategies/ directory
            try:
                importlib.import_module(f"strategies.{name.lower()}")
            except ImportError:
                raise ValueError(f"Unknown strategy: {name}. Registered: {list(cls._strategies.keys())}")
        
        if name not in cls._strategies:
            raise ValueError(f"Strategy '{name}' not registered after import attempt")
        
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def get_registered_names(cls) -> list:
        return list(cls._strategies.keys())


# Register built-in strategies
StrategyRegistry.register("STRONG_CANDLE_CLOSE_REFINED")(StrongCandleCloseRefinedStrategy)