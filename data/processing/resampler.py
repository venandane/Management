"""
Reusable resampling helper for config-driven timeframe preparation.
"""
from typing import Dict, List, Set
from trading_system.data.processing.candle_manager import CandleManager
import logging

logger = logging.getLogger(__name__)


class ConfigDrivenResampler:
    """
    Handles resampling of base M1 data to strategy-required timeframes.
    Eliminates duplication between plotting/backtesting scripts.
    """
    
    @staticmethod
    def extract_instruments_and_timeframes(config: dict) -> tuple[Set[str], Dict[str, Set[str]]]:
        """
        Extract unique instruments and their required timeframes from strategy config.
        
        Returns:
            Tuple of (instruments_set, {instrument: {timeframe1, timeframe2, ...}})
        """
        instruments = set()
        timeframes_by_instrument: Dict[str, Set[str]] = {}
        
        for strat in config.get('strategies', []):
            for symbol in strat.get('symbols', []):
                instruments.add(symbol)
                if symbol not in timeframes_by_instrument:
                    timeframes_by_instrument[symbol] = set()
                for tf in strat.get('timeframes', []):
                    timeframes_by_instrument[symbol].add(tf)
        
        return instruments, timeframes_by_instrument
    
    @staticmethod
    def resample_all_required(
        candle_mgr: CandleManager,
        instruments: List[str],
        timeframes_by_instrument: Dict[str, Set[str]],
        base_granularity: str = "M1"
    ) -> None:
        """
        Load base data and resample to all required timeframes.
        
        Args:
            candle_mgr: Initialized CandleManager
            instruments: List of instruments to process
            timeframes_by_instrument: Mapping of instrument → required timeframes
            base_granularity: Base granularity to load (typically "M1")
        """
        # Step 1: Load base M1 data for all instruments
        logger.info(f"Loading base {base_granularity} data for {len(instruments)} instruments...")
        loaded = candle_mgr.load_from_pickle(list(instruments), granularity=base_granularity)
        
        if not loaded:
            raise RuntimeError("No base data loaded. Check pickle files exist in data/raw/oanda/")
        
        # Step 2: Resample each instrument to its required timeframes
        logger.info("Resampling to strategy-required timeframes...")
        for instrument in sorted(instruments):
            if instrument not in candle_mgr.available_instruments():
                logger.warning(f"Skipping {instrument}: no base data available")
                continue
            
            required_tfs = timeframes_by_instrument.get(instrument, set())
            if not required_tfs:
                logger.warning(f"Skipping {instrument}: no timeframes configured")
                continue
            
            # This populates _resampled[instrument][base_granularity] for get_candles() to work
            if base_granularity not in required_tfs:
                required_tfs = set(required_tfs)  # Copy to avoid mutating original
                required_tfs.add(base_granularity)
                logger.debug(f"Added base granularity '{base_granularity}' to resampling list for {instrument}")
            
            # Resample to each required timeframe
            candle_mgr.resample_to_timeframes(instrument, list(required_tfs))
            
            available = candle_mgr.available_timeframes(instrument)
            logger.info(
                f"{instrument}: base={base_granularity}, "
                f"resampled={sorted(available)}, "
                f"requested={sorted(required_tfs)}"
            )