#!/usr/bin/env python3
"""
Simple breakout strategy with event simulation:
  1. Detect strong bull candles on higher timeframe (1H)
  2. Generate buy signals on 1-minute breakout above strong candle high
  3. Plot synchronized multi-timeframe view with signals
  
Uses event-driven pattern to demonstrate integration with core/event_types.py
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timezone
from trading_system.data.processing.candle_manager import CandleManager
from trading_system.data.processing.indicators import StrongCandleDetector
from trading_system.data.processing.resampler import ConfigDrivenResampler
from trading_system.visualization.candle_plotter import CandlePlotter
from trading_system.strategies.registry import StrategyRegistry
from trading_system.core.event_types import CandleEvent, create_candle_event
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.utils.project_paths import DATA_DIR
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_name: str = "strategies.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def simulate_events_and_generate_signals(
    higher_tf_candles: pd.DataFrame,
    lower_tf_candles: pd.DataFrame,
    instrument: str,
    strategy_params: dict
) -> pd.DataFrame:
    """
    Simulate event-driven strategy execution:
      1. Create CandleEvents for higher timeframe
      2. Strategy detects strong candles and stores breakout levels
      3. Create CandleEvents for lower timeframe
      4. Strategy generates signals on breakout
    
    Returns:
        DataFrame of generated signals with columns ['price', 'breakout_level', 'strong_candle_time']
    """
    # Initialize strategy
    strategy = StrategyRegistry.create(
        name="BREAKOUT_ON_STRONG_CANDLE",
        symbol=instrument,
        timeframes=["M1", "H1"],
        params=strategy_params
    )
    
    # Step 1: Process higher timeframe candles to detect strong bulls
    logger.info("Processing higher timeframe candles for strong bull detection...")
    for idx, row in higher_tf_candles.iterrows():
        event = create_candle_event(
            symbol=instrument,
            timeframe="H1",
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=int(row.get('volume', 0)),
            timestamp=idx
        )
        strategy.on_candle(event)  # Stores breakout levels internally
    
    # Step 2: Process lower timeframe candles to generate breakout signals
    logger.info("Processing lower timeframe candles for breakout signals...")
    signals = []
    
    for idx, row in lower_tf_candles.iterrows():
        event = create_candle_event(
            symbol=instrument,
            timeframe="M1",
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=int(row.get('volume', 0)),
            timestamp=idx
        )
        signal = strategy.on_candle(event)
        
        if signal:
            # Extract breakout level from signal meta
            breakout_level = signal.meta.get('breakout_level', 0.0)
            strong_candle_time = signal.meta.get('strong_candle_time', '')
            
            signals.append({
                'timestamp': signal.timestamp,
                'price': signal.price,
                'breakout_level': breakout_level,
                'strong_candle_time': pd.Timestamp(strong_candle_time) if strong_candle_time else pd.NaT,
                'body_pips': signal.meta.get('body_pips', None)
            })
    
    if not signals:
        logger.info("No breakout signals generated")
        return pd.DataFrame(columns=['price', 'breakout_level', 'strong_candle_time'])
    
    signals_df = pd.DataFrame(signals).set_index('timestamp')
    logger.info(f"Generated {len(signals_df)} breakout signals")
    return signals_df


def main():
    # Load config
    config = load_config()
    
    # Strategy parameters
    instrument = "EUR_USD"
    strategy_params = {
        'higher_timeframe': 'H1',
        'lower_timeframe': 'M1',
        'min_body_pips': 5.0,
        'min_range_pips': 3.0,
        'buffer_pips': 0.5
    }
    
    # Initialize candle manager
    data_root_path = DATA_DIR / "acquisition" / "data" / "raw" / "oanda"
    candle_mgr = CandleManager(data_root_path)
    
    # Step 1: Define required timeframes (M1 is base granularity but MUST be included)
    instruments = [instrument]
    timeframes_by_instrument = {
        instrument: {"M1", "H1"}  # MUST include M1 even though it's base granularity
    }

    # Step 2: Resample properly (handles base M1 loading + H1 resampling)
    logger.info(f"Loading base M1 data and resampling to required timeframes for {instrument}...")
    try:
        ConfigDrivenResampler.resample_all_required(
            candle_mgr=candle_mgr,
            instruments=instruments,
            timeframes_by_instrument=timeframes_by_instrument,
            base_granularity="M1"
        )
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return

    # Step 3: Get candles (now safely available in _resampled dict)
    try:
        higher_tf_candles = candle_mgr.get_candles(instrument, "H1")
        lower_tf_candles = candle_mgr.get_candles(instrument, "M1")  # ✅ Now works!
    except ValueError as e:
        logger.error(f"Failed to retrieve candles: {e}")
        return

    # Add timeframe metadata for plotter
    higher_tf_candles.attrs['timeframe'] = "H1"
    lower_tf_candles.attrs['timeframe'] = "M1"

    # Detect strong candles (for visualization)
    logger.info("Detecting strong candles for visualization...")
    detector = StrongCandleDetector(min_body_pips=5.0, min_range_pips=3.0)
    strong_candles = detector.detect(higher_tf_candles, instrument)

    # Generate signals via event simulation
    buy_signals = simulate_events_and_generate_signals(
        higher_tf_candles=higher_tf_candles,
        lower_tf_candles=lower_tf_candles,
        instrument=instrument,
        strategy_params=strategy_params
    )


    ##########      DEBUG
    # Add AFTER buy_signals = simulate_events_and_generate_signals(...)
    print("\n" + "="*80)
    print("DEBUG: Signal Analysis for Feb 8, 2026")
    print("="*80)

    # Filter signals around the event time (convert EST to UTC for comparison)
    target_utc_start = pd.Timestamp("2026-02-08 23:50:00", tz="UTC")  # 18:50 EST
    target_utc_end = pd.Timestamp("2026-02-09 00:10:00", tz="UTC")   # 19:10 EST

    signals_window = buy_signals[
        (buy_signals.index >= target_utc_start) & 
        (buy_signals.index <= target_utc_end)
    ].copy()

    if not signals_window.empty:
        signals_window['est_time'] = signals_window.index.tz_convert('US/Eastern')
        print("\nSignals in window (UTC → EST):")
        for idx, row in signals_window.iterrows():
            est = row['est_time']
            print(f"  {idx} UTC → {est.strftime('%H:%M:%S')} EST | "
                f"Price: {row['price']:.5f} | "
                f"Breakout: {row['breakout_level']:.5f} | "
                f"Strong Candle: {row['strong_candle_time']} UTC")
        
        # Show the strong candles that generated these signals
        print("\nSource Strong Candles (H1):")
        strong_bulls = strong_candles[
            (strong_candles.index >= pd.Timestamp("2026-02-08 22:00:00", tz="UTC")) &
            (strong_candles.index <= pd.Timestamp("2026-02-09 01:00:00", tz="UTC")) &
            (strong_candles['strong_bull'])
        ]
        
        for idx, row in strong_bulls.iterrows():
            est_close = idx.tz_convert('US/Eastern')
            print(f"  {idx} UTC → {est_close.strftime('%H:%M')} EST | "
                f"Body: {row['body_pips']:.1f} pips | "
                f"High: {row['high']:.5f} | "
                f"Tail Ratio: {row['upper_tail_ratio']:.1%}")
    else:
        print("No signals found in target window - check timezone conversion")
    print("="*80)
    ##########      DEBUG


    days_to_plot = 15  # ← CONFIGURABLE: Change to 30 for month, 90 for quarter
    cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_to_plot)

    # Filter candles to recent period only
    higher_tf_candles = higher_tf_candles[higher_tf_candles.index >= cutoff_time]
    lower_tf_candles = lower_tf_candles[lower_tf_candles.index >= cutoff_time]
    buy_signals = buy_signals[buy_signals.index >= cutoff_time] if not buy_signals.empty else buy_signals

    # Plot
    plotter = CandlePlotter(theme="dark", timezone="US/Eastern")
    fig = plotter.plot_strategy_breakout(
        instrument=instrument,
        higher_tf_candles=higher_tf_candles,
        lower_tf_candles=lower_tf_candles,
        strong_candles=strong_candles[strong_candles.index >= cutoff_time], #strong_candles,
        buy_signals=buy_signals,
        title_suffix=f"(Last {days_to_plot} Days | H1 Strong Candles → M1 Breakouts)",
        height=1900
    )
    
    # Save plot
    data_output_path = DATA_DIR / "output" / "strategy_breakout"
    output_dir = Path(data_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{instrument}_breakout_strategy.html"
    plotter.save_html(fig, str(filepath))

    # Summary statistics
    total_strong_bulls = strong_candles['strong_bull'].sum()
    total_signals = len(buy_signals)
    logger.info(f"\nStrategy Summary for {instrument}:")
    logger.info(f"  • Strong bull candles detected (H1): {total_strong_bulls:,}")
    logger.info(f"  • Breakout buy signals generated (M1): {total_signals:,}")
    if total_strong_bulls > 0:
        conversion_rate = (total_signals / total_strong_bulls) * 100
        logger.info(f"  • Signal conversion rate: {conversion_rate:.1f}%")

    logger.info(f"\nStrategy plot saved to: {filepath.absolute()}")
    logger.info("\nPlot legend:")
    logger.info("Green arrow (top panel) = Strong bull candle on H1")
    logger.info("Blue triangle (middle panel) = BUY signal on M1 breakout")
    logger.info("Gold dotted line (top panel) = Breakout level (high + buffer)")


if __name__ == "__main__":
    main()