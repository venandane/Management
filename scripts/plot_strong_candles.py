#!/usr/bin/env python3
"""
Plot candles with strong candle markers overlaid.
Demonstrates full pipeline: config → data loading → detection → visualization.
"""

import yaml
import pandas as pd
from pathlib import Path
from trading_system.data.processing.indicators import StrongCandleDetector
from trading_system.data.processing.candle_manager import CandleManager
from trading_system.visualization.candle_plotter import CandlePlotter
from trading_system.data.processing.resampler import ConfigDrivenResampler
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.utils.project_paths import DATA_DIR

def load_strategy_config(config_name: str = "strategies.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_strategy_config()
    
    # Extract instruments and timeframes from strong candle strategies
    instruments = set()
    timeframes_by_instrument = {}
    
    for strat in config['strategies']:
        if strat['name'] == "STRONG_CANDLE":
            for symbol in strat['symbols']:
                instruments.add(symbol)
                if symbol not in timeframes_by_instrument:
                    timeframes_by_instrument[symbol] = set()
                for tf in strat['timeframes']:
                    timeframes_by_instrument[symbol].add(tf)
    
    if not instruments:
        print("No STRONG_CANDLE strategies found in config")
        return
    
    print(f"Preparing data for strong candle analysis: {sorted(instruments)}")
    
    # 2. Initialize candle manager and load data
    data_root_path = DATA_DIR / "acquisition" / "data" / "raw" / "oanda"
    candle_mgr = CandleManager(data_root_path)
    candle_mgr.load_from_pickle(list(instruments), granularity="M1")
    
    try:
        ConfigDrivenResampler.resample_all_required(
            candle_mgr=candle_mgr,
            instruments=list(instruments),
            timeframes_by_instrument=timeframes_by_instrument,
            base_granularity="M1"
        )
    except Exception as e:
        print(f"Resampling failed: {e}")
        return

    # Initialize detector with config params
    strat_params = next(s for s in config['strategies'] if s['name'] == "STRONG_CANDLE")['params']
    detector = StrongCandleDetector(
        min_body_pips=strat_params.get('min_body_pips', 5.0),
        ##max_tail_ratio=strat_params.get('max_tail_ratio', 0.20),
        min_range_pips=strat_params.get('min_range_pips', 3.0)
    )
    
    # Setup plotter
    plotter = CandlePlotter(theme=config.get('plotting', {}).get('theme', 'light'), timezone="US/Eastern")
    data_output_path = DATA_DIR / "output" / "plots" / "strong_candles"
    output_dir = Path(data_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each instrument
    for instrument in sorted(instruments):
        timeframes = sorted(timeframes_by_instrument[instrument])
        if not timeframes:
            continue
        
        # Use smallest timeframe for detailed view
        base_tf = min(timeframes, key=lambda x: _parse_timeframe_duration(x))
        candles = candle_mgr.get_candles(instrument, base_tf)
        
        # Detect strong candles
        analyzed = detector.detect(candles, instrument)
        bull_count = analyzed['strong_bull'].sum()
        bear_count = analyzed['strong_bear'].sum()
        total = len(analyzed)
        
        print(
            f"{instrument} ({base_tf}): "
            f"{bull_count} bull / {bear_count} bear strong candles "
            f"({(bull_count + bear_count) / total * 100:.1f}% of all candles)"
        )

        # Create plot with markers
        fig = plotter.plot_single_timeframe(
            instrument=instrument,
            candles=analyzed,
            indicators={},
            signals=None,  # Strong candles handled internally by plotter
            title_suffix=f"Strong Candle Analysis ({base_tf})",
            show_volume=False
        )
        
        # Save
        filepath = output_dir / f"{instrument}_{base_tf}_strong_candles.html"
        plotter.save_html(fig, str(filepath))
        print(f"  → Saved to {filepath.name}")
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")


def _parse_timeframe_duration(tf: str) -> int:
    """Convert OANDA timeframe to minutes for sorting"""
    mapping = {
        'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5, 'M10': 10, 'M15': 15,
        'M30': 30, 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240, 'H6': 360,
        'H8': 480, 'H12': 720, 'D': 1440
    }
    return mapping.get(tf, 9999)


if __name__ == "__main__":
    main()