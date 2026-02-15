#!/usr/bin/env python3
"""
Plot candles based on strategy configuration without executing strategies.
Demonstrates config → data loading → resampling → visualization pipeline.
"""

import yaml
import pandas as pd
from pathlib import Path
from trading_system.data.processing.candle_manager import CandleManager
from trading_system.visualization.candle_plotter import CandlePlotter
from trading_system.data.processing.resampler import ConfigDrivenResampler
from trading_system.utils.project_paths import CONFIG_DIR
from trading_system.utils.project_paths import DATA_DIR
import yaml

def load_strategy_config(config_name: str = "strategies.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load config
    config = load_strategy_config()
    print("Loaded strategy configuration:")
    for strat in config['strategies']:
        print(f"{strat['name']}: {strat['symbols']} @ {strat['timeframes']}")
    
    # 2. Initialize candle manager and load data
    data_root_path = DATA_DIR / "acquisition" / "data" / "raw" / "oanda"
    candle_mgr = CandleManager(data_root_path)

    # 2. Extract instruments/timeframes
    instruments, timeframes_by_instrument = ConfigDrivenResampler.extract_instruments_and_timeframes(config)

    if not instruments:
        print("No instruments found in config")
        return

    # 2. Collect all unique instruments from config
    # instruments = set()
    # timeframes_by_instrument = {}

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


    #This is now done in the ConfigDrivenResampler Class in the resampler file
    # for strat in config['strategies']:
    #     for symbol in strat['symbols']:
    #         instruments.add(symbol)
    #         if symbol not in timeframes_by_instrument:
    #             timeframes_by_instrument[symbol] = set()
    #         for tf in strat['timeframes']:
    #             timeframes_by_instrument[symbol].add(tf)
    
    # # Load base 1-minute data
    # print(f"\nLoading data for instruments: {sorted(instruments)}")
    # candle_mgr.load_from_pickle(list(instruments), granularity="M1")
    
    # # 3. Resample to required timeframes
    # print("\nResampling to strategy timeframes:")
    # for instrument in instruments:
    #     tfs = sorted(timeframes_by_instrument[instrument])
    #     print(f"{instrument}: {', '.join(tfs)}")
    #     candle_mgr.resample_to_timeframes(instrument, tfs)




    # 4. Plot each instrument with its strategy timeframes
    plotter = CandlePlotter(theme="dark", timezone="US/Eastern")  # or "dark"
    data_output_path = DATA_DIR / "output" / "plots"
    output_dir = Path(data_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for instrument in instruments:
        # Get all timeframes for this instrument
        timeframes = sorted(timeframes_by_instrument[instrument])
        
        if not timeframes:
            print(f"No timeframes configured for {instrument}")
            continue
        
        # Use smallest timeframe as base chart
        #base_tf = min(timeframes, key=lambda x: pd.Timedelta(_parse_timeframe(x)))
        base_tf = min(timeframes, key=lambda x: _parse_timeframe_duration(x))
        base_candles = candle_mgr.get_candles(instrument, base_tf)
        base_candles.attrs['timeframe'] = base_tf
        
        # Get other timeframes for subplots
        other_tfs = [tf for tf in timeframes if tf != base_tf]
        other_candles = {
            tf: candle_mgr.get_candles(instrument, tf)
            for tf in other_tfs
        }
        
        # Optional: Add example indicators (SMA) for visualization
        indicators = {
            base_tf: {
                'SMA_20': base_candles['close'].rolling(20).mean(),
                'SMA_50': base_candles['close'].rolling(50).mean()
            }
        }
        
        # Create multi-timeframe plot
        fig = plotter.plot_multi_timeframe(
            instrument=instrument,
            base_candles=base_candles,
            timeframe_candles=other_candles,
            #indicators=indicators,
            title_suffix=f"(Strategy Timeframes: {', '.join(timeframes)})",
            show_volume=False
        )
        
        # Save to HTML
        filepath = output_dir / f"{instrument}_multi_timeframe.html"
        plotter.save_html(fig, str(filepath))
        
        # # Also save a single-timeframe plot for the base TF
        # single_fig = plotter.plot_single_timeframe(
        #     instrument=instrument,
        #     candles=base_candles,
        #     #indicators=indicators[base_tf],
        #     title_suffix=f"Base Timeframe ({base_tf})",
        #     show_volume=False
        # )
        # single_filepath = output_dir / f"{instrument}_{base_tf}.html"
        # plotter.save_html(single_fig, str(single_filepath))
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")


# def _parse_timeframe(tf: str) -> str:
#     """Convert OANDA timeframe to pandas duration string for sorting"""
#     mapping = {
#         'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
#         'H1': '1h', 'H4': '4h', 'D': '1d'
#     }
#     return mapping.get(tf, tf)

def _parse_timeframe_duration(tf: str) -> int:
    mapping = {
        'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5, 'M10': 10, 'M15': 15,
        'M30': 30, 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240, 'H6': 360,
        'H8': 480, 'H12': 720, 'D': 1440
    }
    return mapping.get(tf, 9999)


if __name__ == "__main__":
    main()