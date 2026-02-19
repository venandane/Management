import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from datetime import datetime, timezone as dt_timezone
import pytz  # Required for timezone conversion
from trading_system.core.utils import calculate_pip_distance, get_pip_size


class CandlePlotter:
    """
    Interactive multi-timeframe candlestick plotter with timezone-aware display.
    All timestamps are converted from UTC storage to user-friendly display timezone.
    """
    
    def __init__(self, theme: str = "light", timezone: str = "UTC", strong_candle_offset_pips: float = 1.5):
        self.theme = theme
        self.display_timezone = timezone
        self.strong_candle_offset_pips = strong_candle_offset_pips

        # Validate and store timezone object
        try:
            self._tz_obj = pytz.timezone(timezone) if timezone != "UTC" else dt_timezone.utc
        except pytz.UnknownTimeZoneError:
            raise ValueError(
                f"Unknown timezone: '{timezone}'. Valid examples: 'UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo'"
            )

        self.color_scheme = {
            "light": {
                "bull": "#26a69a",    # Green
                "bear": "#ef5350",    # Red
                "volume": "#757575",
                "grid": "#e0e0e0",
                "bg": "#ffffff",
                "text": "#212121"
            },
            "dark": {
                "bull": "#00e676",
                "bear": "#ff5252",
                "volume": "#9e9e9e",
                "grid": "#424242",
                "bg": "#121212",
                "text": "#e0e0e0"
            }
        }[theme]

    def _prepare_display_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame index from UTC storage to display timezone.
        Returns a COPY with converted index to avoid mutating original data.
        
        Handles:
          - Naive indexes (assumes UTC)
          - Timezone-aware UTC indexes
          - Automatic DST transitions (e.g., EST/EDT via 'US/Eastern')
        """
        if df.empty:
            return df.copy()
        
        df_display = df.copy()
        
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_display.index):
            raise ValueError("DataFrame index must be datetime type")
        
        # Localize naive index to UTC first
        if df_display.index.tz is None:
            df_display.index = df_display.index.tz_localize('UTC')
        # Convert to UTC if in another timezone
        elif df_display.index.tz != dt_timezone.utc:
            df_display.index = df_display.index.tz_convert('UTC')
        
        # Convert to display timezone
        if self.display_timezone != "UTC":
            df_display.index = df_display.index.tz_convert(self._tz_obj)
        
        return df_display

    def _prepare_display_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signal timestamps to display timezone.
        Returns a COPY with converted index.
        """
        if signals.empty:
            return signals.copy()
        
        signals_display = signals.copy()
        
        if signals_display.index.tz is None:
            signals_display.index = signals_display.index.tz_localize('UTC')
        elif signals_display.index.tz != dt_timezone.utc:
            signals_display.index = signals_display.index.tz_convert('UTC')
        
        if self.display_timezone != "UTC":
            signals_display.index = signals_display.index.tz_convert(self._tz_obj)
        
        return signals_display

    def plot_multi_timeframe(
        self,
        instrument: str,
        base_candles: pd.DataFrame,
        timeframe_candles: Dict[str, pd.DataFrame],
        indicators: Optional[Dict[str, Dict[str, pd.Series]]] = None,
        title_suffix: str = "",
        show_volume: bool = True,
        height_per_timeframe: int = 1900,
        width: int = 3600,
        height: Optional[int] = None
    ) -> go.Figure:
        """Plot base timeframe with multiple higher timeframes below it."""
        timeframes = [base_candles.attrs.get('timeframe', 'base')] + list(timeframe_candles.keys())
        n_rows = 1 + len(timeframe_candles) + (1 if show_volume else 0)
        row_heights = [0.7] + ([0.15] * (len(timeframe_candles))) + ([0.15] if show_volume else [])
        
        total_height = height if height is not None else height_per_timeframe * n_rows
        
        # Convert all DataFrames to display timezone FIRST
        base_display = self._prepare_display_df(base_candles)
        timeframe_display = {
            tf: self._prepare_display_df(df) 
            for tf, df in timeframe_candles.items()
        }
        
        # Create subplots
        subplot_titles = [f"{instrument} - {tf}" for tf in timeframes]
        if show_volume:
            subplot_titles.append("Volume")
            
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )

        # Plot base timeframe (row 1)
        self._add_candlestick_trace(
            fig, base_display, row=1, col=1, 
            indicators=indicators.get(timeframes[0], {}) if indicators else {},
            instrument=instrument
        )
        
        # Plot additional timeframes
        current_row = 2
        for tf in timeframe_candles.keys():
            df_display = timeframe_display[tf]
            self._add_candlestick_trace(
                fig, df_display, row=current_row, col=1,
                indicators=indicators.get(tf, {}) if indicators else {},
                instrument=instrument
            )
            current_row += 1
        
        # Volume subplot (last row if enabled)
        if show_volume:
            self._add_volume_trace(fig, base_display, row=current_row, col=1)
        
        # Layout
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} Multi-Timeframe Analysis{tz_label} {title_suffix}",
            height=total_height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Grid lines and timezone label on x-axis (only bottom row)
        for i in range(1, n_rows + 1):
            show_xaxis_label = (i == n_rows)
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                title_text=f"Time{tz_label}" if show_xaxis_label else None,
                row=i,
                col=1,
                rangeslider_visible=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                row=i,
                col=1
            )
        
        return fig

    def plot_single_timeframe(
        self,
        instrument: str,
        candles: pd.DataFrame,
        indicators: Optional[Dict[str, pd.Series]] = None,
        signals: Optional[pd.DataFrame] = None,
        title_suffix: str = "",
        show_volume: bool = True,
        width: int = 3600,
        height: int = 1900
    ) -> go.Figure:
        """Plot single timeframe with indicators and trade signals."""
        rows = 2 if show_volume else 1
        row_heights = [0.75, 0.25] if show_volume else [1.0]
        
        # Convert data to display timezone
        candles_display = self._prepare_display_df(candles)
        signals_display = self._prepare_display_signals(signals) if signals is not None else None
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )

        # Main candlestick chart
        self._add_candlestick_trace(
            fig, candles_display, row=1, col=1,
            indicators=indicators or {},
            instrument=instrument
        )
        
        # Trade signals (if any)
        if signals_display is not None and not signals_display.empty:
            self._add_signal_markers(fig, signals_display, row=1, col=1)
        
        # Volume subplot
        if show_volume:
            self._add_volume_trace(fig, candles_display, row=2, col=1)
        
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} {candles.attrs.get('timeframe', '')}{tz_label} {title_suffix}",
            height=height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Grid lines with timezone label
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.color_scheme["grid"],
            title_text=f"Time{tz_label}",
            rangeslider_visible=False,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor'
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.color_scheme["grid"])
        
        return fig

    def plot_strategy_breakout(
        self,
        instrument: str,
        higher_tf_candles: pd.DataFrame,
        lower_tf_candles: pd.DataFrame,
        strong_candles: pd.DataFrame,
        buy_signals: Optional[pd.DataFrame] = None,
        title_suffix: str = "",
        width: int = 3600,
        height: int = 900
    ) -> go.Figure:
        """
        Plot synchronized multi-timeframe breakout strategy:
          Row 1: Higher timeframe with strong candle markers
          Row 2: Lower timeframe with breakout buy signals
        
        CRITICAL FIXES:
          ✅ Both timeframes converted to display timezone
          ✅ Hovertemplate syntax fixed (%{customdata:.5f} not {customdata:.5f})
          ✅ Volume subplot removed (was causing layout issues when commented)
          ✅ Configurable dimensions with sensible defaults
        """
        # Convert ALL data to display timezone FIRST
        higher_display = self._prepare_display_df(higher_tf_candles)
        lower_display = self._prepare_display_df(lower_tf_candles)
        strong_display = self._prepare_display_df(strong_candles)
        signals_display = self._prepare_display_signals(buy_signals) if buy_signals is not None else None
        
        # Create 2-row subplot (removed volume row for cleaner breakout view)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.45],
            subplot_titles=[
                f"{instrument} - {higher_tf_candles.attrs.get('timeframe', 'H1')} (Signal Generator)",
                f"{instrument} - {lower_tf_candles.attrs.get('timeframe', 'M1')} (Execution)"
            ]
        )
        
        # Row 1: Higher timeframe candles + strong candle markers
        self._add_candlestick_trace(
            fig, strong_display, row=1, col=1, 
            indicators={}, 
            instrument=instrument
        )
        
        # ENHANCEMENT: Draw targeted segments ONLY if required columns exist
        has_breakout_segments = (
            signals_display is not None 
            and not signals_display.empty
            and 'breakout_level' in signals_display.columns
            and 'strong_candle_time' in signals_display.columns
        )
        
        if has_breakout_segments:
            self._add_targeted_breakout_segments(
                fig, signals_display, instrument, row=1, col=1
            )
        elif signals_display is not None and not signals_display.empty and 'breakout_level' in signals_display.columns:
            # Fallback: full-width lines if strong_candle_time missing
            self._add_full_width_breakout_levels(
                fig, signals_display, row=1, col=1
            )
        # Else: no breakout levels to draw
        
        # Row 2: Lower timeframe candles + buy signals
        self._add_candlestick_trace(
            fig, lower_display, row=2, col=1,
            indicators={},
            instrument=instrument
        )








        # Add buy signal markers (blue triangles)
        if signals_display is not None and not signals_display.empty:
            self._add_buy_signal_markers(
                fig, signals_display, instrument, row=2, col=1
            )
        
        # Layout
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} Breakout Strategy{tz_label} {title_suffix}",
            height=height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Grid lines
        for i in range(1, 3):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                title_text=f"Time{tz_label}" if i == 2 else None,
                row=i,
                col=1,
                rangeslider_visible=False
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                row=i,
                col=1
            )
        
        return fig

    def _add_targeted_breakout_segments(
        self,
        fig,
        signals: pd.DataFrame,
        instrument: str,
        row: int,
        col: int
    ):
        """Draw segments from strong candle close → breakout time (defensive column checks)."""
        pip_size = get_pip_size(instrument)
        label_offset = 5.0 * pip_size
        
        for idx, signal in signals.iterrows():
            breakout_level = signal.get('breakout_level', 0.0)
            strong_time_raw = signal.get('strong_candle_time')
            
            # Skip if critical data missing
            if breakout_level <= 0 or pd.isna(strong_time_raw) or pd.isna(idx):
                continue
            
            # Convert strong_time to timezone-aware UTC timestamp
            try:
                strong_time = pd.Timestamp(strong_time_raw)
                if strong_time.tz is None:
                    strong_time = strong_time.tz_localize('UTC')
                elif strong_time.tz != dt_timezone.utc:
                    strong_time = strong_time.tz_convert('UTC')
                
                # Convert to display timezone for plotting
                if self.display_timezone != "UTC":
                    strong_time = strong_time.tz_convert(self._tz_obj)
            except Exception:
                continue  # Skip invalid timestamps
            
            # Skip if segment would be invisible
            if strong_time >= idx:
                continue
            
            # Draw segment
            fig.add_trace(
                go.Scatter(
                    x=[strong_time, idx],
                    y=[breakout_level, breakout_level],
                    mode='lines',
                    line=dict(color='gold', width=2.5, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Add price label
            fig.add_trace(
                go.Scatter(
                    x=[strong_time],
                    y=[breakout_level + label_offset],
                    mode='text',
                    text=[f"{breakout_level:.5f}"],
                    textposition='top center',
                    textfont=dict(size=9, color='gold'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    def _add_full_width_breakout_levels(
        self,
        fig,
        signals: pd.DataFrame,
        row: int,
        col: int
    ):
        """Fallback: draw full-width horizontal lines when strong_candle_time unavailable."""
        drawn_levels = set()
        for _, signal in signals.iterrows():
            level = round(signal.get('breakout_level', 0), 5)
            if level > 0 and level not in drawn_levels:
                drawn_levels.add(level)
                fig.add_hline(
                    y=level,
                    row=row, col=col,
                    line=dict(color="gold", width=1.5, dash="dot"),
                    opacity=0.7
                )

    def _add_buy_signal_markers(
        self,
        fig,
        signals: pd.DataFrame,
        instrument: str,
        row: int,
        col: int
    ):
        """Add blue triangle markers for buy signals with defensive customdata handling."""
        pip_size = get_pip_size(instrument)
        y_offset = 3.0 * pip_size
        
        # Build customdata defensively (handle missing columns)
        has_breakout_level = 'breakout_level' in signals.columns
        has_body_pips = 'body_pips' in signals.columns
        
        if has_breakout_level and has_body_pips:
            customdata = np.column_stack([
                signals['breakout_level'].values,
                signals['body_pips'].values
            ])
            hovertemplate = (
                '<b>BUY Signal</b><br>'
                'Time: %{x}<br>'
                'Price: %{y:.5f}<br>'
                'Breakout Level: %{customdata[0]:.5f}<br>'
                'Body Size: %{customdata[1]:.1f} pips<extra></extra>'
            )
        elif has_breakout_level:
            customdata = signals['breakout_level'].values.reshape(-1, 1)
            hovertemplate = (
                '<b>BUY Signal</b><br>'
                'Time: %{x}<br>'
                'Price: %{y:.5f}<br>'
                'Breakout Level: %{customdata[0]:.5f}<extra></extra>'
            )
        else:
            customdata = None
            hovertemplate = (
                '<b>BUY Signal</b><br>'
                'Time: %{x}<br>'
                'Price: %{y:.5f}<extra></extra>'
            )
        
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['price'] + y_offset,
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color='blue',
                    line=dict(width=2, color='white')
                ),
                hovertemplate=hovertemplate,
                customdata=customdata
            ),
            row=row, col=col
        )

    def _add_candlestick_trace(self, fig, df: pd.DataFrame, row: int, col: int, 
                              indicators: Dict[str, pd.Series], instrument: str):
        """Add candlestick trace with optional indicators and strong candle markers."""
        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Candles",
                increasing_line_color=self.color_scheme["bull"],
                decreasing_line_color=self.color_scheme["bear"],
                opacity=0.8
            ),
            row=row, col=col
        )
        
        # Indicators (lines)
        for name, series in indicators.items():
            # Align indicator index with candles (handle NaNs)
            aligned = series.reindex(df.index)
            fig.add_trace(
                go.Scatter(
                    x=aligned.index,
                    y=aligned.values,
                    mode='lines',
                    name=name,
                    line=dict(width=1.5, dash='dash' if 'signal' in name.lower() else 'solid')
                ),
                row=row, col=col
            )
        
        # Strong candle markers (if detected) - df is already timezone-converted
        if 'strong_bull' in df.columns and 'strong_bear' in df.columns:
            self._add_strong_candle_markers(
                fig, df, row=row, col=col,
                instrument=instrument,
                offset_pips=self.strong_candle_offset_pips
            )

    def _add_strong_candle_markers(
        self,
        fig,
        df: pd.DataFrame,
        row: int,
        col: int,
        instrument: str,
        marker_size: int = 14,
        offset_pips: float = 1.5
    ):
        """Add arrows anchored to candle high/low with FIXED PIP OFFSET."""
        if 'strong_bull' not in df.columns or 'strong_bear' not in df.columns:
            return
        
        # Convert pip offset to price units (critical for JPY vs non-JPY pairs)
        pip_size = get_pip_size(instrument)
        price_offset = offset_pips * pip_size
        
        # Bull candles: green up arrows ABOVE candle high
        bull_candles = df[df['strong_bull']]
        if not bull_candles.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull_candles.index,
                    y=bull_candles['high'] + price_offset,
                    mode='markers',
                    name='Strong Bull',
                    marker=dict(
                        symbol='triangle-up',
                        size=marker_size,
                        color=self.color_scheme["bull"],
                        line=dict(width=1.5, color='white')
                    ),
                    hovertemplate=(
                        '<b>Strong Bull Candle</b><br>'
                        'Time: %{x}<br>'
                        'Body: %{customdata[0]:.1f} pips<br>'
                        'Upper Tail: %{customdata[1]:.1%}<br>'
                        'Total Range: %{customdata[2]:.1f} pips<br>'
                        '<span style="color:gray;font-size:0.9em">Arrow offset: +' + 
                        f'{offset_pips}' + ' pips</span><extra></extra>'
                    ),
                    customdata=np.stack([
                        bull_candles['body_pips'],
                        bull_candles['upper_tail_ratio'],
                        bull_candles['range_pips']
                    ], axis=-1)
                ),
                row=row, col=col
            )
        
        # Bear candles: red down arrows BELOW candle low
        bear_candles = df[df['strong_bear']]
        if not bear_candles.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear_candles.index,
                    y=bear_candles['low'] - price_offset,
                    mode='markers',
                    name='Strong Bear',
                    marker=dict(
                        symbol='triangle-down',
                        size=marker_size,
                        color=self.color_scheme["bear"],
                        line=dict(width=1.5, color='white')
                    ),
                    hovertemplate=(
                        '<b>Strong Bear Candle</b><br>'
                        'Time: %{x}<br>'
                        'Body: %{customdata[0]:.1f} pips<br>'
                        'Lower Tail: %{customdata[1]:.1%}<br>'
                        'Total Range: %{customdata[2]:.1f} pips<br>'
                        '<span style="color:gray;font-size:0.9em">Arrow offset: -' + 
                        f'{offset_pips}' + ' pips</span><extra></extra>'
                    ),
                    customdata=np.stack([
                        bear_candles['body_pips'],
                        bear_candles['lower_tail_ratio'],
                        bear_candles['range_pips']
                    ], axis=-1)
                ),
                row=row, col=col
            )

    def _add_volume_trace(self, fig, df: pd.DataFrame, row: int, col: int):
        """Add volume bars colored by price direction."""
        if df.empty or 'volume' not in df.columns:
            return
            
        colors = [
            self.color_scheme["bull"] if df['close'].iloc[i] > df['open'].iloc[i] 
            else self.color_scheme["bear"]
            for i in range(len(df))
        ]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Volume", row=row, col=col)

    def _add_signal_markers_old(self, fig, signals: pd.DataFrame, row: int, col: int):
        """Add entry/exit markers for backtest visualization (signals already timezone-converted)."""
        if signals.empty:
            return
            
        # Long entries
        long_entries = signals[signals['direction'] > 0]
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries.index,
                    y=long_entries['entry_price'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1.5, color='white'))
                ),
                row=row, col=col
            )
        
        # Short entries
        short_entries = signals[signals['direction'] < 0]
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries.index,
                    y=short_entries['entry_price'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1.5, color='white'))
                ),
                row=row, col=col
            )
        
        # Exits (all)
        exits = signals[signals['exit_price'].notna()]
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=exits['exit_price'],
                    mode='markers',
                    name='Exit',
                    marker=dict(symbol='x', size=10, color='gray')
                ),
                row=row, col=col
            )


    def _add_signal_markers(self, fig, signals: pd.DataFrame, row: int, col: int):
        """Add direction-aware signal markers (blue ▲ for buys, red ▼ for sells)."""
        if signals.empty:
            return
        
        # Get pip size for vertical offset
        # Infer instrument from figure data (hacky but works for single-instrument plots)
        instrument = "EUR_USD"  # Default fallback
        for trace in fig.data:
            if hasattr(trace, 'name') and 'EUR' in str(trace.name):
                instrument = "EUR_USD"
                break
        
        pip_size = get_pip_size(instrument)
        y_offset = 3.0 * pip_size
        
        # BUY signals (direction > 0)
        buy_signals = signals[signals['direction'] > 0]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'] + y_offset,
                    mode='markers',
                    name='BUY Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=16,
                        color='blue',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=(
                        '<b>BUY Signal</b><br>'
                        'Time: %{x}<br>'
                        'Price: %{y:.5f}<br>'
                        'Body: %{customdata[0]:.1f} pips<br>'
                        'Type: %{customdata[1]}<extra></extra>'
                    ),
                    customdata=np.column_stack([
                        buy_signals.get('body_pips', pd.Series(0, index=buy_signals.index)).values,
                        buy_signals.get('candle_direction', pd.Series('bull', index=buy_signals.index)).values
                    ])
                ),
                row=row, col=col
            )
        
        # SELL signals (direction < 0)
        sell_signals = signals[signals['direction'] < 0]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'] - y_offset,
                    mode='markers',
                    name='SELL Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=16,
                        color='red',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=(
                        '<b>SELL Signal</b><br>'
                        'Time: %{x}<br>'
                        'Price: %{y:.5f}<br>'
                        'Body: %{customdata[0]:.1f} pips<br>'
                        'Type: %{customdata[1]}<extra></extra>'
                    ),
                    customdata=np.column_stack([
                        sell_signals.get('body_pips', pd.Series(0, index=sell_signals.index)).values,
                        sell_signals.get('candle_direction', pd.Series('bear', index=sell_signals.index)).values
                    ])
                ),
                row=row, col=col
            )

    def plot_strong_candle_refined(
        self,
        instrument: str,
        candles: pd.DataFrame,
        all_signals: pd.DataFrame,
        filtered_signals: pd.DataFrame,
        bad_signals: Optional[pd.DataFrame] = None,
        title_suffix: str = "",
        show_volume: bool = True,
        width: int = 3600,
        height: int = 1900
    ) -> go.Figure:
        """
        Plot refined strong candle strategy with bad signal markers.
        
        Visual elements:
        - Green ▲ / Red ▼ arrows: Strong candles
        - Small triangles: Raw signals rejected by bad signal filter
        - Large triangles with border: Filtered signals accepted by strategy
        - Red 'x': Bad SELL signals (price rose 12+ pips after signal)
        - Blue 'x': Bad BUY signals (price fell 12+ pips after signal)
        """

        
        # Convert to display timezone
        candles_display = self._prepare_display_df(candles)
        all_signals_display = self._prepare_display_signals(all_signals) if not all_signals.empty else pd.DataFrame()
        filtered_signals_display = self._prepare_display_signals(filtered_signals) if not filtered_signals.empty else pd.DataFrame()
        bad_signals_display = self._prepare_display_signals(bad_signals) if bad_signals is not None and not bad_signals.empty else None
        
        # Create subplot with volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=[
                f"{instrument} - {candles.attrs.get('timeframe', 'H1')} (Refined Signals)",
                "Volume"
            ]
        )
        
        # Main candlestick chart
        self._add_candlestick_trace(
            fig, candles_display, row=1, col=1,
            indicators={},
            instrument=instrument
        )
        
        # Add bad signal markers ('x' at candle low/high)
        if bad_signals_display is not None and not bad_signals_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 2.0 * pip_size
            
            # Red 'x' for bad SELL signals (at candle low)
            bad_sells = bad_signals_display[bad_signals_display['direction'] < 0]
            if not bad_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_sells.index,
                        y=bad_sells['price'] - y_offset,
                        mode='markers+text',
                        name='Bad SELL',
                        marker=dict(symbol='x', size=16, color='red', line=dict(width=2, color='darkred')),
                        text=['✗'] * len(bad_sells),
                        textposition='bottom center',
                        textfont=dict(size=20, color='red'),
                        hovertemplate=(
                            '<b>Bad SELL Signal</b><br>'
                            'Time: %{x}<br>'
                            'Invalidated by: %{customdata}<extra></extra>'
                        ),
                        customdata=bad_sells['reason'].values
                    ),
                    row=1, col=1
                )
            
            # Blue 'x' for bad BUY signals (at candle high)
            bad_buys = bad_signals_display[bad_signals_display['direction'] > 0]
            if not bad_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_buys.index,
                        y=bad_buys['price'] + y_offset,
                        mode='markers+text',
                        name='Bad BUY',
                        marker=dict(symbol='x', size=16, color='blue', line=dict(width=2, color='darkblue')),
                        text=['✗'] * len(bad_buys),
                        textposition='top center',
                        textfont=dict(size=20, color='blue'),
                        hovertemplate=(
                            '<b>Bad BUY Signal</b><br>'
                            'Time: %{x}<br>'
                            'Invalidated by: %{customdata}<extra></extra>'
                        ),
                        customdata=bad_buys['reason'].values
                    ),
                    row=1, col=1
                )
        
        # Add raw signals (rejected by filter) - small triangles
        if not all_signals_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 3.0 * pip_size
            
            # Identify rejected signals (in all_signals but not in filtered_signals)
            rejected_buys = all_signals_display[
                (all_signals_display['direction'] > 0) & 
                (~all_signals_display.index.isin(filtered_signals_display.index))
            ]
            rejected_sells = all_signals_display[
                (all_signals_display['direction'] < 0) & 
                (~all_signals_display.index.isin(filtered_signals_display.index))
            ]
            
            # Small blue triangles for rejected buys
            if not rejected_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rejected_buys.index,
                        y=rejected_buys['price'] + y_offset,
                        mode='markers',
                        name='Rejected BUY',
                        marker=dict(symbol='triangle-up', size=10, color='lightblue', opacity=0.7),
                        hovertemplate=(
                            '<b>Rejected BUY</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            rejected_buys['body_pips'].values
                        ])
                    ),
                    row=1, col=1
                )
            
            # Small red triangles for rejected sells
            if not rejected_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rejected_sells.index,
                        y=rejected_sells['price'] - y_offset,
                        mode='markers',
                        name='Rejected SELL',
                        marker=dict(symbol='triangle-down', size=10, color='lightcoral', opacity=0.7),
                        hovertemplate=(
                            '<b>Rejected SELL</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            rejected_sells['body_pips'].values
                        ])
                    ),
                    row=1, col=1
                )
        
        # Add filtered signals (accepted) - large triangles with border
        if not filtered_signals_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 3.0 * pip_size
            
            # Large blue triangles with white border for accepted buys
            accepted_buys = filtered_signals_display[filtered_signals_display['direction'] > 0]
            if not accepted_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_buys.index,
                        y=accepted_buys['price'] + y_offset,
                        mode='markers',
                        name='Accepted BUY',
                        marker=dict(
                            symbol='triangle-up',
                            size=18,
                            color='blue',
                            line=dict(width=2.5, color='white')
                        ),
                        hovertemplate=(
                            '<b>Accepted BUY</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            accepted_buys['body_pips'].values
                        ])
                    ),
                    row=1, col=1
                )
            
            # Large red triangles with white border for accepted sells
            accepted_sells = filtered_signals_display[filtered_signals_display['direction'] < 0]
            if not accepted_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_sells.index,
                        y=accepted_sells['price'] - y_offset,
                        mode='markers',
                        name='Accepted SELL',
                        marker=dict(
                            symbol='triangle-down',
                            size=18,
                            color='red',
                            line=dict(width=2.5, color='white')
                        ),
                        hovertemplate=(
                            '<b>Accepted SELL</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            accepted_sells['body_pips'].values
                        ])
                    ),
                    row=1, col=1
                )
        
        # Volume subplot
        if show_volume:
            self._add_volume_trace(fig, candles_display, row=2, col=1)
        
        # Layout
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} Refined Strong Candle Strategy{tz_label} {title_suffix}",
            height=height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Grid lines
        for i in range(1, 3):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                title_text=f"Time{tz_label}" if i == 2 else None,
                row=i,
                col=1,
                rangeslider_visible=False
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.color_scheme["grid"],
                row=i,
                col=1
            )
        
        return fig



    def plot_strong_candle_refined_v2(
        self,
        instrument: str,
        candles: pd.DataFrame,
        all_strong_candles: pd.DataFrame,
        accepted_signals: pd.DataFrame,
        bad_signals: Optional[pd.DataFrame] = None,
        title_suffix: str = "",
        width: int = 3600,
        height: int = 1900
    ) -> go.Figure:
        """
        V2: Implements EXACT visual representation of specification logic.
        
        Visual markers:
        - ✗ Red 'x' at candle LOW = Bad SELL signal (reference level for filtering)
        - ✗ Blue 'x' at candle HIGH = Bad BUY signal (reference level for filtering)
        - Large red ▼ = ACCEPTED SELL signal (passed filter: current_close < L_bad OR > L_bad+12pips)
        - Large blue ▲ = ACCEPTED BUY signal (passed filter)
        - Small gray markers = Strong candles that were REJECTED by filter (not plotted by default)
        """

        # Convert to display timezone
        candles_display = self._prepare_display_df(candles)
        accepted_display = self._prepare_display_signals(accepted_signals) if not accepted_signals.empty else pd.DataFrame()
        bad_display = self._prepare_display_signals(bad_signals) if bad_signals is not None and not bad_signals.empty else None
        
        # Create subplot
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{instrument} - {candles.attrs.get('timeframe', 'H1')} (Exact Filter Logic)"]
        )
        
        # Main candlestick chart
        self._add_candlestick_trace(
            fig, candles_display, row=1, col=1,
            indicators={},
            instrument=instrument
        )
        
        # Add BAD SIGNAL markers ('x' at reference levels)
        if bad_display is not None and not bad_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 2.0 * pip_size
            
            # Red 'x' at CANDLE LOW for bad SELL signals (reference level L_bad)
            bad_sells = bad_display[bad_display['direction'] < 0]
            if not bad_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_sells.index,
                        y=bad_sells['price'] - y_offset,
                        mode='markers',
                        name='Bad SELL Reference',
                        marker=dict(symbol='x', size=18, color='red', line=dict(width=2.5, color='darkred')),
                        text=['✗'] * len(bad_sells),
                        textposition='bottom center',
                        textfont=dict(size=24, color='red'),
                        hovertemplate=(
                            '<b>Bad SELL Reference Level</b><br>'
                            'Time: %{x}<br>'
                            'Candle Low (L_bad): %{y:.5f}<br>'
                            '<span style="color:gray">Used for filtering new SELL signals</span><extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # Blue 'x' at CANDLE HIGH for bad BUY signals (reference level H_bad)
            bad_buys = bad_display[bad_display['direction'] > 0]
            if not bad_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_buys.index,
                        y=bad_buys['price'] + y_offset,
                        mode='markers',
                        name='Bad BUY Reference',
                        marker=dict(symbol='x', size=18, color='blue', line=dict(width=2.5, color='darkblue')),
                        text=['✗'] * len(bad_buys),
                        textposition='top center',
                        textfont=dict(size=24, color='blue'),
                        hovertemplate=(
                            '<b>Bad BUY Reference Level</b><br>'
                            'Time: %{x}<br>'
                            'Candle High (H_bad): %{y:.5f}<br>'
                            '<span style="color:gray">Used for filtering new BUY signals</span><extra></extra>'
                        )
                    ),
                    row=1, col=1
                )



        # Add ACCEPTED SIGNALS (large triangles with border)
        if not accepted_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 3.5 * pip_size
            
            # Large blue triangles for accepted BUYs
            accepted_buys = accepted_display[accepted_display['direction'] > 0]
            if not accepted_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_buys.index,
                        y=accepted_buys['price'] + y_offset,
                        mode='markers',
                        name='Accepted BUY',
                        marker=dict(
                            symbol='triangle-up',
                            size=20,
                            color='cyan',
                            line=dict(width=3, color='black')
                        ),
                        hovertemplate=(
                            '<b>ACCEPTED BUY</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<br>'
                            '<span style="color:green">Passed bad signal filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([accepted_buys['body_pips'].values])
                    ),
                    row=1, col=1
                )


            # Large red triangles for accepted SELLs
            accepted_sells = accepted_display[accepted_display['direction'] < 0]
            if not accepted_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_sells.index,
                        y=accepted_sells['price'] - y_offset,
                        mode='markers',
                        name='Accepted SELL',
                        marker=dict(
                            symbol='triangle-down',
                            size=20,
                            color='yellow',
                            line=dict(width=3, color='black')
                        ),
                        hovertemplate=(
                            '<b>ACCEPTED SELL</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<br>'
                            '<span style="color:green">Passed bad signal filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([accepted_sells['body_pips'].values])
                    ),
                    row=1, col=1
                )
        
        
        # Layout
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} Refined Strategy V2{tz_label} {title_suffix}",
            height=height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Grid lines
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.color_scheme["grid"],
            title_text=f"Time{tz_label}",
            rangeslider_visible=False,
            row=1, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.color_scheme["grid"],
            row=1, col=1
        )
        
        return fig



    def save_html(self, fig: go.Figure, filepath: str):
        """Save interactive plot to HTML file."""
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            full_html=True,
            config={'scrollZoom': True, 'displayModeBar': True}
        )
        print(f"Saved interactive chart to: {filepath}")



    def plot_strong_candle_refined_v3(
        self,
        instrument: str,
        candles: pd.DataFrame,
        accepted_signals: pd.DataFrame,
        rejected_signals: pd.DataFrame,
        bad_signals: Optional[pd.DataFrame] = None,
        title_suffix: str = "",
        width: int = 3600,
        height: int = 1900
    ) -> go.Figure:
        """
        V3: Adds markers for REJECTED strong candles (failed filter test).
        
        Visual hierarchy:
        1. Candlestick chart (OHLC)
        2. Strong candle arrows (green ▲ / red ▼) - auto from candles DataFrame
        3. Bad reference markers (red/blue ✗) - at L_bad/H_bad levels
        4. Rejected signal markers (small gray ◯) - at candle close
        5. Accepted signal markers (large blue/red ▲/▼) - at candle close
        """
        
        # Convert to display timezone
        candles_display = self._prepare_display_df(candles)
        accepted_display = self._prepare_display_signals(accepted_signals) if not accepted_signals.empty else pd.DataFrame()
        rejected_display = self._prepare_display_signals(rejected_signals) if not rejected_signals.empty else pd.DataFrame()
        bad_display = self._prepare_display_signals(bad_signals) if bad_signals is not None and not bad_signals.empty else None
        
        # Create SINGLE-PANEL plot
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{instrument} - {candles.attrs.get('timeframe', 'H1')} (Filter Results)"]
        )
        
        # Main candlestick chart WITH STRONG CANDLE ARROWS (auto-detected)
        self._add_candlestick_trace(
            fig, candles_display, row=1, col=1,
            indicators={},
            instrument=instrument
        )
        
        # Add BAD SIGNAL markers ('x' at reference levels)
        if bad_display is not None and not bad_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 2.0 * pip_size
            
            # Red 'x' at CANDLE LOW for bad SELL signals
            bad_sells = bad_display[bad_display['direction'] < 0]
            if not bad_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_sells.index,
                        y=bad_sells['price'] - y_offset,
                        mode='markers',
                        name='Bad SELL Reference',
                        marker=dict(symbol='x', size=18, color='red', line=dict(width=2.5, color='darkred')),
                        text=['✗'] * len(bad_sells),
                        textposition='bottom center',
                        textfont=dict(size=24, color='red'),
                        hovertemplate=(
                            '<b>Bad SELL Reference</b><br>'
                            'Time: %{x}<br>'
                            'Candle Low: %{y:.5f}<br>'
                            '<span style="color:gray">Filter reference level</span><extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
            
            # Blue 'x' at CANDLE HIGH for bad BUY signals
            bad_buys = bad_display[bad_display['direction'] > 0]
            if not bad_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bad_buys.index,
                        y=bad_buys['price'] + y_offset,
                        mode='markers',
                        name='Bad BUY Reference',
                        marker=dict(symbol='x', size=18, color='blue', line=dict(width=2.5, color='darkblue')),
                        text=['✗'] * len(bad_buys),
                        textposition='top center',
                        textfont=dict(size=24, color='blue'),
                        hovertemplate=(
                            '<b>Bad BUY Reference</b><br>'
                            'Time: %{x}<br>'
                            'Candle High: %{y:.5f}<br>'
                            '<span style="color:gray">Filter reference level</span><extra></extra>'
                        )
                    ),
                    row=1, col=1
                )
        
        # REPLACE the rejected signal section with this enhanced implementation:
        if not rejected_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 4.5 * pip_size  # Increased offset to avoid marker overlap
            
            # Split rejected signals by direction for distinct styling
            rejected_buys = rejected_display[rejected_display['direction'] > 0]
            rejected_sells = rejected_display[rejected_display['direction'] < 0]
            
            # ════════════════════════════════════════════════════════════════════
            # REJECTED BUY SIGNALS: Orange downward arrow ABOVE candle
            # (Points down to indicate "rejected long entry")
            # ════════════════════════════════════════════════════════════════════
            if not rejected_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rejected_buys.index,
                        y=rejected_buys['price'] + y_offset,
                        mode='markers',
                        name='Rejected BUY',
                        marker=dict(
                            symbol='arrow-bar-down',  # Long downward arrow with base bar
                            size=22,                  # Larger than accepted signals
                            color='orange',           # Distinct orange for rejected buys
                            line=dict(width=2.5, color='darkorange'),
                            opacity=1.0
                        ),
                        hovertemplate=(
                            '<b style="color:orange">❌ REJECTED BUY</b><br>'
                            '<b>Time:</b> %{x}<br>'
                            '<b>Price:</b> %{y:.5f}<br>'
                            '<b>Body:</b> %{customdata[0]:.1f} pips<br>'
                            '<span style="color:gray">Failed acceptance filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([rejected_buys['body_pips'].values])
                    ),
                    row=1, col=1
                )
            
            # ════════════════════════════════════════════════════════════════════
            # REJECTED SELL SIGNALS: Purple upward arrow BELOW candle
            # (Points up to indicate "rejected short entry")
            # ════════════════════════════════════════════════════════════════════
            if not rejected_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rejected_sells.index,
                        y=rejected_sells['price'] - y_offset,
                        mode='markers',
                        name='Rejected SELL',
                        marker=dict(
                            symbol='arrow-bar-up',    # Long upward arrow with base bar
                            size=22,                  # Larger than accepted signals
                            color='purple',           # Distinct purple for rejected sells
                            line=dict(width=2.5, color='darkviolet'),
                            opacity=1.0
                        ),
                        hovertemplate=(
                            '<b style="color:purple">❌ REJECTED SELL</b><br>'
                            '<b>Time:</b> %{x}<br>'
                            '<b>Price:</b> %{y:.5f}<br>'
                            '<b>Body:</b> %{customdata[0]:.1f} pips<br>'
                            '<span style="color:gray">Failed acceptance filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([rejected_sells['body_pips'].values])
                    ),
                    row=1, col=1
                )
        
        # Add ACCEPTED SIGNALS (large triangles with border)
        if not accepted_display.empty:
            pip_size = get_pip_size(instrument)
            y_offset = 3.5 * pip_size
            
            # Large blue triangles for accepted BUYs
            accepted_buys = accepted_display[accepted_display['direction'] > 0]
            if not accepted_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_buys.index,
                        y=accepted_buys['price'] + y_offset,
                        mode='markers',
                        name='Accepted BUY',
                        marker=dict(
                            symbol='triangle-up',
                            size=20,
                            color='blue',
                            line=dict(width=3, color='white')
                        ),
                        hovertemplate=(
                            '<b>✅ ACCEPTED BUY</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<br>'
                            '<span style="color:green">Passed bad signal filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([accepted_buys['body_pips'].values])
                    ),
                    row=1, col=1
                )
            
            # Large red triangles for accepted SELLs
            accepted_sells = accepted_display[accepted_display['direction'] < 0]
            if not accepted_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accepted_sells.index,
                        y=accepted_sells['price'] - y_offset,
                        mode='markers',
                        name='Accepted SELL',
                        marker=dict(
                            symbol='triangle-down',
                            size=20,
                            color='red',
                            line=dict(width=3, color='white')
                        ),
                        hovertemplate=(
                            '<b>✅ ACCEPTED SELL</b><br>'
                            'Time: %{x}<br>'
                            'Price: %{y:.5f}<br>'
                            'Body: %{customdata[0]:.1f} pips<br>'
                            '<span style="color:green">Passed bad signal filter</span><extra></extra>'
                        ),
                        customdata=np.column_stack([accepted_sells['body_pips'].values])
                    ),
                    row=1, col=1
                )


        # Optimized layout
        tz_label = f" ({self.display_timezone})" if self.display_timezone != "UTC" else ""
        fig.update_layout(
            title=f"{instrument} Refined Strategy V3{tz_label} {title_suffix}",
            height=height,
            width=width,
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=self.color_scheme["bg"],
            paper_bgcolor=self.color_scheme["bg"],
            font=dict(color=self.color_scheme["text"]),
            margin=dict(t=80, b=60, l=60, r=60)
        )


        # CRITICAL FIX: Eliminate EXACT 48-hour weekend gap (Friday 17:00 EST → Sunday 17:00 EST)
        # Winter (EST = UTC-5): Friday 22:00 UTC → Sunday 22:00 UTC
        # Summer (EDT = UTC-4): Friday 21:00 UTC → Sunday 21:00 UTC
        # Using winter bounds (22:00 UTC) works year-round (over-hides 1 hour in summer, but that hour is non-trading)
        # fig.update_xaxes(
        #     rangebreaks=[
        #         dict(bounds=[22, 24], pattern="day of week"),  # Hide Friday 22:00-24:00 UTC
        #         dict(bounds=[0, 22], pattern="day of week")    # Hide Saturday + Sunday 00:00-22:00 UTC
        #     ],
        #     row=1, col=1
        # )

        # Grid lines with timezone label
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.color_scheme["grid"],
            title_text=f"Time{tz_label}",
            rangeslider_visible=False,
            row=1, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.color_scheme["grid"],
            row=1, col=1
        )
        
        return fig


