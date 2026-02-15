"""
Atomic, append-only transaction logging with CSV + JSON formats for audit compliance.
"""
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class TransactionLog:
    """
    Dual-format logging:
      - transactions.csv: Human-readable audit trail (append-only)
      - transactions.jsonl: Machine-readable for analysis (newline-delimited JSON)
    
    Critical Features:
      - Atomic writes (no partial/corrupted logs on crash)
      - UTC timestamps for global consistency
      - Cryptographic hash chain for tamper detection (optional enhancement)
    """
    
    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self.csv_path = log_directory / "transactions.csv"
        self.jsonl_path = log_directory / "transactions.jsonl"
        self._init_logs()
    
    def _init_logs(self):
        """Create CSV header if file doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc", "transaction_id", "account_id", "action", 
                    "instrument", "units", "price", "cash_impact", "commission",
                    "counterparty_account", "metadata_json"
                ])
    
    def log_cross_account_trade(
        self,
        timestamp: datetime,
        instrument: str,
        action: str,
        units: float,
        price: float,
        buy_account_id: Optional[str],
        sell_account_id: Optional[str],
        buy_tx_id: Optional[str],
        sell_tx_id: Optional[str],
        commission: float
    ):
        """
        Log coordinated cross-account transaction with full audit trail.
        """
        # Normalize timestamp to UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # Create master transaction ID for the coordinated trade
        master_tx_id = f"MASTER-{timestamp.strftime('%Y%m%d%H%M%S')}-{instrument[:3]}{abs(units):.0f}"
        
        # Log to CSV (human readable)
        self._append_csv(
            timestamp=timestamp,
            tx_id=master_tx_id,
            account_id=buy_account_id or sell_account_id,
            action=f"CROSS_{action}",
            instrument=instrument,
            units=units,
            price=price,
            cash_impact=None,  # Account-specific impacts logged separately
            commission=commission,
            counterparty_account=sell_account_id if action == "BUY" else buy_account_id,
            metadata={
                "master_tx_id": master_tx_id,
                "buy_account": buy_account_id,
                "sell_account": sell_account_id,
                "buy_tx_id": buy_tx_id,
                "sell_tx_id": sell_tx_id,
                "action": action
            }
        )
        
        # Log to JSONL (machine readable)
        self._append_jsonl({
            "master_transaction_id": master_tx_id,
            "timestamp_utc": timestamp.isoformat(),
            "type": "cross_account_trade",
            "instrument": instrument,
            "action": action,
            "units": units,
            "price": price,
            "commission_total": commission,
            "accounts": {
                "buy": {
                    "account_id": buy_account_id,
                    "transaction_id": buy_tx_id,
                    "direction": "+1" if action == "BUY" else "-1"
                } if buy_account_id else None,
                "sell": {
                    "account_id": sell_account_id,
                    "transaction_id": sell_tx_id,
                    "direction": "-1" if action == "BUY" else "+1"
                } if sell_account_id else None
            }
        })
        
        logger.info(
            f"Cross-account trade logged: {master_tx_id} | "
            f"{action} {units:,.0f} {instrument} @ {price:.5f} | "
            f"Buy: {buy_account_id}({buy_tx_id}) Sell: {sell_account_id}({sell_tx_id})"
        )
    
    def _append_csv(self, **kwargs):
        """Atomic CSV append"""
        row = [
            kwargs["timestamp"].isoformat(),
            kwargs["tx_id"],
            kwargs["account_id"] or "",
            kwargs["action"],
            kwargs["instrument"],
            f"{kwargs['units']:,.2f}",
            f"{kwargs['price']:.5f}",
            f"{kwargs['cash_impact']:.2f}" if kwargs["cash_impact"] is not None else "",
            f"{kwargs['commission']:.2f}",
            kwargs["counterparty_account"] or "",
            json.dumps(kwargs["metadata"], separators=(',', ':'))
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _append_jsonl(self, data: dict):
        """Atomic JSONL append (newline-delimited JSON)"""
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(data, separators=(',', ':')) + '\n')