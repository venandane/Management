"""
Cross-account portfolio manager for coordinated trading strategies.
Handles the "buy from Account A, sell from Account B" workflow with full audit trail.
"""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import logging
from trading_system.portfolio.account import Account
from trading_system.portfolio.transaction_log import TransactionLog
from trading_system.portfolio.currency_converter import CurrencyConverter
from trading_system.utils.project_paths import LOGS_DIR

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Orchestrates multi-account trading with explicit cross-account coordination.
    
    Critical Workflow for Your Requirement:
      ┌───────────────────────────────────────────────────────┐
      │  Strategy signals "BUY EUR_USD"                       │
      │    ↓                                                  │
      │  PortfolioManager.buy_from_account(                   │
      │      instrument="EUR_USD",                            │
      │      units=10000,                                     │
      │      price=1.0800,                                    │
      │      buy_account="USD_ACCOUNT",    ← Account A        │
      │      sell_account="EUR_ACCOUNT"    ← Account B (optional hedge)│
      │  )                                                    │
      └───────────────────────────────────────────────────────┘
    
    Why two accounts?
      - Account A (USD): Provides quote currency for buys
      - Account B (EUR): Could hold base currency for hedging or separate strategy
      - Prevents accidental netting of unrelated strategies
    """
    
    def __init__(
        self,
        accounts: Dict[str, Account],
        log_directory = LOGS_DIR / "portfolio_logs",
        state_file: str = "portfolio_state.json"
    ):
        self.accounts = accounts
        self.log_directory = Path(log_directory)
        self.state_file = Path(log_directory) / state_file
        self.transaction_log = TransactionLog(self.log_directory)
        self.converter = CurrencyConverter()
        
        # Ensure log directory exists
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
    
    def buy_from_account(
        self,
        instrument: str,
        units: float,
        price: float,
        buy_account_id: str,
        sell_account_id: Optional[str] = None,
        commission: float = 0.0
    ) -> Tuple[str, Optional[str]]:
        """
        Execute BUY order funded by specified account.
        
        Cross-Account Logic:
          - Primary: Deduct quote currency from buy_account_id (e.g., USD for EUR_USD)
          - Optional hedge: Simultaneously sell base currency from sell_account_id
            (e.g., sell EUR from EUR account to hedge exposure)
        
        Returns:
            (buy_tx_id, optional_sell_tx_id)
        """
        if buy_account_id not in self.accounts:
            raise ValueError(f"Buy account not found: {buy_account_id}")
        
        # Execute primary buy
        buy_tx_id = self.accounts[buy_account_id].execute_trade(
            instrument=instrument,
            direction=+1,  # Buy
            units=units,
            price=price,
            commission=commission,
            account_counterparty=sell_account_id
        )
        
        # Optional cross-account hedge (sell base currency from other account)
        sell_tx_id = None
        if sell_account_id:
            if sell_account_id not in self.accounts:
                raise ValueError(f"Sell account not found: {sell_account_id}")
            
            # Parse instrument to get base currency to sell
            base_ccy, quote_ccy = self._parse_instrument(instrument)
            
            # Verify sell account holds base currency
            if self.accounts[sell_account_id].base_currency != base_ccy:
                raise ValueError(
                    f"Sell account {sell_account_id} holds {self.accounts[sell_account_id].base_currency}, "
                    f"but instrument {instrument} requires {base_ccy} for hedging"
                )
            
            # Execute hedge sell (sell base currency equivalent)
            sell_tx_id = self.accounts[sell_account_id].execute_trade(
                instrument=instrument,
                direction=-1,  # Sell
                units=units,
                price=price,
                commission=commission,
                account_counterparty=buy_account_id
            )
            
            logger.info(
                f"Cross-account hedge: Sold {units:,.0f} {base_ccy} from {sell_account_id} "
                f"to hedge buy in {buy_account_id}"
            )
        
        # Log coordinated transaction
        self.transaction_log.log_cross_account_trade(
            timestamp=datetime.now(timezone.utc),
            instrument=instrument,
            action="BUY",
            units=units,
            price=price,
            buy_account_id=buy_account_id,
            sell_account_id=sell_account_id,
            buy_tx_id=buy_tx_id,
            sell_tx_id=sell_tx_id,
            commission=commission
        )
        
        # Persist state after trade
        self._save_state()
        
        return buy_tx_id, sell_tx_id
    
    def sell_from_account(
        self,
        instrument: str,
        units: float,
        price: float,
        sell_account_id: str,
        buy_account_id: Optional[str] = None,
        commission: float = 0.0
    ) -> Tuple[str, Optional[str]]:
        """
        Execute SELL order with proceeds to specified account.
        Mirror logic to buy_from_account() for symmetry.
        """
        if sell_account_id not in self.accounts:
            raise ValueError(f"Sell account not found: {sell_account_id}")
        
        # Execute primary sell
        sell_tx_id = self.accounts[sell_account_id].execute_trade(
            instrument=instrument,
            direction=-1,  # Sell
            units=units,
            price=price,
            commission=commission,
            account_counterparty=buy_account_id
        )
        
        # Optional cross-account hedge (buy base currency into other account)
        buy_tx_id = None
        if buy_account_id:
            if buy_account_id not in self.accounts:
                raise ValueError(f"Buy account not found: {buy_account_id}")
            
            base_ccy, quote_ccy = self._parse_instrument(instrument)
            
            if self.accounts[buy_account_id].base_currency != base_ccy:
                raise ValueError(
                    f"Buy account {buy_account_id} holds {self.accounts[buy_account_id].base_currency}, "
                    f"but instrument {instrument} requires {base_ccy}"
                )
            
            buy_tx_id = self.accounts[buy_account_id].execute_trade(
                instrument=instrument,
                direction=+1,
                units=units,
                price=price,
                commission=commission,
                account_counterparty=sell_account_id
            )
            
            logger.info(
                f"Cross-account hedge: Bought {units:,.0f} {base_ccy} into {buy_account_id} "
                f"to hedge sell from {sell_account_id}"
            )
        
        # Log coordinated transaction
        self.transaction_log.log_cross_account_trade(
            timestamp=datetime.now(timezone.utc),
            instrument=instrument,
            action="SELL",
            units=units,
            price=price,
            buy_account_id=buy_account_id,
            sell_account_id=sell_account_id,
            buy_tx_id=buy_tx_id,
            sell_tx_id=sell_tx_id,
            commission=commission
        )
        
        self._save_state()
        return sell_tx_id, buy_tx_id
    
    def get_account_summary(self, account_id: str, current_prices: Dict[str, float]) -> dict:
        """Get comprehensive account snapshot including P&L"""
        if account_id not in self.accounts:
            raise ValueError(f"Account not found: {account_id}")
        
        account = self.accounts[account_id]
        unrealized_pnl = account.get_unrealized_pnl(current_prices)
        
        return {
            "account_id": account_id,
            "account_name": account.account_name,
            "base_currency": account.base_currency,
            "cash_balance": account.current_balance,
            "unrealized_pnl": unrealized_pnl,
            "total_equity": account.get_total_equity(current_prices),
            "positions": {
                instr: {
                    "net_units": pos.net_units,
                    "average_price": pos.average_price,
                    "unrealized_pnl": pos.calculate_unrealized_pnl(current_prices.get(instr, 0))
                }
                for instr, pos in account.positions.items()
            }
        }
    
    def get_consolidated_pnl(self, current_prices: Dict[str, float], reporting_currency: str = "USD") -> dict:
        """Calculate consolidated P&L across all accounts in reporting currency"""
        total_cash = 0.0
        total_unrealized = 0.0
        
        for account_id, account in self.accounts.items():
            # Convert cash balance to reporting currency
            cash_in_reporting = self.converter.convert(
                amount=account.current_balance,
                from_currency=account.base_currency,
                to_currency=reporting_currency,
                rates=current_prices  # Use implied FX rates from instrument prices
            )
            total_cash += cash_in_reporting
            
            # Convert unrealized P&L
            unrealized = account.get_unrealized_pnl(current_prices)
            unrealized_in_reporting = self.converter.convert(
                amount=unrealized,
                from_currency=account.base_currency,  # P&L denominated in account currency
                to_currency=reporting_currency,
                rates=current_prices
            )
            total_unrealized += unrealized_in_reporting
        
        return {
            "reporting_currency": reporting_currency,
            "total_cash": total_cash,
            "total_unrealized_pnl": total_unrealized,
            "total_equity": total_cash + total_unrealized,
            "account_breakdown": {
                aid: self.get_account_summary(aid, current_prices)
                for aid in self.accounts.keys()
            }
        }
    
    def _parse_instrument(self, instrument: str) -> Tuple[str, str]:
        parts = instrument.split('_')
        return parts[0], parts[1]
    
    def _save_state(self):
        """Atomically persist portfolio state to JSON"""
        state = {
            "accounts": {aid: acct.to_dict() for aid, acct in self.accounts.items()},
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Write to temp file then atomically replace
        temp_path = self.state_file.with_suffix(".tmp")
        with open(temp_path, 'w') as f:
            json.dump(state, f, indent=2)
        temp_path.replace(self.state_file)
        
        logger.debug(f"Portfolio state saved to {self.state_file}")
    
    def _load_state(self):
        """Load portfolio state from JSON if exists"""
        if not self.state_file.exists():
            logger.info("No existing portfolio state found - starting fresh")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Reconstruct accounts
            for aid, acct_data in state["accounts"].items():
                self.accounts[aid] = Account.from_dict(acct_data)
            
            logger.info(f"Loaded portfolio state from {self.state_file} "
                       f"(last updated: {state['last_updated']})")
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            raise