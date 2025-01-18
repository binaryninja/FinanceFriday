import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
import math
from enum import Enum
import json
from pathlib import Path
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import threading
from queue import Queue
import time
import queue

from strategy_classes import (
    Position, OptionPosition, Config, DataIngestion,
    StrategyParameters, RiskManager, PositionManager,
    StrategyEngine, VolatilityRegime, TrendDirection
)

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = "covered_call_data.db"

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables for data persistence
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        timestamp DATETIME PRIMARY KEY,
        symbol TEXT,
        price FLOAT,
        volume INTEGER,
        high FLOAT,
        low FLOAT,
        historical_volatility FLOAT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_chains (
        timestamp DATETIME,
        symbol TEXT,
        expiration_date DATE,
        strike FLOAT,
        call_bid FLOAT,
        call_ask FLOAT,
        call_volume INTEGER,
        call_open_interest INTEGER,
        implied_volatility FLOAT,
        delta FLOAT,
        PRIMARY KEY (timestamp, symbol, expiration_date, strike)
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_decisions (
        timestamp DATETIME PRIMARY KEY,
        symbol TEXT,
        action TEXT,
        reason TEXT,
        market_price FLOAT,
        volatility_regime TEXT,
        trend TEXT,
        selected_strike FLOAT,
        selected_expiration DATE,
        premium FLOAT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        position_id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        entry_date DATETIME,
        exit_date DATETIME,
        entry_stock_price FLOAT,
        exit_stock_price FLOAT,
        stock_quantity INTEGER,
        option_strike FLOAT,
        option_expiration DATE,
        option_premium FLOAT,
        exit_reason TEXT,
        profit_loss FLOAT
    )""")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

class DataRecorder:
    """Handles persistence of market and strategy data."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.params = StrategyParameters()  # Initialize strategy parameters

    def save_market_data(self, symbol: str, price_data: pd.DataFrame, hist_vol: float):
        """Record market data to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            latest = price_data.iloc[-1]

            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO market_data
            (timestamp, symbol, price, volume, high, low, historical_volatility)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                symbol,
                float(latest['Close']),
                int(latest['Volume']),
                float(latest['High']),
                float(latest['Low']),
                hist_vol
            ))

            conn.commit()
            conn.close()
            logger.info(f"Saved market data for {symbol}")
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
            raise

    def save_option_chain(self, symbol: str, chain_data: Dict[str, pd.DataFrame],
                         expiry_date: pd.Timestamp):
        """Record option chain data to database."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Process call options
            calls = chain_data['calls']
            records = []
            timestamp = datetime.now()

            for _, row in calls.iterrows():
                records.append((
                    timestamp,
                    symbol,
                    expiry_date.date(),
                    float(row['strike']),
                    float(row['bid']),
                    float(row['ask']),
                    int(row['volume']) if not pd.isna(row['volume']) else 0,
                    int(row['openInterest']) if not pd.isna(row['openInterest']) else 0,
                    float(row['impliedVolatility']),
                    self.params.target_delta
                ))

            cursor = conn.cursor()
            cursor.executemany("""
            INSERT INTO option_chains
            (timestamp, symbol, expiration_date, strike, call_bid, call_ask,
             call_volume, call_open_interest, implied_volatility, delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            conn.commit()
            conn.close()
            logger.info(f"Saved option chain data for {symbol}")
        except Exception as e:
            logger.error(f"Error saving option chain data: {str(e)}")
            raise

    def save_strategy_decision(self, symbol: str, decision: Dict):
        """Record strategy decisions to database."""
        try:
            conn = sqlite3.connect(self.db_path)  # Use self.db_path directly, not self.recorder.db_path

            # Convert Timestamp to string for expiration date
            option_expiration = None
            if 'option' in decision and 'expiration' in decision['option']:
                option_expiration = pd.to_datetime(decision['option']['expiration']).strftime('%Y-%m-%d')

            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO strategy_decisions
            (timestamp, symbol, action, reason, market_price, volatility_regime,
             trend, selected_strike, selected_expiration, premium)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol,
                decision['action'],
                decision.get('reason', ''),
                float(decision['stock_price']),
                str(decision['market_conditions']['volatility_regime'].value),
                str(decision['market_conditions']['trend'].value),
                decision.get('option', {}).get('strike'),
                option_expiration,
                decision.get('option', {}).get('premium')
            ))

            conn.commit()
            conn.close()
            logger.info(f"Saved strategy decision for {symbol}")
        except Exception as e:
            logger.error(f"Error saving strategy decision: {str(e)}")
            raise

    def save_position(self, position: Position, exit_price: Optional[float] = None,
                     exit_reason: Optional[str] = None) -> int:
        """Record position data to database."""
        try:
            conn = sqlite3.connect(self.db_path)

            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO positions
            (symbol, entry_date, exit_date, entry_stock_price, exit_stock_price,
             stock_quantity, option_strike, option_expiration, option_premium,
             exit_reason, profit_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.entry_date,
                datetime.now() if exit_price else None,
                position.entry_stock_price,
                exit_price,
                position.stock_quantity,
                position.option_position.strike if position.option_position else None,
                position.option_position.expiration if position.option_position else None,
                position.option_position.premium if position.option_position else None,
                exit_reason,
                self.calculate_position_pl(position, exit_price) if exit_price else None
            ))

            position_id = cursor.lastrowid
            position.position_id = position_id

            conn.commit()
            conn.close()
            logger.info(f"Saved position data with ID: {position_id}")
            return position_id

        except Exception as e:
            logger.error(f"Error saving position data: {str(e)}")
            raise

    def calculate_position_pl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L for a position."""
        if not exit_price:
            return 0.0

        stock_pl = (exit_price - position.entry_stock_price) * position.stock_quantity
        option_pl = position.option_position.premium if position.option_position else 0
        return stock_pl + option_pl

class CoveredCallService:
    """Service wrapper for the covered call strategy."""

    def __init__(self, config: Config):
        self.config = config
        self.recorder = DataRecorder()
        self.strategy = self._init_strategy()
        self.scheduler = BackgroundScheduler()
        self.event_queue = Queue()
        self._stop_flag = False

    def _init_strategy(self) -> StrategyEngine:
        """Initialize strategy components."""
        data_ingestion = DataIngestion(self.config)
        strategy_params = StrategyParameters()
        risk_manager = RiskManager(strategy_params)
        position_manager = PositionManager()

        return StrategyEngine(
            data_ingestion=data_ingestion,
            params=strategy_params,
            risk_manager=risk_manager,
            position_manager=position_manager
        )

    def start(self):
        """Start the service."""
        init_database()

        # Schedule to run every minute for testing
        self.scheduler.add_job(
            self.check_market_data,
            'interval',
            minutes=1,
            id='market_data_job'
        )

        # Schedule end-of-day analysis
        self.scheduler.add_job(
            self.run_eod_analysis,
            CronTrigger(day_of_week='mon-fri', hour=16, minute=30),
            id='eod_analysis_job'
        )

        # Start the scheduler
        self.scheduler.start()

        # Start event processing thread
        self.event_thread = threading.Thread(target=self._process_events)
        self.event_thread.start()

        logger.info("Covered Call Service started successfully")

    def stop(self):
        """Stop the service."""
        self._stop_flag = True
        self.scheduler.shutdown()
        self.event_queue.put(None)  # Signal event thread to stop
        self.event_thread.join()
        logger.info("Covered Call Service stopped")

    def check_market_data(self):
        """Regular market data check and strategy execution."""
        try:
            logger.info("Starting market data check...")

            # Get current market data
            price_data = self.strategy.data_ingestion.get_historical_data(
                start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            )
            logger.info(f"Retrieved {len(price_data)} price records")

            # Update market environment
            self.strategy.update_market_environment(price_data)
            market_conditions = self.strategy.market_environment.conditions

            # Record market data
            self.recorder.save_market_data(
                self.config.symbol,
                price_data,
                market_conditions['historical_volatility']
            )

            # Get and record option chain data
            chain_data, expiry_date = self.strategy.data_ingestion.get_option_chain()
            self.recorder.save_option_chain(
                self.config.symbol,
                chain_data,
                expiry_date
            )

            # Execute strategy
            result = self.strategy.execute_covered_call_strategy()

            # Record strategy decision
            self.recorder.save_strategy_decision(
                self.config.symbol,
                result
            )

            # Queue any necessary actions
            self.event_queue.put(('strategy_result', result))

            logger.info("Completed market data check")

        except Exception as e:
            logger.error(f"Error in market data check: {str(e)}")

    def run_eod_analysis(self):
        """End-of-day analysis and reporting."""
        try:
            logger.info("Starting end-of-day analysis...")

            # Generate daily summary
            conn = sqlite3.connect(self.recorder.db_path)

            # Analyze today's data
            today = datetime.now().date()

            # Get day's market data
            market_data = pd.read_sql("""
                SELECT * FROM market_data
                WHERE date(timestamp) = date('now')
                AND symbol = ?
            """, conn, params=[self.config.symbol])

            # Get day's strategy decisions
            decisions = pd.read_sql("""
                SELECT * FROM strategy_decisions
                WHERE date(timestamp) = date('now')
                AND symbol = ?
            """, conn, params=[self.config.symbol])

            # Generate and log summary
            summary = {
                'date': today,
                'symbol': self.config.symbol,
                'price_range': f"{market_data['low'].min():.2f} - {market_data['high'].max():.2f}",
                'total_decisions': len(decisions),
                'actions_taken': decisions['action'].value_counts().to_dict()
            }

            logger.info(f"End of day summary: {json.dumps(summary, indent=2, default=str)}")

        except Exception as e:
            logger.error(f"Error in end-of-day analysis: {str(e)}")

    def _process_events(self):
        """Process events from the queue."""
        while not self._stop_flag:
            try:
                event = self.event_queue.get(timeout=1)
                if event is None:  # Stop signal
                    break

                event_type, data = event

                if event_type == 'strategy_result':
                    self._handle_strategy_result(data)

            except Exception as e:
                if not isinstance(e, queue.Empty):
                    logger.error(f"Error processing event: {str(e)}")

    def _handle_strategy_result(self, result: Dict):
        """Handle strategy execution results."""
        try:
            if result['action'] == 'new_position':
                # Record new position
                position = Position(
                    entry_date=datetime.now(),
                    entry_stock_price=result['stock_price'],
                    stock_quantity=100,
                    option_position=OptionPosition(
                        strike=result['option']['strike'],
                        expiration=pd.to_datetime(result['option']['expiration']).to_pydatetime(),
                        premium=result['option']['premium']
                    ),
                    symbol=self.config.symbol
                )
                self.recorder.save_position(position)

            elif result['action'] == 'close':
                # Record position closure
                self.recorder.save_position(
                    result['position'],
                    exit_price=result['stock_price'],
                    exit_reason=result['reason']
                )

        except Exception as e:
            logger.error(f"Error handling strategy result: {str(e)}")

def main():
    """Main function to run the service."""
    try:
        # Initialize configuration
        config = Config(symbol="QYLD")

        # Create and start service
        service = CoveredCallService(config)
        service.start()

        logger.info("Service started. Press Ctrl+C to stop...")

        # Run indefinitely until keyboard interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal, stopping service...")
            service.stop()
            logger.info("Service stopped")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
