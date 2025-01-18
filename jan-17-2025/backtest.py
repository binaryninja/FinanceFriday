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

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrendDirection(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"

@dataclass
class StrategyParameters:
    """Configuration parameters for the covered call strategy."""
    target_delta: float = 0.25
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 45
    min_premium: float = 0.10
    max_strike_percent: float = 1.15
    roll_dte_threshold: int = 5
    min_volume: int = 100
    min_open_interest: int = 500
    max_positions: int = 1
    profit_target_percent: float = 0.50  # Close at 50% profit
    stop_loss_percent: float = 0.200     # Stop loss at 200% of premium

@dataclass
class Config:
    """Global configuration settings."""
    symbol: str
    data_source: str = 'yfinance'
    logging_level: str = 'INFO'
    database_url: Optional[str] = None
    max_positions: int = 1
    risk_parameters: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'data_source': self.data_source,
            'logging_level': self.logging_level,
            'max_positions': self.max_positions,
            'risk_parameters': self.risk_parameters
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Config':
        return cls(**data)

@dataclass
class OptionPosition:
    """Represents an option position"""
    strike: float
    expiration: datetime
    premium: float
    entry_date: datetime = field(default_factory=datetime.now)
    quantity: int = 1
    position_type: str = "short_call"

    def days_to_expiry(self) -> int:
        return (self.expiration - datetime.now()).days

    def is_near_expiry(self, threshold: int = 5) -> bool:
        return self.days_to_expiry() <= threshold

@dataclass
class Position:
    """Represents a complete covered call position"""
    entry_date: datetime
    entry_stock_price: float
    stock_quantity: int
    option_position: Optional[OptionPosition]
    total_premium_collected: float = 0
    assignments: int = 0
    rolls: int = 0

    def calculate_current_value(self, current_stock_price: float) -> float:
        stock_value = self.stock_quantity * current_stock_price
        if self.option_position:
            # Simple approximation of option value, would need Black-Scholes for accuracy
            option_value = max(0, current_stock_price - self.option_position.strike)
            return stock_value - (option_value * 100 * self.option_position.quantity)
        return stock_value

class MarketEnvironment:
    """Analyzes market conditions and volatility regime."""

    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data
        self.update_market_conditions()

    def calculate_historical_volatility(self, window: int = 20) -> float:
        """Calculate rolling historical volatility."""
        returns = self.price_data['Close'].pct_change()
        hist_vol = returns.std() * np.sqrt(252) * 100
        return hist_vol

    def determine_trend(self, short_window: int = 20, long_window: int = 50) -> TrendDirection:
        """Determine market trend using moving averages."""
        short_ma = self.price_data['Close'].rolling(window=short_window).mean()
        long_ma = self.price_data['Close'].rolling(window=long_window).mean()

        latest_short = short_ma.iloc[-1]
        latest_long = long_ma.iloc[-1]

        if latest_short > latest_long * 1.02:
            return TrendDirection.UPTREND
        elif latest_short < latest_long * 0.98:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def determine_volatility_regime(self) -> VolatilityRegime:
        """Categorize current volatility regime."""
        hist_vol = self.calculate_historical_volatility()

        if hist_vol < 15:
            return VolatilityRegime.LOW
        elif hist_vol > 30:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.MEDIUM

    def calculate_relative_strength(self, window: int = 14) -> float:
        """Calculate RSI."""
        delta = self.price_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def update_market_conditions(self) -> Dict:
        """Update and return current market conditions."""
        self.conditions = {
            'historical_volatility': self.calculate_historical_volatility(),
            'trend': self.determine_trend(),
            'relative_strength': self.calculate_relative_strength(),
            'volatility_regime': self.determine_volatility_regime()
        }
        return self.conditions

class RiskManager:
    """Manages position risk and exposure."""

    def __init__(self, params: StrategyParameters):
        self.params = params

    def validate_new_position(self,
                            current_positions: List[Position],
                            proposed_position: Position,
                            market_conditions: Dict) -> Tuple[bool, str]:
        """Validate if new position meets risk parameters."""

        # Check maximum positions
        if len(current_positions) >= self.params.max_positions:
            return False, "Maximum positions reached"

        # Check market conditions
        if market_conditions['volatility_regime'] == VolatilityRegime.HIGH:
            return False, "Volatility too high for new positions"

        # Check position size
        position_value = proposed_position.stock_quantity * proposed_position.entry_stock_price
        if position_value > 100000:  # Example position size limit
            return False, "Position size too large"

        return True, "Position validated"

    def check_stop_loss(self, position: Position, current_stock_price: float) -> bool:
        """Check if position has hit stop loss levels."""
        if not position.option_position:
            return False

        current_value = position.calculate_current_value(current_stock_price)
        entry_value = position.stock_quantity * position.entry_stock_price
        loss_percent = (entry_value - current_value) / entry_value

        return loss_percent > self.params.stop_loss_percent

    def check_profit_target(self, position: Position, current_stock_price: float) -> bool:
        """Check if position has hit profit target."""
        if not position.option_position:
            return False

        current_value = position.calculate_current_value(current_stock_price)
        entry_value = position.stock_quantity * position.entry_stock_price
        profit_percent = (current_value - entry_value) / entry_value

        return profit_percent >= self.params.profit_target_percent

class PositionManager:
    """Manages active and closed positions."""

    def __init__(self):
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.performance_metrics = PerformanceMetrics()

    def add_position(self, position: Position):
        """Add new position."""
        self.active_positions.append(position)
        logger.info(f"Added new position: Strike {position.option_position.strike}")

    def close_position(self, position: Position, close_price: float):
        """Close position and update metrics."""
        self.active_positions.remove(position)
        self.closed_positions.append(position)
        self.performance_metrics.add_trade(position, close_price)
        logger.info(f"Closed position: Strike {position.option_position.strike}")

    def get_total_exposure(self) -> float:
        """Calculate total position exposure."""
        return sum(pos.stock_quantity * pos.entry_stock_price
                  for pos in self.active_positions)

    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary."""
        return self.performance_metrics.get_summary_statistics()

class PerformanceMetrics:
    """Tracks and calculates strategy performance metrics."""

    def __init__(self):
        self.trades: List[Dict] = []

    def add_trade(self, position: Position, close_price: float):
        """Add completed trade to history."""
        profit_loss = self.calculate_trade_pl(position, close_price)
        self.trades.append({
            'entry_date': position.entry_date,
            'exit_date': datetime.now(),
            'duration': (datetime.now() - position.entry_date).days,
            'profit_loss': profit_loss,
            'premium_collected': position.total_premium_collected
        })

    def calculate_trade_pl(self, position: Position, close_price: float) -> float:
        """Calculate P&L for a single trade."""
        stock_pl = (close_price - position.entry_stock_price) * position.stock_quantity
        option_pl = position.total_premium_collected
        return stock_pl + option_pl

    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics for all trades."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'average_duration': 0.0,
                'total_premium_collected': 0.0
            }

        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['profit_loss'] > 0)

        return {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'average_duration': np.mean([t['duration'] for t in self.trades]),
            'total_premium_collected': sum(t['premium_collected'] for t in self.trades),
            'total_profit_loss': sum(t['profit_loss'] for t in self.trades)
        }

class DataIngestion:
    """Handles data ingestion for stock and options data."""

    def __init__(self, config: Config):
        self.config = config
        self.symbol = config.symbol
        self.ticker = yf.Ticker(self.symbol)

    def get_historical_data(self,
                          start_date: str,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical price data."""
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = self.ticker.history(start=start_date, end=end_date)

            self.validate_price_data(df)

            logger.info(f"Successfully retrieved {len(df)} days of historical data")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def validate_price_data(self, df: pd.DataFrame) -> None:
        """Validate retrieved price data."""
        if df.empty:
            raise ValueError("No data retrieved")

        if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            raise ValueError("Negative prices detected")

    def get_option_chain(self, date: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
        """Fetch option chain data."""
        try:
            current_price = self.ticker.history(period='1d')['Close'].iloc[-1]
            logger.info(f"Current {self.symbol} price: ${current_price:.2f}")

            expirations = self.ticker.options

            if not expirations:
                raise ValueError("No option expiration dates available")

            valid_dates = [exp for exp in expirations
                          if (pd.to_datetime(exp) - pd.Timestamp.now()).days >= 7]

            if not valid_dates:
                raise ValueError("No valid expiration dates found")

            selected_date = date if date else valid_dates[0]
            expiry_date = pd.to_datetime(selected_date)

            logger.info(f"Selected expiration date: {selected_date} " +
                       f"({(expiry_date - pd.Timestamp.now()).days} days out)")

            chain = self.ticker.option_chain(selected_date)
            logger.info(f"Retrieved {len(chain.calls)} calls for {selected_date}")

            return {'calls': chain.calls, 'puts': chain.puts}, expiry_date

        except Exception as e:
            logger.error(f"Error fetching option chain: {str(e)}")
            raise

class StrategyEngine:
    """Implements the covered call strategy logic."""

    def __init__(self,
                 data_ingestion: DataIngestion,
                 params: StrategyParameters = None,
                 risk_manager: Optional[RiskManager] = None,
                 position_manager: Optional[PositionManager] = None):
        self.data_ingestion = data_ingestion
        self.params = params or StrategyParameters()
        self.risk_manager = risk_manager or RiskManager(self.params)
        self.position_manager = position_manager or PositionManager()
        self.market_environment = None

    def update_market_environment(self, price_data: pd.DataFrame):
        """Update market environment analysis."""
        self.market_environment = MarketEnvironment(price_data)



    def validate_option_data(self, df: pd.DataFrame) -> bool:
            """Validate option chain data quality."""
            required_columns = ['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return False

            # Check for data quality
            if df.empty:
                logger.error("Empty option chain data")
                return False

            if (df['bid'] < 0).any() or (df['ask'] < 0).any():
                logger.error("Negative bid/ask prices found")
                return False

            return True

    def calculate_option_metrics(self,
                               stock_price: float,
                               option_data: pd.DataFrame,
                               expiry_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate relevant metrics for option selection."""
        try:
            df = option_data.copy()
            logger.info(f"Initial options count: {len(df)}")

            if not self.validate_option_data(df):
                raise ValueError("Option data validation failed")

            df['mid_price'] = (df['bid'] + df['ask']) / 2
            now = pd.Timestamp.now()

            logger.info(f"Current time: {now}")
            logger.info(f"Selected expiration date: {expiry_date}")

            df['expiration'] = expiry_date
            df['days_to_expiry'] = (expiry_date - now).days

            logger.info(f"Days to expiry: {df['days_to_expiry'].iloc[0]}")

            # Enhanced metrics
            df['premium_yield'] = df['mid_price'] / stock_price * 100
            df['moneyness'] = stock_price / df['strike']
            df['annualized_yield'] = (df['premium_yield'] * 365 / df['days_to_expiry'])
            df['bid_ask_spread'] = (df['ask'] - df['bid']) / df['mid_price']
            df['volume_score'] = df['volume'] / df['volume'].mean()
            df['oi_score'] = df['openInterest'] / df['openInterest'].mean()

            # Log key statistics
            self.log_option_metrics(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating option metrics: {str(e)}")
            raise

    def log_option_metrics(self, df: pd.DataFrame):
        """Log key statistics about the option chain."""
        logger.info(f"Strike prices range: {df['strike'].min():.2f} to {df['strike'].max():.2f}")
        logger.info(f"Premium yields range: {df['premium_yield'].min():.2f}% to {df['premium_yield'].max():.2f}%")
        logger.info(f"Annualized yields range: {df['annualized_yield'].min():.2f}% to {df['annualized_yield'].max():.2f}%")
        logger.info(f"Implied volatility range: {df['impliedVolatility'].min():.2f} to {df['impliedVolatility'].max():.2f}")
        logger.info(f"Average bid-ask spread: {df['bid_ask_spread'].mean():.2%}")
        logger.info(f"Average volume: {df['volume'].mean():.0f}")

    def filter_options(self,
                      df: pd.DataFrame,
                      stock_price: float,
                      market_conditions: Dict) -> pd.DataFrame:
        """Apply filters based on strategy parameters and market conditions."""

        # Basic filters
        df = df[
            (df['strike'] > stock_price) &
            (df['strike'] <= stock_price * self.params.max_strike_percent) &
            (df['impliedVolatility'] > 0) &
            (df['impliedVolatility'] < 2.0) &
            (df['volume'] >= self.params.min_volume) &
            (df['openInterest'] >= self.params.min_open_interest) &
            (df['bid_ask_spread'] <= 0.10)  # Maximum 10% spread
        ]

        # Adjust filters based on market conditions
        if market_conditions['volatility_regime'] == VolatilityRegime.HIGH:
            # Be more conservative in high volatility
            df = df[df['strike'] >= stock_price * 1.02]  # Further OTM
        elif market_conditions['volatility_regime'] == VolatilityRegime.LOW:
            # Can be more aggressive in low volatility
            df = df[df['strike'] >= stock_price * 1.01]  # Closer to ATM

        return df

    def score_options(self,
                     df: pd.DataFrame,
                     stock_price: float,
                     market_conditions: Dict) -> pd.DataFrame:
        """Score options based on multiple criteria."""
        if df.empty:
            return df

        # Define weights based on market conditions
        weights = self.get_scoring_weights(market_conditions)

        # Calculate individual scores (0-1 scale)
        df['yield_score'] = (df['annualized_yield'] - df['annualized_yield'].min()) / \
                           (df['annualized_yield'].max() - df['annualized_yield'].min())

        df['strike_score'] = 1 - abs(df['strike']/stock_price - 1.05)

        df['liquidity_score'] = (
            (df['volume_score'] * 0.5) +
            (df['oi_score'] * 0.5)
        )

        df['iv_score'] = 1 - (
            (df['impliedVolatility'] - df['impliedVolatility'].mean()).abs() /
            df['impliedVolatility'].std()
        )

        # Calculate weighted total score
        df['total_score'] = (
            df['yield_score'] * weights['yield'] +
            df['strike_score'] * weights['strike'] +
            df['liquidity_score'] * weights['liquidity'] +
            df['iv_score'] * weights['iv']
        )

        return df

    def get_scoring_weights(self, market_conditions: Dict) -> Dict[str, float]:
        """Define scoring weights based on market conditions."""
        if market_conditions['volatility_regime'] == VolatilityRegime.HIGH:
            return {
                'yield': 0.2,      # Less emphasis on yield
                'strike': 0.4,     # More emphasis on strike selection
                'liquidity': 0.3,  # More emphasis on liquidity
                'iv': 0.1
            }
        elif market_conditions['volatility_regime'] == VolatilityRegime.LOW:
            return {
                'yield': 0.4,      # More emphasis on yield
                'strike': 0.3,     # Less emphasis on strike selection
                'liquidity': 0.2,  # Less emphasis on liquidity
                'iv': 0.1
            }
        else:
            return {
                'yield': 0.3,
                'strike': 0.3,
                'liquidity': 0.3,
                'iv': 0.1
            }

    def select_best_option(self, df: pd.DataFrame) -> Optional[Dict]:
        """Select the best option based on scoring."""
        if df.empty:
            logger.warning("No suitable options found after filtering")
            return None

        best_match = df.nlargest(1, 'total_score').iloc[0]

        return {
            'strike': float(best_match['strike']),
            'premium': float(best_match['mid_price']),
            'expiration': best_match['expiration'].strftime('%Y-%m-%d'),
            'implied_vol': float(best_match['impliedVolatility']),
            'delta': self.params.target_delta,
            'volume': int(best_match['volume']) if not pd.isna(best_match['volume']) else 0,
            'open_interest': int(best_match['openInterest']) if not pd.isna(best_match['openInterest']) else 0,
            'days_to_expiry': int(best_match['days_to_expiry']),
            'annualized_yield': float(best_match['annualized_yield']),
            'bid_ask_spread': float(best_match['bid_ask_spread']),
            'total_score': float(best_match['total_score'])
        }

    def should_roll_position(self,
                          position: Position,
                          current_price: float,
                          market_conditions: Dict) -> Tuple[bool, str]:
        """Determine if current option position should be rolled."""
        if not position.option_position:
            return False, "No option position to roll"

        try:
            days_to_expiry = position.option_position.days_to_expiry()
            is_itm = current_price > position.option_position.strike

            # Check various roll conditions
            if days_to_expiry <= self.params.roll_dte_threshold:
                return True, "Near expiration"

            if is_itm and days_to_expiry < 10:
                return True, "ITM and near expiration"

            if (market_conditions['volatility_regime'] == VolatilityRegime.HIGH and
                is_itm and days_to_expiry < 15):
                return True, "ITM in high volatility"

            return False, "No roll needed"

        except Exception as e:
            logger.error(f"Error checking roll condition: {str(e)}")
            return False, f"Error: {str(e)}"

    def execute_covered_call_strategy(self) -> Dict[str, Any]:
        """Execute one iteration of the covered call strategy."""
        try:
            # Get current market data
            current_price = self.data_ingestion.ticker.history(period='1d')['Close'].iloc[-1]
            option_chain, expiry_date = self.data_ingestion.get_option_chain()

            # Update market environment analysis
            price_data = self.data_ingestion.get_historical_data(
                start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            )
            self.update_market_environment(price_data)
            market_conditions = self.market_environment.conditions

            # Check existing positions
            for position in self.position_manager.active_positions:
                should_roll, reason = self.should_roll_position(
                    position, current_price, market_conditions
                )

                if should_roll:
                    return {
                        'action': 'roll',
                        'reason': reason,
                        'current_position': position,
                        'stock_price': current_price,
                        'market_conditions': market_conditions
                    }

                # Check stop loss and profit targets
                if self.risk_manager.check_stop_loss(position, current_price):
                    return {
                        'action': 'close',
                        'reason': 'stop_loss',
                        'position': position,
                        'stock_price': current_price
                    }

                if self.risk_manager.check_profit_target(position, current_price):
                    return {
                        'action': 'close',
                        'reason': 'profit_target',
                        'position': position,
                        'stock_price': current_price
                    }

            # Look for new position if we have capacity
            if len(self.position_manager.active_positions) < self.params.max_positions:
                # Calculate and filter options
                df = self.calculate_option_metrics(
                    current_price,
                    option_chain['calls'],
                    expiry_date
                )

                df = self.filter_options(df, current_price, market_conditions)
                df = self.score_options(df, current_price, market_conditions)

                selected_option = self.select_best_option(df)

                if selected_option:
                    # Validate with risk manager
                    proposed_position = Position(
                        entry_date=datetime.now(),
                        entry_stock_price=current_price,
                        stock_quantity=100,
                        option_position=OptionPosition(
                            strike=selected_option['strike'],
                            expiration=pd.to_datetime(selected_option['expiration']).to_pydatetime(),
                            premium=selected_option['premium']
                        )
                    )

                    is_valid, reason = self.risk_manager.validate_new_position(
                        self.position_manager.active_positions,
                        proposed_position,
                        market_conditions
                    )

                    if is_valid:
                        return {
                            'action': 'new_position',
                            'option': selected_option,
                            'stock_price': current_price,
                            'market_conditions': market_conditions
                        }
                    else:
                        return {
                            'action': 'hold',
                            'reason': f"Risk validation failed: {reason}",
                            'stock_price': current_price,
                            'market_conditions': market_conditions
                        }

            return {
                'action': 'hold',
                'reason': 'no_opportunities',
                'stock_price': current_price,
                'market_conditions': market_conditions
            }

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            return {'action': 'error', 'error': str(e)}

class Backtester:
    """Backtests the covered call strategy using historical data."""

    def __init__(self,
                 strategy_engine: StrategyEngine,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000.0):
        self.strategy = strategy_engine
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades_history = []
        self.daily_returns = []

    def run_backtest(self) -> Dict:
        """Execute backtest over the specified period."""
        try:
            # Get historical data
            historical_data = self.strategy.data_ingestion.get_historical_data(
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d')
            )

            # Iterate through each trading day
            for date in historical_data.index:
                try:
                    # Update current price data
                    current_price = historical_data.loc[date, 'Close']

                    # Get historical data up to current date for market analysis
                    lookback_start = date - pd.Timedelta(days=60)
                    historical_slice = historical_data[lookback_start:date]

                    # Update market environment
                    self.strategy.update_market_environment(historical_slice)

                    # Manage existing positions
                    self.manage_existing_positions(date, current_price)

                    # Look for new positions
                    self.check_new_positions(date, current_price)

                    # Record daily portfolio value
                    portfolio_value = self.calculate_portfolio_value(current_price)
                    daily_return = (portfolio_value - self.current_capital) / self.current_capital
                    self.daily_returns.append({
                        'date': date,
                        'portfolio_value': portfolio_value,
                        'return': daily_return
                    })
                    self.current_capital = portfolio_value

                except Exception as e:
                    logger.error(f"Error processing date {date}: {str(e)}")

            return self.generate_backtest_results()

        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise

    def manage_existing_positions(self, date: pd.Timestamp, current_price: float):
        """Manage existing positions for the current date."""
        for position in self.positions[:]:  # Use slice copy to allow modification during iteration
            # Check if position should be closed
            if self.strategy.risk_manager.check_stop_loss(position, current_price):
                self.close_position(position, date, current_price, 'stop_loss')
            elif self.strategy.risk_manager.check_profit_target(position, current_price):
                self.close_position(position, date, current_price, 'profit_target')
            elif position.option_position.expiration <= date:
                self.close_position(position, date, current_price, 'expiration')

    def check_new_positions(self, date: pd.Timestamp, current_price: float):
        """Check for and open new positions."""
        if len(self.positions) < self.strategy.params.max_positions:
            try:
                result = self.strategy.execute_covered_call_strategy()
                if result['action'] == 'new_position':
                    self.open_position(date, current_price, result['option'])
            except Exception as e:
                logger.error(f"Error checking new positions: {str(e)}")

    def open_position(self, date: pd.Timestamp, stock_price: float, option_data: Dict):
        """Open a new position."""
        position = Position(
            entry_date=date,
            entry_stock_price=stock_price,
            stock_quantity=100,  # Standard lot size
            option_position=OptionPosition(
                strike=option_data['strike'],
                expiration=pd.to_datetime(option_data['expiration']),
                premium=option_data['premium']
            )
        )
        self.positions.append(position)
        self.trades_history.append({
            'type': 'open',
            'date': date,
            'stock_price': stock_price,
            'strike': option_data['strike'],
            'premium': option_data['premium'],
            'expiration': option_data['expiration']
        })

    def close_position(self, position: Position, date: pd.Timestamp,
                      current_price: float, reason: str):
        """Close an existing position."""
        self.positions.remove(position)
        self.trades_history.append({
            'type': 'close',
            'date': date,
            'stock_price': current_price,
            'reason': reason,
            'profit_loss': self.calculate_position_pl(position, current_price)
        })

    def calculate_position_pl(self, position: Position, current_price: float) -> float:
        """Calculate P&L for a position."""
        stock_pl = (current_price - position.entry_stock_price) * position.stock_quantity
        option_pl = position.total_premium_collected
        return stock_pl + option_pl

    def calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.current_capital
        for position in self.positions:
            portfolio_value += self.calculate_position_pl(position, current_price)
        return portfolio_value

    def generate_backtest_results(self) -> Dict:
        """Generate comprehensive backtest results."""
        daily_returns_df = pd.DataFrame(self.daily_returns)

        if daily_returns_df.empty:
            return {"error": "No trading data available"}

        returns = daily_returns_df['return'].values

        results = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'annual_return': self.calculate_annual_return(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(daily_returns_df['portfolio_value']),
            'total_trades': len(self.trades_history),
            'win_rate': self.calculate_win_rate(),
            'average_trade_duration': self.calculate_average_trade_duration(),
            'profit_factor': self.calculate_profit_factor()
        }

        return results

    @staticmethod
    def calculate_annual_return(returns: np.array) -> float:
        return np.mean(returns) * 252  # Assuming 252 trading days per year

    @staticmethod
    def calculate_sharpe_ratio(returns: np.array, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        rolling_max = portfolio_values.expanding().max()
        drawdowns = portfolio_values / rolling_max - 1
        return drawdowns.min()

    def calculate_win_rate(self) -> float:
        closed_trades = [t for t in self.trades_history if t['type'] == 'close']
        if not closed_trades:
            return 0.0
        winning_trades = sum(1 for t in closed_trades if t['profit_loss'] > 0)
        return winning_trades / len(closed_trades)

    def calculate_average_trade_duration(self) -> float:
        # Implementation depends on your trade history structure
        pass

    def calculate_profit_factor(self) -> float:
        closed_trades = [t for t in self.trades_history if t['type'] == 'close']
        if not closed_trades:
            return 0.0

        gross_profits = sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] > 0)
        gross_losses = abs(sum(t['profit_loss'] for t in closed_trades if t['profit_loss'] < 0))

        return gross_profits / gross_losses if gross_losses != 0 else float('inf')

def main():
    """Main function to demonstrate the complete system."""
    try:
        # Initialize configuration
        config = Config(symbol="GOOGL")

        # Initialize components
        data_ingestion = DataIngestion(config)
        strategy_params = StrategyParameters()
        risk_manager = RiskManager(strategy_params)
        position_manager = PositionManager()

        strategy = StrategyEngine(
            data_ingestion=data_ingestion,
            params=strategy_params,
            risk_manager=risk_manager,
            position_manager=position_manager
        )

        # Setup backtest parameters
        start_date = "2024-01-01"  # Adjust as needed
        end_date = "2024-12-31"    # Adjust as needed
        initial_capital = 100000

        # Initialize backtester
        backtester = Backtester(
            strategy_engine=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )

        # Run backtest
        print(f"\nRunning backtest from {start_date} to {end_date}...")
        results = backtester.run_backtest()

        # Display backtest results
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
