from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import logging
from scipy.stats import norm

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
class OptionPosition:
    strike: float
    expiration: datetime
    premium: float

@dataclass
class Position:
    symbol: str
    entry_date: datetime
    entry_stock_price: float
    stock_quantity: int
    option_position: Optional[OptionPosition] = None
    position_id: Optional[int] = None

    def calculate_pl(self, current_price: float) -> float:
        """Calculate profit/loss for the position."""
        stock_pl = (current_price - self.entry_stock_price) * self.stock_quantity
        option_pl = self.option_position.premium * 100 if self.option_position else 0
        return stock_pl + option_pl

@dataclass
class Config:
    symbol: str
    lookback_days: int = 60
    min_option_dte: int = 20
    max_option_dte: int = 45
    target_delta: float = 0.3

class DataIngestion:
    def __init__(self, config: Config):
        self.config = config
        self.symbol = config.symbol
        self.ticker = yf.Ticker(self.symbol)  # Initialize ticker

    def calculate_delta(self, row: pd.Series, current_price: float,
                       expiry_date: pd.Timestamp, risk_free_rate: float = 0.05) -> float:
        """
        Calculate option delta using Black-Scholes formula
        """
        try:
            S = current_price  # Stock price
            K = row['strike']  # Strike price
            r = risk_free_rate  # Risk-free rate
            t = (expiry_date - pd.Timestamp.now()).days / 365.0  # Time to expiration in years
            sigma = row['impliedVolatility']  # Implied volatility

            d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
            delta = norm.cdf(d1)

            logger.debug(f"Delta calculation - Strike: {K}, IV: {sigma}, Time: {t}, Delta: {delta}")

            return delta
        except Exception as e:
            logger.error(f"Error calculating delta: {str(e)}")
            return 0.3  # Return default delta if calculation fails


    def get_historical_data(self, start_date: str) -> pd.DataFrame:
        """Fetch historical price data from yfinance."""
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(start=start_date)
        return data


    def get_option_chain(self, date: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
        """Fetch option chain data."""
        try:
            current_price = self.ticker.history(period='1d')['Close'].iloc[-1]
            logger.info(f"Current {self.symbol} price: ${current_price:.2f}")

            # Get available expiration dates
            expirations = self.ticker.options

            if not expirations:
                raise ValueError("No option expiration dates available")

            # Filter for valid dates
            valid_dates = []
            for exp in expirations:
                exp_date = pd.to_datetime(exp)
                if (exp_date - pd.Timestamp.now()).days >= 7:
                    valid_dates.append(exp)

            if not valid_dates:
                raise ValueError("No valid expiration dates found")

            # Select expiration date
            selected_date = date if date else valid_dates[0]
            expiry_date = pd.to_datetime(selected_date)

            logger.info(f"Selected expiration date: {selected_date}")

            # Get option chain for selected date
            chain = self.ticker.option_chain(selected_date)

            logger.debug("Options data structure:")
            logger.debug(f"Calls columns: {chain.calls.columns.tolist()}")
            logger.debug(f"Sample call option row:\n{chain.calls.iloc[0]}")

            # Add delta calculation
            calls = chain.calls.copy()
            calls['delta'] = calls.apply(
                lambda row: self.calculate_delta(
                    row,
                    current_price,
                    expiry_date=expiry_date  # Pass the expiration date here
                ),
                axis=1
            )

            # Filter calls based on current price
            calls = calls[
                (calls['strike'] >= current_price * 0.9) &
                (calls['strike'] <= current_price * 1.2)
            ]

            logger.info(f"Retrieved {len(calls)} filtered calls for {selected_date}")

            return {'calls': calls, 'puts': chain.puts}, expiry_date

        except Exception as e:
            logger.error(f"Error fetching option chain: {str(e)}")
            raise

@dataclass
class StrategyParameters:
    target_delta: float = 0.3
    min_premium_threshold: float = 0.002  # 0.2% minimum premium yield
    stop_loss_threshold: float = -0.05  # -5% total position value
    profit_target: float = 0.02  # 2% total position value
    position_size: int = 100  # Number of shares per position
    vol_lookback: int = 20  # Days for historical volatility calculation

class MarketEnvironment:
    def __init__(self):
        self.conditions = {
            'volatility_regime': VolatilityRegime.MEDIUM,
            'trend': TrendDirection.SIDEWAYS,
            'historical_volatility': 0.0
        }

    def update(self, price_data: pd.DataFrame):
        """Update market environment based on price data."""
        # Calculate historical volatility
        returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
        hist_vol = returns.std() * np.sqrt(252)

        # Determine volatility regime
        if hist_vol < 0.15:
            vol_regime = VolatilityRegime.LOW
        elif hist_vol > 0.25:
            vol_regime = VolatilityRegime.HIGH
        else:
            vol_regime = VolatilityRegime.MEDIUM

        # Determine trend using simple moving averages
        ma20 = price_data['Close'].rolling(20).mean()
        ma50 = price_data['Close'].rolling(50).mean()

        current_price = price_data['Close'].iloc[-1]
        current_ma20 = ma20.iloc[-1]
        current_ma50 = ma50.iloc[-1]

        logger.info(f"Trend analysis - Price: ${current_price:.2f}, "
                   f"MA20: ${current_ma20:.2f}, MA50: ${current_ma50:.2f}")

        if ma20.iloc[-1] > ma50.iloc[-1] and price_data['Close'].iloc[-1] > ma20.iloc[-1]:
            trend = TrendDirection.UPTREND
        elif ma20.iloc[-1] < ma50.iloc[-1] and price_data['Close'].iloc[-1] < ma20.iloc[-1]:
            trend = TrendDirection.DOWNTREND
        else:
            trend = TrendDirection.SIDEWAYS

        logger.info(f"Determined trend: {trend}")

        self.conditions.update({
            'volatility_regime': vol_regime,
            'trend': trend,
            'historical_volatility': hist_vol
        })

class RiskManager:
    def __init__(self, params: StrategyParameters):
        self.params = params

    def check_position(self, position: Position, current_price: float) -> tuple[bool, str]:
        """Check if position needs to be closed based on risk parameters."""
        pl_pct = position.calculate_pl(current_price) / (position.entry_stock_price * position.stock_quantity)

        if pl_pct <= self.params.stop_loss_threshold:
            return True, "stop_loss"
        elif pl_pct >= self.params.profit_target:
            return True, "profit_target"

        return False, ""

class PositionManager:
    def __init__(self):
        self.active_positions: Dict[str, Position] = {}

    def add_position(self, position: Position):
        """Add a new position."""
        self.active_positions[position.symbol] = position

    def remove_position(self, symbol: str):
        """Remove a position."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol if it exists."""
        return self.active_positions.get(symbol)

class StrategyEngine:
    def __init__(self, data_ingestion: DataIngestion, params: StrategyParameters,
                 risk_manager: RiskManager, position_manager: PositionManager):
        self.data_ingestion = data_ingestion
        self.params = params
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.market_environment = MarketEnvironment()

    def update_market_environment(self, price_data: pd.DataFrame):
        """Update market environment conditions."""
        self.market_environment.update(price_data)

    def execute_covered_call_strategy(self) -> Dict[str, Any]:
        """Execute the covered call strategy logic."""
        current_position = self.position_manager.get_position(self.data_ingestion.symbol)
        latest_price = float(self.data_ingestion.get_historical_data(
            start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        ).iloc[-1]['Close'])

        logger.info(f"Checking strategy for {self.data_ingestion.symbol} at price ${latest_price:.2f}")
        logger.info(f"Current market conditions: {self.market_environment.conditions}")

        # Check existing position
        if current_position:
            logger.info("Evaluating existing position...")
            should_close, reason = self.risk_manager.check_position(current_position, latest_price)
            if should_close:
                logger.info(f"Closing position due to {reason}")
                self.position_manager.remove_position(current_position.symbol)
                return {
                    'action': 'close',
                    'position': current_position,
                    'stock_price': latest_price,
                    'reason': reason,
                    'market_conditions': self.market_environment.conditions
                }
        else:
            logger.info("No current position exists")

        # Check for new position
        if not current_position:
            logger.info(f"Evaluating new position opportunity. Trend: {self.market_environment.conditions['trend']}")

            if self.market_environment.conditions['trend'] != TrendDirection.DOWNTREND:
                logger.info("Market trend acceptable for new position")
                chain_data, expiry = self.data_ingestion.get_option_chain()
                calls = chain_data['calls']

                logger.info(f"Retrieved {len(calls)} calls before filtering")

                # Filter for appropriate strikes
                target_calls = calls[
                    (calls['delta'].between(0.25, 0.35)) &
                    (calls['volume'] > 0)
                ]

                logger.info(f"Found {len(target_calls)} calls after delta/volume filtering")

                if not target_calls.empty:
                    selected_call = target_calls.iloc[0]
                    premium_pct = selected_call['ask'] / latest_price

                    logger.info(f"Best candidate: Strike=${selected_call['strike']:.2f}, "
                              f"Premium %={premium_pct*100:.2f}%, "
                              f"Delta={selected_call['delta']:.2f}")
                    logger.info(f"Minimum premium threshold: {self.params.min_premium_threshold*100:.2f}%")

                    if premium_pct >= self.params.min_premium_threshold:
                        logger.info("Taking new position")
                        return {
                            'action': 'new_position',
                            'stock_price': latest_price,
                            'option': {
                                'strike': float(selected_call['strike']),
                                'expiration': expiry,
                                'premium': float(selected_call['ask'])
                            },
                            'market_conditions': self.market_environment.conditions
                        }
                    else:
                        logger.info(f"Premium {premium_pct*100:.2f}% below threshold "
                                  f"{self.params.min_premium_threshold*100:.2f}%")
                else:
                    logger.info("No calls met the delta and volume criteria")
            else:
                logger.info("Market in downtrend, avoiding new positions")

        logger.info("No action taken, holding current position")
        return {
            'action': 'hold',
            'stock_price': latest_price,
            'market_conditions': self.market_environment.conditions
        }
