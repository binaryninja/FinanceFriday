import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OptionPosition:
    """Represents an option position"""
    strike: float
    expiration: datetime
    premium: float
    quantity: int = 1
    position_type: str = "short_call"

@dataclass
class StockPosition:
    """Represents a stock position"""
    symbol: str
    quantity: int
    entry_price: float

class DataIngestion:
    """Handles data ingestion for SPY stock and options data."""

    def __init__(self):
        """Initialize the DataIngestion class."""
        self.symbol = "SPY"
        self.ticker = yf.Ticker(self.symbol)

    def get_historical_data(self,
                          start_date: str,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical price data for SPY.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = self.ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError("No data retrieved")

            if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
                raise ValueError("Negative prices detected")

            logger.info(f"Successfully retrieved {len(df)} days of historical data")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def get_option_chain(self, date: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
        """
        Fetch option chain data for SPY with enhanced validation.
        Returns both the option chain and the selected expiration date.
        """
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

    def __init__(self, data_ingestion: DataIngestion):
        """Initialize the strategy engine."""
        self.data_ingestion = data_ingestion
        self.current_stock_position: Optional[StockPosition] = None
        self.current_option_position: Optional[OptionPosition] = None

    def calculate_option_metrics(self,
                               stock_price: float,
                               option_data: pd.DataFrame,
                               expiry_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate relevant metrics for option selection.
        """
        df = option_data.copy()
        logger.info(f"Initial options count: {len(df)}")

        df['mid_price'] = (df['bid'] + df['ask']) / 2
        now = pd.Timestamp.now()

        logger.info(f"Current time: {now}")
        logger.info(f"Selected expiration date: {expiry_date}")

        df['expiration'] = expiry_date
        df['days_to_expiry'] = (expiry_date - now).days

        logger.info(f"Days to expiry: {df['days_to_expiry'].iloc[0]}")

        logger.info(f"Before filtering - Bid range: {df['bid'].min():.2f} to {df['bid'].max():.2f}")
        logger.info(f"Before filtering - Ask range: {df['ask'].min():.2f} to {df['ask'].max():.2f}")
        logger.info(f"Before filtering - Strike range: {df['strike'].min():.2f} to {df['strike'].max():.2f}")

        df = df[
            (df['bid'] >= 0) &
            (df['ask'] > 0)
        ]

        logger.info(f"Options count after bid/ask filter: {len(df)}")
        logger.info(f"Current stock price: {stock_price:.2f}")

        df['premium_yield'] = df['mid_price'] / stock_price * 100
        df['moneyness'] = stock_price / df['strike']
        df['annualized_yield'] = (df['premium_yield'] * 365 / df['days_to_expiry'])

        if not df.empty:
            logger.info(f"Strike prices range: {df['strike'].min():.2f} to {df['strike'].max():.2f}")
            logger.info(f"Premium yields range: {df['premium_yield'].min():.2f}% to {df['premium_yield'].max():.2f}%")
            logger.info(f"Annualized yields range: {df['annualized_yield'].min():.2f}% to {df['annualized_yield'].max():.2f}%")
            logger.info(f"Implied volatility range: {df['impliedVolatility'].min():.2f} to {df['impliedVolatility'].max():.2f}")

        return df

    def select_strike(self,
                     stock_price: float,
                     option_data: pd.DataFrame,
                     expiry_date: pd.Timestamp,
                     target_delta: float = 0.25) -> Optional[dict]:
        """
        Select the best strike price based on strategy criteria.
        """
        try:
            df = self.calculate_option_metrics(stock_price, option_data, expiry_date)

            logger.info(f"Options available after metrics calculation: {len(df)}")

            df = df[
                (df['strike'] > stock_price) &
                (df['strike'] <= stock_price * 1.15) &
                (df['impliedVolatility'] > 0) &
                (df['impliedVolatility'] < 2.0)
            ]

            logger.info(f"Options available after strike/IV filtering: {len(df)}")

            if not df.empty:
                logger.info(f"Filtered strikes range: {df['strike'].min():.2f} to {df['strike'].max():.2f}")
                logger.info(f"Filtered IV range: {df['impliedVolatility'].min():.2f} to {df['impliedVolatility'].max():.2f}")

            if len(df) > 3:
                df = df[
                    (df['mid_price'] >= 0.1) &
                    (df['volume'].fillna(0) >= 0)
                ]
                logger.info(f"Options available after volume/premium filtering: {len(df)}")

            if df.empty:
                logger.warning("No suitable strikes found after filtering")
                return None

            df['total_score'] = (
                (df['annualized_yield'] / df['annualized_yield'].max() * 0.4) +
                (1 - abs(df['strike']/stock_price - 1.05) * 0.4) +
                (1 - (df['impliedVolatility'] - df['impliedVolatility'].mean()).abs() /
                 df['impliedVolatility'].std() * 0.2)
            )

            best_match = df.nlargest(1, 'total_score').iloc[0]

            selected_option = {
                'strike': float(best_match['strike']),
                'premium': float(best_match['mid_price']),
                'expiration': best_match['expiration'].strftime('%Y-%m-%d'),
                'implied_vol': float(best_match['impliedVolatility']),
                'delta': target_delta,
                'volume': int(best_match['volume']) if not pd.isna(best_match['volume']) else 0,
                'open_interest': int(best_match['openInterest']) if not pd.isna(best_match['openInterest']) else 0,
                'days_to_expiry': int(best_match['days_to_expiry']),
                'annualized_yield': float(best_match['annualized_yield'])
            }

            logger.info(f"Selected option: {selected_option}")
            return selected_option

        except Exception as e:
            logger.error(f"Error selecting strike: {str(e)}")
            logger.error(f"Available columns: {option_data.columns.tolist()}")
            return None

    def should_roll_option(self) -> bool:
        """Determine if current option position should be rolled."""
        if not self.current_option_position:
            return False

        try:
            current_price = self.data_ingestion.ticker.history(period='1d')['Close'].iloc[-1]
            days_to_expiry = (self.current_option_position.expiration - datetime.now()).days
            is_itm = current_price > self.current_option_position.strike

            return is_itm and days_to_expiry < 5

        except Exception as e:
            logger.error(f"Error checking roll condition: {str(e)}")
            return False

    def execute_covered_call_strategy(self) -> Dict[str, any]:
        """Execute one iteration of the covered call strategy."""
        try:
            current_price = self.data_ingestion.ticker.history(period='1d')['Close'].iloc[-1]
            option_chain, expiry_date = self.data_ingestion.get_option_chain()

            if self.should_roll_option():
                logger.info("Rolling position recommended")
                return {
                    'action': 'roll',
                    'current_position': self.current_option_position,
                    'stock_price': current_price
                }

            if not self.current_option_position:
                selected_option = self.select_strike(current_price, option_chain['calls'], expiry_date)

                if selected_option:
                    self.current_option_position = OptionPosition(
                        strike=selected_option['strike'],
                        expiration=pd.to_datetime(selected_option['expiration']).to_pydatetime(),
                        premium=selected_option['premium']
                    )

                    return {
                        'action': 'new_position',
                        'option': selected_option,
                        'stock_price': current_price
                    }

            return {
                'action': 'hold',
                'current_position': self.current_option_position,
                'stock_price': current_price
            }

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            return {'action': 'error', 'error': str(e)}

class DataStorage:
    """Handles storage and retrieval of market data and strategy results."""

    def __init__(self):
        """Initialize the DataStorage class."""
        self.price_data = None
        self.option_chains = {}
        self.strategy_results = []

    def store_price_data(self, df: pd.DataFrame) -> None:
        """Store price data in memory."""
        self.price_data = df
        logger.info(f"Stored {len(df)} rows of price data")

    def store_option_chain(self, date: str, chain: Dict[str, pd.DataFrame]) -> None:
        """Store option chain data in memory."""
        self.option_chains[date] = chain
        logger.info(f"Stored option chain for {date}")

    def store_strategy_result(self, result: Dict[str, any]) -> None:
        """Store strategy execution result."""
        self.strategy_results.append({
            'timestamp': datetime.now(),
            **result
        })
        logger.info(f"Stored strategy result: {result['action']}")

    def get_price_data(self) -> pd.DataFrame:
        """Retrieve stored price data."""
        if self.price_data is None:
            raise ValueError("No price data stored")
        return self.price_data

    def get_option_chain(self, date: str) -> Dict[str, pd.DataFrame]:
        """Retrieve stored option chain data."""
        if date not in self.option_chains:
            raise ValueError(f"No option chain data stored for {date}")
        return self.option_chains[date]

    def get_strategy_results(self) -> List[Dict[str, any]]:
        """Retrieve stored strategy results."""
        return self.strategy_results

def main():
    """Main function to demonstrate the complete system."""

    # Initialize components
    data_ingestion = DataIngestion()
    storage = DataStorage()
    strategy = StrategyEngine(data_ingestion)

    try:
        # Get historical price data (2 years)
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        price_data = data_ingestion.get_historical_data(start_date)
        storage.store_price_data(price_data)

        # Get current option chain
        option_chain = data_ingestion.get_option_chain()
        current_date = datetime.now().strftime('%Y-%m-%d')
        storage.store_option_chain(current_date, option_chain)

        # Execute strategy
        result = strategy.execute_covered_call_strategy()
        storage.store_strategy_result(result)

        # Display results
        print("\nLatest price data:")
        print(storage.get_price_data().tail())

        print("\nStrategy execution result:")
        print(result)

        if result['action'] == 'new_position':
            print("\nSelected option details:")
            option = result['option']
            print(f"Strike: ${option['strike']:.2f}")
            print(f"Premium: ${option['premium']:.2f}")
            print(f"Expiration: {option['expiration']}")
            print(f"Days to expiration: {(pd.to_datetime(option['expiration']) - pd.Timestamp.now()).days}")
            print(f"Delta: {option['delta']:.3f}")
            print(f"Implied Volatility: {option['implied_vol']:.2%}")
            print(f"Volume: {option['volume']:,}")
            print(f"Open Interest: {option['open_interest']:,}")
            print(f"Premium Yield: {(option['premium']/result['stock_price']*100):.2f}%")
            print(f"Annualized Yield: {(option['premium']/result['stock_price']*365/(pd.to_datetime(option['expiration']) - pd.Timestamp.now()).days*100):.2f}%")
        elif result['action'] == 'roll':
            print("\nRoll recommendation:")
            print(f"Current position strike: ${result['current_position'].strike:.2f}")
            print(f"Current position expiration: {result['current_position'].expiration.strftime('%Y-%m-%d')}")
            print(f"Current stock price: ${result['stock_price']:.2f}")

        # Additional analysis
        print("\nMarket Overview:")
        current_price = result['stock_price']
        historical_volatility = price_data['Close'].pct_change().std() * np.sqrt(252) * 100
        print(f"Current SPY Price: ${current_price:.2f}")
        print(f"20-day Historical Volatility: {historical_volatility:.1f}%")

        if len(storage.get_strategy_results()) > 1:
            print("\nStrategy History:")
            for past_result in storage.get_strategy_results()[:-1]:  # Exclude current result
                print(f"Date: {past_result['timestamp'].strftime('%Y-%m-%d')}")
                print(f"Action: {past_result['action']}")
                if 'option' in past_result:
                    print(f"Strike: ${past_result['option']['strike']:.2f}")
                    print(f"Premium: ${past_result['option']['premium']:.2f}")
                print("---")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
