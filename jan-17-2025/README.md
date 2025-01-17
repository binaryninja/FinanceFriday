# Covered Call Strategy for SPY

This repository implements a systematic covered call options trading strategy for the SPY ETF, combining theoretical documentation with practical implementation.

## Repository Contents

### Documentation
1. [Complete Guide to Covered Calls](001-guide.md) - A comprehensive guide explaining:
   - What are covered calls and how they work
   - Strategy benefits and considerations
   - Step-by-step implementation guide
   - Best practices for successful covered call writing

2. [Project Implementation Details](002-project.md) - Technical documentation covering:
   - System architecture and components
   - Data ingestion and storage
   - Strategy engine design
   - Broker integration
   - Backtesting framework

3. [Development Stories](003-project-stories.md) - Agile development plan including:
   - Phase 1: MVP Implementation
   - Phase 2: Extended Functionality
   - Phase 3: Advanced Features
   - Detailed user stories and acceptance criteria

### Implementation

The repository includes a Python implementation of the covered call strategy with the following components:

- `main.py`: Core strategy implementation
- `requirements.txt`: Required Python packages

## Getting Started

### Prerequisites

1. Python 3.6 or later
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Setup & Execution

Clone the repository:

```bash
git clone https://github.com/dyngnosis/FinanceFriday.git
cd jan-17-2025/
```

Execute the code:
```
~ jan-17-2025$ python main.py
2025-01-17 13:57:34,914 - INFO - Successfully retrieved 502 days of historical data
2025-01-17 13:57:34,914 - INFO - Stored 502 rows of price data
2025-01-17 13:57:35,039 - INFO - Current SPY price: $598.32
2025-01-17 13:57:35,164 - INFO - Selected expiration date: 2025-01-31 (13 days out)
2025-01-17 13:57:35,279 - INFO - Retrieved 205 calls for 2025-01-31
2025-01-17 13:57:35,280 - INFO - Stored option chain for 2025-01-17
2025-01-17 13:57:35,361 - INFO - Current SPY price: $598.32
2025-01-17 13:57:35,367 - INFO - Selected expiration date: 2025-01-31 (13 days out)
2025-01-17 13:57:35,413 - INFO - Retrieved 205 calls for 2025-01-31
2025-01-17 13:57:35,413 - INFO - Initial options count: 205
2025-01-17 13:57:35,413 - INFO - Current time: 2025-01-17 13:57:35.413373
2025-01-17 13:57:35,413 - INFO - Selected expiration date: 2025-01-31 00:00:00
2025-01-17 13:57:35,413 - INFO - Days to expiry: 13
2025-01-17 13:57:35,413 - INFO - Before filtering - Bid range: 0.00 to 259.61
2025-01-17 13:57:35,413 - INFO - Before filtering - Ask range: 0.01 to 260.40
2025-01-17 13:57:35,414 - INFO - Before filtering - Strike range: 340.00 to 700.00
2025-01-17 13:57:35,414 - INFO - Options count after bid/ask filter: 205
2025-01-17 13:57:35,414 - INFO - Current stock price: 598.32
2025-01-17 13:57:35,415 - INFO - Strike prices range: 340.00 to 700.00
2025-01-17 13:57:35,415 - INFO - Premium yields range: 0.00% to 43.46%
2025-01-17 13:57:35,415 - INFO - Annualized yields range: 0.02% to 1220.11%
2025-01-17 13:57:35,415 - INFO - Implied volatility range: 0.00 to 1.63
2025-01-17 13:57:35,415 - INFO - Options available after metrics calculation: 205
2025-01-17 13:57:35,415 - INFO - Options available after strike/IV filtering: 39
2025-01-17 13:57:35,416 - INFO - Filtered strikes range: 599.00 to 685.00
2025-01-17 13:57:35,416 - INFO - Filtered IV range: 0.12 to 0.22
2025-01-17 13:57:35,416 - INFO - Options available after volume/premium filtering: 27
2025-01-17 13:57:35,418 - INFO - Selected option: {'strike': 610.0, 'premium': 2.05, 'expiration': '2025-01-31', 'implied_vol': 0.1276332354736328, 'delta': 0.25, 'volume': 6577, 'open_interest': 17310, 'days_to_expiry': 13, 'annualized_yield': 9.619884276492668}
2025-01-17 13:57:35,418 - INFO - Stored strategy result: new_position

Latest price data:
                                 Open        High         Low       Close    Volume  Dividends  Stock Splits  Capital Gains
Date                                                                                                                       
2025-01-10 00:00:00-05:00  585.880005  585.950012  578.549988  580.489990  73105000        0.0           0.0            0.0
2025-01-13 00:00:00-05:00  575.770020  581.750000  575.349976  581.390015  47910100        0.0           0.0            0.0
2025-01-14 00:00:00-05:00  584.359985  585.000000  578.349976  582.190002  48420600        0.0           0.0            0.0
2025-01-15 00:00:00-05:00  590.330017  593.940002  589.200012  592.780029  56900200        0.0           0.0            0.0
2025-01-16 00:00:00-05:00  594.169983  594.349976  590.929993  591.640015  43265800        0.0           0.0            0.0

Strategy execution result:
{'action': 'new_position', 'option': {'strike': 610.0, 'premium': 2.05, 'expiration': '2025-01-31', 'implied_vol': 0.1276332354736328, 'delta': 0.25, 'volume': 6577, 'open_interest': 17310, 'days_to_expiry': 13, 'annualized_yield': 9.619884276492668}, 'stock_price': np.float64(598.3200073242188)}

Selected option details:
Strike: $610.00
Premium: $2.05
Expiration: 2025-01-31
Days to expiration: 13
Delta: 0.250
Implied Volatility: 12.76%
Volume: 6,577
Open Interest: 17,310
Premium Yield: 0.34%
Annualized Yield: 9.62%

Market Overview:
Current SPY Price: $598.32
20-day Historical Volatility: 12.8%

```


## Features

- **Data Ingestion**: Fetches historical price and option chain data for SPY
- **Option Selection**: Analyzes options based on multiple metrics
- **Strategy Execution**: Implements covered call strategy with automatic decision-making
- **Market Analysis**: Provides relevant market insights and performance metrics

## Development Roadmap

The project is being developed in three phases:

1. **Phase 1 (MVP)**:
   - Basic data ingestion
   - Simple strike selection
   - Monthly covered call execution
   - Essential reporting

2. **Phase 2**:
   - Multiple strategy variations
   - Advanced rolling logic
   - Enhanced risk management
   - Improved analytics

3. **Phase 3**:
   - Multi-asset support
   - Machine learning integration
   - Advanced portfolio management
   - Real-time monitoring

See [Project Stories](003-project-stories.md) for detailed development plans.

## Architecture

For detailed technical architecture and implementation details, refer to the [Project Implementation Details](002-project.md) document.

## Strategy Guide

For a comprehensive understanding of covered calls and how to use this strategy effectively, see the [Complete Guide to Covered Calls](001-guide.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Acknowledgments

- Thanks to Royace for the inspiration and guidance
- Special thanks to the open-source community for providing essential tools and libraries

---

**Disclaimer**: This software is for educational purposes only. Trading options involves significant risk and may not be suitable for all investors. Always conduct your own due diligence and consider seeking advice from a qualified financial professional.
