# Setup Instructions

## Prerequisites

1. Install Julia 1.10 or later from [julialang.org](https://julialang.org/downloads/)
2. Verify installation: `julia --version`

## Installation

1. Clone or navigate to the project directory
2. Install dependencies:
   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

## Data Setup

### Download Market Data

Download 30-year S&P 500 historical data:
```bash
julia --project=. scripts/download_spx_30years.jl
```

This downloads approximately 7,500 daily price records from 1995-2024.

### Build Soliton Features

Process the market data to create soliton features:
```bash
julia --project=. test_30year_pipeline_fixed.jl
```

This will:
- Compute technical oscillators (RSI, Stochastic, CCI, MACD)
- Run soliton PDE simulations for each trading day
- Generate features file: `data/processed/soliton_features_30years.arrow`

Expected runtime: 15-30 minutes depending on system performance.

## Running Analysis

### Main Analysis

Run the comprehensive trading system analysis:
```bash
julia --project=. comprehensive_ml_trading_system_enhanced.jl
```

This tests multiple ML algorithms and risk management strategies. Results are saved to `results/` directory.

Expected runtime: 30-60 minutes.

### Visualization

Generate charts for the best strategy:
```bash
julia --project=. champion_strategy_visualization.jl
```

This creates performance charts comparing the best soliton strategy vs S&P 500 buy-and-hold.

## Output Files

After running the analysis, you'll find:

- `results/comprehensive_ml_results_enhanced.csv` - Complete results
- `results/top_10_strategies.csv` - Best performing strategies
- `results/champion_strategy.csv` - Single best strategy details
- `results/summary_statistics.json` - Overall statistics
- `champion_soliton_strategy_performance.png` - Performance chart

## Testing

Test core functionality:
```bash
julia --project=. test/runtests.jl
```

Quick integration test:
```bash
julia --project=. run_smoke_test.jl
```

## Troubleshooting

### Common Issues

1. **Package installation fails**: Ensure you have internet connection and Julia 1.10+
2. **Data download fails**: Check internet connection, may need to retry
3. **Out of memory**: Reduce grid size in PDE simulations or use fewer features
4. **Slow performance**: Consider using fewer time horizons or algorithms

### System Requirements

- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB for data and results
- CPU: Multi-core recommended for faster processing

### Performance Tuning

To reduce runtime:
- Edit `comprehensive_ml_trading_system_enhanced.jl` to test fewer algorithms
- Reduce the number of time horizons (e.g., only test 5-day horizon)
- Use smaller feature sets

### Getting Help

- Check Julia documentation: [docs.julialang.org](https://docs.julialang.org/)
- Review package documentation in the source files
- Ensure all dependencies are correctly installed 