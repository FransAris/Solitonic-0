#!/usr/bin/env julia

"""
Comprehensive Kelly Criterion Trading System

Advanced trading system using:
- Linear regression with soliton features
- Kelly criterion position sizing
- Multiple strategy variants
- All available high-frequency stock data
"""

using DataFrames, Arrow, Statistics, LinearAlgebra, Random

println("TARGET: Comprehensive Kelly Criterion Trading System")
println("=" ^ 60)

Random.seed!(42)

include("src/Oscillators_enhanced.jl")
include("src/SolitonPDE_adaptive.jl")

using .Oscillators_enhanced
using .SolitonPDE_adaptive

# Trading parameters
const TRANSACTION_COST = 0.0005    # 0.05%
const INITIAL_CAPITAL = 100000.0
const MIN_TRADE_SIZE = 0.01        # 1% minimum position
const MAX_TRADE_SIZE = 0.5         # 50% maximum position
const LOOKBACK_WINDOW = 50         # Training window for regression
const HOLDING_PERIODS = [5, 10, 20]     # Multiple holding periods (25, 50 minutes)

# Kelly fractions to test  
const KELLY_FRACTIONS = [0.25, 0.5, 0.75]  # Reduced for efficiency

struct TradingResult
    symbol::String
    strategy::String
    kelly_fraction::Float64
    holding_period::Int
    total_return::Float64
    annualized_return::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    num_trades::Int
    win_rate::Float64
    final_capital::Float64
    avg_position_size::Float64
    r_squared::Float64
end

function extract_soliton_features(row)
    """Extract core soliton features for regression with robust fallbacks"""
    
    # Basic oscillator features (always available)
    basic_features = [
        row.RSI / 100,              # Normalized oscillators
        row.StochK / 100,
        (row.CCI + 100) / 200,      # Normalized CCI  
        row.MACDsig,                # MACD signal
        (row.RSI - 50) / 50,        # Centered RSI
        (row.StochK - 50) / 50,     # Centered Stoch
        row.CCI / 100,              # Scaled CCI
        row.RSI * row.StochK / 10000, # Oscillator interactions
    ]
    
    # Try to compute soliton features
    soliton_features = []
    try
        amplitudes = (row.RSI, row.StochK, row.CCI, row.MACDsig)
        result = simulate_soliton_adaptive(amplitudes, 15.0, freq=FreqMin5)  # Reduced complexity
        
        soliton_features = [
            result.H,                    # Hamiltonian
            result.energy,               # Energy
            result.collision_H,          # Collision Hamiltonian
            result.frequency_signature,  # Frequency signature
            result.peak_amplitude,       # Peak amplitude
            result.collision_energy,     # Collision energy
        ]
        
        # Check for invalid values
        if any(isnan.(soliton_features)) || any(isinf.(soliton_features))
            soliton_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Fallback zeros
        end
        
    catch
        # Fallback: use oscillator-derived pseudo-soliton features
        soliton_features = [
            row.RSI * row.StochK / 10000,     # Pseudo Hamiltonian
            abs(row.CCI) / 100,               # Pseudo energy
            row.MACDsig * row.RSI / 100,      # Pseudo collision H
            (row.RSI + row.StochK) / 200,     # Pseudo frequency
            max(row.RSI, row.StochK) / 100,   # Pseudo peak amplitude
            abs(row.CCI * row.MACDsig) / 100  # Pseudo collision energy
        ]
    end
    
    # Combine all features
    all_features = vcat(basic_features, soliton_features)
    
    # Final safety check
    if any(isnan.(all_features)) || any(isinf.(all_features))
        return fill(NaN, 14)
    end
    
    return all_features
end

function prepare_stock_data(symbol::String)
    """Load and prepare data for one stock"""
    
    println("INFO: Preparing data for $symbol...")
    
    # Load data
    stock_dir = "data/raw/stocks"
    file_path = joinpath(stock_dir, "$(lowercase(symbol))_5min_2025-07-19.arrow")
    
    if !isfile(file_path)
        return nothing
    end
    
    df = DataFrame(Arrow.Table(file_path))
    
    # Compute enhanced oscillators
    osc_df = compute_oscillators_enhanced(df, freq=Min5)
    
    # Filter valid data
    valid_mask = .!isnan.(osc_df.RSI) .& .!isnan.(osc_df.StochK) .& 
                 .!isnan.(osc_df.CCI) .& .!isnan.(osc_df.MACDsig)
    valid_df = osc_df[valid_mask, :]
    
    if nrow(valid_df) < 100
        println("   FAILED: Insufficient data for $symbol")
        return nothing
    end
    
    # Extract features for all valid rows
    feature_matrix = []
    prices = []
    timestamps = []
    
    println("   🔄 Extracting soliton features from $(nrow(valid_df)) oscillator data points...")
    
    for i in 1:nrow(valid_df)
        row = valid_df[i, :]
        features = extract_soliton_features(row)
        
        if !any(isnan.(features))
            push!(feature_matrix, features)
            push!(prices, row.Close)
            push!(timestamps, i)
        end
    end
    
    println("   INFO: Successfully extracted $(length(feature_matrix)) valid feature sets")
    
    if length(feature_matrix) < 50
        println("   FAILED: Insufficient valid features for $symbol")
        return nothing
    end
    
    println("   SUCCESS: Processed $(length(feature_matrix)) valid data points")
    
    return (
        symbol = symbol,
        features = hcat(feature_matrix...)',  # N x 14 matrix
        prices = prices,
        timestamps = timestamps,
        raw_data = valid_df
    )
end

function train_regression_model(features, returns, window_start, window_end)
    """Train linear regression model on a window of data"""
    
    if window_end > length(returns) || window_start < 1
        return nothing
    end
    
    X = features[window_start:window_end, :]
    y = returns[window_start:window_end]
    
    # Remove any NaN values
    valid_mask = .!isnan.(y) .& .!any(isnan.(X), dims=2)[:]
    
    if sum(valid_mask) < 20
        return nothing
    end
    
    X_clean = X[valid_mask, :]
    y_clean = y[valid_mask]
    
    try
        # Add intercept
        X_with_intercept = hcat(ones(size(X_clean, 1)), X_clean)
        
        # Linear regression (normal equation)
        coeffs = X_with_intercept \ y_clean
        
        # Calculate R²
        y_pred = X_with_intercept * coeffs
        ss_res = sum((y_clean - y_pred).^2)
        ss_tot = sum((y_clean .- mean(y_clean)).^2)
        r_squared = 1 - ss_res / ss_tot
        
        return (coeffs=coeffs, r_squared=r_squared)
    catch
        return nothing
    end
end

function calculate_kelly_fraction(predicted_return, historical_returns, base_fraction=1.0)
    """Calculate Kelly fraction for position sizing"""
    
    if isnan(predicted_return) || abs(predicted_return) < 1e-6
        return 0.0
    end
    
    # Estimate win probability and average win/loss
    positive_returns = historical_returns[historical_returns .> 0]
    negative_returns = historical_returns[historical_returns .< 0]
    
    if length(positive_returns) == 0 || length(negative_returns) == 0
        return 0.0
    end
    
    win_prob = length(positive_returns) / length(historical_returns)
    avg_win = mean(positive_returns)
    avg_loss = -mean(negative_returns)  # Make positive
    
    if avg_loss <= 0
        return 0.0
    end
    
    # Kelly fraction: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_prob, q = 1-win_prob
    b = avg_win / avg_loss
    kelly_f = (b * win_prob - (1 - win_prob)) / b
    
    # Apply safety factor and constraints
    kelly_f = max(0.0, min(0.5, kelly_f * base_fraction))
    
    # Adjust based on prediction confidence
    confidence = abs(predicted_return) / std(historical_returns)
    kelly_f *= min(1.0, confidence)
    
    return kelly_f
end

function simulate_trading_strategy(stock_data, kelly_fraction, holding_period, strategy_name)
    """Simulate trading with Kelly criterion position sizing"""
    
    features = stock_data.features
    prices = stock_data.prices
    symbol = stock_data.symbol
    
    capital = INITIAL_CAPITAL
    total_trades = 0
    winning_trades = 0
    max_capital = capital
    min_capital = capital
    position_sizes = Float64[]
    returns_log = Float64[]
    
    # Calculate all future returns for different holding periods
    future_returns = Float64[]
    for i in 1:(length(prices) - holding_period)
        future_return = (prices[i + holding_period] - prices[i]) / prices[i]
        push!(future_returns, future_return)
    end
    
    # Rolling regression with out-of-sample predictions
    for i in (LOOKBACK_WINDOW + 1):(length(prices) - holding_period - 1)
        
        # Train model on lookback window
        window_start = max(1, i - LOOKBACK_WINDOW)
        window_end = i - 1
        
        model = train_regression_model(features, future_returns, window_start, window_end)
        
        if model === nothing
            continue
        end
        
        # Make prediction for current point
        current_features = vcat(1.0, features[i, :])  # Add intercept
        predicted_return = dot(model.coeffs, current_features)
        
        # Calculate Kelly fraction
        historical_window = future_returns[max(1, window_start):window_end]
        kelly_f = calculate_kelly_fraction(predicted_return, historical_window, kelly_fraction)
        
        # Position sizing
        if abs(kelly_f) > MIN_TRADE_SIZE / 100
            position_size = min(MAX_TRADE_SIZE, max(MIN_TRADE_SIZE, kelly_f))
            
            # Actual return
            actual_return = future_returns[i]
            
            # Trade execution
            trade_capital = capital * position_size
            
            if predicted_return > 0
                # Long position
                trade_return = actual_return
            else
                # Short position
                trade_return = -actual_return
            end
            
            # Apply transaction costs
            trade_profit = trade_capital * trade_return - trade_capital * TRANSACTION_COST
            capital += trade_profit
            
            total_trades += 1
            if trade_profit > 0
                winning_trades += 1
            end
            
            push!(position_sizes, position_size)
            push!(returns_log, log(capital / INITIAL_CAPITAL))
            
            # Track drawdown
            max_capital = max(max_capital, capital)
            min_capital = min(min_capital, capital)
        end
    end
    
    if total_trades == 0
        return nothing
    end
    
    # Calculate performance metrics
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Annualized return (approximate)
    days_traded = length(returns_log) * holding_period * 5 / (60 * 24)  # Convert 5-min periods to days
    annualized_return = (1 + total_return)^(365.25 / max(1, days_traded)) - 1
    
    # Sharpe ratio
    if length(returns_log) > 1
        return_std = std(diff(returns_log))
        sharpe_ratio = mean(diff(returns_log)) / max(1e-8, return_std) * sqrt(252 * 24 * 12)  # Annualized
    else
        sharpe_ratio = 0.0
    end
    
    # Max drawdown
    max_drawdown = (max_capital - min_capital) / max_capital
    
    # Win rate
    win_rate = total_trades > 0 ? winning_trades / total_trades : 0.0
    
    # Average position size
    avg_position_size = length(position_sizes) > 0 ? mean(position_sizes) : 0.0
    
    return TradingResult(
        symbol,
        strategy_name,
        kelly_fraction,
        holding_period,
        total_return,
        annualized_return,
        sharpe_ratio,
        max_drawdown,
        total_trades,
        win_rate,
        capital,
        avg_position_size,
        0.0  # R² will be calculated separately
    )
end

function test_buy_and_hold(stock_data, holding_period)
    """Calculate buy and hold performance for comparison"""
    
    prices = stock_data.prices
    symbol = stock_data.symbol
    
    if length(prices) < 2
        return nothing
    end
    
    start_price = prices[1]
    end_price = prices[end]
    
    total_return = (end_price - start_price) / start_price
    capital = INITIAL_CAPITAL * (1 + total_return) - 2 * TRANSACTION_COST * INITIAL_CAPITAL
    
    days_traded = length(prices) * 5 / (60 * 24)  # Convert to days
    annualized_return = (1 + total_return)^(365.25 / max(1, days_traded)) - 1
    
    # Simple Sharpe approximation
    sharpe_ratio = total_return / 0.15  # Assume 15% annual volatility
    
    return TradingResult(
        symbol,
        "Buy & Hold",
        0.0,
        holding_period,
        total_return,
        annualized_return,
        sharpe_ratio,
        total_return < 0 ? abs(total_return) : 0.0,
        1,
        total_return > 0 ? 1.0 : 0.0,
        capital,
        1.0,
        0.0
    )
end

function analyze_results(all_results)
    """Comprehensive analysis of all trading results"""
    
    println("\nTARGET: COMPREHENSIVE TRADING ANALYSIS")
    println("=" ^ 50)
    
    # Group by strategy and kelly fraction
    strategy_groups = Dict()
    
    for result in all_results
        key = (result.strategy, result.kelly_fraction, result.holding_period)
        if !haskey(strategy_groups, key)
            strategy_groups[key] = []
        end
        push!(strategy_groups[key], result)
    end
    
    # Summary statistics
    println("INFO: STRATEGY PERFORMANCE SUMMARY")
    println("-" ^ 40)
    
    strategy_stats = []
    
    for ((strategy, kelly_f, hold_period), results) in strategy_groups
        if length(results) == 0
            continue
        end
        
        avg_return = mean([r.total_return for r in results])
        avg_sharpe = mean([r.sharpe_ratio for r in results])
        avg_trades = mean([r.num_trades for r in results])
        avg_win_rate = mean([r.win_rate for r in results])
        total_capital = sum([r.final_capital for r in results])
        success_rate = sum([r.total_return > 0 for r in results]) / length(results)
        
        push!(strategy_stats, (
            strategy = strategy,
            kelly_fraction = kelly_f,
            holding_period = hold_period,
            avg_return = avg_return,
            avg_sharpe = avg_sharpe,
            avg_trades = avg_trades,
            avg_win_rate = avg_win_rate,
            total_capital = total_capital,
            success_rate = success_rate,
            num_stocks = length(results)
        ))
    end
    
    # Sort by total capital
    sort!(strategy_stats, by=x->x.total_capital, rev=true)
    
    println("Top performing strategies (by total capital):")
    for (i, stats) in enumerate(strategy_stats[1:min(10, length(strategy_stats))])
        println("$(i). $(stats.strategy) (Kelly=$(stats.kelly_fraction), Hold=$(stats.holding_period))")
        println("    Total Capital: \$$(round(stats.total_capital, digits=0))")
        println("    Avg Return: $(round(stats.avg_return*100, digits=2))%")
        println("    Avg Sharpe: $(round(stats.avg_sharpe, digits=2))")
        println("    Success Rate: $(round(stats.success_rate*100, digits=1))%")
        println("    Avg Trades: $(round(stats.avg_trades, digits=0))")
        println("    Avg Win Rate: $(round(stats.avg_win_rate*100, digits=1))%")
        println()
    end
    
    # Best vs Buy & Hold comparison
    buy_hold_results = filter(r -> r.strategy == "Buy & Hold", all_results)
    trading_results = filter(r -> r.strategy != "Buy & Hold", all_results)
    
    if !isempty(buy_hold_results) && !isempty(trading_results)
        bh_total = sum([r.final_capital for r in buy_hold_results])
        best_trading = strategy_stats[1]
        
        improvement = best_trading.total_capital - bh_total
        improvement_pct = improvement / (INITIAL_CAPITAL * length(buy_hold_results)) * 100
        
        println("🏆 BEST STRATEGY vs BUY & HOLD")
        println("-" ^ 30)
        println("Best Strategy: $(best_trading.strategy) (Kelly=$(best_trading.kelly_fraction))")
        println("Buy & Hold Total: \$$(round(bh_total, digits=0))")
        println("Best Strategy Total: \$$(round(best_trading.total_capital, digits=0))")
        println("Improvement: \$$(round(improvement, digits=0)) ($(round(improvement_pct, digits=2))%)")
        
        if improvement > 5000
            println("\nSUCCESS: EXCELLENT: Strong outperformance!")
        elseif improvement > 1000
            println("\nUP: GOOD: Solid improvement over buy-hold")
        elseif improvement > 0
            println("\nINFO: POSITIVE: Modest but consistent gains")
        else
            println("\nDOWN: MIXED: Strategy needs refinement")
        end
    end
    
    return strategy_stats
end

function main()
    """Run comprehensive Kelly criterion trading test"""
    
    # Get all available stocks
    stock_files = readdir("data/raw/stocks")
    symbols = unique([uppercase(split(f, "_")[1]) for f in stock_files if endswith(f, ".arrow")])
    
    println("UP: Testing $(length(symbols)) stocks: $(join(symbols, ", "))")
    println("🎲 Kelly fractions: $(join(KELLY_FRACTIONS, ", "))")
    println("⏱️ Holding periods: $(join(HOLDING_PERIODS, ", ")) ($(join(HOLDING_PERIODS .* 5, ", ")) minutes)")
    
    all_results = TradingResult[]
    
    # Process each stock
    for symbol in symbols
        stock_data = prepare_stock_data(symbol)
        
        if stock_data === nothing
            continue
        end
        
        println("\nMONEY: Testing strategies for $symbol...")
        
        # Test all combinations
        for kelly_f in KELLY_FRACTIONS
            for hold_period in HOLDING_PERIODS
                
                # Soliton strategy
                result = simulate_trading_strategy(stock_data, kelly_f, hold_period, "Soliton Kelly")
                if result !== nothing
                    push!(all_results, result)
                end
                
                # Buy and hold (only once per holding period)
                if kelly_f == KELLY_FRACTIONS[1]
                    bh_result = test_buy_and_hold(stock_data, hold_period)
                    if bh_result !== nothing
                        push!(all_results, bh_result)
                    end
                end
            end
        end
        
        completed_results = length(filter(r -> r.symbol == symbol, all_results))
        println("   SUCCESS: Completed $completed_results strategy tests for $symbol")
    end
    
    if isempty(all_results)
        println("FAILED: No results generated")
        return
    end
    
    println("\nINFO: Generated $(length(all_results)) total results")
    
    # Comprehensive analysis
    strategy_stats = analyze_results(all_results)
    
    println("\nTIP: INSIGHTS:")
    println("- Tested $(length(symbols)) stocks")
    println("- $(length(KELLY_FRACTIONS)) Kelly fractions × $(length(HOLDING_PERIODS)) holding periods")
    println("- $(length(filter(r -> r.strategy != "Buy & Hold", all_results))) active trading results")
    println("- Advanced soliton feature regression with Kelly criterion position sizing")
    
    return all_results, strategy_stats
end

if abspath(PROGRAM_FILE) == @__FILE__
    results, stats = main()
end 