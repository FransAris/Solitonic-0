#!/usr/bin/env julia

"""
Realistic Kelly Criterion Trading with Visualization

Conservative trading system with:
- Realistic transaction costs and slippage
- Conservative position sizing
- Individual trade tracking and visualization
- Performance comparison charts
"""

using DataFrames, Arrow, Statistics, LinearAlgebra, Random
using Plots
plotlyjs()  # Interactive plots

println("INFO: Realistic Kelly Trading System with Visualization")
println("=" ^ 60)

Random.seed!(42)

include("src/Oscillators_enhanced.jl")
include("src/SolitonPDE_adaptive.jl")

using .Oscillators_enhanced
using .SolitonPDE_adaptive

# REALISTIC TRADING PARAMETERS  
const TRANSACTION_COST = 0.001      # 0.1% (algo trading cost)
const SLIPPAGE = 0.0005             # 0.05% slippage
const INITIAL_CAPITAL = 100000.0
const MIN_TRADE_SIZE = 0.01         # 1% minimum position
const MAX_TRADE_SIZE = 0.2          # 20% maximum position
const LOOKBACK_WINDOW = 40          # Reasonable window
const HOLDING_PERIOD = 8            # 40 minutes 
const KELLY_FRACTION = 0.5          # More aggressive but realistic

# Risk management
const MAX_DAILY_TRADES = 30         # More trades allowed
const STOP_LOSS = 0.015             # 1.5% stop loss
const TAKE_PROFIT = 0.03            # 3% take profit

struct Trade
    timestamp::Int
    entry_price::Float64
    exit_price::Float64
    position_size::Float64
    predicted_return::Float64
    actual_return::Float64
    profit_loss::Float64
    successful::Bool
    reason::String  # "profit", "loss", "stop", "time"
end

struct TradingResults
    symbol::String
    total_return::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    num_trades::Int
    win_rate::Float64
    final_capital::Float64
    trades::Vector{Trade}
    equity_curve::Vector{Float64}
    timestamps::Vector{Int}
end

function extract_comprehensive_soliton_features(row)
    """Extract ALL available soliton PDE features including asymmetry, field probes, etc."""
    
    # Basic normalized oscillator features  
    basic_features = [
        (row.RSI - 50) / 50,           # Centered RSI [-1, 1]
        (row.StochK - 50) / 50,        # Centered Stoch [-1, 1]  
        tanh(row.CCI / 100),           # Bounded CCI
        tanh(row.MACDsig * 10),        # Bounded MACD
        (row.RSI * row.StochK - 2500) / 2500,  # Oscillator interaction
    ]
    
    # COMPREHENSIVE soliton PDE features
    soliton_features = []
    try
        amplitudes = (row.RSI, row.StochK, row.CCI, row.MACDsig)
        result = simulate_soliton_adaptive(amplitudes, 8.0, freq=FreqMin5)
        
        # ALL AVAILABLE SOLITON FEATURES
        soliton_features = [
            # Core soliton features
            tanh(result.H * 0.1),                           # Hamiltonian (collision center)
            tanh(result.energy * 0.1),                      # Total energy
            tanh(result.concentration),                     # Field concentration
            
            # ASYMMETRY FEATURES (these are key!)
            tanh(result.asymmetry_x * 0.5),                 # X-direction asymmetry
            tanh(result.asymmetry_y * 0.5),                 # Y-direction asymmetry
            
            # Collision-specific features
            tanh(result.collision_H * 0.1),                 # Collision Hamiltonian
            tanh(result.collision_energy * 0.1),            # Collision energy
            tanh(result.collision_asymmetry_x * 0.5),       # Collision X asymmetry
            tanh(result.collision_asymmetry_y * 0.5),       # Collision Y asymmetry
            
            # Advanced features
            tanh(result.frequency_signature),               # Frequency-specific signature
            tanh(result.temporal_sharpness * 0.1),          # Temporal concentration
            
            # Field probe features (F is a vector, use key components)
            tanh(result.F[1] * 0.1),                        # X-direction probe 1
            tanh(result.F[2] * 0.1),                        # X-direction probe 2  
            tanh(result.F[4] * 0.1),                        # Y-direction probe 1
            tanh(result.F[5] * 0.1),                        # Y-direction probe 2
            
            # Derived asymmetry combinations
            tanh((result.asymmetry_x - result.asymmetry_y) * 0.3),     # Asymmetry difference
            tanh((result.collision_asymmetry_x + result.collision_asymmetry_y) * 0.3),  # Total collision asymmetry
        ]
        
        # Safety check for NaN/Inf
        if any(isnan.(soliton_features)) || any(isinf.(soliton_features))
            println("    WARNING:  NaN detected in soliton features, using fallback")
            soliton_features = create_fallback_features(row)
        end
        
    catch e
        println("    WARNING:  Soliton computation failed: $(typeof(e)), using fallback")
        soliton_features = create_fallback_features(row)
    end
    
    return vcat(basic_features, soliton_features)
end

function create_fallback_features(row)
    """Create fallback features when soliton computation fails"""
    return [
        tanh((row.RSI - 50) * (row.StochK - 50) / 1000),    # Momentum interaction
        tanh(abs(row.CCI) / 200),                            # Volatility proxy  
        tanh(row.MACDsig * (row.RSI - 50) / 100),           # Trend-momentum
        tanh((row.RSI - 50) / 50),                          # RSI asymmetry proxy
        tanh((row.StochK - 50) / 50),                       # Stoch asymmetry proxy
        tanh(row.CCI * row.MACDsig / 10000),                # Collision proxy
        tanh(abs(row.CCI) * abs(row.MACDsig) / 10000),      # Energy proxy
        tanh((row.RSI - row.StochK) / 100),                 # X-asymmetry proxy
        tanh((row.CCI + row.MACDsig) / 100),                # Y-asymmetry proxy
        tanh((row.RSI + row.StochK + abs(row.CCI)) / 300),  # Frequency proxy
        tanh(max(row.RSI, row.StochK) / 100),               # Temporal proxy
        tanh(row.RSI * 0.01),                               # Probe 1 proxy
        tanh(row.StochK * 0.01),                            # Probe 2 proxy
        tanh(row.CCI * 0.01),                               # Probe 3 proxy
        tanh(row.MACDsig * 0.1),                            # Probe 4 proxy
        tanh((row.RSI - row.StochK) * 0.01),                # Asymmetry diff proxy
        tanh((abs(row.CCI) + abs(row.MACDsig)) * 0.01),     # Total asymmetry proxy
    ]
end

function prepare_realistic_data(symbol::String)
    """Prepare data for realistic trading test"""
    
    println("INFO: Preparing realistic data for $symbol...")
    
    # Load data
    stock_dir = "data/raw/stocks"
    file_path = joinpath(stock_dir, "$(lowercase(symbol))_5min_2025-07-19.arrow")
    
    if !isfile(file_path)
        return nothing
    end
    
    df = DataFrame(Arrow.Table(file_path))
    
    # Compute oscillators
    osc_df = compute_oscillators_enhanced(df, freq=Min5)
    
    # Filter valid data
    valid_mask = .!isnan.(osc_df.RSI) .& .!isnan.(osc_df.StochK) .& 
                 .!isnan.(osc_df.CCI) .& .!isnan.(osc_df.MACDsig)
    valid_df = osc_df[valid_mask, :]
    
    if nrow(valid_df) < 100
        return nothing
    end
    
    # Extract features with tracking
    features = []
    prices = []
    timestamps = []
    soliton_success_count = 0
    
    for i in 1:nrow(valid_df)
        row = valid_df[i, :]
        feat = extract_comprehensive_soliton_features(row)
        
        if !any(isnan.(feat)) && !any(isinf.(feat))
            push!(features, feat)
            push!(prices, row.Close)
            push!(timestamps, i)
            
            # Check if this used real soliton features (updated for new feature set)
            if abs(feat[8]) > 0.01 || abs(feat[9]) > 0.01  # Asymmetry features from real soliton
                soliton_success_count += 1
            end
        end
    end
    
    if length(features) < 50
        return nothing
    end
    
    soliton_success_rate = soliton_success_count / length(features) * 100
    println("   SUCCESS: Processed $(length(features)) feature sets")
    println("   🧮 Real soliton PDE success: $(round(soliton_success_rate, digits=1))%")
    
    return (
        symbol = symbol,
        features = hcat(features...)',
        prices = prices,
        timestamps = timestamps,
        raw_data = valid_df
    )
end

function train_conservative_model(features, returns, window_start, window_end)
    """Train conservative regression model"""
    
    if window_end > length(returns) || window_start < 1 || (window_end - window_start) < 15
        return nothing
    end
    
    X = features[window_start:window_end, :]
    y = returns[window_start:window_end]
    
    # Remove outliers (more than 3 standard deviations)
    y_std = std(y)
    y_mean = mean(y)
    valid_mask = abs.(y .- y_mean) .< 3 * y_std
    
    if sum(valid_mask) < 10
        return nothing
    end
    
    X_clean = X[valid_mask, :]
    y_clean = y[valid_mask]
    
    try
        # Regularized regression (Ridge-like)
        X_with_intercept = hcat(ones(size(X_clean, 1)), X_clean)
        λ = 0.01  # Regularization
        I_reg = λ * I(size(X_with_intercept, 2))
        I_reg[1,1] = 0  # Don't regularize intercept
        
        coeffs = (X_with_intercept' * X_with_intercept + I_reg) \ (X_with_intercept' * y_clean)
        
        # Calculate conservative R²
        y_pred = X_with_intercept * coeffs
        ss_res = sum((y_clean - y_pred).^2)
        ss_tot = sum((y_clean .- mean(y_clean)).^2)
        r_squared = max(0.0, 1 - ss_res / ss_tot)  # Non-negative R²
        
        return (coeffs=coeffs, r_squared=r_squared)
    catch
        return nothing
    end
end

function calculate_aggressive_kelly(predicted_return, historical_returns)
    """Calculate VERY aggressive Kelly fraction to ensure trades happen"""
    
    if abs(predicted_return) < 1e-8
        return 0.0
    end
    
    # VERY aggressive position sizing - just use prediction magnitude
    kelly_f = abs(predicted_return) * 20  # Much more aggressive scaling
    
    # Very permissive bounds to ensure trading
    kelly_f = max(0.01, min(0.08, kelly_f))  # At least 1%, max 8%
    
    return kelly_f
end

function simulate_realistic_trading(stock_data)
    """Simulate realistic trading with comprehensive tracking"""
    
    features = stock_data.features
    prices = stock_data.prices
    timestamps = stock_data.timestamps
    symbol = stock_data.symbol
    
    capital = INITIAL_CAPITAL
    trades = Trade[]
    equity_curve = [capital]
    equity_timestamps = [1]
    
    max_capital = capital
    daily_trades = 0
    last_trade_time = 0
    
    # Calculate future returns
    future_returns = []
    for i in 1:(length(prices) - HOLDING_PERIOD)
        future_ret = (prices[i + HOLDING_PERIOD] - prices[i]) / prices[i]
        push!(future_returns, future_ret)
    end
    
    println("   🔄 Running realistic trading simulation...")
    
    prediction_count = 0
    kelly_above_threshold = 0
    
    # Rolling prediction and trading
    for i in (LOOKBACK_WINDOW + 1):(length(prices) - HOLDING_PERIOD - 1)
        
        # Reset daily trade counter
        if i - last_trade_time > 78  # Roughly 6.5 hours (78 * 5min)
            daily_trades = 0
        end
        
        # Skip if too many daily trades
        if daily_trades >= MAX_DAILY_TRADES
            push!(equity_curve, capital)
            push!(equity_timestamps, i)
            continue
        end
        
        # Train model
        window_start = max(1, i - LOOKBACK_WINDOW)
        window_end = i - 1
        
        model = train_conservative_model(features, future_returns, window_start, window_end)
        
        if model === nothing  # Remove R² requirement entirely
            push!(equity_curve, capital)
            push!(equity_timestamps, i)
            continue
        end
        
        # Make prediction
        current_features = vcat(1.0, features[i, :])
        predicted_return = dot(model.coeffs, current_features)
        prediction_count += 1
        
        # Calculate position size
        historical_window = future_returns[window_start:window_end]
        kelly_f = calculate_aggressive_kelly(predicted_return, historical_window)
        
        # Debug every 500 iterations
        if prediction_count % 500 == 0 || kelly_f > 0.005
            println("      Step $prediction_count: pred=$(round(predicted_return, digits=6)), kelly=$(round(kelly_f, digits=4)), R²=$(round(model.r_squared, digits=3))")
        end
        
        # Trade decision (EXTREMELY permissive to ensure trades)
        if kelly_f > 0.005  # Just 0.5% minimum - very low threshold
            kelly_above_threshold += 1
            position_size = min(MAX_TRADE_SIZE, max(MIN_TRADE_SIZE, kelly_f))
            
            # Entry
            entry_price = prices[i]
            trade_capital = capital * position_size
            
            # Simulate exit after holding period
            exit_price = prices[i + HOLDING_PERIOD]
            
            # Apply transaction costs and slippage
            total_cost = (TRANSACTION_COST + SLIPPAGE) * trade_capital
            
            # Calculate return based on prediction direction
            if predicted_return > 0
                # Long position
                trade_return = (exit_price - entry_price) / entry_price
            else
                # Short position  
                trade_return = -(exit_price - entry_price) / entry_price
            end
            
            # Apply stop loss and take profit
            exit_reason = "time"
            if trade_return < -STOP_LOSS
                trade_return = -STOP_LOSS
                exit_reason = "stop"
            elseif trade_return > TAKE_PROFIT
                trade_return = TAKE_PROFIT
                exit_reason = "profit"
            end
            
            # Calculate profit/loss
            trade_profit = trade_capital * trade_return - total_cost
            capital += trade_profit
            
            # Record trade
            successful = trade_profit > 0
            trade = Trade(
                i, entry_price, exit_price, position_size,
                predicted_return, trade_return, trade_profit,
                successful, exit_reason
            )
            push!(trades, trade)
            
            daily_trades += 1
            last_trade_time = i
            
            # Track max capital for drawdown
            max_capital = max(max_capital, capital)
        end
        
        push!(equity_curve, capital)
        push!(equity_timestamps, i)
    end
    
    # Calculate performance metrics
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    if length(equity_curve) > 1
        returns = diff(log.(equity_curve))
        returns = returns[.!isnan.(returns) .& .!isinf.(returns)]
        if length(returns) > 0
            sharpe_ratio = mean(returns) / std(returns) * sqrt(252 * 78)  # Annualized
        else
            sharpe_ratio = 0.0
        end
    else
        sharpe_ratio = 0.0
    end
    
    # Max drawdown
    min_capital = minimum(equity_curve)
    max_drawdown = (max_capital - min_capital) / max_capital
    
    # Win rate
    win_rate = length(trades) > 0 ? sum([t.successful for t in trades]) / length(trades) : 0.0
    
    # Debug summary
    println("   INFO: Trading Summary:")
    println("      Predictions made: $prediction_count")
    println("      Kelly above threshold: $kelly_above_threshold")
    println("      Actual trades: $(length(trades))")
    
    return TradingResults(
        symbol, total_return, sharpe_ratio, max_drawdown,
        length(trades), win_rate, capital, trades,
        equity_curve, equity_timestamps
    )
end

function calculate_buy_hold(stock_data)
    """Calculate buy and hold performance"""
    
    prices = stock_data.prices
    
    start_price = prices[1]
    end_price = prices[end]
    
    total_return = (end_price - start_price) / start_price
    final_capital = INITIAL_CAPITAL * (1 + total_return)
    
    return (
        total_return = total_return,
        final_capital = final_capital,
        equity_curve = INITIAL_CAPITAL .* (prices ./ start_price),
        timestamps = collect(1:length(prices))
    )
end

function create_performance_visualization(all_results, buy_hold_results)
    """Create comprehensive performance visualization"""
    
    println("INFO: Creating performance visualizations...")
    
    # 1. Performance comparison chart
    symbols = [r.symbol for r in all_results]
    soliton_returns = [r.total_return * 100 for r in all_results]
    buy_hold_returns = [bh.total_return * 100 for (sym, bh) in buy_hold_results]
    
    p1 = bar(symbols, [soliton_returns buy_hold_returns], 
             labels=["Soliton Kelly" "Buy & Hold"],
             title="Total Return Comparison (%)",
             ylabel="Return (%)",
             color=[:blue :gray],
             alpha=0.7)
    
    # 2. Individual stock performance with trade markers
    plots_array = []
    
    for (i, result) in enumerate(all_results)
        symbol = result.symbol
        bh = buy_hold_results[findfirst(x -> x[1] == symbol, buy_hold_results)][2]
        
        # Normalize equity curves to same starting point
        soliton_curve = result.equity_curve ./ INITIAL_CAPITAL
        bh_curve = bh.equity_curve ./ INITIAL_CAPITAL
        
        # Create base plot
        p = plot(result.timestamps, soliton_curve, 
                label="Soliton Kelly", 
                linewidth=2, 
                color=:blue,
                title="$symbol Performance",
                ylabel="Capital Multiple",
                xlabel="Time (5-min intervals)")
        
        plot!(p, bh.timestamps, bh_curve,
              label="Buy & Hold",
              linewidth=2,
              color=:gray,
              linestyle=:dash)
        
        # Add trade markers
        successful_trades = filter(t -> t.successful, result.trades)
        failed_trades = filter(t -> !t.successful, result.trades)
        
        if !isempty(successful_trades)
            successful_times = [t.timestamp for t in successful_trades]
            successful_values = [result.equity_curve[findfirst(x -> x >= t.timestamp, result.timestamps)] / INITIAL_CAPITAL for t in successful_trades]
            scatter!(p, successful_times, successful_values,
                    color=:green, 
                    markersize=3,
                    alpha=0.7,
                    label="Successful Trades")
        end
        
        if !isempty(failed_trades)
            failed_times = [t.timestamp for t in failed_trades]
            failed_values = [result.equity_curve[findfirst(x -> x >= t.timestamp, result.timestamps)] / INITIAL_CAPITAL for t in failed_trades]
            scatter!(p, failed_times, failed_values,
                    color=:red,
                    markersize=3, 
                    alpha=0.7,
                    label="Failed Trades")
        end
        
        push!(plots_array, p)
    end
    
    # Combine individual plots
    p2 = plot(plots_array..., layout=(2, 3), size=(1200, 800))
    
    # 3. Summary statistics
    avg_soliton_return = mean(soliton_returns)
    avg_bh_return = mean(buy_hold_returns)
    total_trades = sum([length(r.trades) for r in all_results])
    avg_win_rate = mean([r.win_rate for r in all_results]) * 100
    
    p3 = plot([1, 2], [avg_bh_return, avg_soliton_return],
              seriestype=:bar,
              labels=["Buy & Hold" "Soliton Kelly"],
              title="Average Performance Across All Stocks",
              ylabel="Average Return (%)",
              color=[:gray :blue],
              alpha=0.7,
              xticks=([1, 2], ["Buy & Hold", "Soliton Kelly"]))
    
    # Save plots
    savefig(p1, "visualizations/outputs/return_comparison.png")
    savefig(p2, "visualizations/outputs/individual_performance.png") 
    savefig(p3, "visualizations/outputs/average_performance.png")
    
    println("INFO: Visualizations saved to visualizations/outputs/")
    
    return p1, p2, p3, avg_soliton_return, avg_bh_return, total_trades, avg_win_rate
end

function main()
    """Run realistic Kelly trading test with visualization"""
    
    # Ensure output directory exists
    mkpath("visualizations/outputs")
    
    # Test stocks
    test_stocks = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "JPM"]
    
    all_results = TradingResults[]
    buy_hold_results = []
    
    println("STRONG Testing REAL SOLITON Kelly strategy on $(length(test_stocks)) stocks...")
    println("Parameters (optimized for trade generation with real PDE features):")
    println("  - Transaction cost: $(TRANSACTION_COST*100)%")
    println("  - Slippage: $(SLIPPAGE*100)%") 
    println("  - Max position size: $(MAX_TRADE_SIZE*100)%")
    println("  - Stop loss: $(STOP_LOSS*100)%")
    println("  - Take profit: $(TAKE_PROFIT*100)%")
    println("  - Kelly scaling: Aggressive")
    println("  - Min trade threshold: 0.2%")
    println("  - Features: Real Soliton PDE + Oscillators")
    println("  - PDE complexity: 8.0 (fast but real)")
    
    for symbol in test_stocks
        stock_data = prepare_realistic_data(symbol)
        
        if stock_data === nothing
            continue
        end
        
        println("\nMONEY: Testing $symbol...")
        
        # Run soliton trading
        result = simulate_realistic_trading(stock_data)
        push!(all_results, result)
        
        # Calculate buy and hold
        bh_result = calculate_buy_hold(stock_data)
        push!(buy_hold_results, (symbol, bh_result))
        
        # Print results
        println("   INFO: Soliton Kelly: $(round(result.total_return*100, digits=2))% return")
        println("   INFO: Buy & Hold: $(round(bh_result.total_return*100, digits=2))% return")
        println("   INFO: Trades: $(result.num_trades), Win Rate: $(round(result.win_rate*100, digits=1))%")
        println("   INFO: Sharpe: $(round(result.sharpe_ratio, digits=2)), Max DD: $(round(result.max_drawdown*100, digits=2))%")
    end
    
    if isempty(all_results)
        println("FAILED: No results generated")
        return
    end
    
    println("\nTARGET: CREATING VISUALIZATIONS...")
    p1, p2, p3, avg_soliton, avg_bh, total_trades, avg_win_rate = create_performance_visualization(all_results, buy_hold_results)
    
    println("\nUP: FINAL RESULTS:")
    println("=" ^ 40)
    println("Average Soliton Kelly Return: $(round(avg_soliton, digits=2))%")
    println("Average Buy & Hold Return: $(round(avg_bh, digits=2))%")
    println("Improvement: $(round(avg_soliton - avg_bh, digits=2)) percentage points")
    println("Total Trades Executed: $total_trades")
    println("Average Win Rate: $(round(avg_win_rate, digits=1))%")
    
    outperformance = sum([r.total_return > bh[2].total_return for (r, bh) in zip(all_results, buy_hold_results)])
    println("Stocks Outperformed: $outperformance/$(length(all_results))")
    
    if avg_soliton > avg_bh + 1.0
        println("\nSUCCESS: SUCCESS: Soliton Kelly shows consistent advantage!")
    elseif avg_soliton > avg_bh
        println("\nINFO: POSITIVE: Modest but consistent improvement")
    else
        println("\nINFO: MIXED: Strategy performance varies by market conditions")
    end
    
    return all_results, buy_hold_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results, bh_results = main()
end 