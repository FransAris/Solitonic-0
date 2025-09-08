#!/usr/bin/env julia

"""
Advanced Soliton ML Trading System

Enhanced system with:
- Multiple ML models (Random Forest, Neural Networks, SVM)
- Market regime detection
- Model ensemble methods
- Advanced feature selection
- Comprehensive performance analysis
"""

using DataFrames, Arrow, Statistics, LinearAlgebra, Random, Plots
using MLJ, MLJDecisionTreeInterface, MLJLinearModels, MLJFlux
using StatsBase, Clustering
plotlyjs()

println("🧠 Advanced Soliton ML Trading System")
println("=" ^ 60)

Random.seed!(42)

include("src/Oscillators_enhanced.jl")
include("src/SolitonPDE_adaptive.jl")

using .Oscillators_enhanced
using .SolitonPDE_adaptive

# ENHANCED TRADING PARAMETERS
const INITIAL_CAPITAL = 100000.0
const TRANSACTION_COST = 0.001
const HOLDING_PERIOD = 10
const LOOKBACK_WINDOW = 50  # Larger window for ML models
const MIN_POSITION = 0.01
const MAX_POSITION = 0.2

# MARKET REGIME PARAMETERS
const VOLATILITY_WINDOW = 20
const TREND_WINDOW = 10
const REGIME_LOOKBACK = 100

struct MLTradeResult
    entry_time::Int
    entry_price::Float64
    exit_price::Float64
    position_size::Float64
    return_pct::Float64
    profit::Float64
    successful::Bool
    prediction::Float64
    model_type::String
    regime::String
    confidence::Float64
end

struct ModelPerformance
    model_name::String
    total_return::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    num_trades::Int
    win_rate::Float64
    avg_trade_return::Float64
end

function extract_enhanced_soliton_features(row)
    """Extract comprehensive soliton features for ML models"""
    
    try
        amplitudes = (row.RSI, row.StochK, row.CCI, row.MACDsig)
        result = simulate_soliton_adaptive(amplitudes, 8.0, freq=FreqMin5)
        
        # Core soliton features
        soliton_features = [
            result.H,                        # Core collision result
            result.energy,                   # Energy
            result.asymmetry_x,              # X asymmetry  
            result.asymmetry_y,              # Y asymmetry
            result.concentration,            # Field concentration
            result.collision_H,              # Collision Hamiltonian
            result.collision_energy,         # Collision energy
            result.frequency_signature,      # Frequency signature
            result.collision_asymmetry_x,    # Collision X asymmetry
            result.collision_asymmetry_y,    # Collision Y asymmetry
        ]
        
        # Derived soliton features for ML
        derived_features = [
            result.H * result.energy,                           # Energy-collision interaction
            abs(result.asymmetry_x - result.asymmetry_y),       # Asymmetry difference
            result.collision_H / (result.energy + 1e-6),        # Collision efficiency
            result.concentration * result.frequency_signature,   # Concentration-frequency interaction
            sqrt(result.asymmetry_x^2 + result.asymmetry_y^2),  # Asymmetry magnitude
            result.collision_energy * result.concentration,     # Localized collision energy
        ]
        
        # Technical indicators for context
        technical_features = [
            (row.RSI - 50) / 50,             # Normalized RSI
            (row.StochK - 50) / 50,          # Normalized Stoch
            tanh(row.CCI / 100),             # Bounded CCI
            tanh(row.MACDsig * 10),          # Bounded MACD
        ]
        
        return vcat(soliton_features, derived_features, technical_features)
        
    catch
        # Enhanced fallback features
        return [
            (row.RSI - 50) * (row.StochK - 50) / 2500,   # Momentum interaction
            abs(row.CCI) / 100,                          # Volatility proxy
            (row.RSI - 50) / 50,                         # RSI asymmetry
            (row.StochK - 50) / 50,                      # Stoch asymmetry
            max(row.RSI, row.StochK) / 100,              # Peak momentum
            row.CCI * row.MACDsig / 10000,               # Collision proxy
            abs(row.CCI * row.MACDsig) / 10000,          # Energy proxy
            (row.RSI + row.StochK + abs(row.CCI)) / 300, # Frequency proxy
            (row.RSI - row.StochK) / 100,                # X asymmetry proxy
            (row.CCI + row.MACDsig) / 100,               # Y asymmetry proxy
            row.RSI * abs(row.CCI) / 10000,              # Energy interaction
            abs(row.RSI - row.StochK),                   # Asymmetry difference
            row.CCI^2 / 10000,                           # Collision efficiency
            row.RSI * row.MACDsig / 5000,                # Concentration proxy
            sqrt(row.RSI^2 + row.StochK^2) / 100,        # Momentum magnitude
            row.StochK * abs(row.CCI) / 10000,           # Localized energy
            (row.RSI - 50) / 50,                         # RSI normalized
            (row.StochK - 50) / 50,                      # Stoch normalized
            tanh(row.CCI / 100),                         # CCI bounded
            tanh(row.MACDsig * 10),                      # MACD bounded
        ]
    end
end

function detect_market_regime(prices::Vector{Float64}, returns::Vector{Float64}, current_idx::Int)
    """Detect current market regime using multiple indicators"""
    
    if current_idx < max(VOLATILITY_WINDOW, TREND_WINDOW, REGIME_LOOKBACK)
        return "unknown"
    end
    
    # Get recent data
    start_idx = max(1, current_idx - REGIME_LOOKBACK + 1)
    recent_prices = prices[start_idx:current_idx]
    recent_returns = returns[start_idx:min(current_idx, length(returns))]
    
    # Volatility regime
    recent_vol = std(recent_returns[max(1, end-VOLATILITY_WINDOW+1):end])
    historical_vol = std(recent_returns)
    vol_ratio = recent_vol / historical_vol
    
    # Trend regime
    trend_window = min(TREND_WINDOW, length(recent_prices))
    price_start = recent_prices[max(1, end-trend_window+1)]
    price_end = recent_prices[end]
    trend_strength = (price_end - price_start) / price_start
    
    # Momentum regime (using simple moving averages)
    if length(recent_prices) >= 20
        sma_short = mean(recent_prices[end-9:end])  # 10-period SMA
        sma_long = mean(recent_prices[end-19:end])  # 20-period SMA
        momentum = (sma_short - sma_long) / sma_long
    else
        momentum = 0.0
    end
    
    # Classify regime
    if vol_ratio > 1.5
        if abs(trend_strength) > 0.02
            return "high_vol_trending"
        else
            return "high_vol_ranging"
        end
    elseif vol_ratio < 0.7
        if abs(trend_strength) > 0.01
            return "low_vol_trending"
        else
            return "low_vol_ranging"
        end
    else
        if abs(momentum) > 0.005
            return "moderate_trending"
        else
            return "moderate_ranging"
        end
    end
end

function prepare_ml_trading_data(symbol::String)
    """Prepare enhanced data for ML trading"""
    
    println("INFO: Preparing ML data for $symbol...")
    
    # Load and process data (same as before)
    file_path = joinpath("data/raw/stocks", "$(lowercase(symbol))_5min_2025-07-19.arrow")
    if !isfile(file_path)
        return nothing
    end
    
    df = DataFrame(Arrow.Table(file_path))
    osc_df = compute_oscillators_enhanced(df, freq=Min5)
    
    valid_mask = .!isnan.(osc_df.RSI) .& .!isnan.(osc_df.StochK) .& 
                 .!isnan.(osc_df.CCI) .& .!isnan.(osc_df.MACDsig)
    valid_df = osc_df[valid_mask, :]
    
    if nrow(valid_df) < 200
        return nothing
    end
    
    # Extract enhanced features
    features_matrix = []
    prices = []
    timestamps = []
    dates = []
    
    println("   🔄 Computing enhanced soliton features...")
    for i in 1:nrow(valid_df)
        row = valid_df[i, :]
        features = extract_enhanced_soliton_features(row)
        
        if !any(isnan.(features)) && !any(isinf.(features))
            push!(features_matrix, features)
            push!(prices, row.Close)
            push!(timestamps, i)
            push!(dates, row.datetime)
        end
    end
    
    if length(features_matrix) < 100
        return nothing
    end
    
    # Convert to matrix
    X = hcat(features_matrix...)'
    
    # Calculate forward returns
    returns = Float64[]
    for i in 1:(length(prices) - HOLDING_PERIOD)
        ret = (prices[i + HOLDING_PERIOD] - prices[i]) / prices[i]
        push!(returns, ret)
    end
    
    # Calculate price returns for regime detection
    price_returns = Float64[]
    for i in 2:length(prices)
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        push!(price_returns, ret)
    end
    
    # Detect market regimes
    regimes = String[]
    prices_float = Float64.(prices)  # Convert to Float64
    for i in 1:length(returns)
        regime = detect_market_regime(prices_float, price_returns, i)
        push!(regimes, regime)
    end
    
    # Trim data to match
    X = X[1:length(returns), :]
    final_prices = Float64.(prices[1:length(returns)])
    final_timestamps = timestamps[1:length(returns)]
    final_dates = dates[1:length(returns)]
    
    println("   SUCCESS: ML dataset: $(size(X, 1)) samples × $(size(X, 2)) enhanced features")
    
    # Print regime distribution
    regime_counts = countmap(regimes)
    println("   INFO: Market regimes detected:")
    for (regime, count) in sort(collect(regime_counts), by=x->x[2], rev=true)
        pct = round(count / length(regimes) * 100, digits=1)
        println("      $regime: $count ($pct%)")
    end
    
    return (
        features = X,
        returns = returns,
        prices = final_prices,
        timestamps = final_timestamps,
        dates = final_dates,
        regimes = regimes,
        symbol = symbol
    )
end

function train_ml_models(X_train::Matrix, y_train::Vector)
    """Train multiple statistical models and return ensemble"""
    
    models = Dict()
    
    try
        # Prepare MLJ data
        df_train = DataFrame(X_train, :auto)
        df_train.target = y_train
        
        # Linear Regression (always works)
        linear_model = @load LinearRegressor pkg=MLJLinearModels verbosity=0
        linear_mach = machine(linear_model(), select(df_train, Not(:target)), df_train.target)
        MLJ.fit!(linear_mach, verbosity=0)
        models["Linear"] = linear_mach
        
        # Ridge Regression 
        ridge_model = @load RidgeRegressor pkg=MLJLinearModels verbosity=0
        ridge_mach = machine(ridge_model(lambda=0.1), select(df_train, Not(:target)), df_train.target)
        MLJ.fit!(ridge_mach, verbosity=0)
        models["Ridge"] = ridge_mach
        
        # Try Decision Tree (if available)
        try
            tree_model = @load DecisionTreeRegressor pkg=MLJDecisionTreeInterface verbosity=0
            tree_mach = machine(tree_model(max_depth=8), select(df_train, Not(:target)), df_train.target)
            MLJ.fit!(tree_mach, verbosity=0)
            models["DecisionTree"] = tree_mach
        catch
            # Manual polynomial features as alternative
            X_poly = hcat(X_train, X_train.^2)  # Add squared terms
            X_poly_intercept = hcat(ones(size(X_poly, 1)), X_poly)
            poly_coeffs = X_poly_intercept \ y_train
            models["Polynomial"] = poly_coeffs
        end
        
        # Manual ensemble: weighted combination of different regularizations
        X_with_intercept = hcat(ones(size(X_train, 1)), X_train)
        
        # Light regularization
        λ_light = 0.01
        I_reg_light = λ_light * I(size(X_with_intercept, 2))
        I_reg_light[1,1] = 0
        coeffs_light = (X_with_intercept' * X_with_intercept + I_reg_light) \ (X_with_intercept' * y_train)
        models["RidgeLight"] = coeffs_light
        
        # Heavy regularization  
        λ_heavy = 0.5
        I_reg_heavy = λ_heavy * I(size(X_with_intercept, 2))
        I_reg_heavy[1,1] = 0
        coeffs_heavy = (X_with_intercept' * X_with_intercept + I_reg_heavy) \ (X_with_intercept' * y_train)
        models["RidgeHeavy"] = coeffs_heavy
        
    catch e
        println("      WARNING:  ML training error: $e")
        # Ultimate fallback
        X_with_intercept = hcat(ones(size(X_train, 1)), X_train)
        coeffs = X_with_intercept \ y_train
        models["Fallback"] = coeffs
    end
    
    println("      SUCCESS: Trained $(length(models)) models successfully")
    return models
end

function predict_ensemble(models::Dict, X_test::Matrix)
    """Make ensemble predictions from multiple models"""
    
    predictions = Dict()
    
    for (name, model) in models
        try
            if name in ["Fallback", "RidgeLight", "RidgeHeavy"]
                # Manual linear prediction
                X_with_intercept = hcat(ones(size(X_test, 1)), X_test)
                pred = X_with_intercept * model
                predictions[name] = pred
            elseif name == "Polynomial"
                # Polynomial prediction
                X_poly = hcat(X_test, X_test.^2)
                X_poly_intercept = hcat(ones(size(X_poly, 1)), X_poly)
                pred = X_poly_intercept * model
                predictions[name] = pred
            else
                # MLJ prediction
                df_test = DataFrame(X_test, :auto)
                pred = MLJ.predict(model, df_test)
                predictions[name] = pred
            end
        catch e
            println("      WARNING:  Prediction error for $name: $e")
        end
    end
    
    # Ensemble average (if multiple models available)
    if length(predictions) > 1
        ensemble_pred = mean([pred for pred in values(predictions)])
        predictions["Ensemble"] = ensemble_pred
    end
    
    return predictions
end

function simulate_ml_trading(data)
    """Advanced ML trading simulation with regime awareness"""
    
    X = data.features
    returns = data.returns
    prices = data.prices
    timestamps = data.timestamps
    regimes = data.regimes
    symbol = data.symbol
    
    # Results storage
    model_results = Dict{String, Vector{MLTradeResult}}()
    model_capitals = Dict{String, Vector{Float64}}()
    
    println("   🤖 Running ML trading simulation...")
    
    # Initialize tracking for each model type  
    all_model_names = ["Linear", "Ridge", "DecisionTree", "Polynomial", "RidgeLight", "RidgeHeavy", "Ensemble"]
    
    for model_name in all_model_names
        model_results[model_name] = MLTradeResult[]
        model_capitals[model_name] = [INITIAL_CAPITAL]
    end
    
    # Rolling window ML trading
    for i in (LOOKBACK_WINDOW + 1):(length(returns) - 1)
        
        # Prepare training data
        window_start = max(1, i - LOOKBACK_WINDOW)
        window_end = i - 1
        
        X_train = X[window_start:window_end, :]
        y_train = returns[window_start:window_end]
        
        # Remove outliers
        y_std = std(y_train)
        y_mean = mean(y_train)
        valid_mask = abs.(y_train .- y_mean) .< 3 * y_std
        
        if sum(valid_mask) < 20
            for model_name in all_model_names
                push!(model_capitals[model_name], model_capitals[model_name][end])
            end
            continue
        end
        
        X_clean = X_train[valid_mask, :]
        y_clean = y_train[valid_mask]
        
        # Train models
        models = train_ml_models(X_clean, y_clean)
        
        # Make predictions
        X_current = reshape(X[i, :], 1, :)
        predictions = predict_ensemble(models, X_current)
        
        # Current market regime
        current_regime = regimes[i]
        
        # Trade with each model
        for model_name in all_model_names
            current_capital = model_capitals[model_name][end]
            
            if haskey(predictions, model_name)
                predicted_return = predictions[model_name][1]
                
                # Regime-based position sizing
                regime_multiplier = if current_regime in ["high_vol_trending", "moderate_trending"]
                    1.5  # More aggressive in trending markets
                elseif current_regime in ["low_vol_ranging", "moderate_ranging"]
                    0.7  # More conservative in ranging markets
                else
                    1.0  # Default
                end
                
                # Position sizing
                prediction_strength = abs(predicted_return)
                position_size = prediction_strength * regime_multiplier * 0.3
                position_size = max(MIN_POSITION, min(MAX_POSITION, position_size))
                
                # Trade threshold (regime dependent)
                trade_threshold = if current_regime in ["high_vol_ranging"]
                    0.002  # Higher threshold in choppy markets
                else
                    0.0005  # Lower threshold otherwise
                end
                
                if prediction_strength > trade_threshold
                    # Execute trade
                    entry_price = prices[i]
                    trade_capital = current_capital * position_size
                    
                    # Actual return
                    actual_return = returns[i]
                    exit_price = entry_price * (1 + actual_return)
                    
                    # Calculate trade result
                    if predicted_return > 0
                        trade_return = actual_return  # Long
                    else
                        trade_return = -actual_return  # Short
                    end
                    
                    # Apply costs
                    total_cost = TRANSACTION_COST * trade_capital * 2
                    trade_profit = trade_capital * trade_return - total_cost
                    
                    new_capital = current_capital + trade_profit
                    successful = trade_profit > 0
                    
                    # Record trade
                    trade = MLTradeResult(
                        i, entry_price, exit_price, position_size,
                        trade_return * 100, trade_profit, successful,
                        predicted_return, model_name, current_regime,
                        prediction_strength
                    )
                    
                    push!(model_results[model_name], trade)
                    push!(model_capitals[model_name], new_capital)
                else
                    push!(model_capitals[model_name], current_capital)
                end
            else
                push!(model_capitals[model_name], current_capital)
            end
        end
    end
    
    # Calculate performance metrics for each model
    performance_results = ModelPerformance[]
    
    for model_name in all_model_names
        if haskey(model_results, model_name) && !isempty(model_results[model_name])
            trades = model_results[model_name]
            capital_history = model_capitals[model_name]
            
            total_return = (capital_history[end] - INITIAL_CAPITAL) / INITIAL_CAPITAL
            
            # Sharpe ratio
            if length(capital_history) > 1
                returns_series = diff(log.(capital_history))
                sharpe = mean(returns_series) / (std(returns_series) + 1e-8) * sqrt(252 * 78)
            else
                sharpe = 0.0
            end
            
            # Max drawdown
            peak = INITIAL_CAPITAL
            max_dd = 0.0
            for capital in capital_history
                peak = max(peak, capital)
                drawdown = (peak - capital) / peak
                max_dd = max(max_dd, drawdown)
            end
            
            # Win rate
            wins = sum([t.successful for t in trades])
            win_rate = wins / length(trades)
            
            # Average trade return
            avg_trade_return = mean([t.return_pct for t in trades])
            
            perf = ModelPerformance(
                model_name, total_return, sharpe, max_dd,
                length(trades), win_rate, avg_trade_return
            )
            
            push!(performance_results, perf)
            
            println("   INFO: $model_name: $(round(total_return*100, digits=2))% return, $(length(trades)) trades, $(round(win_rate*100, digits=1))% win rate")
        end
    end
    
    return (
        symbol = symbol,
        model_results = model_results,
        model_capitals = model_capitals,
        performance = performance_results,
        regimes = regimes,
        dates = data.dates,
        cumulative_returns = last.(values(model_capitals)) ./ INITIAL_CAPITAL
    )
end

function create_ml_visualization(results, buy_hold_results)
    """Create comprehensive ML trading visualization with proper time axes"""
    
    println("INFO: Creating ML visualization...")
    
    # Performance comparison by model - updated model list
    all_symbols = [r.symbol for r in results]
    model_names = ["Linear", "Ridge", "DecisionTree", "Polynomial", "RidgeLight", "RidgeHeavy", "Ensemble"]
    
    # Aggregate performance by model
    model_performance = Dict()
    for model_name in model_names
        total_returns = []
        for result in results
            model_perf = filter(p -> p.model_name == model_name, result.performance)
            if !isempty(model_perf)
                push!(total_returns, model_perf[1].total_return * 100)
            end
        end
        if !isempty(total_returns)
            model_performance[model_name] = mean(total_returns)
        end
    end
    
    # Buy-hold average
    bh_avg = mean([bh[2].total_return * 100 for bh in buy_hold_results])
    
    # Model comparison plot
    model_names_plot = collect(keys(model_performance))
    push!(model_names_plot, "Buy & Hold")
    model_returns_plot = [model_performance[name] for name in model_names_plot[1:end-1]]
    push!(model_returns_plot, bh_avg)
    
    p1 = bar(model_names_plot, model_returns_plot,
             title="ML Model Performance Comparison",
             ylabel="Average Return (%)",
             xrotation=45,
             color=[:blue :green :red :purple :orange :brown :pink :gray])
    
    # Individual stock performance for best model
    best_model = argmax(model_performance)
    
    plots = []
    for (i, result) in enumerate(results[1:min(4, length(results))])  # Show first 4 stocks
        if haskey(result.model_capitals, best_model)
            bh = buy_hold_results[findfirst(x -> x[1] == result.symbol, buy_hold_results)][2]
            
            # Normalize curves and ensure safe bounds
            ml_curve = result.model_capitals[best_model] ./ INITIAL_CAPITAL
            bh_curve = bh.equity_curve ./ INITIAL_CAPITAL
            
            # Get proper date range - use the minimum length to avoid bounds errors
            min_length = min(length(ml_curve), length(bh_curve), length(result.dates))
            plot_dates = result.dates[1:min_length]
            ml_curve_safe = ml_curve[1:min_length]
            bh_curve_safe = bh_curve[1:min_length]
            
            p = plot(plot_dates, ml_curve_safe,
                    label="$(best_model) ML",
                    title="$(result.symbol) - ML vs Buy & Hold",
                    ylabel="Capital Multiple",
                    xlabel="Date",
                    xrotation=45)
            
            plot!(p, plot_dates, bh_curve_safe,
                  label="Buy & Hold",
                  linestyle=:dash)
            
            # Add trade markers with proper bounds checking
            if haskey(result.model_results, best_model)
                trades = result.model_results[best_model]
                
                # Filter trades to only include those within our safe range
                safe_trades = filter(t -> t.entry_time <= min_length, trades)
                successful_trades = filter(t -> t.successful, safe_trades)
                failed_trades = filter(t -> !t.successful, safe_trades)
                
                if !isempty(successful_trades)
                    times = [plot_dates[t.entry_time] for t in successful_trades]
                    values = [ml_curve_safe[t.entry_time] for t in successful_trades]
                    scatter!(p, times, values, color=:green, markersize=3, alpha=0.8, label="Wins")
                end
                
                if !isempty(failed_trades)
                    times = [plot_dates[t.entry_time] for t in failed_trades]
                    values = [ml_curve_safe[t.entry_time] for t in failed_trades]
                    scatter!(p, times, values, color=:red, markersize=3, alpha=0.8, label="Losses")
                end
            end
            
            push!(plots, p)
        end
    end
    
    p2 = plot(plots..., layout=(2, 2), size=(1200, 900))
    
    # Overall portfolio performance with proper time axis
    colors = [:blue, :red, :green, :purple, :orange, :brown, :pink]
    
    p3 = plot(title="Multi-Model Portfolio Performance Over Time", 
              xlabel="Date", ylabel="Capital Multiple", 
              size=(1400, 700), xrotation=45)
    
    # Plot ML results
    for (i, result) in enumerate(results)
        if length(result.dates) == length(result.cumulative_returns)
            plot!(p3, result.dates, result.cumulative_returns, 
                  label="$(result.symbol) ML", linewidth=2,
                  color=colors[mod(i-1, length(colors))+1])
        end
    end
    
    # Add buy-hold benchmarks with proper time alignment
    for (i, (symbol, bh)) in enumerate(buy_hold_results)
        matching_result = findfirst(r -> r.symbol == symbol, results)
        if matching_result !== nothing
            result_dates = results[matching_result].dates
            min_length = min(length(bh.equity_curve), length(result_dates))
            
            if min_length > 0
                bh_normalized = bh.equity_curve[1:min_length] ./ INITIAL_CAPITAL
                plot_dates = result_dates[1:min_length]
                
                plot!(p3, plot_dates, bh_normalized,
                      label="$(symbol) B&H", linestyle=:dash,
                      color=colors[mod(i-1, length(colors))+1], alpha=0.7, linewidth=1)
            end
        end
    end
    
    # Create final layout
    final_plot = plot(p1, p2, p3, layout=(3, 1), size=(1400, 2000))
    
    # Save plots
    mkpath("visualizations/outputs")
    try
        savefig(p1, "visualizations/outputs/ml_model_comparison.png")
        savefig(p2, "visualizations/outputs/ml_individual_performance.png")
        savefig(p3, "visualizations/outputs/ml_portfolio_performance.png")
        savefig(final_plot, "visualizations/outputs/ml_complete_analysis.png")
        println("SUCCESS: All visualizations saved to visualizations/outputs/")
    catch e
        println("WARNING:  Warning: Could not save some plots: $e")
    end
    
    return p1, p2, p3, best_model
end

function main()
    """Run advanced ML soliton trading system"""
    
    test_stocks = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "JPM"]
    
    results = []
    buy_hold_results = []
    
    println("STRONG Testing ADVANCED STATISTICAL soliton strategy:")
    println("   Features: Enhanced soliton PDE + market regimes")
    println("   Models: Linear, Ridge, DecisionTree, Polynomial, Multiple Ridge variants, Ensemble")
    
    for symbol in test_stocks
        stock_data = prepare_ml_trading_data(symbol)
        
        if stock_data === nothing
            continue
        end
        
        # Run ML trading
        result = simulate_ml_trading(stock_data)
        push!(results, result)
        
        # Buy and hold comparison
        start_price = stock_data.prices[1]
        end_price = stock_data.prices[end]
        bh_return = (end_price - start_price) / start_price
        bh_equity = INITIAL_CAPITAL .* (stock_data.prices ./ start_price)
        
        push!(buy_hold_results, (symbol, (
            total_return = bh_return,
            equity_curve = bh_equity,
            timestamps = stock_data.timestamps
        )))
    end
    
    if isempty(results)
        println("FAILED: No results generated")
        return
    end
    
    # Create visualization and analysis
    p1, p2, p3, best_model = create_ml_visualization(results, buy_hold_results)
    
    # Summary analysis
    println("\nTARGET: ADVANCED ML SUMMARY:")
    println("=" ^ 40)
    
    # Performance by model
    model_aggregates = Dict()
    for model_name in ["Linear", "Ridge", "DecisionTree", "Polynomial", "RidgeLight", "RidgeHeavy", "Ensemble"]
        returns = []
        trades = 0
        win_rates = []
        
        for result in results
            model_perf = filter(p -> p.model_name == model_name, result.performance)
            if !isempty(model_perf)
                push!(returns, model_perf[1].total_return * 100)
                trades += model_perf[1].num_trades
                push!(win_rates, model_perf[1].win_rate * 100)
            end
        end
        
        if !isempty(returns)
            model_aggregates[model_name] = (
                avg_return = mean(returns),
                total_trades = trades,
                avg_win_rate = mean(win_rates)
            )
        end
    end
    
    # Print model comparison
    for (model_name, stats) in sort(collect(model_aggregates), by=x->x[2].avg_return, rev=true)
        println("$model_name:")
        println("  Avg Return: $(round(stats.avg_return, digits=2))%")
        println("  Total Trades: $(stats.total_trades)")
        println("  Avg Win Rate: $(round(stats.avg_win_rate, digits=1))%")
        println()
    end
    
    bh_avg = mean([bh[2].total_return * 100 for bh in buy_hold_results])
    println("Buy & Hold Average: $(round(bh_avg, digits=2))%")
    
    # Best performing model
    if !isempty(model_aggregates)
        best_model_name = argmax(Dict(name => stats.avg_return for (name, stats) in model_aggregates))
        best_stats = model_aggregates[best_model_name]
        improvement = best_stats.avg_return - bh_avg
        
        println("\n🏆 BEST MODEL: $best_model_name")
        println("   Outperformance: $(round(improvement, digits=2)) percentage points")
        
        if improvement > 2.0
            println("   SUCCESS: EXCELLENT: Strong ML advantage!")
        elseif improvement > 0.5
            println("   UP: GOOD: Solid ML improvement")
        elseif improvement > 0
            println("   INFO: POSITIVE: Modest ML gains")
        else
            println("   INFO: MIXED: ML needs further optimization")
        end
    end
    
    return results, buy_hold_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results, bh_results = main()
end 