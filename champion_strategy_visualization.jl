#!/usr/bin/env julia --compiled-modules=no

"""
Strategy Visualization
Recreate and visualize the winning strategy: Linear + KellyCriterion (5d, SolitonCore)
vs S&P 500 Buy-and-Hold
"""

println("Strategy Visualization")
println("=" ^ 50)

using Arrow, DataFrames, Statistics, Dates, StatsBase, LinearAlgebra, Plots
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Load dataset
df = DataFrame(Arrow.Table("data/processed/soliton_features_30years.arrow"))
println("✅ Loaded $(nrow(df)) trading days")

# Strategy configuration
println("\nStrategy Configuration:")
println("   Algorithm: Linear Regression")
println("   Risk Management: Kelly Criterion") 
println("   Features: SolitonCore (SolitonHeight, SolitonProbeMean, SolitonProbeStd, SolitonProbeMax)")
println("   Horizon: 5 days")
println("   Expected excess CAGR: +5.1% vs S&P 500")

# Prepare data for strategy
soliton_core_features = ["SolitonHeight", "SolitonProbeMean", "SolitonProbeStd", "SolitonProbeMax"]
available_features = filter(f -> f in names(df), soliton_core_features)
essential_cols = ["Date", "Close", "ForwardReturn5d"]

println("\nAvailable SolitonCore features: $(available_features)")

# Clean dataset
all_cols = unique(vcat(available_features, essential_cols))
df_clean = df[!, all_cols]
df_clean = dropmissing(df_clean)
sort!(df_clean, :Date)

println("   Clean samples: $(nrow(df_clean))")

# Utility functions
function cummax(x::Vector{T}) where T
    result = similar(x)
    if isempty(x)
        return result
    end
    result[1] = x[1]
    for i in 2:length(x)
        result[i] = max(result[i-1], x[i])
    end
    return result
end

# Simple Ridge Regression (our winning algorithm)
function fit_ridge_simple(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64=1.0)
    try
        X_ridge = hcat(ones(size(X, 1)), X)  # Add intercept
        I_reg = Matrix{Float64}(I, size(X_ridge, 2), size(X_ridge, 2))
        I_reg[1, 1] = 0.0  # Don't regularize intercept
        
        coeffs = (X_ridge' * X_ridge + lambda * I_reg) \ (X_ridge' * y)
        return coeffs
    catch e
        println("   Ridge error: $e")
        return zeros(size(X, 2) + 1)
    end
end

function predict_ridge_simple(X::Matrix{Float64}, coeffs::Vector{Float64})
    try
        X_with_intercept = hcat(ones(size(X, 1)), X)
        return X_with_intercept * coeffs
    catch e
        println("   Prediction error: $e")
        return zeros(size(X, 1))
    end
end

# Kelly Criterion position sizing (our winning risk strategy)
function kelly_position_sizing(signal, returns_history)
    if length(returns_history) < 20
        return clamp(signal * 0.1, -0.2, 0.2)
    end
    
    variance = var(returns_history[max(1, end-252):end])
    if variance <= 0
        return 0.0
    end
    
    kelly_fraction = signal / variance
    kelly_adjusted = kelly_fraction * 0.25  # Conservative Kelly
    return clamp(kelly_adjusted, -0.3, 0.3)
end

# Time series cross-validation
function create_simple_splits(n_total::Int, n_splits::Int=3)
    splits = []
    test_size = n_total ÷ (n_splits + 1)
    
    for i in 1:n_splits
        test_start = i * test_size
        test_end = min((i + 1) * test_size - 1, n_total)
        train_start = max(1, test_start - 2 * test_size)
        train_end = test_start - 1
        
        if train_end > train_start && test_end > test_start
            push!(splits, (
                train_indices = train_start:train_end,
                test_indices = test_start:test_end
            ))
        end
    end
    
    return splits
end

# Strategy backtest
println("\nRunning Strategy Backtest...")

# Prepare feature matrix
feature_data = df_clean[!, available_features]
X = Matrix{Float64}(feature_data)
y = Vector{Float64}(df_clean.ForwardReturn5d)
dates = df_clean.Date
prices = df_clean.Close

# Time series splits
splits = create_simple_splits(length(y), 3)

# Results storage
initial_capital = 100000.0
strategy_returns = Float64[]
strategy_dates = Date[]
strategy_values = Float64[]

# S&P 500 buy-and-hold
spx_values = Float64[]

println("   Processing $(length(splits)) time series splits...")

for (split_idx, split) in enumerate(splits)
    try
        # Training data
        X_train = X[split.train_indices, :]
        y_train = y[split.train_indices]
        
        # Test data
        X_test = X[split.test_indices, :]
        y_test = y[split.test_indices]
        test_dates = dates[split.test_indices]
        test_prices = prices[split.test_indices]
        
        # Train Linear model (no regularization for pure linear)
        coeffs = fit_ridge_simple(X_train, y_train, 0.0)
        predictions = predict_ridge_simple(X_test, coeffs)
        
        # Trading simulation with Kelly Criterion
        current_position = 0.0
        local_capital = length(strategy_values) > 0 ? strategy_values[end] : initial_capital
        
        for i in 1:length(predictions)
            signal = isfinite(predictions[i]) ? predictions[i] : 0.0
            
            # Kelly Criterion position sizing
            target_position = kelly_position_sizing(signal, strategy_returns)
            
            # Transaction costs
            position_change = abs(target_position - current_position)
            transaction_cost = position_change * local_capital * 0.0003
            
            # Update position
            current_position = target_position
            
            # Portfolio return
            actual_return = y_test[i]
            portfolio_return = current_position * actual_return
            
            # Update capital
            local_capital = local_capital * (1 + portfolio_return) - transaction_cost
            
            # Store results
            push!(strategy_returns, portfolio_return)
            push!(strategy_dates, test_dates[i])
            push!(strategy_values, local_capital)
        end
        
        println("   Split $split_idx completed: $(length(strategy_values)) trades")
        
    catch e
        println("   Error in split $split_idx: $e")
        continue
    end
end

# Create S&P 500 buy-and-hold for same period
if !isempty(strategy_dates) && !isempty(strategy_values)
    # Get S&P prices for the same dates as our strategy
    spx_start_price = prices[findfirst(d -> d == strategy_dates[1], dates)]
    spx_shares = initial_capital / spx_start_price
    
    for date in strategy_dates
        price_idx = findfirst(d -> d == date, dates)
        if price_idx !== nothing
            spx_value = spx_shares * prices[price_idx]
            push!(spx_values, spx_value)
        end
    end
end

# Calculate performance metrics
if !isempty(strategy_values) && !isempty(spx_values)
    
    strategy_total_return = (strategy_values[end] - initial_capital) / initial_capital
    spx_total_return = (spx_values[end] - initial_capital) / initial_capital
    
    years = (strategy_dates[end] - strategy_dates[1]).value / 365.25
    strategy_cagr = (strategy_values[end] / initial_capital)^(1/years) - 1
    spx_cagr = (spx_values[end] / initial_capital)^(1/years) - 1
    
    excess_cagr = strategy_cagr - spx_cagr
    
    println("\n📊 PERFORMANCE COMPARISON:")
    println("   Period: $(strategy_dates[1]) to $(strategy_dates[end]) ($(round(years, digits=1)) years)")
    println("   ")
    println("   🏆 CHAMPION SOLITON STRATEGY:")
    println("      Final Value: \$$(round(strategy_values[end], digits=0))")
    println("      Total Return: $(round(strategy_total_return*100, digits=1))%")
    println("      CAGR: $(round(strategy_cagr*100, digits=1))%")
    println("   ")
    println("   📈 S&P 500 BUY & HOLD:")
    println("      Final Value: \$$(round(spx_values[end], digits=0))")
    println("      Total Return: $(round(spx_total_return*100, digits=1))%")
    println("      CAGR: $(round(spx_cagr*100, digits=1))%")
    println("   ")
    println("   ⚡ EXCESS PERFORMANCE:")
    println("      Excess CAGR: $(round(excess_cagr*100, digits=1))%")
    println("      Outperformance: $(excess_cagr > 0 ? "✅ WINNER!" : "❌ Underperformed")")
    
    # CREATE VISUALIZATION
    println("\n📈 Creating performance visualization...")
    
    # Convert to percentage returns from initial capital
    strategy_pct = (strategy_values .- initial_capital) ./ initial_capital .* 100
    spx_pct = (spx_values .- initial_capital) ./ initial_capital .* 100
    
    # Create the plot
    p = plot(strategy_dates, strategy_pct, 
             label="Soliton Strategy (Linear+Kelly+SolitonCore)", 
             linewidth=3, 
             color=:blue,
             title="Champion Soliton Strategy vs S&P 500 Buy-and-Hold",
             xlabel="Date",
             ylabel="Cumulative Return (%)",
             size=(1200, 600),
             dpi=300)
    
    plot!(p, strategy_dates, spx_pct, 
          label="S&P 500 Buy & Hold", 
          linewidth=2, 
          color=:red,
          linestyle=:dash)
    
    # Add performance annotations
    annotate!(p, strategy_dates[end÷2], maximum(strategy_pct)*0.8, 
             text("Soliton CAGR: $(round(strategy_cagr*100, digits=1))%\nS&P 500 CAGR: $(round(spx_cagr*100, digits=1))%\nExcess: $(round(excess_cagr*100, digits=1))%", 
                  12, :left))
    
    # Customize plot
    plot!(p, legend=:topleft, grid=true, gridwidth=1, gridcolor=:lightgray)
    
    # Save plot
    savefig(p, "champion_soliton_strategy_performance.png")
    println("   ✅ Chart saved: champion_soliton_strategy_performance.png")
    
    # Also save as PDF for publication quality
    try
        savefig(p, "champion_soliton_strategy_performance.pdf")
        println("   ✅ PDF saved: champion_soliton_strategy_performance.pdf")
    catch
        println("   ⚠️  PDF save failed (no problem)")
    end
    
    # Display the plot
    display(p)
    
    # Save data for further analysis
    results_df = DataFrame(
        Date = strategy_dates,
        Soliton_Strategy_Value = strategy_values,
        SPX_Value = spx_values,
        Soliton_Return_Pct = strategy_pct,
        SPX_Return_Pct = spx_pct,
        Daily_Strategy_Return = strategy_returns
    )
    
    Arrow.write("champion_strategy_results.arrow", results_df)
    println("   ✅ Data saved: champion_strategy_results.arrow")
    
else
    println("❌ No data generated for visualization")
end

println("\nStrategy Visualization Complete!") 