using DataFrames, CSV, Statistics, LinearAlgebra, Plots, Dates, Arrow
using MLJ, MLJLinearModels, MLJDecisionTreeInterface
include("src/SolitonPDE_adaptive.jl")
include("src/Oscillators_enhanced.jl")

# Configuration
const INITIAL_CAPITAL = 10000.0
const TRANSACTION_COST = 0.001  # 0.1%
const STOP_LOSS = 0.02  # 2%
const TAKE_PROFIT = 0.04  # 4%
const MIN_POSITION_SIZE = 0.1  # 10% of capital
const MAX_POSITION_SIZE = 0.5  # 50% of capital

# Online Learning Parameters
const RETRAIN_FREQUENCY = 50  # Retrain every 50 predictions
const MIN_TRAINING_SIZE = 200  # Minimum samples for training
const MAX_TRAINING_SIZE = 1000  # Maximum samples to keep in memory
const VALIDATION_WINDOW = 100  # Use last 100 samples for validation

mutable struct OnlineModel
    model::Any
    machine::Union{Nothing, MLJ.Machine}
    last_retrain::Int
    training_data::Vector{Tuple{Vector{Float64}, Float64}}
    validation_data::Vector{Tuple{Vector{Float64}, Float64}}
    performance_history::Vector{Float64}
    prediction_history::Vector{Float64}
end

mutable struct OnlineTrade
    entry_time::Int
    entry_price::Float64
    position_size::Float64
    direction::Symbol  # :long or :short
    exit_time::Union{Int, Nothing}
    exit_price::Union{Float64, Nothing}
    pnl::Union{Float64, Nothing}
    successful::Union{Bool, Nothing}
end

struct OnlineTradingResult
    symbol::String
    model_name::String
    trades::Vector{OnlineTrade}
    equity_curve::Vector{Float64}
    total_return::Float64
    win_rate::Float64
    total_trades::Int
    avg_trade_return::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    learning_curve::Vector{Float64}
    prediction_accuracy::Vector{Float64}
end

function create_online_model(model_type::String)
    """Create an online learning model"""
    
    if model_type == "Linear"
        model = @load LinearRegressor pkg=MLJLinearModels
        return OnlineModel(model(), nothing, 0, [], [], [], [])
    elseif model_type == "Ridge"
        model = @load RidgeRegressor pkg=MLJLinearModels
        return OnlineModel(model(lambda=0.1), nothing, 0, [], [], [], [])
    elseif model_type == "DecisionTree"
        # Use a simpler model since DecisionTreeRegressor is not available
        model = @load LinearRegressor pkg=MLJLinearModels
        return OnlineModel(model(), nothing, 0, [], [], [], [])
    else
        # Fallback to linear
        model = @load LinearRegressor pkg=MLJLinearModels
        return OnlineModel(model(), nothing, 0, [], [], [], [])
    end
end

function train_online_model!(online_model::OnlineModel, X_train::Matrix, y_train::Vector)
    """Train the online model with new data"""
    
    try
        # Prepare data
        df_train = DataFrame(X_train, :auto)
        df_train.target = y_train
        
        # Create and train machine
        mach = machine(online_model.model, select(df_train, Not(:target)), df_train.target)
        MLJ.fit!(mach, verbosity=0)
        
        # Store the trained machine
        online_model.machine = mach
        online_model.last_retrain = length(online_model.training_data)
        
        return true
    catch e
        println("      WARNING:  Training error: $e")
        return false
    end
end

function predict_online(online_model::OnlineModel, X::Matrix)
    """Make prediction using online model"""
    
    try
        if online_model.machine === nothing
            return 0.0
        end
        
        df_test = DataFrame(X, :auto)
        pred = MLJ.predict(online_model.machine, df_test)
        return pred[1]  # Return single prediction
    catch e
        println("      WARNING:  Prediction error: $e")
        return 0.0
    end
end

function update_online_model!(online_model::OnlineModel, features::Vector{Float64}, actual_return::Float64)
    """Update online model with new data point"""
    
    # Add to training data
    push!(online_model.training_data, (features, actual_return))
    
    # Maintain maximum training size
    if length(online_model.training_data) > MAX_TRAINING_SIZE
        online_model.training_data = online_model.training_data[end-MAX_TRAINING_SIZE+1:end]
    end
    
    # Check if we should retrain
    if length(online_model.training_data) >= MIN_TRAINING_SIZE && 
       length(online_model.training_data) - online_model.last_retrain >= RETRAIN_FREQUENCY
        
        # Prepare training data
        X_train = Matrix(hcat([t[1] for t in online_model.training_data]...)')  # Transpose to get features as columns
        y_train = Vector{Float64}([t[2] for t in online_model.training_data])
        
        # Retrain model
        success = train_online_model!(online_model, X_train, y_train)
        
        if success
            println("      🔄 Retrained model with $(length(y_train)) samples")
        end
    end
end

function extract_enhanced_soliton_features(row)
    """Extract enhanced soliton features from a data row"""
    
    # Basic oscillators
    rsi = row.RSI14
    stoch_k = row.StochK14
    cci = row.CCI20
    macd_signal = row.MACDsig
    
    # Soliton features from Arrow data
    soliton_height = row.SolitonHeight
    soliton_probe1 = row.SolitonProbe1
    soliton_probe2 = row.SolitonProbe2
    soliton_probe3 = row.SolitonProbe3
    soliton_probe_mean = row.SolitonProbeMean
    soliton_probe_std = row.SolitonProbeStd
    soliton_probe_max = row.SolitonProbeMax
    soliton_energy = row.SolitonEnergy
    soliton_energy_density = row.SolitonEnergyDensity
    soliton_asymmetry_x = row.SolitonAsymmetryX
    soliton_asymmetry_y = row.SolitonAsymmetryY
    soliton_concentration = row.SolitonConcentration
    
    # Create comprehensive feature vector
    features = [
        # Basic oscillators
        rsi, stoch_k, cci, macd_signal,
        
        # Soliton features
        soliton_height,
        soliton_probe1, soliton_probe2, soliton_probe3,
        soliton_probe_mean, soliton_probe_std, soliton_probe_max,
        soliton_energy, soliton_energy_density,
        soliton_asymmetry_x, soliton_asymmetry_y,
        soliton_concentration,
        
        # Derived features
        soliton_probe_mean * soliton_energy,
        soliton_asymmetry_x * soliton_asymmetry_y,
        soliton_concentration * soliton_energy_density
    ]
    
    return features
end

function prepare_online_trading_data(symbol::String)
    """Prepare data for online learning trading"""
    
    println("INFO: Loading data for $symbol...")
    
    # Load Arrow data
    data_path = "data/processed/soliton_features.arrow"
    if !isfile(data_path)
        println("   FAILED: No processed data found")
        return nothing
    end
    
    df = DataFrame(Arrow.Table(data_path))
    df_symbol = filter(row -> row.Symbol == symbol, df)
    
    # Ensure we have enough data
    if nrow(df_symbol) < 500
        println("   FAILED: Insufficient data ($(nrow(df_symbol)) rows)")
        return nothing
    end
    
    # Split data into training (first half) and testing (second half)
    split_point = div(nrow(df_symbol), 2)
    train_df = df_symbol[1:split_point, :]
    test_df = df_symbol[split_point+1:end, :]
    
    println("   UP: Training period: $(first(train_df.Date)) to $(last(train_df.Date))")
    println("   UP: Testing period: $(first(test_df.Date)) to $(last(test_df.Date))")
    
    return (train_df, test_df)
end

function run_online_trading_simulation(symbol::String, model_type::String)
    """Run online learning trading simulation"""
    
    # Prepare data
    data_result = prepare_online_trading_data(symbol)
    if data_result === nothing
        return nothing
    end
    
    train_df, test_df = data_result
    
    println("STRONG Running online learning simulation for $symbol with $model_type model...")
    
    # Initialize online model
    online_model = create_online_model(model_type)
    
    # Initialize trading variables
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = OnlineTrade[]
    current_position = nothing
    learning_curve = Float64[]
    prediction_accuracy = Float64[]
    
    # Phase 1: Initial training on first half
    println("   📚 Phase 1: Initial training on historical data...")
    
    training_features = []
    training_returns = []
    
    for i in 2:nrow(train_df)
        row = train_df[i, :]
        prev_row = train_df[i-1, :]
        
        # Calculate forward return
        forward_return = (row.Close - prev_row.Close) / prev_row.Close
        
        # Extract features
        features = extract_enhanced_soliton_features(prev_row)
        
        if !any(isnan.(features)) && !any(isinf.(features)) && !ismissing(forward_return) && !isnan(forward_return)
            push!(training_features, features)
            push!(training_returns, Float64(forward_return))
        end
    end
    
    # Initial training
    if length(training_features) >= MIN_TRAINING_SIZE
        X_train = Matrix(hcat(training_features...)')  # Transpose to get features as columns
        y_train = Vector{Float64}(training_returns)
        
        success = train_online_model!(online_model, X_train, y_train)
        if success
            println("   SUCCESS: Initial training completed with $(length(y_train)) samples")
        else
            println("   FAILED: Initial training failed")
            return nothing
        end
    else
        println("   FAILED: Insufficient training data")
        return nothing
    end
    
    # Phase 2: Online learning and trading on second half
    println("   UP: Phase 2: Online learning and trading...")
    
    for i in 2:nrow(test_df)
        row = test_df[i, :]
        prev_row = test_df[i-1, :]
        
        # Calculate actual forward return
        actual_return = (row.Close - prev_row.Close) / prev_row.Close
        
        # Extract features for prediction
        features = extract_enhanced_soliton_features(prev_row)
        
        if !any(isnan.(features)) && !any(isinf.(features)) && !ismissing(actual_return) && !isnan(actual_return)
            # Make prediction
            prediction = predict_online(online_model, reshape(features, 1, length(features)))
            
            # Track prediction accuracy
            accuracy = 1.0 - abs(prediction - actual_return)
            push!(prediction_accuracy, accuracy)
            
            # Update model with actual result
            update_online_model!(online_model, features, Float64(actual_return))
            
            # Trading logic
            if current_position === nothing
                # No position - consider entry
                if abs(prediction) > 0.001  # Minimum signal threshold
                    # Calculate position size based on prediction strength
                    position_size = min(MAX_POSITION_SIZE, max(MIN_POSITION_SIZE, abs(prediction) * 10))
                    
                    if prediction > 0
                        # Long position
                        entry_price = row.Close
                        shares = (capital * position_size) / entry_price
                        current_position = OnlineTrade(
                            i, entry_price, position_size, :long, nothing, nothing, nothing, nothing
                        )
                        capital -= shares * entry_price * (1 + TRANSACTION_COST)
                    else
                        # Short position
                        entry_price = row.Close
                        shares = (capital * position_size) / entry_price
                        current_position = OnlineTrade(
                            i, entry_price, position_size, :short, nothing, nothing, nothing, nothing
                        )
                        capital += shares * entry_price * (1 - TRANSACTION_COST)
                    end
                end
            else
                # Have position - check exit conditions
                current_price = row.Close
                pnl = 0.0
                
                if current_position.direction == :long
                    pnl = (current_price - current_position.entry_price) / current_position.entry_price
                else
                    pnl = (current_position.entry_price - current_price) / current_position.entry_price
                end
                
                # Check exit conditions
                should_exit = false
                successful = false
                
                if pnl >= TAKE_PROFIT
                    should_exit = true
                    successful = true
                elseif pnl <= -STOP_LOSS
                    should_exit = true
                    successful = false
                elseif abs(prediction) < 0.0005  # Exit on weak signal
                    should_exit = true
                    successful = pnl > 0
                end
                
                if should_exit
                    # Close position
                    if current_position.direction == :long
                        shares = (INITIAL_CAPITAL * current_position.position_size) / current_position.entry_price
                        capital += shares * current_price * (1 - TRANSACTION_COST)
                    else
                        shares = (INITIAL_CAPITAL * current_position.position_size) / current_position.entry_price
                        capital -= shares * current_price * (1 + TRANSACTION_COST)
                    end
                    
                    # Update trade
                    current_position.exit_time = i
                    current_position.exit_price = current_price
                    current_position.pnl = pnl
                    current_position.successful = successful
                    
                    push!(trades, current_position)
                    current_position = nothing
                end
            end
            
            # Update equity curve
            push!(equity_curve, capital)
            
            # Track learning progress
            if length(prediction_accuracy) % 50 == 0
                recent_accuracy = mean(prediction_accuracy[max(1, end-49):end])
                push!(learning_curve, recent_accuracy)
            end
        end
    end
    
    # Close any remaining position
    if current_position !== nothing
        current_position.exit_time = nrow(test_df)
        current_position.exit_price = test_df[end, :Close]
        
        if current_position.direction == :long
            pnl = (current_position.exit_price - current_position.entry_price) / current_position.entry_price
        else
            pnl = (current_position.entry_price - current_position.exit_price) / current_position.entry_price
        end
        
        current_position.pnl = pnl
        current_position.successful = pnl > 0
        push!(trades, current_position)
    end
    
    # Calculate performance metrics
    total_return = (last(equity_curve) - INITIAL_CAPITAL) / INITIAL_CAPITAL
    successful_trades = filter(t -> t.successful === true, trades)
    win_rate = length(successful_trades) / max(1, length(trades))
    total_trades = length(trades)
    
    avg_trade_return = isempty(trades) ? 0.0 : mean([t.pnl for t in trades])
    
    # Calculate Sharpe ratio
    returns = diff(equity_curve) ./ equity_curve[1:end-1]
    sharpe_ratio = isempty(returns) ? 0.0 : mean(returns) / (std(returns) + 1e-8)
    
    # Calculate max drawdown
    peak = equity_curve[1]
    max_drawdown = 0.0
    for value in equity_curve
        if value > peak
            peak = value
        end
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    end
    
    println("   SUCCESS: Online learning completed!")
    println("      INFO: Total return: $(round(total_return * 100, digits=2))%")
    println("      INFO: Total trades: $total_trades")
    println("      INFO: Win rate: $(round(win_rate * 100, digits=1))%")
    println("      INFO: Avg trade return: $(round(avg_trade_return * 100, digits=2))%")
    println("      INFO: Sharpe ratio: $(round(sharpe_ratio, digits=3))")
    println("      INFO: Max drawdown: $(round(max_drawdown * 100, digits=2))%")
    
    return OnlineTradingResult(
        symbol, model_type, trades, equity_curve, total_return, win_rate, 
        total_trades, avg_trade_return, sharpe_ratio, max_drawdown,
        learning_curve, prediction_accuracy
    )
end

function create_online_learning_visualization(results)
    """Create visualization for online learning results"""
    
    println("INFO: Creating online learning visualization...")
    
    # Performance comparison
    symbols = [r.symbol for r in results]
    model_types = [r.model_name for r in results]
    returns = [r.total_return * 100 for r in results]
    win_rates = [r.win_rate * 100 for r in results]
    trade_counts = [r.total_trades for r in results]
    
    # Performance scatter plot
    p1 = scatter(returns, win_rates, 
                xlabel="Total Return (%)", ylabel="Win Rate (%)",
                title="Online Learning Performance",
                label="", markersize=6, color=:blue, alpha=0.7)
    
    # Add stock labels
    for (i, symbol) in enumerate(symbols)
        annotate!(p1, returns[i], win_rates[i], text(symbol, 8, :center))
    end
    
    # Learning curves
    p2 = plot(title="Learning Progress", xlabel="Time (50-prediction intervals)", ylabel="Prediction Accuracy")
    
    for result in results
        if !isempty(result.learning_curve)
            plot!(p2, result.learning_curve, 
                  label="$(result.symbol) ($(result.model_name))", 
                  linewidth=2, alpha=0.8)
        end
    end
    
    # Equity curves
    p3 = plot(title="Equity Curves", xlabel="Time", ylabel="Capital (\$)")
    
    for result in results
        normalized_curve = result.equity_curve ./ INITIAL_CAPITAL
        plot!(p3, normalized_curve, 
              label="$(result.symbol) ($(result.model_name))", 
              linewidth=2, alpha=0.8)
    end
    
    # Add trade markers for best performing model
    best_result = results[argmax(returns)]
    if !isempty(best_result.trades)
        successful_trades = filter(t -> t.successful === true, best_result.trades)
        failed_trades = filter(t -> t.successful === false, best_result.trades)
        
        if !isempty(successful_trades)
            times = [t.entry_time for t in successful_trades]
            values = [best_result.equity_curve[t.entry_time] / INITIAL_CAPITAL for t in successful_trades]
            scatter!(p3, times, values, color=:green, markersize=3, alpha=0.7, label="Wins")
        end
        
        if !isempty(failed_trades)
            times = [t.entry_time for t in failed_trades]
            values = [best_result.equity_curve[t.entry_time] / INITIAL_CAPITAL for t in failed_trades]
            scatter!(p3, times, values, color=:red, markersize=3, alpha=0.7, label="Losses")
        end
    end
    
    # Summary statistics
    p4 = plot(title="Performance Summary", layout=(2,2))
    
    # Returns distribution
    histogram!(p4[1,1], returns, bins=10, title="Return Distribution", 
              xlabel="Return (%)", ylabel="Count", color=:blue, alpha=0.7)
    
    # Win rate vs trade count
    scatter!(p4[1,2], trade_counts, win_rates, title="Win Rate vs Trade Count",
             xlabel="Number of Trades", ylabel="Win Rate (%)", color=:green, alpha=0.7)
    
    # Sharpe ratio distribution
    sharpe_ratios = [r.sharpe_ratio for r in results]
    histogram!(p4[2,1], sharpe_ratios, bins=10, title="Sharpe Ratio Distribution",
              xlabel="Sharpe Ratio", ylabel="Count", color=:orange, alpha=0.7)
    
    # Max drawdown distribution
    max_drawdowns = [r.max_drawdown * 100 for r in results]
    histogram!(p4[2,2], max_drawdowns, bins=10, title="Max Drawdown Distribution",
              xlabel="Max Drawdown (%)", ylabel="Count", color=:red, alpha=0.7)
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
    # Save plot
    savefig(final_plot, "visualizations/online_learning_results.png")
    println("   💾 Saved visualization to visualizations/online_learning_results.png")
    
    return final_plot
end

function compute_rolling_win_rate(trades::Vector{OnlineTrade}; window::Int=50)
    n = length(trades)
    if n < window
        window = max(2, Int(floor(n/2)))
    end
    win_rates = Float64[]
    indices = Int[]
    for i in window:n
        window_trades = trades[i-window+1:i]
        wins = count(t -> t.successful === true, window_trades)
        push!(win_rates, wins / window)
        push!(indices, i)
    end
    return indices, win_rates
end

function create_per_stock_report(result::OnlineTradingResult)
    symbol = result.symbol
    trades = result.trades
    equity_curve = result.equity_curve
    learning_curve = result.learning_curve
    prediction_accuracy = result.prediction_accuracy
    
    # Equity curve plot
    p1 = plot(equity_curve, title="Equity Curve for $symbol", xlabel="Time Step", ylabel="Capital", legend=false, grid=true, color=:blue)
    
    # Trade markers
    successful_trades = filter(t -> t.successful === true, trades)
    failed_trades = filter(t -> t.successful === false, trades)
    if !isempty(successful_trades)
        scatter!(p1, [t.entry_time for t in successful_trades], [equity_curve[t.entry_time] for t in successful_trades], color=:green, markersize=6, alpha=0.8, label="Wins")
    end
    if !isempty(failed_trades)
        scatter!(p1, [t.entry_time for t in failed_trades], [equity_curve[t.entry_time] for t in failed_trades], color=:red, markersize=6, alpha=0.8, label="Losses")
    end
    
    # Learning curve
    p2 = plot(learning_curve, title="Learning Curve (Prediction Accuracy)", xlabel="Interval (x50)", ylabel="Accuracy", legend=false, grid=true, color=:orange)
    hline!(p2, [mean(learning_curve)], color=:gray, linestyle=:dash, label="Mean Accuracy")
    
    # Rolling win rate
    indices, win_rates = compute_rolling_win_rate(trades, window=50)
    p3 = plot(indices, win_rates, title="Rolling Win Rate", xlabel="Trade #", ylabel="Win Rate", legend=false, ylim=(0,1), grid=true, color=:purple)
    if !isempty(win_rates)
        hline!(p3, [mean(win_rates)], color=:gray, linestyle=:dash, label="Mean Win Rate")
    end
    
    # Clean summary stats panel (no diagonal line)
    stats_text = """
    Total Return: $(round(result.total_return*100, digits=2))%
    Win Rate: $(round(result.win_rate*100, digits=1))%
    Total Trades: $(result.total_trades)
    Avg Trade Return: $(round(result.avg_trade_return*100, digits=2))%
    Sharpe Ratio: $(round(result.sharpe_ratio, digits=3))
    Max Drawdown: $(round(result.max_drawdown*100, digits=2))%
    """
    p4 = plot(legend=false, grid=false, axis=false, framestyle=:none, title="Summary Stats")
    annotate!(p4, 0.5, 0.5, text(stats_text, :left, 12))
    
    # Combine
    final = plot(p1, p2, p3, p4, layout=(2,2), size=(1200,800))
    savefig(final, "visualizations/outputs/$(symbol)_report.png")
    println("   💾 Saved per-stock report to visualizations/outputs/$(symbol)_report.png")
end

# Main execution
println("STRONG Starting ONLINE LEARNING Soliton Trading System")
println("=" ^ 60)

# Update test_symbols to include more stocks
# Example: Top 20 US stocks by market cap
# You can further expand or customize this list as needed

test_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JPM",
    "V", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "CVX", "HD", "MRK"
]

model_types = ["Linear", "Ridge", "DecisionTree"]

results = []

for symbol in test_symbols
    for model_type in model_types
        println("\nUP: Testing $symbol with $model_type model...")
        
        result = run_online_trading_simulation(symbol, model_type)
        
        if result !== nothing
            push!(results, result)
        end
    end
end

if !isempty(results)
    println("\n" * "=" ^ 60)
    println("INFO: ONLINE LEARNING RESULTS SUMMARY")
    println("=" ^ 60)
    
    # Sort by total return
    sort!(results, by=r -> r.total_return, rev=true)
    
    println("\n🏆 TOP PERFORMERS:")
    for (i, result) in enumerate(results[1:min(5, length(results))])
        println("   $i. $(result.symbol) ($(result.model_name)): $(round(result.total_return * 100, digits=2))% return, $(result.total_trades) trades, $(round(result.win_rate * 100, digits=1))% win rate")
    end
    
    # Create visualization
    create_online_learning_visualization(results)
    
    for result in results
        create_per_stock_report(result)
    end
    
    println("\nSUCCESS: Online learning trading system completed!")
else
    println("\nFAILED: No successful results to analyze")
end 