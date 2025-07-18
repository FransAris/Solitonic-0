"""
Oscillators Module

Computes technical oscillators (RSI, Stochastic %K, CCI, MACD-signal) 
and normalizes them to [-1,1] for use as soliton launch amplitudes.
"""
module Oscillators

export compute_oscillators

using DataFrames, MarketTechnicals
using Statistics

"""
    compute_oscillators(df::DataFrame) -> DataFrame

Add columns :RSI14, :StochK14, :CCI20, :MACDsig to an OHLCV frame.
Then min-max scale each to [-1,1] across the *input* frame.

# Arguments
- `df::DataFrame`: Input OHLCV data with columns :Open, :High, :Low, :Close, :Volume

# Returns
- `DataFrame`: Original data plus oscillator columns normalized to [-1,1]

# Example
```julia
df = DataFrame(
    Date = Date.(["2022-01-01", "2022-01-02", "2022-01-03"]),
    Open = [100.0, 101.0, 102.0],
    High = [102.0, 103.0, 104.0], 
    Low = [99.0, 100.0, 101.0],
    Close = [101.0, 102.0, 103.0],
    Volume = [1000, 1100, 1200]
)
result = compute_oscillators(df)
```
"""
function compute_oscillators(df::DataFrame)::DataFrame
    # TODO: Verify required columns exist
    required_cols = [:Open, :High, :Low, :Close, :Volume]
    df_cols = Symbol.(names(df))  # Convert string names to symbols
    missing_cols = setdiff(required_cols, df_cols)
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    
    # Create working copy
    result_df = copy(df)
    n_rows = nrow(result_df)
    
    if n_rows < 26  # Need enough data for indicators
        @warn "Insufficient data for oscillators (need ≥26 rows, got $n_rows)"
        # Fill with zeros for now
        result_df.RSI14 = zeros(n_rows)
        result_df.StochK14 = zeros(n_rows) 
        result_df.CCI20 = zeros(n_rows)
        result_df.MACDsig = zeros(n_rows)
        return result_df
    end
    
    # Extract price arrays for MarketTechnicals
    close_prices = result_df.Close
    high_prices = result_df.High
    low_prices = result_df.Low
    
    # 1. RSI (14-period)
    # TODO: Use MarketTechnicals.rsi when available
    rsi_values = _compute_rsi(close_prices, 14)
    
    # 2. Stochastic %K (14-period)  
    # TODO: Use MarketTechnicals stochastic when available
    stoch_k_values = _compute_stochastic_k(high_prices, low_prices, close_prices, 14)
    
    # 3. CCI (20-period)
    # TODO: Use MarketTechnicals.cci when available  
    cci_values = _compute_cci(high_prices, low_prices, close_prices, 20)
    
    # 4. MACD Signal line
    # TODO: Use MarketTechnicals.macd when available
    macd_sig_values = _compute_macd_signal(close_prices, 12, 26, 9)
    
    # Add raw oscillator values
    result_df.RSI14_raw = rsi_values
    result_df.StochK14_raw = stoch_k_values  
    result_df.CCI20_raw = cci_values
    result_df.MACDsig_raw = macd_sig_values
    
    # Normalize each oscillator to [-1, 1] using min-max scaling
    result_df.RSI14 = _normalize_to_range(rsi_values, -1.0, 1.0)
    result_df.StochK14 = _normalize_to_range(stoch_k_values, -1.0, 1.0)
    result_df.CCI20 = _normalize_to_range(cci_values, -1.0, 1.0)
    result_df.MACDsig = _normalize_to_range(macd_sig_values, -1.0, 1.0)
    
    return result_df
end

"""
Simple RSI implementation (placeholder until MarketTechnicals is confirmed)
"""
function _compute_rsi(prices::Vector{Float64}, period::Int)::Vector{Float64}
    n = length(prices)
    rsi = fill(NaN, n)
    
    if n < period + 1
        return rsi
    end
    
    # Calculate price changes
    changes = diff(prices)
    
    # Separate gains and losses
    gains = max.(changes, 0.0)
    losses = -min.(changes, 0.0)
    
    # Calculate initial averages
    avg_gain = mean(gains[1:period])
    avg_loss = mean(losses[1:period])
    
    # Calculate RSI for each point after initial period
    for i in (period+1):n
        if i > period + 1
            # Smoothed averages (Wilder's smoothing)
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        end
        
        if avg_loss == 0.0
            rsi[i] = 100.0
        else
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        end
    end
    
    return rsi
end

"""
Simple Stochastic %K implementation
"""
function _compute_stochastic_k(highs::Vector{Float64}, lows::Vector{Float64}, 
                              closes::Vector{Float64}, period::Int)::Vector{Float64}
    n = length(closes)
    stoch_k = fill(NaN, n)
    
    for i in period:n
        period_high = maximum(highs[(i-period+1):i])
        period_low = minimum(lows[(i-period+1):i])
        
        if period_high == period_low
            stoch_k[i] = 50.0  # Neutral when no range
        else
            stoch_k[i] = 100.0 * (closes[i] - period_low) / (period_high - period_low)
        end
    end
    
    return stoch_k
end

"""
Simple CCI implementation
"""
function _compute_cci(highs::Vector{Float64}, lows::Vector{Float64}, 
                     closes::Vector{Float64}, period::Int)::Vector{Float64}
    n = length(closes)
    cci = fill(NaN, n)
    
    # Typical Price = (H + L + C) / 3
    typical_prices = (highs .+ lows .+ closes) ./ 3.0
    
    for i in period:n
        tp_subset = typical_prices[(i-period+1):i]
        sma_tp = mean(tp_subset)
        mean_deviation = mean(abs.(tp_subset .- sma_tp))
        
        if mean_deviation == 0.0
            cci[i] = 0.0
        else
            cci[i] = (typical_prices[i] - sma_tp) / (0.015 * mean_deviation)
        end
    end
    
    return cci
end

"""
Simple MACD Signal line implementation
"""
function _compute_macd_signal(prices::Vector{Float64}, fast::Int, slow::Int, signal::Int)::Vector{Float64}
    n = length(prices)
    
    if n < slow + signal
        return fill(NaN, n)
    end
    
    # Calculate EMAs
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    
    # MACD line = EMA_fast - EMA_slow  
    macd_line = fill(NaN, n)
    for i in 1:n
        if !isnan(ema_fast[i]) && !isnan(ema_slow[i])
            macd_line[i] = ema_fast[i] - ema_slow[i]
        end
    end
    
    # Signal line = EMA of MACD line (only use valid MACD values)
    # Find first valid MACD value
    first_valid = findfirst(!isnan, macd_line)
    if first_valid === nothing
        return fill(NaN, n)
    end
    
    # Create a clean MACD series for signal calculation
    valid_macd = macd_line[first_valid:end]
    if length(valid_macd) < signal
        return fill(NaN, n)
    end
    
    # Calculate signal line EMA on valid MACD values
    signal_ema = _ema(valid_macd, signal)
    
    # Insert back into full-length array
    macd_signal = fill(NaN, n)
    for i in 1:length(signal_ema)
        if !isnan(signal_ema[i])
            macd_signal[first_valid + i - 1] = signal_ema[i]
        end
    end
    
    return macd_signal
end

"""
Exponential Moving Average calculation
"""
function _ema(prices::Vector{Float64}, period::Int)::Vector{Float64}
    n = length(prices)
    ema = fill(NaN, n)
    
    if n < period
        return ema
    end
    
    # Smoothing factor
    α = 2.0 / (period + 1.0)
    
    # Initialize with SMA
    ema[period] = mean(prices[1:period])
    
    # Calculate EMA
    for i in (period+1):n
        ema[i] = α * prices[i] + (1.0 - α) * ema[i-1]
    end
    
    return ema
end

"""
Normalize values to specified range using min-max scaling
"""
function _normalize_to_range(values::Vector{Float64}, min_val::Float64, max_val::Float64)::Vector{Float64}
    # Filter out NaN values for min/max calculation
    valid_values = filter(!isnan, values)
    
    if isempty(valid_values)
        return fill(0.0, length(values))  # Return zeros if all NaN
    end
    
    data_min = minimum(valid_values)
    data_max = maximum(valid_values)
    
    if data_max == data_min
        # No variation - return middle of target range
        return fill((min_val + max_val) / 2.0, length(values))
    end
    
    # Min-max normalization: (x - min) / (max - min) * (new_max - new_min) + new_min
    normalized = map(values) do x
        if isnan(x)
            return NaN
        else
            return (x - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        end
    end
    
    return normalized
end

end # module Oscillators 