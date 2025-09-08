"""
Enhanced Oscillators Module

Computes technical oscillators with time-frequency-aware parameter scaling.
Optimized for both daily and intraday data (5min, 15min, 30min, 1hour).
"""
module Oscillators_enhanced

export compute_oscillators_enhanced, get_optimal_periods, TimeFrequency, Daily, Hour1, Min30, Min15, Min5, Min1

using DataFrames, Statistics

# Enum for supported time frequencies
@enum TimeFrequency begin
    Daily = 1440      # Daily = 1440 minutes
    Hour1 = 60        # 1 hour = 60 minutes
    Min30 = 30        # 30 minutes  
    Min15 = 15        # 15 minutes
    Min5 = 5          # 5 minutes
    Min1 = 1          # 1 minute
end

"""
    get_optimal_periods(freq::TimeFrequency) -> NamedTuple

Get optimal oscillator periods for different time frequencies.
Scales periods to capture similar market cycles across timeframes.

# Time Frequency Analysis:
- Daily: Traditional periods (RSI14, StochK14, CCI20, MACD 12/26/9)
- 1hour: Scale up 6x (RSI84, StochK84, CCI120, MACD 72/156/54) 
- 30min: Scale up 12x (RSI168, StochK168, CCI240, MACD 144/312/108)
- 15min: Scale up 24x (RSI336, StochK336, CCI480, MACD 288/624/216)
- 5min: Scale up 72x (RSI1008, StochK1008, CCI1440, MACD 864/1872/648)
- 1min: Scale up 360x (RSI5040, StochK5040, CCI7200, MACD 4320/9360/3240)

Note: Very high-frequency (1min, 5min) may need alternative approaches due to noise.
"""
function get_optimal_periods(freq::TimeFrequency)::NamedTuple
    if freq == Daily
        return (
            rsi = 14,
            stoch = 14, 
            cci = 20,
            macd_fast = 12,
            macd_slow = 26,
            macd_signal = 9
        )
    elseif freq == Hour1
        # 1 hour: 6 trading hours per day, so scale by 6x
        return (
            rsi = 84,           # 14 * 6 = 84 hours ≈ 14 days
            stoch = 84,         # 14 * 6 = 84 hours ≈ 14 days
            cci = 120,          # 20 * 6 = 120 hours ≈ 20 days
            macd_fast = 72,     # 12 * 6 = 72 hours ≈ 12 days
            macd_slow = 156,    # 26 * 6 = 156 hours ≈ 26 days
            macd_signal = 54    # 9 * 6 = 54 hours ≈ 9 days
        )
    elseif freq == Min30
        # 30 min: 12 periods per trading day, so scale by 12x
        return (
            rsi = 168,          # 14 * 12 = 168 periods ≈ 14 days
            stoch = 168,        # 14 * 12 = 168 periods ≈ 14 days  
            cci = 240,          # 20 * 12 = 240 periods ≈ 20 days
            macd_fast = 144,    # 12 * 12 = 144 periods ≈ 12 days
            macd_slow = 312,    # 26 * 12 = 312 periods ≈ 26 days
            macd_signal = 108   # 9 * 12 = 108 periods ≈ 9 days
        )
    elseif freq == Min15
        # 15 min: 24 periods per trading day, so scale by 24x
        return (
            rsi = 336,          # 14 * 24 = 336 periods ≈ 14 days
            stoch = 336,        # 14 * 24 = 336 periods ≈ 14 days
            cci = 480,          # 20 * 24 = 480 periods ≈ 20 days
            macd_fast = 288,    # 12 * 24 = 288 periods ≈ 12 days
            macd_slow = 624,    # 26 * 24 = 624 periods ≈ 26 days
            macd_signal = 216   # 9 * 24 = 216 periods ≈ 9 days
        )
    elseif freq == Min5
        # 5 min: 72 periods per trading day, but cap for practicality
        # Use shorter periods due to noise and computational limits
        return (
            rsi = 280,          # Shorter than 14*72 due to noise
            stoch = 280,        # Shorter than 14*72 due to noise
            cci = 400,          # Shorter than 20*72 due to noise
            macd_fast = 240,    # Shorter than 12*72 due to noise
            macd_slow = 520,    # Shorter than 26*72 due to noise
            macd_signal = 180   # Shorter than 9*72 due to noise
        )
    elseif freq == Min1
        # 1 min: 360 periods per trading day, but use much shorter for noise
        # Very short periods for 1-minute data due to high noise
        return (
            rsi = 60,           # 1 hour of 1-min data
            stoch = 60,         # 1 hour of 1-min data
            cci = 80,           # ~1.3 hours of 1-min data
            macd_fast = 48,     # ~48 minutes
            macd_slow = 104,    # ~1.7 hours
            macd_signal = 36    # ~36 minutes
        )
    else
        error("Unsupported time frequency: $freq")
    end
end

"""
    detect_frequency(df::DataFrame) -> TimeFrequency

Automatically detect time frequency from datetime differences in DataFrame.
Assumes DataFrame has a datetime or Date column.
"""
function detect_frequency(df::DataFrame)::TimeFrequency
    # Find datetime column
    datetime_col = nothing
    for col in names(df)
        if col in ["datetime", "Date", "timestamp", "time"]
            datetime_col = col
            break
        end
    end
    
    if datetime_col === nothing
        @warn "No datetime column found, assuming daily frequency"
        return Daily
    end
    
    if nrow(df) < 2
        @warn "Insufficient data to detect frequency, assuming daily"
        return Daily
    end
    
    # Calculate median time difference
    time_diffs = diff(df[!, datetime_col])
    median_diff = median(time_diffs)
    
    # Convert to minutes for comparison
    if isa(median_diff, Dates.Day)
        minutes = Dates.value(median_diff) * 24 * 60
    elseif isa(median_diff, Dates.Hour)
        minutes = Dates.value(median_diff) * 60
    elseif isa(median_diff, Dates.Minute)
        minutes = Dates.value(median_diff)
    else
        # Try to convert to minutes
        minutes = Dates.value(median_diff) / (1000 * 60)  # Assume milliseconds
    end
    
    # Classify frequency based on median difference
    if minutes >= 1200      # >= 20 hours (daily)
        return Daily
    elseif minutes >= 45    # 45-75 minutes (1 hour)
        return Hour1  
    elseif minutes >= 20    # 20-40 minutes (30 min)
        return Min30
    elseif minutes >= 10    # 10-20 minutes (15 min)
        return Min15
    elseif minutes >= 3     # 3-10 minutes (5 min)
        return Min5
    else                    # < 3 minutes (1 min)
        return Min1
    end
end

"""
    compute_oscillators_enhanced(df::DataFrame; freq::Union{TimeFrequency,Nothing}=nothing) -> DataFrame

Enhanced oscillator computation with automatic frequency detection and parameter scaling.

# Arguments
- `df::DataFrame`: Input OHLCV data
- `freq::Union{TimeFrequency,Nothing}`: Optional frequency override (auto-detected if nothing)

# Returns
- `DataFrame`: Original data plus oscillator columns with frequency-appropriate parameters

# Features
- Automatic time frequency detection
- Optimal period scaling for each frequency
- Noise filtering for high-frequency data  
- Backward compatibility with existing code
"""
function compute_oscillators_enhanced(df::DataFrame; freq::Union{TimeFrequency,Nothing}=nothing)::DataFrame
    # Validate input - check for both capitalized and lowercase column names
    required_cols_caps = [:Open, :High, :Low, :Close, :Volume]
    required_cols_lower = [:open, :high, :low, :close, :volume]
    df_cols = Symbol.(names(df))
    
    # Check if we have capitalized or lowercase columns
    has_caps = all(col in df_cols for col in required_cols_caps)
    has_lower = all(col in df_cols for col in required_cols_lower)
    
    if !has_caps && !has_lower
        missing_caps = setdiff(required_cols_caps, df_cols)
        missing_lower = setdiff(required_cols_lower, df_cols)
        error("Missing required columns. Need either $required_cols_caps or $required_cols_lower. Missing: caps=$missing_caps, lower=$missing_lower")
    end
    
    # Standardize to capitalized column names if needed
    if has_lower && !has_caps
        rename!(df, :open => :Open, :high => :High, :low => :Low, :close => :Close, :volume => :Volume)
    end
    
    # Create working copy
    result_df = copy(df)
    n_rows = nrow(result_df)
    
    # Detect or use provided frequency
    if freq === nothing
        freq = detect_frequency(df)
        println("INFO: Auto-detected frequency: $freq")
    else
        println("INFO: Using provided frequency: $freq")
    end
    
    # Get optimal periods for this frequency
    periods = get_optimal_periods(freq)
    println("INFO: Oscillator periods: RSI=$(periods.rsi), Stoch=$(periods.stoch), CCI=$(periods.cci)")
    println("INFO: MACD periods: $(periods.macd_fast)/$(periods.macd_slow)/$(periods.macd_signal)")
    
    # Check if we have enough data
    min_required = max(periods.rsi, periods.stoch, periods.cci, periods.macd_slow + periods.macd_signal)
    if n_rows < min_required
        @warn "Insufficient data for oscillators (need ≥$min_required rows, got $n_rows)"
        # Fill with zeros for now
        result_df.RSI_raw = zeros(n_rows)
        result_df.StochK_raw = zeros(n_rows) 
        result_df.CCI_raw = zeros(n_rows)
        result_df.MACDsig_raw = zeros(n_rows)
        result_df.RSI = zeros(n_rows)
        result_df.StochK = zeros(n_rows) 
        result_df.CCI = zeros(n_rows)
        result_df.MACDsig = zeros(n_rows)
        return result_df
    end
    
    # Extract price arrays
    close_prices = result_df.Close
    high_prices = result_df.High
    low_prices = result_df.Low
    
    # Compute oscillators with frequency-appropriate periods
    rsi_values = _compute_rsi(close_prices, periods.rsi)
    stoch_k_values = _compute_stochastic_k(high_prices, low_prices, close_prices, periods.stoch)
    cci_values = _compute_cci(high_prices, low_prices, close_prices, periods.cci)
    macd_sig_values = _compute_macd_signal(close_prices, periods.macd_fast, periods.macd_slow, periods.macd_signal)
    
    # Apply noise filtering for high-frequency data
    if freq in [Min1, Min5]
        println("INFO: Applying noise filtering for high-frequency data")
        rsi_values = _smooth_series(rsi_values, 3)
        stoch_k_values = _smooth_series(stoch_k_values, 3)  
        cci_values = _smooth_series(cci_values, 3)
        macd_sig_values = _smooth_series(macd_sig_values, 3)
    end
    
    # Store raw values
    result_df.RSI_raw = rsi_values
    result_df.StochK_raw = stoch_k_values  
    result_df.CCI_raw = cci_values
    result_df.MACDsig_raw = macd_sig_values
    
    # Normalize to [-1, 1] for soliton amplitudes
    result_df.RSI = _normalize_to_range(rsi_values, -1.0, 1.0)
    result_df.StochK = _normalize_to_range(stoch_k_values, -1.0, 1.0)
    result_df.CCI = _normalize_to_range(cci_values, -1.0, 1.0)
    result_df.MACDsig = _normalize_to_range(macd_sig_values, -1.0, 1.0)
    
    # Add metadata
    result_df.frequency = fill(string(freq), n_rows)
    
    println("INFO: Enhanced oscillators computed successfully")
    
    return result_df
end

"""
Simple moving average smoothing for noise reduction
"""
function _smooth_series(values::Vector{Float64}, window::Int)::Vector{Float64}
    n = length(values)
    smoothed = copy(values)
    
    for i in window:n
        if !isnan(values[i])
            # Calculate average of window, excluding NaN values
            window_values = values[max(1, i-window+1):i]
            valid_values = filter(!isnan, window_values)
            if !isempty(valid_values)
                smoothed[i] = mean(valid_values)
            end
        end
    end
    
    return smoothed
end

# Include all the existing oscillator calculation functions from the original module
"""
RSI implementation - same as original but with configurable period
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
Stochastic %K implementation - same as original but with configurable period
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
CCI implementation - same as original but with configurable period
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
MACD Signal line implementation - same as original but with configurable periods
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

end # module Oscillators_enhanced 