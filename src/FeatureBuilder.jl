"""
FeatureBuilder Module

Orchestrates the full pipeline from price/VIX data to soliton collision features.
Handles data joining, oscillator computation, PDE simulation, and feature extraction.
"""
module FeatureBuilder

export build_features

using DataFrames, DataFramesMeta, Parquet
using DrWatson, ProgressMeter
using Statistics

# Import our custom modules
include("Oscillators.jl")
include("SolitonPDE.jl")
using .Oscillators, .SolitonPDE

"""
    build_features(price_df::DataFrame, vix_df::DataFrame; 
                  step="1d", output_file="data/processed/features.parquet",
                  grid=64, T=1.0, λ=1.0, μmax=0.1) -> DataFrame

Build soliton collision features from price and VIX data.

# Arguments
- `price_df`: DataFrame with Date, Open, High, Low, Close, Volume columns
- `vix_df`: DataFrame with Date, VIX columns (or price_df can contain VIX column)
- `step`: Time frequency for feature extraction ("1d" for daily)
- `output_file`: Path to save features as Parquet
- `grid`: Grid size for PDE simulation
- `T`: PDE integration time  
- `λ`: Nonlinearity parameter
- `μmax`: Maximum damping coefficient

# Returns
- `DataFrame`: Features with columns Date, RSI14, StochK14, CCI20, MACDsig, H, F1, F2, F3, energy, forward_return

# Pipeline Steps
1. Join price and VIX data by date
2. Compute technical oscillators  
3. Normalize oscillators to [-1,1]
4. For each row: simulate soliton PDE with oscillator amplitudes
5. Extract collision features (H, F, energy)
6. Compute forward returns for ML target
7. Save features to Parquet for caching
"""
function build_features(price_df::DataFrame, vix_df::DataFrame=DataFrame(); 
                       step::String="1d", 
                       output_file::String="data/processed/features.parquet",
                       grid::Int=64, T::Float64=1.0, λ::Float64=1.0, μmax::Float64=0.1,
                       vix_max_sample::Float64=50.0,
                       forward_days::Int=5)::DataFrame
    
    println("🔧 Building soliton features from market data...")
    println("   Price data: $(nrow(price_df)) rows")
    println("   VIX data: $(nrow(vix_df)) rows")
    println("   PDE params: grid=$grid, T=$T, λ=$λ, μmax=$μmax")
    
    # Step 1: Join price and VIX data
    combined_df = _join_price_vix_data(price_df, vix_df)
    
    # Step 2: Compute oscillators and normalize
    println("📈 Computing technical oscillators...")
    osc_df = compute_oscillators(combined_df)
    
    # Step 3: Compute forward returns for ML target
    println("🎯 Computing forward returns...")
    target_df = _add_forward_returns(osc_df, forward_days)
    
    # Step 4: Filter rows with complete data (no NaN oscillators)
    valid_rows = _filter_complete_data(target_df)
    n_valid = nrow(valid_rows)
    
    println("   Valid rows for PDE simulation: $n_valid")
    
    if n_valid == 0
        @warn "No valid rows for feature extraction!"
        return DataFrame()
    end
    
    # Step 5: Extract soliton features for each row
    println("🌊 Running PDE simulations...")
    feature_rows = Vector{NamedTuple}()
    
    progress = Progress(n_valid, desc="PDE simulations: ")
    
    for (idx, row) in enumerate(eachrow(valid_rows))
        try
            # Extract normalized oscillator amplitudes
            amplitudes = (
                row.RSI14,
                row.StochK14, 
                row.CCI20,
                row.MACDsig
            )
            
            # Get VIX value (with fallback)
            vix_value = haskey(row, :VIX) ? row.VIX : 20.0
            
            # Run soliton simulation
            soliton_result = simulate_soliton(
                amplitudes, vix_value,
                grid=grid, T=T, λ=λ, μmax=μmax, 
                vix_max_sample=vix_max_sample
            )
            
            # Collect features
            feature_row = (
                Date = row.Date,
                Close = row.Close,
                VIX = vix_value,
                RSI14 = row.RSI14,
                StochK14 = row.StochK14,
                CCI20 = row.CCI20, 
                MACDsig = row.MACDsig,
                H = soliton_result.H,
                F1 = length(soliton_result.F) >= 1 ? soliton_result.F[1] : NaN,
                F2 = length(soliton_result.F) >= 2 ? soliton_result.F[2] : NaN,
                F3 = length(soliton_result.F) >= 3 ? soliton_result.F[3] : NaN,
                energy = soliton_result.energy,
                forward_return = row.forward_return,
                grid = grid,
                T = T,
                λ = λ,
                μ = soliton_result.metadata.μ
            )
            
            push!(feature_rows, feature_row)
            
        catch e
            @warn "PDE simulation failed for row $idx: $e"
            # Continue with other rows
        end
        
        next!(progress)
    end
    
    finish!(progress)
    
    if isempty(feature_rows)
        @warn "No successful PDE simulations!"
        return DataFrame()
    end
    
    # Step 6: Convert to DataFrame
    features_df = DataFrame(feature_rows)
    
    println("   ✅ Generated $(nrow(features_df)) feature rows")
    
    # Step 7: Save to Parquet for caching
    println("💾 Saving features to $output_file...")
    mkpath(dirname(output_file))
    Parquet.write_parquet(output_file, features_df)
    
    # Log feature summary
    _log_feature_summary(features_df)
    
    return features_df
end

"""
Join price and VIX data by date
"""
function _join_price_vix_data(price_df::DataFrame, vix_df::DataFrame)::DataFrame
    if isempty(vix_df)
        # Check if VIX column already exists in price_df
        if :VIX in names(price_df)
            println("   Using VIX column from price data")
            return price_df
        else
            # Generate dummy VIX data
            println("   ⚠️  No VIX data provided, using dummy values")
            combined_df = copy(price_df)
            combined_df.VIX = 20.0 .+ 10.0 .* randn(nrow(combined_df))  # VIX ~ N(20, 10²)
            combined_df.VIX = max.(combined_df.VIX, 5.0)  # Floor at 5
            return combined_df
        end
    end
    
    # Join on Date column
    println("   Joining price and VIX data on Date...")
    combined_df = leftjoin(price_df, vix_df, on=:Date)
    
    # Handle missing VIX values
    if any(ismissing, combined_df.VIX)
        println("   ⚠️  Filling missing VIX values with interpolation")
        combined_df.VIX = coalesce.(combined_df.VIX, 20.0)  # Simple fallback
    end
    
    return combined_df
end

"""
Add forward returns as ML target variable
"""
function _add_forward_returns(df::DataFrame, forward_days::Int)::DataFrame
    result_df = copy(df)
    n_rows = nrow(result_df)
    
    forward_returns = fill(NaN, n_rows)
    
    for i in 1:(n_rows - forward_days)
        if !ismissing(result_df.Close[i]) && !ismissing(result_df.Close[i + forward_days])
            forward_returns[i] = (result_df.Close[i + forward_days] - result_df.Close[i]) / result_df.Close[i]
        end
    end
    
    result_df.forward_return = forward_returns
    
    println("   Added $(forward_days)-day forward returns")
    
    return result_df
end

"""
Filter rows with complete oscillator and target data
"""
function _filter_complete_data(df::DataFrame)::DataFrame
    required_cols = [:RSI14, :StochK14, :CCI20, :MACDsig, :forward_return]
    
    # Find rows where all required columns are not NaN
    valid_mask = trues(nrow(df))
    
    for col in required_cols
        if col in names(df)
            valid_mask .&= .!isnan.(df[!, col])
        else
            @warn "Missing required column: $col"
            return DataFrame()  # Return empty if missing critical columns
        end
    end
    
    result_df = df[valid_mask, :]
    
    dropped_rows = nrow(df) - nrow(result_df)
    if dropped_rows > 0
        println("   Dropped $dropped_rows rows with incomplete data")
    end
    
    return result_df
end

"""
Log summary statistics of generated features
"""
function _log_feature_summary(features_df::DataFrame)
    println("\n📊 Feature Summary:")
    println("   Rows: $(nrow(features_df))")
    
    # Summary stats for key features
    soliton_features = [:H, :F1, :F2, :F3, :energy]
    
    for feature in soliton_features
        if feature in names(features_df)
            values = filter(!isnan, features_df[!, feature])
            if !isempty(values)
                μ = mean(values)
                σ = std(values)
                println("   $feature: μ=$μ, σ=$σ, range=[$(minimum(values)), $(maximum(values))]")
            end
        end
    end
    
    # Correlation with forward returns
    if :forward_return in names(features_df)
        println("\n🔗 Correlations with forward returns:")
        for feature in soliton_features
            if feature in names(features_df)
                corr_val = _safe_correlation(features_df[!, feature], features_df.forward_return)
                println("   $feature: r = $corr_val")
            end
        end
    end
end

"""
Safe correlation calculation handling NaN values
"""
function _safe_correlation(x::Vector, y::Vector)::Float64
    # Find valid pairs (no NaN in either)
    valid_mask = .!isnan.(x) .& .!isnan.(y)
    
    if sum(valid_mask) < 2
        return NaN
    end
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    return cor(x_valid, y_valid)
end

end # module FeatureBuilder 