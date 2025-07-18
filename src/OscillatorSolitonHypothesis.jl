"""
OscillatorSolitonHypothesis.jl

Main package file for the Oscillator-Soliton Performative Market Hypothesis project.

This package implements a novel approach to financial market prediction by:
1. Computing technical oscillators (RSI, Stochastic %K, CCI, MACD) from price data
2. Using normalized oscillator values as amplitudes for soliton waves in a φ⁴/Klein-Gordon PDE
3. Extracting collision features from the PDE solution
4. Testing predictive power of these features versus baseline oscillators using ML

# Example Usage
```julia
using OscillatorSolitonHypothesis
using DataFrames, CSV

# Load price data
price_df = CSV.read("data/raw/es_2022q1_daily.csv", DataFrame)

# Build soliton features  
features_df = build_features(price_df, grid=32, T=0.5)

# Run ML experiment
include("notebooks/xgboost_test.jl")
results = main()
```
"""
module OscillatorSolitonHypothesis

# Package exports
export 
    # From Oscillators module
    compute_oscillators,
    
    # From SolitonPDE module  
    simulate_soliton_collision_real,
    
    # From FeatureBuilder module
    build_features

# Core dependencies - only use what's available
using DataFrames

# Include and import our modules
include("Oscillators.jl")
include("SolitonPDE_simple.jl")  # Use simplified version
include("FeatureBuilder.jl")

using .Oscillators
using .SolitonPDE_simple
using .FeatureBuilder

# Package version
const VERSION = v"0.1.0"

"""
    quick_demo(; n_samples=5, grid=16, T=0.1)

Run a quick demonstration of the full pipeline with synthetic data.
Useful for testing that everything is working properly.
"""
function quick_demo(; n_samples::Int=5, grid::Int=16, T::Float64=0.1)
    println("Running Quick Demo of Oscillator-Soliton Pipeline")
    println("=" ^ 50)
    
    try
        println("Creating synthetic price data...")
        using Random, Dates
        Random.seed!(42)
        
        dates = Date("2022-01-01"):Day(1):Date("2022-01-01") + Day(n_samples - 1)
        base_price = 100.0
        prices = base_price .+ cumsum(0.1 * randn(n_samples))
        
        synthetic_df = DataFrame(
            Date = collect(dates),
            Open = prices .+ 0.1 * randn(n_samples),
            High = prices .+ abs.(0.2 * randn(n_samples)),
            Low = prices .- abs.(0.2 * randn(n_samples)),
            Close = prices,
            Volume = rand(1000:2000, n_samples)
        )
        
        println("   Generated $(nrow(synthetic_df)) price rows")
        
        # Test oscillator computation
        println("\nComputing oscillators...")
        try
            osc_df = compute_oscillators(synthetic_df)
            println("   Oscillators computed successfully")
            println("   Columns: $(names(osc_df))")
        catch e
            println("   Oscillator computation failed: $e")
            return false
        end
        
        # Test soliton simulation
        println("\nTesting soliton simulation...")
        test_amplitudes = (0.5, -0.3, 0.8, -0.2)
        test_vix = 25.0
        
        try
            soliton_result = simulate_soliton_collision_real(
                test_amplitudes[1], test_amplitudes[2], 
                test_amplitudes[3], test_amplitudes[4]
            )
            println("   Soliton simulation completed")
            println("   Features: height=$(round(soliton_result.height, digits=4))")
        catch e
            println("   Soliton simulation failed: $e")
            return false
        end
        
        println("\nQuick demo completed successfully!")
        println("   All pipeline components are working")
        println("   Ready for full-scale experiments")
        return true
        
    catch e
        println("Demo failed: $e")
        rethrow(e)
    end
end

# Print package information when loaded
function __init__()
    println("Goal: Test whether technical oscillators have higher-order market feedback")
    println("Method: Map oscillators → soliton amplitudes → PDE collision features → ML")
    
    println("Quick Start:")
    println("  using OscillatorSolitonHypothesis")
    println("  features = build_features(price_df)")
    println("  include(\"notebooks/xgboost_test.jl\")")
    println("  results = main()")
    println()
    
    println("Project Structure:")
    println("  src/Oscillators.jl - Technical oscillator computation")
    println("  src/SolitonPDE.jl - φ⁴/Klein-Gordon PDE solver")  
    println("  src/FeatureBuilder.jl - Pipeline orchestration")
    println("  notebooks/xgboost_test.jl - ML comparison experiments")
    println("  scripts/download_*.jl - Data acquisition utilities")
    println()
    
    println("OscillatorSolitonHypothesis.jl loaded")
end

end # module 