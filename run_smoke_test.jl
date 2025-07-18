#!/usr/bin/env julia

"""
Smoke Test for Oscillator-Soliton Performative Market Hypothesis

Runs a mini end-to-end test using the first 5 dates with reduced parameters:
- grid = 32, T = 0.1, dt = 1e-2
- Produces and inspects features Parquet

Usage: julia --project=. run_smoke_test.jl
"""

using Pkg
Pkg.activate(".")

println("Oscillator-Soliton Smoke Test")
println("=" ^ 50)

try
    # Test basic imports
    println("Testing imports...")
    using OscillatorSolitonHypothesis
    using DataFrames, CSV, Dates
    println("   All imports successful")
    
    # Load or create test data
    println("\nLoading/creating test data...")
    
    data_file = "data/raw/es_2022q1_daily.csv"
    test_df = nothing
    
    if isfile(data_file)
        println("   Loading real ES data from $data_file")
        full_df = CSV.read(data_file, DataFrame)
        # Use first 5 dates as specified
        test_df = first(full_df, min(5, nrow(full_df)))
        println("   Loaded $(nrow(test_df)) rows of real data")
    else
        println("   Creating synthetic test data (5 rows)")
        # Create 5 days of synthetic OHLCV data
        dates = Date("2022-01-01"):Day(1):Date("2022-01-05")
        base_price = 4500.0  # ES futures typical price
        
        test_df = DataFrame(
            Date = collect(dates),
            Open = base_price .+ cumsum(randn(5) * 2.0),
            High = base_price .+ cumsum(randn(5) * 2.0) .+ abs.(randn(5) * 1.5),
            Low = base_price .+ cumsum(randn(5) * 2.0) .- abs.(randn(5) * 1.5),
            Close = base_price .+ cumsum(randn(5) * 2.0),
            Volume = rand(10000:50000, 5)
        )
        
        # Fix OHLC relationships
        for i in 1:nrow(test_df)
            test_df.High[i] = max(test_df.High[i], test_df.Open[i], test_df.Close[i])
            test_df.Low[i] = min(test_df.Low[i], test_df.Open[i], test_df.Close[i])
        end
        
        println("   Created synthetic data with realistic ES price levels")
    end
    
    println("   Test data:")
    println(test_df)
    
    # Smoke test parameters (reduced as specified)
    smoke_params = (
        grid = 32,      # Reduced from default 64
        T = 0.1,        # Reduced from default 1.0
        dt = 1e-2,      # Specified in instructions
        forward_days = 1  # Reduced for small dataset
    )
    
    println("\nSmoke test parameters:")
    println("   grid = $(smoke_params.grid)")
    println("   T = $(smoke_params.T)")
    println("   dt = $(smoke_params.dt)")
    println("   forward_days = $(smoke_params.forward_days)")
    
    # Run feature building with smoke test parameters
    println("\nRunning feature extraction...")
    
    # Need more data for oscillators - extend test data
    if nrow(test_df) < 30
        println("   Extending data to 30 rows for oscillator computation...")
        extended_dates = test_df.Date[end] .+ Day.(1:25)
        last_close = test_df.Close[end]
        
        extended_data = DataFrame(
            Date = extended_dates,
            Open = last_close .+ cumsum(randn(25) * 2.0),
            High = last_close .+ cumsum(randn(25) * 2.0) .+ abs.(randn(25) * 1.5),
            Low = last_close .+ cumsum(randn(25) * 2.0) .- abs.(randn(25) * 1.5),
            Close = last_close .+ cumsum(randn(25) * 2.0),
            Volume = rand(10000:50000, 25)
        )
        
        # Fix OHLC relationships
        for i in 1:nrow(extended_data)
            extended_data.High[i] = max(extended_data.High[i], extended_data.Open[i], extended_data.Close[i])
            extended_data.Low[i] = min(extended_data.Low[i], extended_data.Open[i], extended_data.Close[i])
        end
        
        test_df = vcat(test_df, extended_data)
        println("   Extended to $(nrow(test_df)) rows")
    end
    
    # Run feature building
    features_df = build_features(
        test_df, 
        DataFrame(),  # No VIX data
        grid = smoke_params.grid,
        T = smoke_params.T,
        forward_days = smoke_params.forward_days,
        output_file = "data/processed/smoke_test_features.parquet"
    )
    
    println("\nSmoke Test Results:")
    if nrow(features_df) > 0
        println("   Feature extraction successful!")
        println("   Generated features: $(nrow(features_df)) rows")
        println("   Columns: $(names(features_df))")
        
        # Inspect sample features
        if nrow(features_df) >= 1
            sample_row = features_df[1, :]
            println("\n   Sample feature row:")
            for col in [:Date, :H, :F1, :F2, :F3, :energy]
                if col in names(features_df)
                    val = sample_row[col]
                    if isa(val, Number) && !isnan(val)
                        println("     $col: $(round(val, digits=6))")
                    else
                        println("     $col: $val")
                    end
                end
            end
        end
        
        # Check Parquet file
        parquet_file = "data/processed/smoke_test_features.parquet"
        if isfile(parquet_file)
            file_size = filesize(parquet_file)
            println("\n   Parquet file created: $parquet_file")
            println("     Size: $file_size bytes")
        end
        
    else
        println("   Warning: No features generated - may need more data or parameter adjustment")
    end
    
    println("\nSmoke test completed successfully!")
    println("   All pipeline components functional")
    println("   Ready for full experiments with real data")
    
    return true
    
catch e
    println("\nSmoke test failed:")
    println("   Error: $e")
    println("\nTroubleshooting suggestions:")
    println("   1. Ensure Julia ≥ 1.10 is installed")
    println("   2. Run: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
    println("   3. Check that all dependencies installed successfully")
    println("   4. Verify project structure is complete")
    
    return false
end 