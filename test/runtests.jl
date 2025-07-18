using Test
using DataFrames, Dates
using Statistics

# Include modules to test
include("../src/Oscillators.jl")
using .Oscillators

@testset "Oscillator Tests" begin
    
    @testset "Basic Oscillator Computation" begin
        # Create 10-row toy DataFrame as specified
        toy_df = DataFrame(
            Date = Date("2022-01-01"):Day(1):Date("2022-01-10"),
            Open = [100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0],
            High = [101.0, 102.0, 103.0, 102.5, 104.0, 103.0, 105.0, 104.5, 106.0, 105.0],
            Low = [99.0, 100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0],
            Close = [100.5, 101.5, 102.5, 101.0, 103.5, 102.5, 104.5, 103.0, 105.5, 104.5],
            Volume = [1000, 1100, 1200, 950, 1300, 1050, 1400, 1150, 1500, 1250]
        )
        
        println("Testing with toy DataFrame:")
        println(toy_df)
        
        # Test basic computation
        result_df = compute_oscillators(toy_df)
        
        # Check that all original columns are preserved
        for col in names(toy_df)
            @test col in names(result_df)
            @test result_df[!, col] == toy_df[!, col]
        end
        
        # Check that oscillator columns are added
        expected_cols = [:RSI14, :StochK14, :CCI20, :MACDsig]
        for col in expected_cols
            @test col in names(result_df)
        end
        
        # Check that raw columns are also added
        expected_raw_cols = [:RSI14_raw, :StochK14_raw, :CCI20_raw, :MACDsig_raw]
        for col in expected_raw_cols
            @test col in names(result_df)
        end
        
        println("✅ Basic oscillator computation test passed")
    end
    
    @testset "Normalization Tests" begin
        # Test that normalized values are in [-1, 1] range
        toy_df = DataFrame(
            Date = Date("2022-01-01"):Day(1):Date("2022-01-30"),
            Open = 100.0 .+ cumsum(randn(30)),
            High = 100.0 .+ cumsum(randn(30)) .+ 1.0,
            Low = 100.0 .+ cumsum(randn(30)) .- 1.0,
            Close = 100.0 .+ cumsum(randn(30)),
            Volume = rand(1000:2000, 30)
        )
        
        # Ensure price relationships are correct
        for i in 1:nrow(toy_df)
            if toy_df.High[i] < toy_df.Close[i]
                toy_df.High[i] = toy_df.Close[i] + rand()
            end
            if toy_df.Low[i] > toy_df.Close[i]
                toy_df.Low[i] = toy_df.Close[i] - rand()
            end
            if toy_df.High[i] < toy_df.Open[i]
                toy_df.High[i] = toy_df.Open[i] + rand()
            end
            if toy_df.Low[i] > toy_df.Open[i]
                toy_df.Low[i] = toy_df.Open[i] - rand()
            end
        end
        
        result_df = compute_oscillators(toy_df)
        
        # Test normalization bounds
        normalized_cols = [:RSI14, :StochK14, :CCI20, :MACDsig]
        
        for col in normalized_cols
            values = result_df[!, col]
            valid_values = filter(!isnan, values)
            
            if !isempty(valid_values)
                @test all(valid_values .>= -1.0)
                @test all(valid_values .<= 1.0)
                println("✅ $col normalization bounds test passed")
            end
        end
    end
    
    @testset "Edge Cases" begin
        # Test with insufficient data
        small_df = DataFrame(
            Date = Date("2022-01-01"):Day(1):Date("2022-01-05"),
            Open = [100.0, 101.0, 102.0, 103.0, 104.0],
            High = [101.0, 102.0, 103.0, 104.0, 105.0],
            Low = [99.0, 100.0, 101.0, 102.0, 103.0],
            Close = [100.5, 101.5, 102.5, 103.5, 104.5],
            Volume = [1000, 1100, 1200, 1300, 1400]
        )
        
        result_df = compute_oscillators(small_df)
        
        # Should return zeros for insufficient data
        @test all(result_df.RSI14 .== 0.0)
        @test all(result_df.StochK14 .== 0.0)
        @test all(result_df.CCI20 .== 0.0)
        @test all(result_df.MACDsig .== 0.0)
        
        println("✅ Insufficient data test passed")
        
        # Test with missing columns
        incomplete_df = DataFrame(
            Date = Date("2022-01-01"):Day(1):Date("2022-01-10"),
            Open = rand(10),
            High = rand(10),
            Low = rand(10)
            # Missing Close and Volume
        )
        
        @test_throws ErrorException compute_oscillators(incomplete_df)
        println("✅ Missing columns test passed")
        
        # Test with constant prices (no variation)
        constant_df = DataFrame(
            Date = Date("2022-01-01"):Day(1):Date("2022-01-30"),
            Open = fill(100.0, 30),
            High = fill(100.0, 30),
            Low = fill(100.0, 30),
            Close = fill(100.0, 30),
            Volume = fill(1000, 30)
        )
        
        # Should handle constant prices without error
        result_df = compute_oscillators(constant_df)
        @test nrow(result_df) == 30
        
        # Normalized values should be 0 (middle of [-1,1]) for no variation
        normalized_cols = [:RSI14, :StochK14, :CCI20, :MACDsig]
        for col in normalized_cols
            valid_values = filter(!isnan, result_df[!, col])
            if !isempty(valid_values)
                @test all(abs.(valid_values) .< 1e-10)  # Should be ~0
            end
        end
        
        println("✅ Constant prices test passed")
    end
    
    @testset "Internal Functions" begin
        # Test RSI calculation
        prices = [44.0, 44.25, 44.5, 43.75, 44.5, 44.75, 44.5, 44.0, 44.25]
        rsi_result = Oscillators._compute_rsi(prices, 4)
        
        # Should have NaN for first few values, then valid RSI
        @test all(isnan.(rsi_result[1:4]))
        @test !isnan(rsi_result[5])
        @test rsi_result[5] >= 0.0 && rsi_result[5] <= 100.0
        
        println("✅ RSI calculation test passed")
        
        # Test normalization function
        test_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        normalized = Oscillators._normalize_to_range(test_values, -1.0, 1.0)
        
        @test minimum(normalized) ≈ -1.0
        @test maximum(normalized) ≈ 1.0
        @test length(normalized) == length(test_values)
        
        println("✅ Normalization function test passed")
        
        # Test with NaN values
        test_with_nan = [10.0, NaN, 30.0, 40.0, NaN]
        normalized_with_nan = Oscillators._normalize_to_range(test_with_nan, -1.0, 1.0)
        
        @test isnan(normalized_with_nan[2])
        @test isnan(normalized_with_nan[5])
        @test !isnan(normalized_with_nan[1])
        @test !isnan(normalized_with_nan[3])
        @test !isnan(normalized_with_nan[4])
        
        println("✅ NaN handling test passed")
    end
end

println("\n🎉 All Oscillator tests completed!") 