#!/usr/bin/env julia

"""
Download Individual Stock Data at Higher Frequencies from Alpha Vantage
Supports 1min, 5min, 15min, 30min, and 60min intervals for individual stocks
Saves data as Arrow files to data/raw/stocks/

Usage: julia scripts/download_individual_stocks.jl

Configuration:
- Edit STOCK_SYMBOLS to add/remove stocks
- Edit INTERVAL to change time frequency 
- Get free API key from https://www.alphavantage.co/support/#api-key
"""

using HTTP, JSON3, DataFrames, Arrow, Dates, SHA
using CSV # For parsing CSV responses

# Configuration
const API_KEY = get(ENV, "ALPHAVANTAGE_API_KEY", "demo")  # Get from environment or use demo
const BASE_URL = "https://www.alphavantage.co/query"
const INTERVAL = "5min"  # Options: 1min, 5min, 15min, 30min, 60min
const OUTPUT_SIZE = "full"  # "compact" (100 points) or "full" (up to 2 years)

# Stock symbols to download (diverse selection across sectors)
const STOCK_SYMBOLS = [
    "AAPL",    # Technology - Apple
    "MSFT",    # Technology - Microsoft  
    "GOOGL",   # Technology - Google
    "TSLA",    # Auto/Energy - Tesla
    "NVDA",    # Semiconductors - NVIDIA
    "JPM",     # Finance - JPMorgan Chase
    "JNJ",     # Healthcare - Johnson & Johnson
    "PG",      # Consumer Goods - Procter & Gamble
    "XOM",     # Energy - Exxon Mobil
    "DIS",     # Entertainment - Disney
]

function main()
    println("Individual Stock High-Frequency Data Download")
    println("=" ^ 60)
    println("Source: Alpha Vantage")
    println("Interval: $INTERVAL")
    println("Stocks: $(length(STOCK_SYMBOLS)) symbols")
    println("API Key: $(API_KEY == "demo" ? "DEMO (limited)" : "Custom")")
    
    if API_KEY == "demo"
        println("\nWARNING:  Using demo API key - limited to ~5 calls per minute")
        println("   Get free key: https://www.alphavantage.co/support/#api-key")
        println("   Set environment variable: export ALPHAVANTAGE_API_KEY=your_key")
    end
    
    # Create output directory
    output_dir = joinpath(@__DIR__, "..", "data", "raw", "stocks")
    mkpath(output_dir)
    
    successful_downloads = 0
    failed_downloads = String[]
    
    println("\nDownloading stock data...")
    
    for (i, symbol) in enumerate(STOCK_SYMBOLS)
        println("\n[$i/$(length(STOCK_SYMBOLS))] Processing $symbol...")
        
        try
            # Download and process stock data
            df = download_stock_data(symbol, INTERVAL)
            
            if nrow(df) == 0
                println("   FAILED: No data returned for $symbol")
                push!(failed_downloads, symbol)
                continue
            end
            
            # Save to Arrow file
            output_file = joinpath(output_dir, "$(lowercase(symbol))_$(INTERVAL)_$(today()).arrow")
            Arrow.write(output_file, df)
            
            file_size = filesize(output_file)
            println("   SUCCESS: Saved $(nrow(df)) records ($(round(file_size/1024, digits=1)) KB)")
            println("      Range: $(minimum(df.datetime)) to $(maximum(df.datetime))")
            println("      File: $output_file")
            
            successful_downloads += 1
            
            # Rate limiting for demo API key
            if API_KEY == "demo" && i < length(STOCK_SYMBOLS)
                println("   ⏳ Waiting 12 seconds (rate limit)...")
                sleep(12)  # Demo key allows ~5 calls per minute
            elseif i < length(STOCK_SYMBOLS)
                println("   ⏳ Waiting 1 second...")
                sleep(1)   # Be polite to API
            end
            
        catch e
            println("   FAILED: Error downloading $symbol: $e")
            push!(failed_downloads, symbol)
            
            # Wait a bit longer on errors
            if i < length(STOCK_SYMBOLS)
                sleep(5)
            end
        end
    end
    
    # Summary
    println("\n" ^ 2 * "=" ^ 60)
    println("DOWNLOAD SUMMARY")
    println("=" ^ 60)
    println("SUCCESS: Successful: $successful_downloads/$(length(STOCK_SYMBOLS)) stocks")
    println("FAILED: Failed: $(length(failed_downloads)) stocks")
    
    if !isempty(failed_downloads)
        println("\nFailed downloads:")
        for symbol in failed_downloads
            println("   - $symbol")
        end
    end
    
    println("\nData saved to: $output_dir")
    println("Next steps:")
    println("   1. Adjust oscillator parameters for $INTERVAL data")
    println("   2. Run soliton analysis on higher frequency data")
    println("   3. Compare results with daily data")
    
    return successful_downloads > 0
end

function download_stock_data(symbol::String, interval::String)::DataFrame
    """Download intraday stock data from Alpha Vantage"""
    
    # Build API URL
    params = Dict(
        "function" => "TIME_SERIES_INTRADAY",
        "symbol" => symbol,
        "interval" => interval,
        "outputsize" => OUTPUT_SIZE,
        "datatype" => "csv",  # CSV is easier to parse than JSON
        "apikey" => API_KEY
    )
    
    url = build_url(BASE_URL, params)
    
    # Download data
    response = HTTP.get(url; headers = ["User-Agent" => "Julia/$(VERSION) Alpha Vantage Client"])
    
    if response.status != 200
        error("HTTP request failed with status: $(response.status)")
    end
    
    # Parse CSV response
    csv_content = String(response.body)
    
    # Check for API errors (Alpha Vantage returns error messages in CSV format too)
    if occursin("Error Message", csv_content) || occursin("Note:", csv_content)
        error("API Error: $(first(split(csv_content, '\n'), 2))")
    end
    
    # Parse CSV to DataFrame
    df = CSV.read(IOBuffer(csv_content), DataFrame)
    
    # Standardize column names (Alpha Vantage uses different names)
    if "timestamp" in names(df)
        rename!(df, "timestamp" => "datetime")
    end
    
    # Convert datetime column
    if "datetime" in names(df)
        df.datetime = DateTime.(df.datetime, "yyyy-mm-dd HH:MM:SS")
    end
    
    # Convert price columns to Float64
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols
        if col in names(df)
            # Only parse if not already numeric
            if eltype(df[!, col]) <: AbstractString
                df[!, col] = parse.(Float64, df[!, col])
            elseif !(eltype(df[!, col]) <: AbstractFloat)
                df[!, col] = Float64.(df[!, col])
            end
        end
    end
    
    # Convert volume to Int64
    if "volume" in names(df)
        # Only parse if not already numeric
        if eltype(df.volume) <: AbstractString
            df.volume = parse.(Int64, df.volume)
        elseif !(eltype(df.volume) <: Integer)
            df.volume = Int64.(df.volume)
        end
    end
    
    # Sort by datetime (oldest first)
    sort!(df, :datetime)
    
    return df
end

function build_url(base_url::String, params::Dict)::String
    """Build URL with query parameters"""
    param_strings = ["$k=$v" for (k, v) in params]
    return base_url * "?" * join(param_strings, "&")
end

# Run the script if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = main()
    exit(success ? 0 : 1)
end 