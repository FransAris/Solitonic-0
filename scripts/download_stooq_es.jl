#!/usr/bin/env julia

"""
Download S&P 500 Index (^SPX) data from Stooq (public, key-free)
^SPX is highly correlated to ES futures but with cleaner data and longer history
Saves 2022 full year daily data as Arrow to data/raw/spx_2022_stooq.arrow

Usage: julia scripts/download_stooq_es.jl
"""

using CSV, DataFrames, HTTP, Dates, Arrow, SHA

function main()
    println("Starting S&P 500 Index (^SPX) data download from Stooq...")
    
    # Fetch raw data
    url = "https://stooq.com/q/d/l/?s=^SPX&i=d"  # S&P 500 Index daily
    raw_path = joinpath(@__DIR__, "..", "data", "raw", "spx_stooq_full.csv")
    
    println("Downloading from: $url")
    println("   Saving to: $raw_path")
    
    # Ensure data/raw directory exists
    mkpath(dirname(raw_path))
    
    # Download with proper User-Agent
    resp = HTTP.get(url; headers = ["User-Agent" => "Mozilla/5.0 (compatible; Julia HTTP.jl)"])
    
    if resp.status != 200
        error("HTTP request failed with status: $(resp.status)")
    end
    
    # Write raw CSV
    open(raw_path, "w") do io
        write(io, resp.body)
    end
    
    file_size = filesize(raw_path)
    println("Downloaded $(file_size) bytes")
    
    # Compute checksum
    checksum = open(raw_path) do io
        bytes2hex(sha256(io))
    end
    println("SHA256: $checksum")
    
    # Load & slice
    println("Loading and processing data...")
    
    # Load CSV with proper date parsing
    df = CSV.read(raw_path, DataFrame; 
                  dateformat="yyyy-mm-dd",
                  types=Dict("Date" => Date))
    
    total_rows = nrow(df)
    println("   Loaded $(total_rows) total rows")
    println("   Date range: $(minimum(df.Date)) to $(maximum(df.Date))")
    
    # Slice 2022 full year: January 1 to December 31, 2022
    rng = Date(2022,1,1):Day(1):Date(2022,12,31)
    df_2022 = filter(:Date => d -> d in rng, df)
    
    rows_2022 = nrow(df_2022)
    println("   2022 full year slice: $(rows_2022) rows")
    
    if rows_2022 == 0
        error("No data found for 2022! Check date format and range.")
    end
    
    # Sanity checks
    println("Running sanity checks...")
    
    # Check for duplicate dates
    unique_dates = length(unique(df_2022.Date))
    if unique_dates != rows_2022
        error("Duplicate dates found! Expected $rows_2022 unique dates, got $unique_dates")
    end
    println("   No duplicate dates")
    
    # Check required OHLC columns
    required = ["Open", "High", "Low", "Close"]
    missing_cols = setdiff(required, names(df_2022))
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    println("   All OHLC columns present: $(names(df_2022))")
    
    # Check for missing values in OHLC
    for col in required
        if any(ismissing, df_2022[!, col])
            error("Missing values found in column: $col")
        end
    end
    println("   No missing values in OHLC data")
    
    # Basic data validation
    if any(df_2022.High .< df_2022.Low)
        error("Invalid OHLC data: High < Low detected")
    end
    if any((df_2022.Open .< df_2022.Low) .| (df_2022.Open .> df_2022.High))
        error("Invalid OHLC data: Open outside High-Low range")
    end
    if any((df_2022.Close .< df_2022.Low) .| (df_2022.Close .> df_2022.High))
        error("Invalid OHLC data: Close outside High-Low range")
    end
    println("   OHLC data validation passed")
    
    # Write Arrow
    arrow_path = joinpath(@__DIR__, "..", "data", "raw", "spx_2022_stooq.arrow")
    
    println("Saving Arrow file...")
    
    # Arrow.jl natively supports Date types - no conversion needed!
    Arrow.write(arrow_path, df_2022)
    
    arrow_size = filesize(arrow_path)
    println("Saved $arrow_path")
    println("   Rows: $(nrow(df_2022))")
    println("   Size: $(arrow_size) bytes")
    
    # Summary
    println("\nSummary:")
    println("   Source: Stooq S&P 500 Index (^SPX)")
    println("   Period: 2022 ($(minimum(df_2022.Date)) to $(maximum(df_2022.Date)))")
    println("   Raw CSV: $raw_path ($(file_size) bytes)")
    println("   SHA-256: $checksum")
    println("   Arrow: $arrow_path ($(arrow_size) bytes)")
    println("   Records: $(nrow(df_2022)) daily bars")
    println("   Columns: $(names(df_2022))")
    
    # Display sample data
    println("\nSample data (first 3 rows):")
    for (i, row) in enumerate(eachrow(df_2022[1:min(3, nrow(df_2022)), :]))
        println("   $(row.Date): O=$(row.Open) H=$(row.High) L=$(row.Low) C=$(row.Close)")
    end
    
    return true
end

# Run the script if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        success = main()
        exit(success ? 0 : 1)
    catch e
        println("Download failed: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        exit(1)
    end
end 