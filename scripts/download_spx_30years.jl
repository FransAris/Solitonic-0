#!/usr/bin/env julia

"""
Download S&P 500 Index (^SPX) 30-Year Historical Data from Stooq
Expands analysis from 2-year window to 30 years (1995-2025)
Saves as Arrow to data/raw/spx_30years_stooq.arrow

Usage: julia scripts/download_spx_30years.jl
"""

using CSV, DataFrames, HTTP, Dates, Arrow, SHA

function main()
    println("Starting S&P 500 Index (^SPX) 30-year data download from Stooq...")
    
    # Fetch raw data
    url = "https://stooq.com/q/d/l/?s=^SPX&i=d"  # S&P 500 Index daily (full history)
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
    
    # Load & slice to 30 years
    println("Loading and processing data...")
    
    # Load CSV with proper date parsing
    df = CSV.read(raw_path, DataFrame; 
                  dateformat="yyyy-mm-dd",
                  types=Dict("Date" => Date))
    
    total_rows = nrow(df)
    println("   Loaded $(total_rows) total rows")
    println("   Full date range: $(minimum(df.Date)) to $(maximum(df.Date))")
    
    # Slice to 30 years: January 1, 1995 to December 31, 2024
    start_date = Date(1995, 1, 1)
    end_date = Date(2024, 12, 31)
    
    df_30years = filter(:Date => d -> start_date <= d <= end_date, df)
    
    rows_30years = nrow(df_30years)
    println("   30-year slice (1995-2024): $(rows_30years) rows")
    
    if rows_30years == 0
        error("No data found for 1995-2024! Check date format and range.")
    end
    
    actual_start = minimum(df_30years.Date)
    actual_end = maximum(df_30years.Date)
    years_span = (actual_end - actual_start).value / 365.25
    
    println("   Actual range: $actual_start to $actual_end")
    println("   Span: $(round(years_span, digits=1)) years")
    
    # Sanity checks
    println("Running sanity checks...")
    
    # Check for duplicate dates
    unique_dates = length(unique(df_30years.Date))
    if unique_dates != rows_30years
        error("Duplicate dates found! Expected $rows_30years unique dates, got $unique_dates")
    end
    println("   No duplicate dates")
    
    # Check required OHLC columns
    required = ["Open", "High", "Low", "Close"]
    missing_cols = setdiff(required, names(df_30years))
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    println("   All OHLC columns present: $(names(df_30years))")
    
    # Check for missing values in OHLC
    for col in required
        if any(ismissing, df_30years[!, col])
            error("Missing values found in column: $col")
        end
    end
    println("   No missing values in OHLC data")
    
    # Basic data validation
    if any(df_30years.High .< df_30years.Low)
        error("Invalid OHLC data: High < Low detected")
    end
    if any((df_30years.Open .< df_30years.Low) .| (df_30years.Open .> df_30years.High))
        error("Invalid OHLC data: Open outside High-Low range")
    end
    if any((df_30years.Close .< df_30years.Low) .| (df_30years.Close .> df_30years.High))
        error("Invalid OHLC data: Close outside High-Low range")
    end
    println("   OHLC data validation passed")
    
    # Market cycles check
    min_price = minimum(df_30years.Close)
    max_price = maximum(df_30years.Close)
    total_return = (max_price - min_price) / min_price * 100
    
    println("   Price range: \$$(round(min_price, digits=2)) - \$$(round(max_price, digits=2))")
    println("   Total return span: $(round(total_return, digits=1))%")
    
    # Write Arrow
    arrow_path = joinpath(@__DIR__, "..", "data", "raw", "spx_30years_stooq.arrow")
    
    println("Saving Arrow file...")
    
    # Arrow.jl natively supports Date types - no conversion needed!
    Arrow.write(arrow_path, df_30years)
    
    arrow_size = filesize(arrow_path)
    println("Saved $arrow_path")
    println("   Rows: $(nrow(df_30years))")
    println("   Size: $(arrow_size) bytes")
    
    # Summary
    println("\n30-Year Dataset Summary:")
    println("   Source: Stooq S&P 500 Index (^SPX)")
    println("   Period: 30 years ($actual_start to $actual_end)")
    println("   Raw CSV: $raw_path ($(file_size) bytes)")
    println("   SHA-256: $checksum")
    println("   Arrow: $arrow_path ($(arrow_size) bytes)")
    println("   Records: $(nrow(df_30years)) daily bars")
    println("   Columns: $(names(df_30years))")
    println("   Coverage: Multiple bull/bear cycles, tech bubble, 2008 crisis, COVID, etc.")
    
    # Display sample data from different decades
    println("\nSample data across decades:")
    sample_years = [1995, 2000, 2008, 2020, 2024]
    
    for year in sample_years
        year_data = filter(:Date => d -> Dates.year(d) == year, df_30years)
        if !isempty(year_data)
            sample_row = year_data[1, :]
            println("   $year: $(sample_row.Date) - Close: \$$(round(sample_row.Close, digits=2))")
        end
    end
    
    println("\nReady for 30-year soliton analysis!")
    println("   This dataset covers multiple market cycles for robust soliton validation.")
    
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