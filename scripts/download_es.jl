#!/usr/bin/env julia

"""
Download ES continuous futures data from Yahoo Finance
Saves 2022 Q1 daily data to data/raw/es_2022q1_daily.csv

Usage: julia scripts/download_es.jl
"""

using HTTP, CSV, DataFrames, Downloads
using Retry, RateLimiter
using SHA

# Configuration
const YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v7/finance/download"
const SYMBOL = "ES=F"
const PERIOD1 = 1640995200  # 1-Jan-2022 UTC
const PERIOD2 = 1648684800  # 31-Mar-2022 UTC  
const INTERVAL = "1d"
const OUTPUT_FILE = "data/raw/es_2022q1_daily.csv"
const MAX_RETRIES = 3
const RETRY_DELAY = 1.0  # seconds

function build_yahoo_url()::String
    return "$(YAHOO_BASE_URL)/$(SYMBOL)?" *
           "period1=$(PERIOD1)&period2=$(PERIOD2)&interval=$(INTERVAL)&events=history"
end

function download_with_retry()::String
    url = build_yahoo_url()
    println("Downloading ES futures data from Yahoo Finance...")
    println("   URL: $url")
    
    @repeat MAX_RETRIES try
        println("   Attempting download...")
        
        # Rate limiting: sleep between attempts
        sleep(RETRY_DELAY)
        
        response = HTTP.get(url)
        
        if response.status != 200
            error("HTTP request failed with status: $(response.status)")
        end
        
        return String(response.body)
    catch e
        # Let automatic retry logic handle retries
        rethrow(e)
    end
end

function save_and_verify(csv_data::String)
    # Ensure output directory exists
    mkpath(dirname(OUTPUT_FILE))
    
    # Write CSV data
    open(OUTPUT_FILE, "w") do f
        write(f, csv_data)
    end
    
    # Calculate file size and SHA-256 for reproducibility
    file_size = filesize(OUTPUT_FILE)
    file_hash = bytes2hex(sha256(csv_data))
    
    # Verify data by reading back
    df = CSV.read(OUTPUT_FILE, DataFrame)
    row_count = nrow(df)
    
    println("Download successful!")
    println("   File: $OUTPUT_FILE")
    println("   Size: $file_size bytes")
    println("   SHA-256: $file_hash")
    println("   Rows: $row_count")
    println("   Columns: $(names(df))")
    
    # Quick data validation
    if row_count < 50 || row_count > 80
        @warn "Unexpected row count: $row_count (expected ~63 for Q1 2022 daily)"
    end
    
    return (file_size, file_hash, row_count)
end

function main()
    try
        println("Starting ES futures data download...")
        
        csv_data = download_with_retry()
        file_size, file_hash, row_count = save_and_verify(csv_data)
        
        println("Summary:")
        println("   Downloaded: ES=F (Q1 2022 daily)")
        println("   File size: $file_size bytes")
        println("   SHA-256: $file_hash")
        println("   Records: $row_count")
        
        return true
        
    catch e
        println("Download failed: $e")
        return false
    end
end

# Run the script if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = main()
    exit(success ? 0 : 1)
end 