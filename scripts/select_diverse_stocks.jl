#!/usr/bin/env julia

"""
Diverse Stock Selection Framework
Selects a representative portfolio of individual stocks across:
- Multiple sectors (Technology, Healthcare, Finance, Energy, etc.)
- Different market capitalizations (Large, Mid, Small cap)
- Various volatility profiles (High, Medium, Low volatility)
- Different trading volumes (High, Medium liquidity)

Usage: julia scripts/select_diverse_stocks.jl
"""

using DataFrames, CSV, Statistics, HTTP, JSON3, Dates
using StatsBase  # For sampling

# Configuration
const MAX_STOCKS_PER_CATEGORY = 3
const MIN_MARKET_CAP = 1e9      # $1B minimum market cap
const MIN_AVG_VOLUME = 1e6      # 1M shares average daily volume
const OUTPUT_FILE = "data/stock_universe.csv"

# Predefined stock universe organized by sector and characteristics
# This is a curated list of liquid, well-established stocks across sectors
const STOCK_UNIVERSE = Dict(
    "Technology" => Dict(
        "large_cap" => ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "CRM", "ADBE"],
        "mid_cap" => ["SNOW", "PLTR", "ROKU", "TWLO", "ZM", "DOCU", "OKTA", "CRWD", "NET", "DDOG"],
        "high_vol" => ["TSLA", "NVDA", "AMD", "NFLX", "SNAP", "ROKU", "ZM", "PLTR", "GME", "AMC"],
        "low_vol" => ["MSFT", "AAPL", "GOOGL", "ORCL", "IBM", "HPQ", "INTC", "CSCO", "ADBE", "TXN"]
    ),
    "Healthcare" => Dict(
        "large_cap" => ["JNJ", "UNH", "PFE", "ABBV", "TMO", "DHR", "BMY", "MDT", "ABT", "LLY"],
        "mid_cap" => ["REGN", "VRTX", "GILD", "BIIB", "ILMN", "MRNA", "MODERNA", "ZTS", "EW", "VAR"],
        "high_vol" => ["MRNA", "BNTX", "NVAX", "SGEN", "BIIB", "REGN", "VRTX", "GILD", "BMRN", "SRPT"],
        "low_vol" => ["JNJ", "UNH", "ABT", "MDT", "TMO", "DHR", "BDX", "SYK", "BSX", "EW"]
    ),
    "Finance" => Dict(
        "large_cap" => ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB"],
        "mid_cap" => ["COF", "PNC", "TFC", "BK", "STT", "NTRS", "FITB", "RF", "CFG", "KEY"],
        "high_vol" => ["GS", "MS", "C", "WFC", "BAC", "AXP", "COF", "JPM", "BLK", "AFL"],
        "low_vol" => ["USB", "PNC", "TFC", "BK", "STT", "NTRS", "SCHW", "BBT", "ZION", "FITB"]
    ),
    "Energy" => Dict(
        "large_cap" => ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "KMI", "OKE"],
        "mid_cap" => ["DVN", "FANG", "APA", "EQT", "CNX", "AR", "MRO", "CLR", "NBL", "PXD"],
        "high_vol" => ["DVN", "FANG", "APA", "CNX", "AR", "MRO", "CLR", "NBL", "SM", "MTDR"],
        "low_vol" => ["XOM", "CVX", "COP", "PSX", "VLO", "MPC", "KMI", "OKE", "EPD", "ET"]
    ),
    "Consumer" => Dict(
        "large_cap" => ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW"],
        "mid_cap" => ["COST", "DG", "DLTR", "BBY", "GPS", "M", "JWN", "KSS", "ANF", "AEO"],
        "high_vol" => ["NFLX", "DIS", "SBUX", "NKE", "GPS", "M", "JWN", "KSS", "ANF", "AEO"],
        "low_vol" => ["PG", "KO", "PEP", "WMT", "HD", "MCD", "COST", "TGT", "LOW", "CL"]
    ),
    "Industrial" => Dict(
        "large_cap" => ["BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "MMM", "GD", "NOC"],
        "mid_cap" => ["IR", "ROK", "EMR", "ETN", "ITW", "PH", "CMI", "DE", "FDX", "DAL"],
        "high_vol" => ["BA", "GE", "DAL", "AAL", "UAL", "LUV", "JBLU", "ALK", "HA", "SAVE"],
        "low_vol" => ["HON", "UPS", "MMM", "ITW", "PH", "EMR", "ETN", "ROK", "IR", "CMI"]
    )
)

# Volatility and volume characteristics (approximate)
const STOCK_CHARACTERISTICS = Dict(
    # High volatility stocks (>30% annual volatility)
    "high_volatility" => ["TSLA", "NVDA", "AMD", "MRNA", "ROKU", "ZM", "PLTR", "SNOW", "NET", "CRWD",
                          "BNTX", "NVAX", "SGEN", "DVN", "FANG", "APA", "BA", "GE", "NFLX", "DIS"],
    
    # Low volatility stocks (<20% annual volatility)  
    "low_volatility" => ["MSFT", "AAPL", "JNJ", "PG", "KO", "WMT", "HD", "UNH", "JPM", "USB",
                         "XOM", "CVX", "MMM", "HON", "TMO", "ABT", "COST", "LOW", "PEP", "MCD"],
    
    # High volume stocks (>10M shares daily average)
    "high_volume" => ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "SPY", "QQQ", "AMZN", "GOOGL", "META",
                     "F", "BAC", "GE", "PFE", "T", "WFC", "XOM", "CVX", "JPM", "DIS"],
    
    # Medium volume stocks (1M-10M shares daily)
    "medium_volume" => ["ORCL", "CRM", "ADBE", "NFLX", "UNH", "JNJ", "PG", "HD", "MCD", "KO",
                       "ABBV", "TMO", "DHR", "ABT", "BMY", "LLY", "REGN", "GILD", "HON", "MMM"]
)

function main()
    println("Diverse Stock Selection Framework")
    println("=" ^ 50)
    
    # Step 1: Define selection criteria
    selection_criteria = [
        ("Technology Large Cap", "Technology", "large_cap", 3),
        ("Technology Mid Cap", "Technology", "mid_cap", 2),
        ("Technology High Vol", "Technology", "high_vol", 2),
        
        ("Healthcare Large Cap", "Healthcare", "large_cap", 3),
        ("Healthcare Mid Cap", "Healthcare", "mid_cap", 2),
        ("Healthcare High Vol", "Healthcare", "high_vol", 2),
        
        ("Finance Large Cap", "Finance", "large_cap", 3),
        ("Finance Mid Cap", "Finance", "mid_cap", 2),
        ("Finance High Vol", "Finance", "high_vol", 1),
        
        ("Energy Large Cap", "Energy", "large_cap", 2),
        ("Energy High Vol", "Energy", "high_vol", 2),
        
        ("Consumer Large Cap", "Consumer", "large_cap", 2),
        ("Consumer High Vol", "Consumer", "high_vol", 1),
        
        ("Industrial Large Cap", "Industrial", "large_cap", 2),
        ("Industrial High Vol", "Industrial", "high_vol", 1),
    ]
    
    # Step 2: Select stocks based on criteria
    selected_stocks = Dict{String, Vector{String}}()
    all_selected = Set{String}()
    
    println("\nSelecting stocks by category:")
    
    for (category_name, sector, subcategory, count) in selection_criteria
        available_stocks = get(get(STOCK_UNIVERSE, sector, Dict()), subcategory, String[])
        
        if length(available_stocks) >= count
            # Randomly sample without replacement
            selected = sample(available_stocks, count, replace=false)
            
            # Ensure no duplicates across categories
            selected = [s for s in selected if s ∉ all_selected]
            
            # If we don't have enough unique stocks, add more
            while length(selected) < count && length(available_stocks) > length(selected)
                remaining = [s for s in available_stocks if s ∉ all_selected && s ∉ selected]
                if !isempty(remaining)
                    push!(selected, first(remaining))
                else
                    break
                end
            end
            
            selected_stocks[category_name] = selected
            union!(all_selected, selected)
            
            println("   $category_name: $(join(selected, ", "))")
        else
            println("   WARNING:  $category_name: Not enough stocks available (need $count, have $(length(available_stocks)))")
        end
    end
    
    # Step 3: Add some additional diversity picks
    println("\nAdding diversity picks:")
    
    # High volume picks for liquidity
    high_vol_candidates = setdiff(STOCK_CHARACTERISTICS["high_volume"], all_selected)
    high_vol_picks = sample(high_vol_candidates, min(3, length(high_vol_candidates)), replace=false)
    selected_stocks["High Volume Picks"] = high_vol_picks
    union!(all_selected, high_vol_picks)
    println("   High Volume: $(join(high_vol_picks, ", "))")
    
    # Low volatility picks for stability
    low_vol_candidates = setdiff(STOCK_CHARACTERISTICS["low_volatility"], all_selected)
    low_vol_picks = sample(low_vol_candidates, min(2, length(low_vol_candidates)), replace=false)
    selected_stocks["Low Volatility Picks"] = low_vol_picks
    union!(all_selected, low_vol_picks)
    println("   Low Volatility: $(join(low_vol_picks, ", "))")
    
    # Step 4: Create comprehensive stock list with metadata
    println("\nCreating comprehensive stock universe...")
    
    stock_list = Vector{NamedTuple}()
    
    for (category, stocks) in selected_stocks
        for stock in stocks
            # Determine characteristics
            sector = get_stock_sector(stock)
            market_cap = get_market_cap_category(stock)
            volatility = get_volatility_category(stock)
            volume = get_volume_category(stock)
            
            push!(stock_list, (
                symbol = stock,
                category = category,
                sector = sector,
                market_cap = market_cap,
                volatility = volatility,
                volume = volume,
                selected_date = Dates.today()
            ))
        end
    end
    
    # Step 5: Create DataFrame and save
    df = DataFrame(stock_list)
    
    # Remove duplicates (keep first occurrence)
    df = unique(df, :symbol)
    
    println("\nINFO: Final Stock Universe Summary:")
    println("   Total stocks: $(nrow(df))")
    println("   Sectors: $(length(unique(df.sector)))")
    println("   Categories: $(length(unique(df.category)))")
    
    # Show breakdown by sector
    println("\nUP: Sector Breakdown:")
    sector_counts = combine(groupby(df, :sector), nrow => :count)
    for row in eachrow(sector_counts)
        println("   $(row.sector): $(row.count) stocks")
    end
    
    # Show breakdown by characteristics
    println("\nTARGET: Characteristic Breakdown:")
    char_breakdowns = [
        ("Market Cap", :market_cap),
        ("Volatility", :volatility),
        ("Volume", :volume)
    ]
    
    for (name, col) in char_breakdowns
        println("   $name:")
        char_counts = combine(groupby(df, col), nrow => :count)
        for row in eachrow(char_counts)
            println("     $(row[col]): $(row.count) stocks")
        end
    end
    
    # Step 6: Save to file
    CSV.write(OUTPUT_FILE, df)
    println("\n💾 Saved stock universe to: $OUTPUT_FILE")
    
    # Step 7: Display sample for verification
    println("\nLIST: Sample stocks:")
    sample_stocks = sample(collect(df.symbol), min(10, nrow(df)), replace=false)
    for symbol in sample_stocks
        row = df[df.symbol .== symbol, :][1, :]
        println("   $symbol: $(row.sector) - $(row.market_cap) cap, $(row.volatility) vol")
    end
    
    println("\nSUCCESS: Stock selection completed!")
    println("   Use this universe for downloading and testing individual stock data")
    println("   Next steps:")
    println("     1. Run download script with selected stocks")
    println("     2. Process with enhanced oscillators")
    println("     3. Run adaptive soliton analysis")
    println("     4. Compare results across frequencies and sectors")
    
    return true
end

function get_stock_sector(symbol::String)::String
    for (sector, subcategories) in STOCK_UNIVERSE
        for (subcategory, stocks) in subcategories
            if symbol in stocks
                return sector
            end
        end
    end
    return "Other"
end

function get_market_cap_category(symbol::String)::String
    for (sector, subcategories) in STOCK_UNIVERSE
        if haskey(subcategories, "large_cap") && symbol in subcategories["large_cap"]
            return "Large"
        elseif haskey(subcategories, "mid_cap") && symbol in subcategories["mid_cap"]
            return "Mid"
        end
    end
    return "Large"  # Default assumption
end

function get_volatility_category(symbol::String)::String
    if symbol in STOCK_CHARACTERISTICS["high_volatility"]
        return "High"
    elseif symbol in STOCK_CHARACTERISTICS["low_volatility"]
        return "Low"
    else
        return "Medium"
    end
end

function get_volume_category(symbol::String)::String
    if symbol in STOCK_CHARACTERISTICS["high_volume"]
        return "High"
    elseif symbol in STOCK_CHARACTERISTICS["medium_volume"]
        return "Medium"
    else
        return "Low"
    end
end

# Run the script if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = main()
    println(success ? "\nSUCCESS: Stock selection successful!" : "\nFAILED: Stock selection failed!")
    exit(success ? 0 : 1)
end 