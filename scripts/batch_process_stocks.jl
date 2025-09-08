using Arrow, DataFrames
using Dates
include("../src/Oscillators_enhanced.jl")
include("../src/SolitonPDE_adaptive.jl")

# Import the modules properly
using .Oscillators_enhanced
using .SolitonPDE_adaptive

const RAW_DIR = "data/raw/stocks"
const OUT_DIR = "data/processed"
const FREQ = "5min"

function extract_symbol(filename)
    # e.g., aapl_5min_2025-07-19.arrow -> AAPL
    return uppercase(split(filename, "_")[1])
end

function process_stock_file(file)
    symbol = extract_symbol(file)
    println("\nProcessing $symbol from $file ...")
    df = DataFrame(Arrow.Table(joinpath(RAW_DIR, file)))
    if nrow(df) < 100
        println("   Skipping $symbol (too few rows)")
        return
    end
    # Compute oscillator features
    df_osc = try
        compute_oscillators_enhanced(df; freq=Min5)
    catch e
        println("   WARNING: Oscillator computation failed for $symbol: $e")
        return
    end
    # Compute soliton features row-wise
    soliton_features = NamedTuple[]
    for row in eachrow(df_osc)
        # Example: use 5 most recent oscillator values for PDE
        try
            osc_tuple = (
                row.RSI14, row.StochK14, row.CCI20, row.MACD, row.MACD_signal
            )
            # Use a dummy VIX value (e.g., 20.0) if not available
            soliton = simulate_soliton_adaptive(osc_tuple, 20.0; freq=FreqMin5)
            push!(soliton_features, soliton)
        catch e
            # Fallback: fill with missing
            push!(soliton_features, (; hamiltonian=missing, energy=missing, asymmetry_x=missing, asymmetry_y=missing, concentration=missing))
        end
    end
    # Merge soliton features into DataFrame
    df_soliton = DataFrame(soliton_features)
    df_final = hcat(df_osc, df_soliton)
    # Save as Arrow
    out_path = joinpath(OUT_DIR, "soliton_features_$(symbol).arrow")
    Arrow.write(out_path, df_final)
    println("   SUCCESS: Saved processed file: $out_path")
end

function main()
    files = filter(f -> endswith(f, ".arrow"), readdir(RAW_DIR))
    for file in files
        process_stock_file(file)
    end
    println("\nAll stocks processed!")
end

main() 