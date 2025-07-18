#!/usr/bin/env julia --compiled-modules=no

"""
30-Year Soliton-Oscillator Pipeline (Fixed)
Complete rebuild with 30 years of S&P 500 data (1995-2024)
Covers multiple market cycles for robust soliton validation
"""

println("30-Year Soliton-Oscillator Market Hypothesis Pipeline (Fixed)")
println("=" ^ 75)

using Arrow, DataFrames, Statistics, Dates, CSV, StatsBase
using LinearAlgebra

# Step 1: Load 30-Year Market Data
println("\nStep 1: Loading 30-year S&P 500 dataset...")
df_market = DataFrame(Arrow.Table("data/raw/spx_30years_stooq.arrow"))

println("Loaded $(nrow(df_market)) trading days")
println("Date range: $(minimum(df_market.Date)) to $(maximum(df_market.Date))")
years_span = (maximum(df_market.Date) - minimum(df_market.Date)).value / 365.25
println("Span: $(round(years_span, digits=1)) years")
println("Price range: \$$(round(minimum(df_market.Close), digits=2)) - \$$(round(maximum(df_market.Close), digits=2))")

total_return = (maximum(df_market.Close) - minimum(df_market.Close)) / minimum(df_market.Close) * 100
println("Total return: $(round(total_return, digits=1))%")

# Step 2: Compute Oscillators
println("\nStep 2: Computing oscillators on 30-year dataset...")
include("src/Oscillators.jl")
using .Oscillators

df_oscillators = compute_oscillators(df_market)
println("RSI, Stoch%K, CCI, MACD computed on $(nrow(df_oscillators)) days")

# Check oscillator ranges
oscillator_cols = ["RSI14", "StochK14", "CCI20", "MACDsig"]
for col in oscillator_cols
    values = df_oscillators[!, col]
    valid_values = values[.!isnan.(values)]
    if !isempty(valid_values)
        println("   $col: [$(round(minimum(valid_values), digits=3)), $(round(maximum(valid_values), digits=3))] ($(length(valid_values)) valid)")
    end
end

# Step 3: Soliton PDE Simulations
println("\nStep 3: Running soliton PDE simulations...")
include("src/SolitonPDE_simple.jl")
using .SolitonPDE_simple

# Initialize soliton feature arrays
n_rows = nrow(df_oscillators)
soliton_dates = Date[]
soliton_heights = Float64[]
soliton_probe_means = Float64[]
soliton_probe_stds = Float64[]
soliton_probe_maxs = Float64[]
soliton_energies = Float64[]
soliton_energy_densities = Float64[]
soliton_asymmetry_x = Float64[]
soliton_asymmetry_y = Float64[]
soliton_concentrations = Float64[]

println("Processing $(n_rows) simulations...")
print("Progress: ")

successful_sims = 0

for (i, row) in enumerate(eachrow(df_oscillators))
    # Progress indicator (every 1000 rows)
    if i % 1000 == 0 || i == n_rows
        progress_pct = round(i / n_rows * 100, digits=1)
        print("$(progress_pct)% ")
    end
    
    # Skip rows with NaN oscillators
    if any(isnan, [row.RSI14, row.StochK14, row.CCI20, row.MACDsig])
        continue
    end
    
    # Run soliton simulation
    try
        # Use fixed VIX value for now (can be enhanced later)
        vix_value = 25.0
        
        result = simulate_soliton(
            (row.RSI14, row.StochK14, row.CCI20, row.MACDsig), 
            vix_value;
            grid=16, T=0.5, dt=1e-3  # Smaller grid for speed
        )
        
        # Store features
        push!(soliton_dates, row.Date)
        push!(soliton_heights, result.H)
        push!(soliton_probe_means, mean(result.F))
        push!(soliton_probe_stds, std(result.F))
        push!(soliton_probe_maxs, maximum(abs.(result.F)))
        push!(soliton_energies, result.energy)
        
        # Calculate energy density (energy per volume)
        volume = (2.0)^3  # [-L,L]³ with L=1.0
        push!(soliton_energy_densities, result.energy / volume)
        
        push!(soliton_asymmetry_x, result.asymmetry_x)
        push!(soliton_asymmetry_y, result.asymmetry_y)
        push!(soliton_concentrations, result.concentration)
        
        successful_sims += 1
        
    catch e
        # Skip problematic simulations silently for speed
        continue
    end
end

println("\nCompleted $(successful_sims) successful soliton simulations")

# Step 4: Create Soliton Features DataFrame
println("\nStep 4: Building soliton features dataset...")

df_soliton_features = DataFrame(
    Date = soliton_dates,
    SolitonHeight = soliton_heights,
    SolitonProbeMean = soliton_probe_means,
    SolitonProbeStd = soliton_probe_stds,
    SolitonProbeMax = soliton_probe_maxs,
    SolitonEnergy = soliton_energies,
    SolitonEnergyDensity = soliton_energy_densities,
    SolitonAsymmetryX = soliton_asymmetry_x,
    SolitonAsymmetryY = soliton_asymmetry_y,
    SolitonConcentration = soliton_concentrations
)

println("   Created $(nrow(df_soliton_features)) soliton feature rows")

# Step 5: Merge with Original Data
println("\nStep 5: Merging datasets...")

# Merge soliton features with market data
df_combined = leftjoin(df_oscillators, df_soliton_features, on=:Date)

println("   Combined dataset: $(nrow(df_combined)) rows")
println("   Soliton coverage: $(sum(.!ismissing.(df_combined.SolitonHeight))) rows")

# Step 6: Add Forward Returns
println("\nStep 6: Adding forward returns...")

# Add forward returns for multiple horizons
horizons = [1, 3, 5, 10]
for horizon in horizons
    col_name = Symbol("ForwardReturn$(horizon)d")
    forward_returns = fill(NaN, nrow(df_combined))
    
    for i in 1:(nrow(df_combined) - horizon)
        if !ismissing(df_combined.Close[i]) && !ismissing(df_combined.Close[i + horizon])
            forward_returns[i] = (df_combined.Close[i + horizon] - df_combined.Close[i]) / df_combined.Close[i]
        end
    end
    
    df_combined[!, col_name] = forward_returns
    
    valid_returns = sum(.!isnan.(forward_returns))
    println("   Added $(horizon)d forward returns: $(valid_returns) valid samples")
end

# Step 7: Save Full Dataset
println("\nStep 7: Saving 30-year feature dataset...")
output_path = "data/processed/soliton_features_30years.arrow"
Arrow.write(output_path, df_combined)

file_size = filesize(output_path)
println("Saved $output_path")
println("   Size: $(round(file_size / 1024 / 1024, digits=1)) MB")
println("   Rows: $(nrow(df_combined))")
println("   Columns: $(length(names(df_combined)))")

# Step 8: Quick Multi-Horizon Analysis
println("\nStep 8: 30-Year Multi-Horizon Analysis Preview...")

# Clean dataset for analysis
df_clean = dropmissing(df_combined, [:SolitonHeight, :ForwardReturn1d, :ForwardReturn3d, :ForwardReturn5d])
println("Clean samples for analysis: $(nrow(df_clean))")

if nrow(df_clean) > 100
    # Feature sets
    baseline_cols = ["RSI14", "StochK14", "CCI20", "MACDsig"]
    soliton_cols = ["SolitonHeight", "SolitonProbeMean", "SolitonProbeStd", 
                   "SolitonProbeMax", "SolitonEnergy", "SolitonEnergyDensity"]
    postcoll_cols = ["SolitonAsymmetryX", "SolitonAsymmetryY", "SolitonConcentration"]
    
    # Quick correlation analysis for each horizon
    target_cols = ["ForwardReturn1d", "ForwardReturn3d", "ForwardReturn5d"]
    horizon_labels = ["1-Day", "3-Day", "5-Day"]
    
    println("\nSolitonAsymmetryY Correlation Analysis (30 Years):")
    asymmetryY_correlations = []
    
    for (horizon_label, target_col) in zip(horizon_labels, target_cols)
        y = df_clean[!, Symbol(target_col)]
        asymmetryY_values = df_clean[!, :SolitonAsymmetryY]
        
        valid_idx = .!isnan.(y) .& .!isnan.(asymmetryY_values)
        
        if sum(valid_idx) > 100
            corr_val = cor(asymmetryY_values[valid_idx], y[valid_idx])
            push!(asymmetryY_correlations, corr_val)
            
            # Compare to MACD
            macd_values = df_clean[!, :MACDsig]
            valid_macd = .!isnan.(y) .& .!isnan.(macd_values)
            macd_corr = cor(macd_values[valid_macd], y[valid_macd])
            
            println("   $horizon_label: SolitonAsymmetryY=$(round(corr_val, digits=4)), MACD=$(round(macd_corr, digits=4))")
        end
    end
    
    # Trend analysis
    if length(asymmetryY_correlations) >= 3
        horizon_nums = [1, 3, 5]
        trend_corr = cor(horizon_nums, abs.(asymmetryY_correlations))
        
        println("\n30-Year Trend Analysis:")
        println("   AsymmetryY absolute correlation trend: $(round(trend_corr, digits=4))")
        
        if trend_corr > 0.8
            println("   Strong positive trend: Post-collision features highly predictive at longer horizons")
        elseif trend_corr > 0.5
            println("   Moderate positive trend: Clear improvement at longer horizons")
        else
            println("   Weak trend: Limited horizon dependency")
        end
        
        # Calculate improvement
        improvement_1_to_5 = ((abs(asymmetryY_correlations[3]) - abs(asymmetryY_correlations[1])) / abs(asymmetryY_correlations[1])) * 100
        println("   SolitonAsymmetryY improvement (1d to 5d): $(round(improvement_1_to_5, digits=1))%")
    end
    
    # Enhancement factors by feature type
    println("\nFeature Type Performance:")
    for (horizon_label, target_col) in zip(horizon_labels, target_cols)
        y = df_clean[!, Symbol(target_col)]
        
        # Calculate average absolute correlations by type
        baseline_corrs = []
        soliton_corrs = []
        postcoll_corrs = []
        
        for col in baseline_cols
            values = df_clean[!, col]
            valid_idx = .!isnan.(y) .& .!isnan.(values)
            if sum(valid_idx) > 100
                push!(baseline_corrs, abs(cor(values[valid_idx], y[valid_idx])))
            end
        end
        
        for col in soliton_cols
            if col in names(df_clean)
                values = df_clean[!, col]
                valid_idx = .!isnan.(y) .& .!isnan.(values)
                if sum(valid_idx) > 100
                    push!(soliton_corrs, abs(cor(values[valid_idx], y[valid_idx])))
                end
            end
        end
        
        for col in postcoll_cols
            if col in names(df_clean)
                values = df_clean[!, col]
                valid_idx = .!isnan.(y) .& .!isnan.(values)
                if sum(valid_idx) > 100
                    push!(postcoll_corrs, abs(cor(values[valid_idx], y[valid_idx])))
                end
            end
        end
        
        if !isempty(baseline_corrs) && !isempty(soliton_corrs) && !isempty(postcoll_corrs)
            baseline_avg = mean(baseline_corrs)
            soliton_avg = mean(soliton_corrs)
            postcoll_avg = mean(postcoll_corrs)
            enhancement_factor = (soliton_avg + postcoll_avg) / baseline_avg
            
            println("   $horizon_label Enhancement: $(round(enhancement_factor, digits=2))x")
        end
    end
end

# Dataset statistics
println("\n30-Year Dataset Summary:")
println("   Period: $(minimum(df_combined.Date)) to $(maximum(df_combined.Date))")
println("   Trading days: $(nrow(df_combined))")
println("   Successful soliton simulations: $(sum(.!ismissing.(df_combined.SolitonHeight)))")
println("   Market cycles covered: Dot-com bubble, 2008 crisis, COVID-19, multiple bull/bear markets")
println("   Price appreciation: $(round(total_return, digits=1))% over $(round(years_span, digits=1)) years")

# Market regime analysis
println("\nMarket Regime Breakdown:")
year_groups = [
    (1995, 1999, "Dot-com buildup"),
    (2000, 2002, "Tech crash"),
    (2003, 2007, "Mid-2000s bull"),
    (2008, 2009, "Financial crisis"),
    (2010, 2019, "Long bull market"),
    (2020, 2024, "COVID & recovery")
]

for (start_year, end_year, description) in year_groups
    regime_data = filter(row -> start_year <= Dates.year(row.Date) <= end_year, df_combined)
    if !isempty(regime_data)
        regime_samples = nrow(regime_data)
        soliton_samples = sum(.!ismissing.(regime_data.SolitonHeight))
        println("   $description ($start_year-$end_year): $regime_samples days, $soliton_samples solitons")
    end
end

println("\n30-Year Pipeline Complete!")
println("Ready for comprehensive soliton analysis across multiple market cycles")
println("Dataset: data/processed/soliton_features_30years.arrow")
println("Next: Run detailed multi-horizon analysis and visualizations") 