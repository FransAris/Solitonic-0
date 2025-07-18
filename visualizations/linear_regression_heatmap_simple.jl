#!/usr/bin/env julia --compiled-modules=no

"""
Simplified Linear Regression Heatmap Visualization
3-Day Horizon Soliton Enhancement Analysis
Focus on core heatmap views without complex plotting conflicts
"""

using Arrow, DataFrames, Statistics, LinearAlgebra
using Plots
using StatsBase

gr() # Use GR backend for reliable heatmaps

println("🎨 CREATING SIMPLIFIED LINEAR REGRESSION HEATMAPS")
println("=" ^ 60)

# Load data
df = DataFrame(Arrow.Table("../data/processed/soliton_features.arrow"))
df_clean = dropmissing(df, :ForwardReturn3d)
println("✅ Loaded $(nrow(df_clean)) samples")

# Feature definitions
baseline_cols = ["RSI14", "StochK14", "CCI20", "MACDsig"]
soliton_cols = ["SolitonHeight", "SolitonProbeMean", "SolitonProbeStd", 
               "SolitonProbeMax", "SolitonEnergy", "SolitonEnergyDensity"]
postcoll_cols = ["SolitonAsymmetryX", "SolitonAsymmetryY", "SolitonConcentration"]
all_feature_cols = vcat(baseline_cols, soliton_cols, postcoll_cols)

# Prepare data for analysis
y = df_clean.ForwardReturn3d
X_all = select(df_clean, all_feature_cols)

println("\n📊 COMPUTING CORRELATIONS AND COEFFICIENTS...")

# 1. Feature-Target Correlations
println("Computing feature-target correlations...")
correlations = []
feature_names = []
for col in all_feature_cols
    values = X_all[!, col]
    valid_idx = .!isnan.(values) .& .!isnan.(y)
    if sum(valid_idx) > 10
        corr_val = cor(values[valid_idx], y[valid_idx])
        push!(correlations, corr_val)
        push!(feature_names, col)
    end
end

# 2. Feature-Feature Correlation Matrix
println("Computing feature-feature correlation matrix...")
n_features = length(feature_names)
correlation_matrix = zeros(n_features, n_features)
for (i, col1) in enumerate(feature_names)
    for (j, col2) in enumerate(feature_names)
        if i == j
            correlation_matrix[i, j] = 1.0
        else
            vals1 = X_all[!, col1]
            vals2 = X_all[!, col2]
            valid_idx = .!isnan.(vals1) .& .!isnan.(vals2)
            if sum(valid_idx) > 10
                correlation_matrix[i, j] = cor(vals1[valid_idx], vals2[valid_idx])
            end
        end
    end
end

# 3. Linear Regression Coefficients
println("Computing linear regression coefficients...")
X_matrix = Matrix(select(X_all, feature_names))
valid_rows = .!any(isnan.(X_matrix), dims=2)[:]
X_clean = X_matrix[valid_rows, :]
y_clean = y[valid_rows]

# Add intercept and solve
X_ext = hcat(ones(size(X_clean, 1)), X_clean)
β = (X_ext' * X_ext + 1e-6*I) \ (X_ext' * y_clean)
coefficients = β[2:end]  # Remove intercept

println("\n🎨 CREATING HEATMAP VISUALIZATIONS...")

# Plot 1: Feature-Target Correlation Bar
println("Creating feature-target correlation visualization...")
p1 = bar(
    1:length(correlations),
    correlations,
    xlabel="Features",
    ylabel="Correlation with 3d Returns",
    title="Feature-Target Correlations (3-Day Horizon)",
    xticks=(1:length(feature_names), feature_names),
    xrotation=45,
    color=ifelse.(correlations .> 0, :blue, :red),
    size=(1200, 400),
    dpi=300,
    legend=false
)

# Plot 2: Feature-Feature Correlation Matrix Heatmap
println("Creating feature-feature correlation matrix...")
p2 = heatmap(
    correlation_matrix,
    xlabel="Features",
    ylabel="Features",
    title="Feature-Feature Correlation Matrix",
    xticks=(1:length(feature_names), feature_names),
    yticks=(1:length(feature_names), feature_names),
    xrotation=45,
    color=:RdBu,
    clims=(-1, 1),
    size=(800, 800),
    dpi=300,
    aspect_ratio=:equal
)

# Plot 3: Linear Regression Coefficients
println("Creating coefficient importance visualization...")
p3 = bar(
    1:length(coefficients),
    coefficients,
    xlabel="Features",
    ylabel="Linear Coefficient",
    title="Linear Regression Coefficients (3-Day Horizon)",
    xticks=(1:length(feature_names), feature_names),
    xrotation=45,
    color=ifelse.(coefficients .> 0, :green, :orange),
    size=(1200, 400),
    dpi=300,
    legend=false
)

# Plot 4: "Cuboid from Above" - Feature Type Grid
println("Creating cuboid from above view...")

# Create importance scores (correlation * coefficient)
importance_scores = abs.(correlations) .* abs.(coefficients)

# Create grid organized by feature type
n_baseline = length(baseline_cols)
n_soliton = length(soliton_cols)
n_postcoll = length(postcoll_cols)
max_features = max(n_baseline, n_soliton, n_postcoll)

grid_data = zeros(3, max_features)
grid_labels = fill("", 3, max_features)

# Fill grid with importance scores
for (i, col) in enumerate(baseline_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing
        grid_data[1, i] = importance_scores[idx]
        grid_labels[1, i] = col
    end
end

for (i, col) in enumerate(soliton_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing && i <= max_features
        grid_data[2, i] = importance_scores[idx]
        grid_labels[2, i] = col
    end
end

for (i, col) in enumerate(postcoll_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing
        grid_data[3, i] = importance_scores[idx]
        grid_labels[3, i] = col
    end
end

p4 = heatmap(
    grid_data,
    xlabel="Feature Index within Type",
    ylabel="Feature Type",
    title="Soliton Feature Importance Grid (Cuboid View from Above)\nCombined |Correlation| × |Coefficient|",
    yticks=(1:3, ["Baseline", "Soliton", "Post-Collision"]),
    color=:plasma,
    size=(1000, 400),
    dpi=300
)

# Combine plots
println("Combining plots into final visualization...")
final_plot = plot(
    p1, p3, p2, p4,
    layout=(2, 2),
    size=(1600, 1200),
    dpi=300,
    plot_title="Linear Regression Soliton Enhancement Analysis (3-Day Horizon)"
)

# Save visualizations
println("\n💾 SAVING VISUALIZATIONS...")

try
    savefig(p1, "feature_target_correlations.png")
    println("✅ Saved feature_target_correlations.png")
    
    savefig(p2, "feature_correlation_matrix.png")
    println("✅ Saved feature_correlation_matrix.png")
    
    savefig(p3, "linear_coefficients.png")
    println("✅ Saved linear_coefficients.png")
    
    savefig(p4, "cuboid_from_above.png")
    println("✅ Saved cuboid_from_above.png")
    
    savefig(final_plot, "linear_regression_heatmap_comprehensive.png")
    println("✅ Saved linear_regression_heatmap_comprehensive.png")
    
catch e
    println("⚠️  Could not save PNG files: $e")
end

# Print summary statistics
println("\n📊 SUMMARY STATISTICS:")
println("-" ^ 40)

# Find feature types
baseline_indices = [i for (i, name) in enumerate(feature_names) if name in baseline_cols]
soliton_indices = [i for (i, name) in enumerate(feature_names) if name in soliton_cols]
postcoll_indices = [i for (i, name) in enumerate(feature_names) if name in postcoll_cols]

println("🎯 Top 5 Most Correlated Features:")
sorted_indices = sortperm(abs.(correlations), rev=true)
for i in 1:min(5, length(sorted_indices))
    idx = sorted_indices[i]
    feature_type = if idx in baseline_indices
        "Baseline"
    elseif idx in soliton_indices
        "Soliton"
    else
        "Post-Collision"
    end
    println("  $i. $(feature_names[idx]) ($feature_type): $(round(correlations[idx], digits=4))")
end

baseline_avg_corr = mean([abs(correlations[i]) for i in baseline_indices])
soliton_avg_corr = mean([abs(correlations[i]) for i in soliton_indices])
postcoll_avg_corr = mean([abs(correlations[i]) for i in postcoll_indices])

println("\n📈 Average Absolute Correlations by Type:")
println("  🔴 Baseline: $(round(baseline_avg_corr, digits=4))")
println("  🔵 Soliton: $(round(soliton_avg_corr, digits=4))")
println("  🟢 Post-Collision: $(round(postcoll_avg_corr, digits=4))")

enhancement_factor = (soliton_avg_corr + postcoll_avg_corr) / baseline_avg_corr
println("\n🌊 Soliton Enhancement Factor: $(round(enhancement_factor, digits=2))x")

println("\n✅ Linear regression heatmap analysis complete!")
println("🎨 Check the generated PNG files for visualizations") 