#!/usr/bin/env julia --compiled-modules=no

"""
Linear Regression Heatmap Visualization
3-Day Horizon Soliton Enhancement Analysis
Showing the "cuboid from above" view of feature relationships
"""

using Arrow, DataFrames, Statistics, LinearAlgebra
using Plots, StatsBase
using StatsPlots: heatmap
using PlotlyJS

plotlyjs() # Use PlotlyJS backend for interactive heatmaps

println("🎨 CREATING LINEAR REGRESSION HEATMAP VISUALIZATION")
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

# Create feature type mapping for coloring
feature_types = Dict()
for col in baseline_cols
    feature_types[col] = "Baseline"
end
for col in soliton_cols
    feature_types[col] = "Soliton"
end
for col in postcoll_cols
    feature_types[col] = "PostCollision"
end

println("\n📊 COMPUTING CORRELATIONS AND COEFFICIENTS...")

# 1. Correlation Heatmap: Features vs Target
println("Computing feature-target correlations...")
feature_target_corr = []
for col in all_feature_cols
    values = X_all[!, col]
    valid_idx = .!isnan.(values) .& .!isnan.(y)
    if sum(valid_idx) > 10
        corr_val = cor(values[valid_idx], y[valid_idx])
        push!(feature_target_corr, (feature=col, correlation=corr_val, type=feature_types[col]))
    end
end

# 2. Feature-Feature Correlation Matrix
println("Computing feature-feature correlation matrix...")
correlation_matrix = zeros(length(all_feature_cols), length(all_feature_cols))
for (i, col1) in enumerate(all_feature_cols)
    for (j, col2) in enumerate(all_feature_cols)
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
X_matrix = Matrix(X_all)
valid_rows = .!any(isnan.(X_matrix), dims=2)[:]
X_clean = X_matrix[valid_rows, :]
y_clean = y[valid_rows]

# Add intercept and solve
X_ext = hcat(ones(size(X_clean, 1)), X_clean)
β = (X_ext' * X_ext + 1e-6*I) \ (X_ext' * y_clean)
coefficients = β[2:end]  # Remove intercept

# 4. Create visualizations
println("\n🎨 CREATING HEATMAP VISUALIZATIONS...")

# Color schemes for different feature types
baseline_color = :reds
soliton_color = :blues  
postcoll_color = :greens

# Plot 1: Feature-Target Correlation Heatmap (vertical bar chart style)
println("Creating feature-target correlation heatmap...")

# Prepare data for plotting
feature_names = [f.feature for f in feature_target_corr]
correlations = [f.correlation for f in feature_target_corr]
types = [f.type for f in feature_target_corr]

# Create color mapping
colors = []
for t in types
    if t == "Baseline"
        push!(colors, :red)
    elseif t == "Soliton"
        push!(colors, :blue)
    else  # PostCollision
        push!(colors, :green)
    end
end

p1 = heatmap(
    reshape(correlations, 1, length(correlations)),
    xlabel="Features",
    ylabel="Correlation with 3d Returns", 
    title="Feature-Target Correlations (3-Day Horizon)\n🔴 Baseline | 🔵 Soliton | 🟢 Post-Collision",
    xticks=(1:length(feature_names), feature_names),
    xrotation=45,
    color=:RdBu,
    clims=(-maximum(abs.(correlations)), maximum(abs.(correlations))),
    size=(1200, 300),
    dpi=300
)

# Plot 2: Feature-Feature Correlation Matrix  
println("Creating feature-feature correlation matrix...")

p2 = heatmap(
    correlation_matrix,
    xlabel="Features",
    ylabel="Features", 
    title="Feature-Feature Correlation Matrix\n🔴 Baseline | 🔵 Soliton | 🟢 Post-Collision",
    xticks=(1:length(all_feature_cols), all_feature_cols),
    yticks=(1:length(all_feature_cols), all_feature_cols),
    xrotation=45,
    yrotation=0,
    color=:RdBu,
    clims=(-1, 1),
    size=(800, 800),
    dpi=300,
    aspect_ratio=:equal
)

# Plot 3: Linear Regression Coefficients Heatmap
println("Creating coefficient importance heatmap...")

p3 = heatmap(
    reshape(coefficients, 1, length(coefficients)),
    xlabel="Features",
    ylabel="Linear Coefficient",
    title="Linear Regression Coefficients (3-Day Horizon)\n🔴 Baseline | 🔵 Soliton | 🟢 Post-Collision",
    xticks=(1:length(all_feature_cols), all_feature_cols),
    xrotation=45,
    color=:RdBu,
    clims=(-maximum(abs.(coefficients)), maximum(abs.(coefficients))),
    size=(1200, 300),
    dpi=300
)

# Plot 4: Combined "Cuboid from Above" View - Feature Importance Grid
println("Creating cuboid from above view...")

# Create a 2D grid showing feature types and their importance
n_baseline = length(baseline_cols)
n_soliton = length(soliton_cols) 
n_postcoll = length(postcoll_cols)

# Create importance matrix (correlation * abs(coefficient))
importance_scores = abs.(correlations) .* abs.(coefficients)

# Arrange in a spatial grid by feature type
grid_data = zeros(3, max(n_baseline, n_soliton, n_postcoll))

# Fill baseline row
for (i, col) in enumerate(baseline_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing
        grid_data[1, i] = importance_scores[idx]
    end
end

# Fill soliton row  
for (i, col) in enumerate(soliton_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing
        grid_data[2, i] = importance_scores[idx]
    end
end

# Fill post-collision row
for (i, col) in enumerate(postcoll_cols)
    idx = findfirst(f -> f == col, feature_names)
    if idx !== nothing
        grid_data[3, i] = importance_scores[idx]
    end
end

p4 = heatmap(
    grid_data,
    xlabel="Feature Index within Type",
    ylabel="Feature Type",
    title="Soliton Feature Importance Grid (Cuboid View from Above)\nCombined Correlation × Coefficient Strength",
    yticks=(1:3, ["🔴 Baseline", "🔵 Soliton", "🟢 Post-Collision"]),
    color=:plasma,
    size=(800, 400),
    dpi=300
)

# Plot 5: Enhanced Spatial View - Feature Space Map
println("Creating feature space map...")

# Create spatial coordinates for features based on their relationships
coords_x = []
coords_y = []
importance_vals = []
feature_labels = []

# Baseline features (bottom row)
for (i, col) in enumerate(baseline_cols)
    push!(coords_x, i)
    push!(coords_y, 1)
    idx = findfirst(f -> f == col, feature_names)
    push!(importance_vals, idx !== nothing ? importance_scores[idx] : 0)
    push!(feature_labels, col)
end

# Soliton features (middle row)  
for (i, col) in enumerate(soliton_cols)
    push!(coords_x, i)
    push!(coords_y, 2)
    idx = findfirst(f -> f == col, feature_names)
    push!(importance_vals, idx !== nothing ? importance_scores[idx] : 0)
    push!(feature_labels, col)
end

# Post-collision features (top row)
for (i, col) in enumerate(postcoll_cols)
    push!(coords_x, i)
    push!(coords_y, 3)
    idx = findfirst(f -> f == col, feature_names)
    push!(importance_vals, idx !== nothing ? importance_scores[idx] : 0)
    push!(feature_labels, col)
end

p5 = scatter(
    coords_x, coords_y,
    marker_z=importance_vals,
    markersize=15,
    xlabel="Feature Position",
    ylabel="Feature Layer",
    title="3D Feature Space (Viewed from Above)\nBubble Color = Predictive Importance",
    yticks=(1:3, ["Baseline", "Soliton", "Post-Collision"]),
    color=:viridis,
    size=(1000, 600),
    dpi=300,
    legend=:right
)

# Combine plots into final layout
println("Combining plots into final visualization...")

final_plot = plot(
    p1, p3, p2, p4, p5,
    layout=(3, 2),
    size=(1600, 1400),
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
    
    savefig(p5, "feature_space_map.png")
    println("✅ Saved feature_space_map.png")
    
    savefig(final_plot, "linear_regression_heatmap_comprehensive.png")
    println("✅ Saved linear_regression_heatmap_comprehensive.png")
    
catch e
    println("⚠️  Could not save PNG files: $e")
    println("Attempting HTML save...")
    
    try
        savefig(final_plot, "linear_regression_heatmap_comprehensive.html")
        println("✅ Saved linear_regression_heatmap_comprehensive.html")
    catch e2
        println("❌ Could not save HTML either: $e2")
    end
end

# Print summary statistics
println("\n📊 SUMMARY STATISTICS:")
println("-" ^ 40)

strongest_features = sort(feature_target_corr, by=x->abs(x.correlation), rev=true)[1:5]
println("🎯 Top 5 Most Correlated Features:")
for (i, f) in enumerate(strongest_features)
    println("  $i. $(f.feature) ($(f.type)): $(round(f.correlation, digits=4))")
end

soliton_avg_corr = mean([abs(f.correlation) for f in feature_target_corr if f.type == "Soliton"])
baseline_avg_corr = mean([abs(f.correlation) for f in feature_target_corr if f.type == "Baseline"])
postcoll_avg_corr = mean([abs(f.correlation) for f in feature_target_corr if f.type == "PostCollision"])

println("\n📈 Average Absolute Correlations by Type:")
println("  🔴 Baseline: $(round(baseline_avg_corr, digits=4))")
println("  🔵 Soliton: $(round(soliton_avg_corr, digits=4))")
println("  🟢 Post-Collision: $(round(postcoll_avg_corr, digits=4))")

enhancement_factor = (soliton_avg_corr + postcoll_avg_corr) / baseline_avg_corr
println("\n🌊 Soliton Enhancement Factor: $(round(enhancement_factor, digits=2))x")

println("\n✅ Linear regression heatmap analysis complete!")
println("🎨 Check the generated PNG/HTML files for interactive visualizations") 