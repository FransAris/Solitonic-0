#!/usr/bin/env julia

"""
Soliton-Oscillator Performance Visualization
Shows the enhancement for 1-day forward returns (highest R²) with intuitive plots
"""

using Arrow, DataFrames, Statistics, Dates
using Plots, LinearAlgebra, StatsBase

# Use GR backend to avoid PlotlyJS issues
gr()

println("📊 Loading Soliton Feature Dataset...")

# Load the processed features
if !isfile("../data/processed/soliton_features.arrow")
    println("❌ Run test_real_soliton.jl first to generate features!")
    exit(1)
end

df = DataFrame(Arrow.Table("../data/processed/soliton_features.arrow"))
println("✅ Loaded $(nrow(df)) feature vectors")

# Focus on 1-day returns (HIGHEST R² = most predictable!)
df_clean = dropmissing(df, :ForwardReturn1d)
println("📈 Analyzing $(nrow(df_clean)) samples with 1-day forward returns (highest R²)")

# Feature sets
baseline_cols = ["RSI14", "StochK14", "CCI20", "MACDsig"]
soliton_cols = ["SolitonHeight", "SolitonProbeMean", "SolitonProbeStd", 
               "SolitonProbeMax", "SolitonEnergy", "SolitonEnergyDensity"]
postcoll_cols = ["SolitonAsymmetryX", "SolitonAsymmetryY", "SolitonConcentration"]
combined_cols = vcat(baseline_cols, soliton_cols, postcoll_cols)

# Target variable
y = df_clean.ForwardReturn1d
dates = df_clean.Date

# Train models
println("🤖 Training models...")
n_train = Int(floor(0.8 * nrow(df_clean)))
train_idx = 1:n_train
test_idx = (n_train+1):nrow(df_clean)

# Helper function to train model
function train_model(feature_cols, name)
    X = Matrix(select(df_clean, feature_cols))
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Regularized linear regression
    X_train_ext = hcat(ones(size(X_train, 1)), X_train)
    X_test_ext = hcat(ones(size(X_test, 1)), X_test)
    
    λ = 1e-6
    I_reg = λ * I(size(X_train_ext, 2))
    β = (X_train_ext' * X_train_ext + I_reg) \ (X_train_ext' * y_train)
    
    y_pred_train = X_train_ext * β
    y_pred_test = X_test_ext * β
    
    # Full dataset predictions for plotting
    X_full_ext = hcat(ones(nrow(df_clean)), X)
    y_pred_full = X_full_ext * β
    
    # Metrics
    r2_test = 1 - sum((y_test .- y_pred_test).^2) / sum((y_test .- mean(y_test)).^2)
    corr_test = cor(y_test, y_pred_test)
    
    println("   $name: R² = $(round(r2_test, digits=4)), Corr = $(round(corr_test, digits=4))")
    
    return (
        name = name,
        y_pred_full = y_pred_full,
        y_pred_test = y_pred_test,
        r2 = r2_test,
        corr = corr_test,
        beta = β
    )
end

# Train all models
baseline_model = train_model(baseline_cols, "Baseline Oscillators")
soliton_model = train_model(soliton_cols, "Soliton Features")
combined_model = train_model(combined_cols, "Combined (Baseline + Soliton)")

enhancement = ((combined_model.r2 - baseline_model.r2) / abs(baseline_model.r2)) * 100
println("🚀 Soliton Enhancement: +$(round(enhancement, digits=1))%")

# ===== PLOT 1: Time Series Performance =====
println("📊 Creating time series visualization...")

p1 = plot(dates, y * 100, 
         label="Actual 1-Day Returns", 
         color=:black, linewidth=2, alpha=0.7,
         title="Soliton vs Baseline: 1-Day SPX Return Prediction",
         xlabel="Date", ylabel="1-Day Forward Return (%)",
         legend=:topright, size=(1200, 600))

plot!(p1, dates, baseline_model.y_pred_full * 100, 
      label="Baseline Prediction (R²=$(round(baseline_model.r2, digits=3)))", 
      color=:red, linewidth=2, alpha=0.8)

plot!(p1, dates, combined_model.y_pred_full * 100, 
      label="Soliton Enhanced (R²=$(round(combined_model.r2, digits=3)))", 
      color=:blue, linewidth=2, alpha=0.8)

# Add enhancement annotation
annotate!(p1, dates[end-200], maximum(y)*80, 
         text("🌊 Soliton Enhancement: +$(round(enhancement, digits=1))%", 
              :blue, :bold, 14))

# ===== PLOT 2: Prediction Accuracy Scatter =====
println("📊 Creating accuracy scatter plots...")

test_dates = dates[test_idx]
y_test = y[test_idx]

p2a = scatter(baseline_model.y_pred_test * 100, y_test * 100,
             alpha=0.6, color=:red, markersize=4,
             title="Baseline Oscillators",
             xlabel="Predicted Return (%)", ylabel="Actual Return (%)",
             legend=false, aspect_ratio=:equal)
plot!(p2a, [-6, 6], [-6, 6], color=:gray, linestyle=:dash, linewidth=2)
annotate!(p2a, -5, 5, text("R² = $(round(baseline_model.r2, digits=3))", :red, :bold, 12))

p2b = scatter(combined_model.y_pred_test * 100, y_test * 100,
             alpha=0.6, color=:blue, markersize=4,
             title="Soliton Enhanced",
             xlabel="Predicted Return (%)", ylabel="Actual Return (%)",
             legend=false, aspect_ratio=:equal)
plot!(p2b, [-6, 6], [-6, 6], color=:gray, linestyle=:dash, linewidth=2)
annotate!(p2b, -5, 5, text("R² = $(round(combined_model.r2, digits=3))", :blue, :bold, 12))

p2 = plot(p2a, p2b, layout=(1,2), size=(1000, 400),
         plot_title="Prediction Accuracy: Out-of-Sample Test Set")

# ===== PLOT 3: Signal Effectiveness Over Time =====
println("📊 Creating signal effectiveness analysis...")

# Calculate rolling prediction errors
window = 50
n_windows = length(test_idx) - window + 1
rolling_dates = test_dates[window:end]
baseline_rolling_r2 = Float64[]
combined_rolling_r2 = Float64[]

for i in 1:n_windows
    idx_window = i:(i+window-1)
    y_window = y_test[idx_window]
    
    # Baseline rolling R²
    pred_baseline = baseline_model.y_pred_test[idx_window]
    r2_base = 1 - sum((y_window .- pred_baseline).^2) / sum((y_window .- mean(y_window)).^2)
    push!(baseline_rolling_r2, r2_base)
    
    # Combined rolling R²
    pred_combined = combined_model.y_pred_test[idx_window]
    r2_comb = 1 - sum((y_window .- pred_combined).^2) / sum((y_window .- mean(y_window)).^2)
    push!(combined_rolling_r2, r2_comb)
end

enhancement_rolling = ((combined_rolling_r2 .- baseline_rolling_r2) ./ abs.(baseline_rolling_r2)) .* 100

p3 = plot(rolling_dates, enhancement_rolling,
         color=:green, linewidth=3, alpha=0.8,
         title="Rolling Soliton Enhancement (50-day windows)",
         xlabel="Date", ylabel="Enhancement over Baseline (%)",
         legend=false, size=(1200, 400))

hline!(p3, [0], color=:gray, linestyle=:dash, linewidth=2)
hline!(p3, [enhancement], color=:blue, linestyle=:dash, linewidth=2, 
       label="Overall Enhancement: +$(round(enhancement, digits=1))%")

# Color regions where soliton helps vs hurts
positive_mask = enhancement_rolling .> 0
plot!(p3, rolling_dates[positive_mask], enhancement_rolling[positive_mask],
      seriestype=:scatter, color=:green, alpha=0.7, markersize=3,
      label="Soliton Helps")
plot!(p3, rolling_dates[.!positive_mask], enhancement_rolling[.!positive_mask],
      seriestype=:scatter, color=:red, alpha=0.7, markersize=3,
      label="Baseline Better")

# ===== PLOT 4: Feature Importance =====
println("📊 Creating feature importance analysis...")

# Extract feature importance from coefficients
feature_names = vcat(baseline_cols, soliton_cols, postcoll_cols)
feature_importance = abs.(combined_model.beta[2:end])  # Skip intercept
sorted_idx = sortperm(feature_importance, rev=true)

p4 = bar(feature_importance[sorted_idx], 
        orientation=:h,
        yticks=(1:length(feature_names), feature_names[sorted_idx]),
        title="Feature Importance (|Coefficient|)",
        xlabel="Absolute Coefficient Value",
        legend=false, size=(800, 600))

# Color code feature types
colors = []
for name in feature_names[sorted_idx]
    if name in baseline_cols
        push!(colors, :red)
    elseif name in soliton_cols
        push!(colors, :blue)
    else
        push!(colors, :green)
    end
end
bar!(p4, feature_importance[sorted_idx], color=colors, orientation=:h)

# ===== PLOT 5: Performance Summary Dashboard =====
println("📊 Creating performance dashboard...")

metrics_data = [
    baseline_model.r2 * 100,
    soliton_model.r2 * 100, 
    combined_model.r2 * 100
]

p5 = bar(["Baseline\nOscillators", "Soliton\nFeatures", "Combined\nModel"], 
        metrics_data,
        color=[:red, :orange, :blue],
        title="Model Performance Comparison (R² %)",
        ylabel="R² Score (%)",
        legend=false, size=(600, 500))

# Add enhancement arrow
annotate!(p5, 2.5, (metrics_data[1] + metrics_data[3])/2,
         text("🚀 +$(round(enhancement, digits=1))%", :green, :bold, 16))

# Add correlation info
for (i, model) in enumerate([baseline_model, soliton_model, combined_model])
    annotate!(p5, i, metrics_data[i] + 0.1, 
             text("ρ=$(round(model.corr, digits=3))", :black, 10))
end

# ===== COMBINE ALL PLOTS =====
println("📊 Assembling final visualization...")

# Create the master layout
final_plot = plot(
    p1,
    plot(p2a, p2b, layout=(1,2)),
    p3, 
    plot(p4, p5, layout=(1,2)),
    layout=(4,1), 
    size=(1400, 1600),
    plot_title="🌊 Soliton-Oscillator Market Hypothesis: 1-Day SPX Returns (+$(round(enhancement, digits=1))% Enhancement)"
)

# Save plots
println("💾 Saving visualizations...")
mkpath("outputs")

savefig(final_plot, "outputs/soliton_performance_dashboard.html")
savefig(p1, "outputs/time_series_comparison.png")
savefig(p3, "outputs/rolling_enhancement.png")

println("✅ Visualizations saved:")
println("   📊 outputs/soliton_performance_dashboard.html (interactive)")
println("   📈 outputs/time_series_comparison.png")
println("   📉 outputs/rolling_enhancement.png")

# ===== SUMMARY STATS =====
println("\n🎯 Key Insights:")
println("   📊 Baseline R²: $(round(baseline_model.r2, digits=4))")
println("   🌊 Combined R²: $(round(combined_model.r2, digits=4))")
println("   🚀 Enhancement: +$(round(enhancement, digits=1))%")
println("   ⏱️  Periods where soliton helps: $(round(100 * sum(positive_mask) / length(positive_mask), digits=1))%")
println("   📈 Best enhancement period: +$(round(maximum(enhancement_rolling), digits=1))%")
println("   📉 Worst period: $(round(minimum(enhancement_rolling), digits=1))%") 