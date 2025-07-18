#!/usr/bin/env julia

"""
XGBoost ML Experiment for Oscillator-Soliton Performative Market Hypothesis

Tests whether soliton collision features extracted from technical oscillators
have predictive power for forward returns beyond baseline oscillator features.

Usage: julia notebooks/xgboost_test.jl
"""

using MLJ, Parquet, DataFrames, DataFramesMeta
using Statistics, StatsBase
using Plots, PlotlyJS  # For interactive plots
using BenchmarkTools

# Set up plotting backend
plotlyjs()

"""
Load features from Parquet file with validation
"""
function load_features(file_path::String)::DataFrame
    if !isfile(file_path)
        error("Features file not found: $file_path")
    end
    
    println("Loading features from $file_path...")
    feat_df = DataFrame(Parquet.File(file_path))
    
    println("   Loaded $(nrow(feat_df)) rows with $(ncol(feat_df)) columns")
    println("   Columns: $(names(feat_df))")
    
    return feat_df
end

"""
Prepare features for ML: handle missing values, create feature sets
"""
function prepare_ml_data(feat_df::DataFrame)
    println("Preparing ML data...")
    
    # Remove rows with NaN in target or key features
    required_cols = [:forward_return, :H, :F1, :F2, :F3, :energy]
    
    valid_mask = trues(nrow(feat_df))
    for col in required_cols
        if col in names(feat_df)
            valid_mask .&= .!isnan.(feat_df[!, col])
        end
    end
    
    clean_df = feat_df[valid_mask, :]
    dropped = nrow(feat_df) - nrow(clean_df)
    
    if dropped > 0
        println("   Dropped $dropped rows with missing values")
    end
    
    if nrow(clean_df) < 10
        error("Insufficient clean data for ML (need ≥10 rows, got $(nrow(clean_df)))")
    end
    
    # Define feature sets for comparison
    baseline_features = [:RSI14, :StochK14, :CCI20, :MACDsig]
    soliton_features = [:H, :F1, :F2, :F3, :energy]
    combined_features = vcat(baseline_features, soliton_features)
    
    # Filter to available features
    baseline_available = filter(f -> f in names(clean_df), baseline_features)
    soliton_available = filter(f -> f in names(clean_df), soliton_features)
    combined_available = vcat(baseline_available, soliton_available)
    
    println("   Clean data: $(nrow(clean_df)) rows")
    println("   Baseline features: $baseline_available")
    println("   Soliton features: $soliton_available")
    
    return clean_df, baseline_available, soliton_features, combined_features
end

"""
Train and evaluate XGBoost model
"""
function train_xgboost(X::DataFrame, y::Vector{Float64}, feature_names::Vector{Symbol}; 
                      cv_folds::Int=5, test_size::Float64=0.2)
    
    println("Training XGBoost with features: $feature_names")
    
    # Extract feature matrix
    X_features = Matrix(X[:, feature_names])
    
    if size(X_features, 2) == 0
        error("No features available for training")
    end
    
    # Train/test split
    n = length(y)
    n_test = round(Int, n * test_size)
    n_train = n - n_test
    
    # Use temporal split (last 20% as test set)
    train_idx = 1:n_train
    test_idx = (n_train + 1):n
    
    X_train = X_features[train_idx, :]
    X_test = X_features[test_idx, :]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    println("   Train set: $(length(y_train)) samples")
    println("   Test set: $(length(y_test)) samples")
    
    # Load XGBoost model
    XGBoostRegressor = @load XGBoostRegressor pkg=MLJXGBoostInterface
    
    # Configure model
    model = XGBoostRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Convert to MLJ format
    X_train_table = MLJ.table(X_train)
    X_test_table = MLJ.table(X_test)
    
    # Train model
    println("   Training...")
    mach = machine(model, X_train_table, y_train)
    MLJ.fit!(mach, verbosity=0)
    
    # Predictions
    y_pred_train = MLJ.predict(mach, X_train_table)
    y_pred_test = MLJ.predict(mach, X_test_table)
    
    # Metrics
    train_rmse = rmse(y_train, y_pred_train)
    test_rmse = rmse(y_test, y_pred_test)
    train_r2 = cor(y_train, y_pred_train)^2
    test_r2 = cor(y_test, y_pred_test)^2
    
    # Cross-validation
    println("   Cross-validation ($cv_folds folds)...")
    cv_result = evaluate!(mach, 
                         resampling=CV(nfolds=cv_folds, shuffle=true, rng=42),
                         measure=rmse, 
                         verbosity=0)
    
    cv_rmse_mean = mean(cv_result.measurement)
    cv_rmse_std = std(cv_result.measurement)
    
    results = (
        model = mach,
        train_rmse = train_rmse,
        test_rmse = test_rmse,
        train_r2 = train_r2,
        test_r2 = test_r2,
        cv_rmse_mean = cv_rmse_mean,
        cv_rmse_std = cv_rmse_std,
        feature_names = feature_names,
        y_train = y_train,
        y_test = y_test,
        y_pred_train = y_pred_train,
        y_pred_test = y_pred_test
    )
    
    return results
end

"""
Compare model performance across feature sets
"""
function compare_models(clean_df::DataFrame, baseline_features::Vector{Symbol}, 
                       soliton_features::Vector{Symbol}, combined_features::Vector{Symbol})
    
    println("\nModel Comparison")
    println("=" ^ 50)
    
    y = clean_df.forward_return
    
    results = Dict{String, NamedTuple}()
    
    # Test different feature sets
    feature_sets = [
        ("Baseline (Oscillators)", baseline_features),
        ("Soliton Features", soliton_features), 
        ("Combined", combined_features)
    ]
    
    for (name, features) in feature_sets
        if isempty(features)
            println("Warning: Skipping $name: no features available")
            continue
        end
        
        try
            println("\nTesting: $name")
            result = train_xgboost(clean_df, y, features)
            results[name] = result
            
            println("   Train RMSE: $(round(result.train_rmse, digits=4))")
            println("   Test RMSE:  $(round(result.test_rmse, digits=4))")
            println("   Train R²:   $(round(result.train_r2, digits=4))")
            println("   Test R²:    $(round(result.test_r2, digits=4))")
            println("   CV RMSE:    $(round(result.cv_rmse_mean, digits=4)) ± $(round(result.cv_rmse_std, digits=4))")
            
        catch e
            println("Failed to train $name: $e")
        end
    end
    
    return results
end

"""
Create diagnostic plots
"""
function create_plots(results::Dict{String, NamedTuple})
    println("\nCreating diagnostic plots...")
    
    plots = []
    
    # Performance comparison bar chart
    model_names = collect(keys(results))
    test_rmse_values = [results[name].test_rmse for name in model_names]
    test_r2_values = [results[name].test_r2 for name in model_names]
    
    p1 = bar(model_names, test_rmse_values, 
             title="Test RMSE Comparison", 
             ylabel="RMSE", 
             color=:lightblue,
             rotation=45)
    
    p2 = bar(model_names, test_r2_values,
             title="Test R² Comparison", 
             ylabel="R²", 
             color=:lightgreen,
             rotation=45)
    
    push!(plots, p1, p2)
    
    # Prediction vs actual plots for best model
    if !isempty(results)
        best_model_name = model_names[argmin(test_rmse_values)]
        best_result = results[best_model_name]
        
        p3 = scatter(best_result.y_test, best_result.y_pred_test,
                    title="Predictions vs Actual ($best_model_name)",
                    xlabel="Actual Forward Return",
                    ylabel="Predicted Forward Return",
                    alpha=0.6)
        
        # Add diagonal line
        min_val = min(minimum(best_result.y_test), minimum(best_result.y_pred_test))
        max_val = max(maximum(best_result.y_test), maximum(best_result.y_pred_test))
        plot!(p3, [min_val, max_val], [min_val, max_val], 
              line=:dash, color=:red, label="Perfect Prediction")
        
        push!(plots, p3)
        
        # Residuals plot
        residuals = best_result.y_test .- best_result.y_pred_test
        p4 = scatter(best_result.y_pred_test, residuals,
                    title="Residuals vs Predicted ($best_model_name)",
                    xlabel="Predicted Forward Return", 
                    ylabel="Residuals",
                    alpha=0.6)
        hline!(p4, [0], line=:dash, color=:red, label="Zero Residual")
        
        push!(plots, p4)
    end
    
    return plots
end

"""
Print summary and conclusions
"""
function print_conclusions(results::Dict{String, NamedTuple})
    println("\n" * "=" ^ 60)
    println("Experiment Conclusions")
    println("=" ^ 60)
    
    if isempty(results)
        println("No models were successfully trained")
        return
    end
    
    # Find best model by test RMSE
    model_names = collect(keys(results))
    test_rmse_values = [results[name].test_rmse for name in model_names]
    best_idx = argmin(test_rmse_values)
    best_model = model_names[best_idx]
    
    println("Best Model: $best_model")
    println("   Test RMSE: $(round(results[best_model].test_rmse, digits=4))")
    println("   Test R²:   $(round(results[best_model].test_r2, digits=4))")
    
    # Performance comparison
    if "Baseline (Oscillators)" in model_names && "Soliton Features" in model_names
        baseline_rmse = results["Baseline (Oscillators)"].test_rmse
        soliton_rmse = results["Soliton Features"].test_rmse
        improvement = (baseline_rmse - soliton_rmse) / baseline_rmse * 100
        
        println("\nSoliton vs Baseline Comparison:")
        println("   Baseline RMSE:    $(round(baseline_rmse, digits=4))")
        println("   Soliton RMSE:     $(round(soliton_rmse, digits=4))")
        println("   Improvement:      $(round(improvement, digits=2))%")
        
        if improvement > 5
            println("Soliton features show meaningful improvement!")
        elseif improvement > 0
            println("Warning: Soliton features show modest improvement")
        else
            println("Soliton features do not improve over baseline")
        end
    end
    
    # Next steps
    println("\nSuggested Next Steps:")
    println("   1. Increase dataset size for more robust testing")
    println("   2. Experiment with different PDE parameters (λ, μmax, T)")
    println("   3. Try different grid resolutions and integration times")
    println("   4. Test other ML models (Random Forest, Neural Networks)")
    println("   5. Implement feature importance analysis")
    println("   6. Add more sophisticated soliton collision features")
end

"""
Main experiment function
"""
function main()
    println("Oscillator-Soliton ML Experiment")
    println("=" ^ 50)
    
    # Load features
    features_file = "data/processed/features.parquet"
    
    try
        feat_df = load_features(features_file)
        
        # Prepare data
        clean_df, baseline_features, soliton_features, combined_features = prepare_ml_data(feat_df)
        
        # Compare models
        results = compare_models(clean_df, baseline_features, soliton_features, combined_features)
        
        # Create plots
        if !isempty(results)
            plots = create_plots(results)
            
            # Display plots
            for (i, p) in enumerate(plots)
                display(p)
                println("   Plot $i displayed")
            end
        end
        
        # Print conclusions
        print_conclusions(results)
        
        return results
        
    catch e
        if isa(e, SystemError) && contains(string(e), "No such file")
            println("Features file not found: $features_file")
            println("   Run the feature extraction pipeline first!")
        else
            println("Experiment failed: $e")
        end
        return nothing
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end 