# ===============  PROJECT BOOTSTRAP PROMPT  ===================
You are my AI pair-programmer inside this IDE.  
Your job is to create a fully working, *offline, reproducible* Julia
environment for testing the **Oscillator-Soliton Performative Market
Hypothesis**.

─────────────────────────
🔑 1. Background & Goals
─────────────────────────
• We explore whether common technical oscillators (RSI, Stoch %K,
  CCI, MACD-signal) feed back into market prices in a higher-order,
  “Keynesian beauty-contest” manner.
• Idea: map the four oscillator readings at a timestamp into launch
  amplitudes of four solitary waves inside a damped, cubical φ⁴/Klein-
  Gordon PDE; capture collision features (height at origin, slice
  fields) and test if they forecast forward returns.
• Pipeline stages  
  1. Ingest HISTORICAL price/VIX data (no live feed).  
  2. Compute oscillators & normalise to [-1,1].  
  3. For each time slice, solve the PDE and emit features.  
  4. Dump features to Parquet → run MLJ models (e.g. XGBoost) to
     measure predictive power against baseline oscillators.  
  5. All artefacts cached on disk so stages are re-entrant.

─────────────────────────
🛠 2. Environment Setup
─────────────────────────
• Julia ≥ 1.10 with project-local `Project.toml` + `Manifest.toml`.
• Required packages (**exact names**):

  # data IO
  HTTP CSV Parquet DataFrames DataFramesMeta Arrow
  MarketTechnicals  RollingFunctions
  # PDE / numerics
  DifferentialEquations DiffEqOperators
  # optional GPU / MPI placeholders
  CUDA MPI
  # ML / statistics
  MLJ MLJXGBoostInterface Statistics
  # experiment management
  DrWatson  BenchmarkTools  ProgressMeter
  # utilities
  JSON3 RateLimit Retry Downloads

• Create a DrWatson-compatible folder tree:

  project_root/
  ├── data/raw/
  ├── data/processed/
  ├── src/
  │   ├── Oscillators.jl        # osc + normalisation helpers
  │   ├── SolitonPDE.jl         # PDE definition & solver
  │   ├── FeatureBuilder.jl     # orchestration glue
  ├── notebooks/                # quick experiments
  ├── runs/                     # saved DrWatson experiment outputs
  ├── Project.toml
  └── Manifest.toml

─────────────────────────
📦 3. Data Acquisition Task
─────────────────────────
Write a Julia script `scripts/download_es.jl` that:

1. Hits Yahoo Finance’s CSV endpoint for ES continuous futures:
   https://query1.finance.yahoo.com/v7/finance/download/ES=F?\
   period1=1640995200&period2=1648684800&interval=1d&events=history
   – period1 = 1-Jan-2022 UTC, period2 = 31-Mar-2022 UTC.
2. Saves to `data/raw/es_2022q1_daily.csv`.
3. Logs download size & SHA-256 for reproducibility.

Include retry (max 3) and polite rate-limit (sleep 1 s between tries).

─────────────────────────
UP: 4. Oscillator Module
─────────────────────────
Create `src/Oscillators.jl` exposing

```julia
module Oscillators
export compute_oscillators

using DataFrames, MarketTechnicals

"""
    compute_oscillators(df::DataFrame) -> DataFrame

Add columns :RSI14, :StochK14, :CCI20, :MACDsig to an OHLCV frame.
Then min-max scale each to [-1,1] across the *input* frame.
"""
function compute_oscillators(df::DataFrame)::DataFrame
    # implementation placeholder
end

end # module

Add unit tests in test/runtests.jl using a 10-row toy DataFrame.

─────────────────────────
🌊 5. Soliton PDE Skeleton
─────────────────────────
Create src/SolitonPDE.jl with a function

simulate_soliton(amplitudes::NTuple{4,Float64},
                 vix::Float64; grid=64, L=1.0, λ=1.0,
                 μmax=0.1, T=1.0, dt=1e-3) -> NamedTuple

that:

    Builds a cubic grid [-L,L]^3 with DiffEqOperators.CenteredDifference.

    Defines damped φ⁴ ℒ(u) = u_tt - c²∇²u + λu³ + μ(t)u_t.

    Sets μ(t) = μmax * (vix / vix_max_sample) (pass vix_max_sample as
    optional global or keyword).

    Places four sech pulses on ±x, ±y faces, aimed at origin, scaled
    by amplitudes.

    Integrates with Tsit5() to T, returning
    (H = u(0,0,0,T), F = [u(x_i,0,0,T) for x_i in probes]).

Stub out heavy loops; we’ll optimise later.

─────────────────────────
🔄 6. Feature Builder
─────────────────────────
src/FeatureBuilder.jl:

function build_features(price_df, vix_df; step="1d")
    # join, compute osc, loop rows, call simulate_soliton,
    # push NamedTuple → collect DataFrame, write Parquet
end

─────────────────────────
🤖 7. ML Experiment Template
─────────────────────────
In notebooks/xgboost_test.jl:

using MLJ, Parquet, DataFrames, Plots
feat_df = DataFrame(Parquet.File("data/processed/features.parquet"))
y, X = unpack(feat_df, ==(:forward_return), colname -> true)
model = @load XGBoostRegressor pkg=MLJXGBoostInterface
e = evaluate(model, X, y,
             resampling=CV(nfolds=5, shuffle=true),
             measure=rms, verbosity=1)
println(e)

Plot residuals.

─────────────────────────
SUCCESS: 8. Tasks for You, AI-IDE
─────────────────────────

    Generate Project.toml with the exact package list and ]instantiate.

    Scaffold every file/folder above; insert TODO: markers where impl.
    detail is still needed.

    Insert inline docstrings & comments so a human teammate can follow.

    Run scripts/download_es.jl; verify CSV row count ≈ 63 (daily bars).

    Run a mini end-to-end smoke test: use only the first 5 dates,
    grid = 32, T = 0.1, dt = 1e-2; produce and inspect features Parquet.

    Echo next-step suggestions (profiling, GPU path, experiment registry).

Whenever you need parameters (e.g. λ, μmax) and they are unset,
ask me via IDE chat instead of guessing.
==============================================================