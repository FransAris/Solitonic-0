"""
SolitonPDE_Adaptive Module

Implements frequency-adaptive damped φ⁴/Klein-Gordon PDE solver.
Automatically adjusts parameters based on data time frequency for optimal collision feature extraction.

The PDE is: u_tt - c²∇²u + λu³ + μ(t)u_t = 0
where μ(t) = μmax * (vix / vix_max_sample) provides VIX-dependent damping.
"""
module SolitonPDE_adaptive

export simulate_soliton_adaptive, get_optimal_pde_params, DataFrequency, FreqDaily, FreqHour1, FreqMin30, FreqMin15, FreqMin5, FreqMin1

using LinearAlgebra, Statistics

# Import time frequency enum (redefine here for independence)
@enum DataFrequency begin
    FreqDaily = 1440     # Daily = 1440 minutes
    FreqHour1 = 60       # 1 hour = 60 minutes
    FreqMin30 = 30       # 30 minutes
    FreqMin15 = 15       # 15 minutes
    FreqMin5 = 5         # 5 minutes
    FreqMin1 = 1         # 1 minute
end

"""
    get_optimal_pde_params(freq::DataFrequency) -> NamedTuple

Get optimal PDE simulation parameters for different data frequencies.
Adjusts timing, grid resolution, and physical parameters to match market dynamics.

# Parameter Philosophy:
- **Integration Time (T)**: Shorter for higher frequencies to match prediction horizons
- **Grid Size**: Smaller for high-freq to reduce computational cost with more frequent simulations  
- **Damping**: Higher for high-freq data due to increased noise
- **Nonlinearity**: Adjusted for frequency-dependent market coupling strength
- **Spatial Scale**: Optimized for collision detection at each frequency

# Frequency Mappings:
- Daily: Long integration time, detailed spatial resolution
- 1-hour: Medium integration time, balanced resolution  
- 30-min: Shorter integration, moderate resolution
- 15-min: Quick integration, compact grid
- 5-min: Very short integration, small grid, high damping
- 1-min: Ultra-short integration, minimal grid, maximum damping
"""
function get_optimal_pde_params(freq::DataFrequency)::NamedTuple
    if freq == FreqDaily
        return (
            grid = 32,              # Standard resolution for daily analysis
            L = 1.0,                # Full spatial domain  
            T = 1.0,                # Full collision development time
            dt = 1e-3,              # Fine time resolution
            λ = 1.0,                # Standard nonlinearity
            μmax = 0.1,             # Moderate damping
            c = 1.0,                # Standard wave speed
            collision_threshold = 0.6  # Time fraction for collision detection
        )
    elseif freq == FreqHour1
        return (
            grid = 24,              # Reduced resolution for speed
            L = 0.8,                # Slightly smaller domain
            T = 0.6,                # Shorter collision time (matches 1-hour prediction horizon)
            dt = 8e-4,              # Slightly coarser time steps
            λ = 0.8,                # Reduced nonlinearity for shorter timescale
            μmax = 0.15,            # Increased damping for noise
            c = 1.2,                # Slightly faster waves
            collision_threshold = 0.5
        )
    elseif freq == FreqMin30
        return (
            grid = 20,              # Smaller grid for computational efficiency
            L = 0.6,                # Compact domain
            T = 0.4,                # Quick collision development
            dt = 6e-4,              # Coarser time stepping
            λ = 0.6,                # Lower nonlinearity
            μmax = 0.2,             # Higher damping for 30-min noise
            c = 1.4,                # Faster wave propagation
            collision_threshold = 0.4
        )
    elseif freq == FreqMin15
        return (
            grid = 16,              # Small grid for speed
            L = 0.5,                # Tight spatial domain
            T = 0.25,               # Very quick collision
            dt = 5e-4,              # Coarse time steps
            λ = 0.4,                # Weak nonlinearity for short timescale
            μmax = 0.25,            # Strong damping
            c = 1.6,                # Fast wave speed
            collision_threshold = 0.3
        )
    elseif freq == FreqMin5
        return (
            grid = 12,              # Minimal viable grid
            L = 0.4,                # Very compact domain
            T = 0.15,               # Ultra-quick collision
            dt = 4e-4,              # Coarse stepping for speed
            λ = 0.2,                # Very weak nonlinearity
            μmax = 0.35,            # High damping for noise control
            c = 2.0,                # Very fast waves
            collision_threshold = 0.2
        )
    elseif freq == FreqMin1
        return (
            grid = 10,              # Absolute minimum grid
            L = 0.3,                # Tiny domain
            T = 0.08,               # Lightning-fast collision
            dt = 3e-4,              # Coarse stepping  
            λ = 0.1,                # Minimal nonlinearity
            μmax = 0.5,             # Maximum damping for 1-min noise
            c = 2.5,                # Extremely fast waves
            collision_threshold = 0.15
        )
    else
        error("Unsupported data frequency: $freq")
    end
end

"""
    detect_data_frequency(time_diff_minutes::Float64) -> DataFrequency

Detect data frequency from median time difference in minutes.
"""
function detect_data_frequency(time_diff_minutes::Float64)::DataFrequency
    if time_diff_minutes >= 1200      # >= 20 hours (daily)
        return FreqDaily
    elseif time_diff_minutes >= 45    # 45-75 minutes (1 hour)
        return FreqHour1
    elseif time_diff_minutes >= 20    # 20-40 minutes (30 min)
        return FreqMin30
    elseif time_diff_minutes >= 10    # 10-20 minutes (15 min)
        return FreqMin15
    elseif time_diff_minutes >= 3     # 3-10 minutes (5 min)
        return FreqMin5
    else                             # < 3 minutes (1 min)
        return FreqMin1
    end
end

"""
    simulate_soliton_adaptive(amplitudes::NTuple{4,Float64}, vix::Float64; 
                             freq::Union{DataFrequency,Nothing}=nothing,
                             time_diff_minutes::Union{Float64,Nothing}=nothing,
                             override_params::NamedTuple=NamedTuple(),
                             custom_params::Union{NamedTuple,Nothing}=nothing) -> NamedTuple

Adaptive soliton simulation that automatically adjusts parameters based on data frequency.

# Arguments
- `amplitudes`: Four soliton launch amplitudes (from normalized oscillators)
- `vix`: VIX value for time-dependent damping
- `freq`: Data frequency (auto-detected if nothing)
- `time_diff_minutes`: Time difference for frequency detection (if freq is nothing)
- `override_params`: Manual parameter overrides (internal format)
- `custom_params`: Custom parameters for optimization (external format with grid_size, lambda, etc.)

# Returns
Enhanced NamedTuple with frequency-specific features and metadata

# Example
```julia
# For 5-minute data
amplitudes = (0.5, -0.3, 0.8, -0.2)
result = simulate_soliton_adaptive(amplitudes, 25.0, freq=FreqMin5)

# Auto-detect from time difference  
result = simulate_soliton_adaptive(amplitudes, 25.0, time_diff_minutes=5.0)

# With custom parameters for optimization
custom_params = (grid_size=12, lambda=0.5, mu_max=0.3, wave_speed=2.0, integration_time=0.1)
result = simulate_soliton_adaptive(amplitudes, 25.0, freq=FreqMin5, custom_params=custom_params)
```
"""
function simulate_soliton_adaptive(amplitudes::NTuple{4,Float64}, vix::Float64; 
                                  freq::Union{DataFrequency,Nothing}=nothing,
                                  time_diff_minutes::Union{Float64,Nothing}=nothing,
                                  override_params::NamedTuple=NamedTuple(),
                                  custom_params::Union{NamedTuple,Nothing}=nothing,
                                  vix_max_sample::Float64=50.0)::NamedTuple
    
    # Determine frequency
    if freq === nothing
        if time_diff_minutes === nothing
            @warn "No frequency specified, assuming daily"
            freq = FreqDaily
        else
            freq = detect_data_frequency(time_diff_minutes)
        end
    end
    
    # Get optimal parameters for this frequency
    base_params = get_optimal_pde_params(freq)
    
    # Apply any manual overrides
    params = merge(base_params, override_params)
    
    # Apply custom parameters if provided (for optimization)
    if custom_params !== nothing
        # Convert custom_params format to internal format
        custom_internal = NamedTuple()
        if haskey(custom_params, :grid_size)
            custom_internal = merge(custom_internal, (grid = custom_params.grid_size,))
        end
        if haskey(custom_params, :domain_length)
            custom_internal = merge(custom_internal, (L = custom_params.domain_length,))
        end
        if haskey(custom_params, :integration_time)
            custom_internal = merge(custom_internal, (T = custom_params.integration_time,))
        end
        if haskey(custom_params, :lambda)
            custom_internal = merge(custom_internal, (λ = custom_params.lambda,))
        end
        if haskey(custom_params, :mu_max)
            custom_internal = merge(custom_internal, (μmax = custom_params.mu_max,))
        end
        if haskey(custom_params, :wave_speed)
            custom_internal = merge(custom_internal, (c = custom_params.wave_speed,))
        end
        
        # Merge custom parameters with computed dt based on new grid/domain
        if haskey(custom_params, :grid_size) || haskey(custom_params, :domain_length) || haskey(custom_params, :integration_time)
            grid_custom = get(custom_params, :grid_size, params.grid)
            L_custom = get(custom_params, :domain_length, params.L)
            T_custom = get(custom_params, :integration_time, params.T)
            dx_custom = 2 * L_custom / (grid_custom - 1)
            dt_custom = min(T_custom / 100, 0.4 * dx_custom / get(custom_params, :wave_speed, params.c))
            custom_internal = merge(custom_internal, (dt = dt_custom,))
        end
        
        params = merge(params, custom_internal)
    end
    
    # Extract parameters
    grid = params.grid
    L = params.L
    T = params.T  
    dt = params.dt
    λ = params.λ
    μmax = params.μmax
    c = params.c
    collision_threshold = params.collision_threshold
    
    # Validate inputs
    @assert grid > 4 "Grid size must be > 4"
    @assert L > 0 "Domain size must be positive" 
    @assert T > 0 "Integration time must be positive"
    @assert dt > 0 "Time step must be positive"
    
    # Time stepping parameters
    n_steps = Int(ceil(T / dt))
    
    # Spatial grid
    dx = 2*L / (grid - 1)
    x = range(-L, L, length=grid)
    y = range(-L, L, length=grid) 
    z = range(-L, L, length=grid)
    
    # CFL stability check
    dt_max = 0.4 * dx / c  # Stability condition
    if dt > dt_max
        dt = 0.8 * dt_max
        n_steps = Int(ceil(T / dt))
    end
    
    # Frequency-adaptive damping
    base_damping = μmax * (vix / vix_max_sample)
    
    # Increase damping for higher frequencies (more noise)
    frequency_damping_factor = if freq == FreqMin1
        2.0  # Double damping for 1-minute data
    elseif freq == FreqMin5
        1.5  # 50% more damping for 5-minute data
    elseif freq == FreqMin15
        1.2  # 20% more damping for 15-minute data
    else
        1.0  # Standard damping for lower frequencies
    end
    
    μ = base_damping * frequency_damping_factor
    
    # Initialize fields
    u_current, u_prev = build_adaptive_initial_conditions(amplitudes, x, y, z, L, dt, c, freq)
    
    # Enhanced feature tracking during evolution
    collision_features = Vector{NamedTuple}()
    collision_time = T * collision_threshold
    collision_step = Int(ceil(collision_time / dt))
    
    # Time integration with feature extraction
    for step in 1:n_steps
        # Compute next time step
        u_next = finite_difference_step(u_current, u_prev, dx^2, (c * dt)^2, λ, μ, dt)
        
        # Extract features at collision time for frequency-specific analysis
        if step == collision_step
            collision_snapshot = extract_collision_features(u_current, x, y, z, grid, L)
            push!(collision_features, (time=step*dt, features=collision_snapshot))
        end
        
        # Update for next iteration
        u_prev = u_current
        u_current = u_next
    end
    
    # Extract final features
    final_features = extract_adaptive_collision_features(u_current, x, y, z, grid, L, freq)
    
    # Combine collision-time and final features
    enhanced_features = if !isempty(collision_features)
        collision_snapshot = collision_features[1].features
        merge(final_features, (
            collision_H = collision_snapshot.H,
            collision_energy = collision_snapshot.energy,
            collision_asymmetry_x = collision_snapshot.asymmetry_x,
            collision_asymmetry_y = collision_snapshot.asymmetry_y
        ))
    else
        final_features
    end
    
    return (
        H = enhanced_features.H,
        F = enhanced_features.F,
        energy = enhanced_features.energy,
        asymmetry_x = enhanced_features.asymmetry_x,
        asymmetry_y = enhanced_features.asymmetry_y,
        concentration = enhanced_features.concentration,
        
        # Enhanced features for frequency adaptation
        collision_H = get(enhanced_features, :collision_H, enhanced_features.H),
        collision_energy = get(enhanced_features, :collision_energy, enhanced_features.energy),
        collision_asymmetry_x = get(enhanced_features, :collision_asymmetry_x, enhanced_features.asymmetry_x),
        collision_asymmetry_y = get(enhanced_features, :collision_asymmetry_y, enhanced_features.asymmetry_y),
        
        # Frequency-specific features
        frequency_signature = compute_frequency_signature(enhanced_features, freq),
        temporal_sharpness = enhanced_features.concentration * (1.0 + 1.0/T),  # Higher for shorter timeframes
        
        metadata = (
            frequency = freq,
            grid = grid,
            L = L,
            λ = λ,
            μ = μ,
            T = T,
            dt = dt,
            n_steps = n_steps,
            amplitudes = amplitudes,
            vix = vix,
            dx = dx,
            stability_ratio = dt / dt_max,
            collision_threshold = collision_threshold,
            frequency_damping_factor = frequency_damping_factor
        )
    )
end

"""
Build frequency-adaptive initial conditions
"""
function build_adaptive_initial_conditions(amplitudes::NTuple{4,Float64}, x, y, z, L::Float64, 
                                          dt::Float64, c::Float64, freq::DataFrequency)
    grid = length(x)
    u_current = zeros(grid, grid, grid)
    u_prev = zeros(grid, grid, grid)
    
    # Frequency-adaptive pulse parameters
    σ_factor = if freq == FreqMin1
        12  # Very narrow pulses for 1-minute (focused collision)
    elseif freq == FreqMin5  
        10  # Narrow pulses for 5-minute
    elseif freq in [FreqMin15, FreqMin30]
        8   # Moderate width for 15-30 minute
    else
        6   # Standard width for hourly/daily
    end
    
    σ = L / σ_factor  # Pulse width
    
    # Frequency-adaptive velocity (faster for higher frequencies)
    v_base = 0.8
    v_factor = if freq in [FreqMin1, FreqMin5]
        1.5  # Faster collision for high-frequency
    elseif freq in [FreqMin15, FreqMin30]
        1.2  # Moderately faster
    else
        1.0  # Standard velocity
    end
    
    v = v_base * v_factor
    
    # Sech pulse positions (adaptive face offset)
    face_offset = L * 0.7
    positions = [
        (face_offset, 0.0, 0.0),
        (-face_offset, 0.0, 0.0),
        (0.0, face_offset, 0.0),
        (0.0, -face_offset, 0.0)
    ]
    
    # Velocity directions (toward origin)
    velocities = [
        (-v, 0.0, 0.0),
        (v, 0.0, 0.0),
        (0.0, -v, 0.0),
        (0.0, v, 0.0)
    ]
    
    for (i, ((x0, y0, z0), (vx, vy, vz), amp)) in enumerate(zip(positions, velocities, amplitudes))
        for (ix, xval) in enumerate(x), (iy, yval) in enumerate(y), (iz, zval) in enumerate(z)
            r = sqrt((xval - x0)^2 + (yval - y0)^2 + (zval - z0)^2)
            
            # Sech profile (soliton shape)
            sech_val = 1.0 / cosh(r / σ)
            pulse_amplitude = amp * sech_val
            
            u_current[ix, iy, iz] += pulse_amplitude
            
            # Initial velocity for traveling wave
            if r < 3*σ
                grad_mag = sech_val * tanh(r / σ) / σ
                velocity_mag = sqrt(vx^2 + vy^2 + vz^2)
                u_dot_0 = -velocity_mag * grad_mag * amp
                
                u_prev[ix, iy, iz] += pulse_amplitude - dt * u_dot_0
            else
                u_prev[ix, iy, iz] += pulse_amplitude
            end
        end
    end
    
    return u_current, u_prev
end

"""
Extract frequency-adaptive collision features
"""
function extract_adaptive_collision_features(u_final::Array{Float64,3}, x, y, z, grid::Int, L::Float64, freq::DataFrequency)
    
    # Standard features
    center = (grid + 1) ÷ 2
    H = u_final[center, center, center]
    
    # Adaptive probe selection based on frequency
    if freq in [FreqMin1, FreqMin5]
        # High-frequency: focus on central region
        probe_indices = [center-1, center, center+1]
    else
        # Lower frequency: standard wide probes  
        probe_indices = [grid÷4, center, 3*grid÷4]
    end
    
    F_x = [u_final[idx, center, center] for idx in probe_indices if 1 <= idx <= grid]
    F_y = [u_final[center, idx, center] for idx in probe_indices if 1 <= idx <= grid]
    F_z = [u_final[center, center, idx] for idx in probe_indices if 1 <= idx <= grid]
    F = vcat(F_x, F_y, F_z)
    
    # Standard energy
    energy = sum(abs2, u_final) / length(u_final)
    
    # Frequency-specific asymmetry measures
    if grid >= 5
        field_asymmetry_x = mean(u_final[center+1:end, center, center]) - mean(u_final[1:center-1, center, center])
        field_asymmetry_y = mean(u_final[center, center+1:end, center]) - mean(u_final[center, 1:center-1, center])
    else
        field_asymmetry_x = 0.0
        field_asymmetry_y = 0.0
    end
    
    # Adaptive concentration measure
    if grid >= 3
        central_region = u_final[max(1,center-1):min(grid,center+1), 
                                max(1,center-1):min(grid,center+1), 
                                max(1,center-1):min(grid,center+1)]
        field_concentration = sum(abs2, central_region) / sum(abs2, u_final)
    else
        field_concentration = 1.0
    end
    
    return (
        H = H,
        F = F,
        energy = energy,
        asymmetry_x = field_asymmetry_x,
        asymmetry_y = field_asymmetry_y,
        concentration = field_concentration
    )
end

"""
Compute frequency-specific signature combining multiple features
"""
function compute_frequency_signature(features, freq::DataFrequency)::Float64
    # Weight features differently based on frequency
    if freq in [FreqMin1, FreqMin5]
        # High-frequency: emphasize concentration and energy
        return 0.4 * abs(features.H) + 0.4 * features.concentration + 0.2 * features.energy
    elseif freq in [FreqMin15, FreqMin30]
        # Medium-frequency: balanced combination
        return 0.3 * abs(features.H) + 0.3 * features.concentration + 
               0.2 * features.energy + 0.2 * (abs(features.asymmetry_x) + abs(features.asymmetry_y))
    else
        # Low-frequency: emphasize asymmetry and complex features
        return 0.2 * abs(features.H) + 0.2 * features.concentration + 
               0.15 * features.energy + 0.45 * (abs(features.asymmetry_x) + abs(features.asymmetry_y))
    end
end

# Include finite difference and extraction functions from SolitonPDE_simple
"""
Finite difference time step: u_tt - c²∇²u + λu³ + μu_t = 0
"""
function finite_difference_step(u_current::Array{Float64,3}, u_prev::Array{Float64,3}, 
                               dx2::Float64, c2_dt2::Float64, λ::Float64, μ::Float64, dt::Float64)
    
    grid = size(u_current, 1)
    u_next = zeros(grid, grid, grid)
    
    for i in 2:(grid-1), j in 2:(grid-1), k in 2:(grid-1)
        u_ijk = u_current[i, j, k]
        
        # 3D Laplacian using central differences
        laplacian = (u_current[i+1, j, k] + u_current[i-1, j, k] +
                    u_current[i, j+1, k] + u_current[i, j-1, k] +
                    u_current[i, j, k+1] + u_current[i, j, k-1] - 6*u_ijk) / dx2
        
        # Time derivative
        u_t = (u_current[i, j, k] - u_prev[i, j, k]) / dt
        
        # PDE: u_tt = c²∇²u - λu³ - μu_t
        acceleration = c2_dt2 * laplacian - λ * dt^2 * u_ijk^3 - μ * dt * u_t
        
        # Explicit time step
        u_next[i, j, k] = 2*u_current[i, j, k] - u_prev[i, j, k] + acceleration
    end
    
    return u_next
end

"""
Standard collision feature extraction (from SolitonPDE_simple)
"""
function extract_collision_features(u_final::Array{Float64,3}, x, y, z, grid::Int, L::Float64)
    center = (grid + 1) ÷ 2
    H = u_final[center, center, center]
    
    probe_indices = [max(1, grid÷4), center, min(grid, 3*grid÷4)]
    F_x = [u_final[idx, center, center] for idx in probe_indices]
    F_y = [u_final[center, idx, center] for idx in probe_indices] 
    F_z = [u_final[center, center, idx] for idx in probe_indices]
    F = vcat(F_x, F_y, F_z)
    
    energy = sum(abs2, u_final) / length(u_final)
    
    if grid >= 3
        field_asymmetry_x = mean(u_final[center+1:end, center, center]) - mean(u_final[1:center-1, center, center])
        field_asymmetry_y = mean(u_final[center, center+1:end, center]) - mean(u_final[center, 1:center-1, center])
        central_region = u_final[center-1:center+1, center-1:center+1, center-1:center+1]
        field_concentration = sum(abs2, central_region) / sum(abs2, u_final)
    else
        field_asymmetry_x = 0.0
        field_asymmetry_y = 0.0
        field_concentration = 1.0
    end
    
    return (
        H = H,
        F = F,
        energy = energy,
        asymmetry_x = field_asymmetry_x,
        asymmetry_y = field_asymmetry_y,
        concentration = field_concentration
    )
end

end # module SolitonPDE_adaptive 