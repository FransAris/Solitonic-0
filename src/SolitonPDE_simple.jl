"""
SolitonPDE_Simple Module

Implements the damped φ⁴/Klein-Gordon PDE solver using finite differences
Avoids DifferentialEquations.jl dependency issues

The PDE is: u_tt - c²∇²u + λu³ + μ(t)u_t = 0
where μ(t) = μmax * (vix / vix_max_sample) provides VIX-dependent damping.
"""
module SolitonPDE_simple

export simulate_soliton

using LinearAlgebra, Statistics

"""
    simulate_soliton(amplitudes::NTuple{4,Float64}, vix::Float64; 
                    grid=32, L=1.0, λ=1.0, μmax=0.1, T=1.0, dt=1e-3, 
                    vix_max_sample=50.0, c=1.0) -> NamedTuple

Simulate soliton collision in damped φ⁴/Klein-Gordon PDE using finite differences.

# Arguments  
- `amplitudes`: Four soliton launch amplitudes (from normalized oscillators)
- `vix`: VIX value for time-dependent damping
- `grid`: Grid points per dimension (total points = grid³)
- `L`: Half-width of cubic domain [-L,L]³  
- `λ`: Nonlinearity parameter in φ⁴ term
- `μmax`: Maximum damping coefficient
- `T`: Integration time
- `dt`: Time step
- `vix_max_sample`: VIX normalization factor
- `c`: Wave speed parameter

# Returns
NamedTuple with:
- `H`: Field value at origin u(0,0,0,T)
- `F`: Field values at probe points
- `energy`: Total energy at final time
- `asymmetry_x`: Directional bias in aftermath (x-axis)
- `asymmetry_y`: Directional bias in aftermath (y-axis)
- `concentration`: Field concentration in aftermath
- `metadata`: Simulation parameters

# Example
```julia
amplitudes = (0.5, -0.3, 0.8, -0.2)  # From normalized oscillators
result = simulate_soliton(amplitudes, 25.0, grid=32, T=0.5)
```
"""
function simulate_soliton(amplitudes::NTuple{4,Float64}, vix::Float64; 
                         grid::Int=32, L::Float64=1.0, λ::Float64=1.0, 
                         μmax::Float64=0.1, T::Float64=1.0, dt::Float64=1e-3,
                         vix_max_sample::Float64=50.0, c::Float64=1.0)::NamedTuple
    
    # Removed verbose logging for performance
    # println("Starting soliton simulation (finite differences)...")
    # println("   Grid: $(grid)³, Domain: [-$L,$L]³, T=$T, dt=$dt")
    # println("   Amplitudes: $amplitudes, VIX: $vix")
    
    # Validate inputs
    @assert grid > 2 "Grid size must be > 2"
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
    c_eff = c
    dt_max = 0.5 * dx / c_eff  # CFL condition
    # println("   Spatial resolution: dx = $dx")
    
    if dt > dt_max
        dt = 0.8 * dt_max
        n_steps = Int(ceil(T / dt))
    end
    
    # VIX-dependent damping (normalized)
    μ = μmax * (vix / vix_max_sample)
    # println("   Damping coefficient: μ = $μ")
    
    # Initialize fields
    u = zeros(grid, grid, grid)
    u_prev = zeros(grid, grid, grid)
    
    # println("   Time steps: $n_steps")
    
    # Build initial conditions with four sech pulses
    u_current, u_prev = build_initial_conditions(amplitudes, x, y, z, L, dt, c)
    
    # Time integration
    for step in 1:n_steps
        # Compute next time step using explicit finite differences
        u_next = finite_difference_step(u_current, u_prev, dx^2, (c * dt)^2, λ, μ, dt)
        
        # Update for next iteration
        u_prev = u_current
        u_current = u_next
        
        # Progress indicator
        if step % (n_steps ÷ 10) == 0
            progress = 100 * step / n_steps
            energy = compute_energy(u_current, u_prev, dx, dt, c, λ)
            # println("   Progress: $(round(progress, digits=1))%, Energy: $(round(energy, digits=6))")
        end
    end
    
    # Extract features from final state
    features = extract_collision_features(u_current, x, y, z, grid, L)
    
    # println("   Simulation complete")
    
    return (
        H = features.H,
        F = features.F, 
        energy = features.energy,
        asymmetry_x = features.asymmetry_x,
        asymmetry_y = features.asymmetry_y,
        concentration = features.concentration,
        metadata = (
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
            stability_ratio = dt / dt_max
        )
    )
end

"""
Build initial conditions: four sech pulses aimed at origin
"""
function build_initial_conditions(amplitudes::NTuple{4,Float64}, x, y, z, L::Float64, dt::Float64, c::Float64)
    grid = length(x)
    u_current = zeros(grid, grid, grid)
    u_prev = zeros(grid, grid, grid)
    
    # println("   Building initial sech pulses...")
    
    # Pulse parameters
    σ = L / 6  # Pulse width (narrower for better solitons)
    v = 0.8    # Initial velocity toward center
    
    # Sech pulse positions on domain faces
    face_offset = L * 0.7  # Position on faces
    positions = [
        (face_offset, 0.0, 0.0),    # +x face → center
        (-face_offset, 0.0, 0.0),   # -x face → center  
        (0.0, face_offset, 0.0),    # +y face → center
        (0.0, -face_offset, 0.0)    # -y face → center
    ]
    
    # Velocity directions (toward origin)
    velocities = [
        (-v, 0.0, 0.0),   # +x pulse moves -x
        (v, 0.0, 0.0),    # -x pulse moves +x
        (0.0, -v, 0.0),   # +y pulse moves -y  
        (0.0, v, 0.0)     # -y pulse moves +y
    ]
    
    for (i, ((x0, y0, z0), (vx, vy, vz), amp)) in enumerate(zip(positions, velocities, amplitudes))
        # println("      Pulse $i: pos=($x0,$y0,$z0), vel=($vx,$vy,$vz), amp=$amp")
        
        # Build sech pulse: amp * sech((r-r0)/σ)
        for (ix, xval) in enumerate(x), (iy, yval) in enumerate(y), (iz, zval) in enumerate(z)
            # Distance from pulse center
            r = sqrt((xval - x0)^2 + (yval - y0)^2 + (zval - z0)^2)
            
            # Sech profile (soliton shape)
            sech_val = 1.0 / cosh(r / σ)
            pulse_amplitude = amp * sech_val
            
            # Add to current field
            u_current[ix, iy, iz] += pulse_amplitude
            
            # Initial velocity: u_prev = u_current - dt * u_dot(t=0)
            # For traveling wave: u_dot ≈ -v · ∇u
            if r < 3*σ  # Only near pulse center
                # Simple approximation of -v · ∇u
                grad_mag = sech_val * tanh(r / σ) / σ  # |∇sech|
                velocity_mag = sqrt(vx^2 + vy^2 + vz^2)
                u_dot_0 = -velocity_mag * grad_mag * amp
                
                u_prev[ix, iy, iz] += pulse_amplitude - dt * u_dot_0
            else
                u_prev[ix, iy, iz] += pulse_amplitude
            end
        end
    end
    
    # println("   Initial conditions: 4 sech pulses positioned")
    
    return u_current, u_prev
end

"""
Finite difference time step: u_tt - c²∇²u + λu³ + μu_t = 0
Using explicit central differences
"""
function finite_difference_step(u_current::Array{Float64,3}, u_prev::Array{Float64,3}, 
                               dx2::Float64, c2_dt2::Float64, λ::Float64, μ::Float64, dt::Float64)
    
    grid = size(u_current, 1)
    u_next = zeros(grid, grid, grid)
    
    # Interior points (avoid boundaries for simplicity)
    for i in 2:(grid-1), j in 2:(grid-1), k in 2:(grid-1)
        # Current field value
        u_ijk = u_current[i, j, k]
        
        # 3D Laplacian using central differences
        laplacian = (u_current[i+1, j, k] + u_current[i-1, j, k] +
                    u_current[i, j+1, k] + u_current[i, j-1, k] +
                    u_current[i, j, k+1] + u_current[i, j, k-1] - 6*u_ijk) / dx2
        
        # Time derivative (backward difference)
        u_t = (u_current[i, j, k] - u_prev[i, j, k]) / dt
        
        # PDE: u_tt = c²∇²u - λu³ - μu_t
        acceleration = c2_dt2 * laplacian - λ * dt^2 * u_ijk^3 - μ * dt * u_t
        
        # Explicit time step: u_next = 2*u_current - u_prev + dt²*acceleration
        u_next[i, j, k] = 2*u_current[i, j, k] - u_prev[i, j, k] + acceleration
    end
    
    # Boundary conditions: zero (absorbing boundaries)
    # u_next[1, :, :] .= 0, etc. (already zero from initialization)
    
    return u_next
end

"""
Compute total energy: kinetic + potential + interaction
"""
function compute_energy(u_current::Array{Float64,3}, u_prev::Array{Float64,3}, 
                       dx::Float64, dt::Float64, c::Float64, λ::Float64)
    
    grid = size(u_current, 1)
    total_energy = 0.0
    dx3 = dx^3  # Volume element
    
    for i in 2:(grid-1), j in 2:(grid-1), k in 2:(grid-1)
        u = u_current[i, j, k]
        
        # Kinetic energy: (1/2) * u_t²
        u_t = (u_current[i, j, k] - u_prev[i, j, k]) / dt
        kinetic = 0.5 * u_t^2
        
        # Gradient energy: (c²/2) * |∇u|²
        grad_x = (u_current[i+1, j, k] - u_current[i-1, j, k]) / (2*dx)
        grad_y = (u_current[i, j+1, k] - u_current[i, j-1, k]) / (2*dx)
        grad_z = (u_current[i, j, k+1] - u_current[i, j, k-1]) / (2*dx)
        gradient = 0.5 * c^2 * (grad_x^2 + grad_y^2 + grad_z^2)
        
        # Interaction energy: (λ/4) * u⁴
        interaction = 0.25 * λ * u^4
        
        total_energy += (kinetic + gradient + interaction) * dx3
    end
    
    return total_energy
end

"""
Extract collision features from final field state AND collision evolution
"""
function extract_collision_features(u_final::Array{Float64,3}, x, y, z, grid::Int, L::Float64)
    
    # Center grid point
    center = (grid + 1) ÷ 2
    
    # H: Field value at origin u(0,0,0,T) - post-collision amplitude
    H = u_final[center, center, center]
    
    # F: Field values at probe points along axes - post-collision field pattern
    probe_indices = [grid÷4, center, 3*grid÷4]
    F_x = [u_final[idx, center, center] for idx in probe_indices]
    F_y = [u_final[center, idx, center] for idx in probe_indices] 
    F_z = [u_final[center, center, idx] for idx in probe_indices]
    F = vcat(F_x, F_y, F_z)  # 9 probe values
    
    # Post-collision energy (residual after damping)
    energy = sum(abs2, u_final) / length(u_final)
    
    # Additional post-collision features
    max_field = maximum(abs, u_final)
    field_variance = var(vec(u_final))
    
    # Collision asymmetry - measure directional bias in aftermath
    field_asymmetry_x = mean(u_final[center+1:end, center, center]) - mean(u_final[1:center-1, center, center])
    field_asymmetry_y = mean(u_final[center, center+1:end, center]) - mean(u_final[center, 1:center-1, center])
    
    # Field concentration - how localized is the post-collision state
    central_region = u_final[center-1:center+1, center-1:center+1, center-1:center+1]
    field_concentration = sum(abs2, central_region) / sum(abs2, u_final)
    
    # println("   Post-collision features: H=$H, max_field=$max_field, concentration=$(round(field_concentration, digits=3))")
    
    return (
        H = H, 
        F = F, 
        energy = energy,
        max_field = max_field,
        field_variance = field_variance,
        asymmetry_x = field_asymmetry_x,
        asymmetry_y = field_asymmetry_y,
        concentration = field_concentration
    )
end

end # module SolitonPDE_simple 