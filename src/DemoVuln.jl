module DemoVuln

using LinearAlgebra
using Statistics
using DataFrames

export MatrixPopulationModel,
       PerturbationGrid,
       SimulationResult,
       GridResult,
       dominant_eigenvalue,
       stable_stage_distribution,
       build_target_mask,
       apply_perturbation,
       apply_perturbation!,
       population_reduction,
       simulate_dynamics,
       perturbation_grid_from_frequencies,
       scenarios,
       run_grid,
       compute_vulnerability

const Target = Union{Symbol, AbstractString}

"""
    dominant_eigenvalue(A)

Return the dominant eigenvalue of a projection matrix, using the largest real
part among the eigenvalues.
"""
function dominant_eigenvalue(A::AbstractMatrix{<:Real})::Float64
    vals = eigvals(Matrix{Float64}(A))
    return maximum(real.(vals))
end

"""
    stable_stage_distribution(A)

Return the stable stage distribution associated with the dominant right
eigenvector of the projection matrix `A`.
"""
function stable_stage_distribution(A::AbstractMatrix{<:Real})::Vector{Float64}
    F = eigen(Matrix{Float64}(A))
    idx = argmax(real.(F.values))
    w = abs.(real.(F.vectors[:, idx]))
    s = sum(w)
    s > 0 || error("The dominant eigenvector has zero norm.")
    return w ./ s
end

"""
    MatrixPopulationModel(A; fecundity_mask=nothing, fecundity_rows=[1],
                          adult_stages=nothing, juvenile_stages=nothing,
                          name=nothing)

Matrix population model and associated demographic targets.

Columns are source stages at time `t`; rows are destination stages at time
`t + 1`. Stage indices follow Julia's one-based indexing.

# Arguments
- `A`: square non-negative projection matrix.
- `fecundity_mask`: Boolean matrix identifying fecundity entries. If omitted,
  non-zero entries in `fecundity_rows` are treated as fecundity entries.
- `fecundity_rows`: rows interpreted as reproductive-output rows. The default
  is the first row.
- `adult_stages`: source-stage columns interpreted as adult or reproductive
  stages. If omitted, adult stages are inferred as columns with at least one
  fecundity entry.
- `juvenile_stages`: source-stage columns interpreted as juvenile or
  pre-reproductive stages. If omitted, juvenile stages are inferred as the
  remaining columns.
- `name`: optional model or species label.
"""
struct MatrixPopulationModel
    A::Matrix{Float64}
    fecundity_mask::BitMatrix
    adult_stages::Vector{Int}
    juvenile_stages::Vector{Int}
    name::Union{Nothing,String}
    lambda::Float64
end

function MatrixPopulationModel(A::AbstractMatrix{<:Real};
    fecundity_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    fecundity_rows::AbstractVector{<:Integer}=[1],
    adult_stages::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    juvenile_stages::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    name::Union{Nothing,String}=nothing)

    nrow, ncol = size(A)
    nrow == ncol || throw(ArgumentError("A must be a square projection matrix."))
    any(A .< 0) && throw(ArgumentError("A must contain non-negative entries."))

    M = Matrix{Float64}(A)
    n = size(M, 1)

    fmask = if fecundity_mask === nothing
        mask = falses(n, n)
        for r in fecundity_rows
            1 <= r <= n || throw(ArgumentError("fecundity_rows contains an invalid row index."))
            @views mask[r, :] .= M[r, :] .!= 0
        end
        mask
    else
        size(fecundity_mask) == size(M) || throw(ArgumentError("fecundity_mask must have the same size as A."))
        BitMatrix(fecundity_mask)
    end

    adult = if adult_stages === nothing
        [j for j in 1:n if any(@view fmask[:, j])]
    else
        collect(Int, adult_stages)
    end

    juvenile = if juvenile_stages === nothing
        [j for j in 1:n if !(j in adult)]
    else
        collect(Int, juvenile_stages)
    end

    for j in adult
        1 <= j <= n || throw(ArgumentError("adult_stages contains an invalid stage index."))
    end
    for j in juvenile
        1 <= j <= n || throw(ArgumentError("juvenile_stages contains an invalid stage index."))
    end

    λ = dominant_eigenvalue(M)
    isfinite(λ) && λ > 0 || throw(ArgumentError("The dominant eigenvalue of A must be positive."))

    return MatrixPopulationModel(M, fmask, adult, juvenile, name, λ)
end

_normalize_target(target::Target)::Symbol = Symbol(target)

@inline function _active_step(step::Int, start::Int, duration::Int, period::Int)::Bool
    duration <= 0 && return false
    step < start && return false
    return mod(step - start, period) < duration
end

"""
    build_target_mask(model, target; survival_affects_fecundity=true, custom_mask=nothing)

Return a Boolean matrix selecting projection-matrix entries affected by a
perturbation.
"""
function build_target_mask(model::MatrixPopulationModel, target::Target;
    survival_affects_fecundity::Bool=true,
    custom_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing)::BitMatrix

    t = _normalize_target(target)
    n = size(model.A, 1)
    mask = falses(n, n)

    if t === :adult_survival
        for j in model.adult_stages
            @views mask[:, j] .= true
        end
        survival_affects_fecundity || (mask .&= .!model.fecundity_mask)
    elseif t === :juvenile_survival
        for j in model.juvenile_stages
            @views mask[:, j] .= true
        end
        survival_affects_fecundity || (mask .&= .!model.fecundity_mask)
    elseif t === :fecundity
        mask .= model.fecundity_mask
    elseif t === :all
        mask .= true
    elseif t === :custom
        custom_mask === nothing && throw(ArgumentError("custom_mask is required when target=:custom."))
        size(custom_mask) == size(model.A) || throw(ArgumentError("custom_mask must have the same size as A."))
        mask .= custom_mask
    else
        throw(ArgumentError("Unknown perturbation target: $target"))
    end

    return mask
end

"""
    apply_perturbation(model, target, magnitude; kwargs...)

Return a perturbed copy of the projection matrix. A magnitude `m` applies
`x -> (1 - m) * x` to targeted entries. For `target=:all`, fecundity entries
are multiplied by `(1 - m)^2` when `survival_affects_fecundity=true`.
"""
function apply_perturbation(model::MatrixPopulationModel, target::Target, magnitude::Real;
    survival_affects_fecundity::Bool=true,
    custom_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing)::Matrix{Float64}

    B = copy(model.A)
    apply_perturbation!(B, model, target, magnitude;
        survival_affects_fecundity=survival_affects_fecundity,
        custom_mask=custom_mask)
    return B
end

"""
    apply_perturbation!(B, model, target, magnitude; kwargs...)

In-place version of [`apply_perturbation`](@ref). The matrix `B` is modified and
returned. This function is useful in performance-sensitive workflows.
"""
function apply_perturbation!(B::AbstractMatrix{<:Real}, model::MatrixPopulationModel,
    target::Target, magnitude::Real;
    survival_affects_fecundity::Bool=true,
    custom_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing)

    0 <= magnitude <= 1 || throw(ArgumentError("magnitude must lie in [0, 1]."))
    size(B) == size(model.A) || throw(ArgumentError("B must have the same size as model.A."))

    t = _normalize_target(target)
    factor = 1.0 - Float64(magnitude)

    if t === :all && survival_affects_fecundity
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            if model.fecundity_mask[i, j]
                B[i, j] *= factor^2
            else
                B[i, j] *= factor
            end
        end
        return B
    end

    mask = build_target_mask(model, t;
        survival_affects_fecundity=survival_affects_fecundity,
        custom_mask=custom_mask)

    @inbounds for j in axes(B, 2), i in axes(B, 1)
        if mask[i, j]
            B[i, j] *= factor
        end
    end

    return B
end

"""
    population_reduction(final_population, baseline_final_population)

Return the percent reduction in final population size relative to the
unperturbed baseline trajectory.
"""
function population_reduction(final_population::Real, baseline_final_population::Real)::Float64
    baseline_final_population > 0 || throw(ArgumentError("baseline_final_population must be positive."))
    return 100.0 * (1.0 - Float64(final_population) / Float64(baseline_final_population))
end

"""
Output of a single perturbation simulation.
"""
struct SimulationResult
    abundance::Vector{Float64}
    baseline_abundance::Vector{Float64}
    stage_vectors::Union{Nothing,Matrix{Float64}}
    baseline_stage_vectors::Union{Nothing,Matrix{Float64}}
    reduction::Float64
    final_population::Float64
    baseline_final_population::Float64
    magnitude::Float64
    duration::Int
    period::Int
    target::Symbol
end

function _as_model(model_or_A)::MatrixPopulationModel
    model_or_A isa MatrixPopulationModel && return model_or_A
    return MatrixPopulationModel(model_or_A)
end

function _normalised_matrices(model::MatrixPopulationModel, Bpert::AbstractMatrix{<:Real}, normalize_by_lambda::Bool)
    if normalize_by_lambda
        return model.A ./ model.lambda, Matrix{Float64}(Bpert) ./ model.lambda
    else
        return model.A, Matrix{Float64}(Bpert)
    end
end

"""
    simulate_dynamics(model; target=:adult_survival, magnitude, duration, period,
                      t_max, recovery_steps=0, start=1, kwargs...)

Simulate population dynamics under a temporally structured perturbation.

`start` is one-based: `start=1` means that a perturbation may affect the first
projection step. Population reduction is computed relative to the unperturbed
baseline at the same final time.
"""
function simulate_dynamics(model_or_A;
    target::Target=:adult_survival,
    magnitude::Real,
    duration::Integer,
    period::Integer,
    t_max::Integer,
    recovery_steps::Integer=0,
    start::Integer=1,
    initial_state::Union{Nothing,AbstractVector{<:Real}}=nothing,
    normalize_by_lambda::Bool=true,
    survival_affects_fecundity::Bool=true,
    custom_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    return_stage_vectors::Bool=false,
    force_during_recovery::Bool=false)::SimulationResult

    model = _as_model(model_or_A)
    duration >= 0 || throw(ArgumentError("duration must be non-negative."))
    period >= 1 || throw(ArgumentError("period must be positive."))
    duration <= period || throw(ArgumentError("duration cannot exceed period."))
    t_max >= 0 || throw(ArgumentError("t_max must be non-negative."))
    recovery_steps >= 0 || throw(ArgumentError("recovery_steps must be non-negative."))
    start >= 1 || throw(ArgumentError("start must be at least 1 in Julia's one-based indexing."))

    nstages = size(model.A, 1)
    total_steps = Int(t_max + recovery_steps)

    Bpert = apply_perturbation(model, target, magnitude;
        survival_affects_fecundity=survival_affects_fecundity,
        custom_mask=custom_mask)
    A0, A1 = _normalised_matrices(model, Bpert, normalize_by_lambda)

    n = if initial_state === nothing
        stable_stage_distribution(model.A)
    else
        length(initial_state) == nstages || throw(ArgumentError("initial_state has incompatible length."))
        x = Vector{Float64}(initial_state)
        sx = sum(x)
        sx > 0 || throw(ArgumentError("initial_state must have positive total abundance."))
        x ./ sx
    end

    nb = copy(n)
    abundance = Vector{Float64}(undef, total_steps + 1)
    baseline_abundance = Vector{Float64}(undef, total_steps + 1)
    abundance[1] = sum(n)
    baseline_abundance[1] = sum(nb)

    stage_vectors = return_stage_vectors ? Matrix{Float64}(undef, total_steps + 1, nstages) : nothing
    baseline_stage_vectors = return_stage_vectors ? Matrix{Float64}(undef, total_steps + 1, nstages) : nothing
    if return_stage_vectors
        @views stage_vectors[1, :] .= n
        @views baseline_stage_vectors[1, :] .= nb
    end

    tmp = similar(n)
    tmpb = similar(nb)

    @inbounds for step in 1:total_steps
        forcing_window = force_during_recovery || step <= t_max
        active = forcing_window && _active_step(step, Int(start), Int(duration), Int(period))
        At = active ? A1 : A0

        mul!(tmp, At, n)
        n, tmp = tmp, n

        mul!(tmpb, A0, nb)
        nb, tmpb = tmpb, nb

        abundance[step + 1] = sum(n)
        baseline_abundance[step + 1] = sum(nb)
        if return_stage_vectors
            @views stage_vectors[step + 1, :] .= n
            @views baseline_stage_vectors[step + 1, :] .= nb
        end
    end

    red = population_reduction(abundance[end], baseline_abundance[end])
    return SimulationResult(abundance, baseline_abundance, stage_vectors, baseline_stage_vectors,
        red, abundance[end], baseline_abundance[end], Float64(magnitude), Int(duration), Int(period),
        _normalize_target(target))
end

"""
    PerturbationGrid(magnitudes, durations, periods)

Grid of perturbation regimes defining the perturbation space.
"""
struct PerturbationGrid
    magnitudes::Vector{Float64}
    durations::Vector{Int}
    periods::Vector{Int}
end

function PerturbationGrid(magnitudes, durations, periods)
    mags = collect(Float64, magnitudes)
    durs = collect(Int, durations)
    pers = collect(Int, periods)
    all(0 .<= mags .<= 1) || throw(ArgumentError("All magnitudes must lie in [0, 1]."))
    all(durs .>= 0) || throw(ArgumentError("All durations must be non-negative."))
    all(pers .>= 1) || throw(ArgumentError("All periods must be positive."))
    return PerturbationGrid(mags, durs, pers)
end

"""
    perturbation_grid_from_frequencies(magnitudes, durations, frequencies;
                                       generation_time=1.0, rounding=:nearest)

Create a `PerturbationGrid` by converting event frequencies to integer periods.
Frequencies are interpreted as events per `generation_time` projection steps.
"""
function perturbation_grid_from_frequencies(magnitudes, durations, frequencies;
    generation_time::Real=1.0,
    rounding::Symbol=:nearest)::PerturbationGrid

    generation_time > 0 || throw(ArgumentError("generation_time must be positive."))
    periods = Int[]
    for f in frequencies
        f > 0 || throw(ArgumentError("frequencies must be positive."))
        p = Float64(generation_time) / Float64(f)
        pint = if rounding === :nearest
            round(Int, p)
        elseif rounding === :floor
            floor(Int, p)
        elseif rounding === :ceil
            ceil(Int, p)
        else
            throw(ArgumentError("rounding must be :nearest, :floor or :ceil."))
        end
        push!(periods, max(pint, 1))
    end
    return PerturbationGrid(magnitudes, durations, sort(unique(periods)))
end

"""
    scenarios(grid; skip_infeasible=true)

Return perturbation scenarios as named tuples.
"""
function scenarios(grid::PerturbationGrid; skip_infeasible::Bool=true)
    out = NamedTuple[]
    for m in grid.magnitudes, d in grid.durations, p in grid.periods
        feasible = d <= p
        if feasible || !skip_infeasible
            push!(out, (magnitude=m, duration=d, period=p, feasible=feasible))
        end
    end
    return out
end

function _simulate_reduction_fast(model::MatrixPopulationModel, A0::Matrix{Float64}, A1::Matrix{Float64},
    init::Vector{Float64}, duration::Int, period::Int, t_max::Int, recovery_steps::Int,
    start::Int, force_during_recovery::Bool)::Tuple{Float64,Float64,Float64}

    total_steps = t_max + recovery_steps
    n = copy(init)
    nb = copy(init)
    tmp = similar(init)
    tmpb = similar(init)

    @inbounds for step in 1:total_steps
        forcing_window = force_during_recovery || step <= t_max
        active = forcing_window && _active_step(step, start, duration, period)
        At = active ? A1 : A0
        mul!(tmp, At, n); n, tmp = tmp, n
        mul!(tmpb, A0, nb); nb, tmpb = tmpb, nb
    end

    final_population = sum(n)
    baseline_final_population = sum(nb)
    red = population_reduction(final_population, baseline_final_population)
    return red, final_population, baseline_final_population
end

"""
Output of a perturbation-grid simulation.
"""
struct GridResult
    table::DataFrame
    trajectories::Dict{Tuple{Float64,Int,Int},SimulationResult}
    vulnerability::Float64
end

"""
    compute_vulnerability(table; column=:population_reduction)

Compute integrated vulnerability as the mean population reduction across the
simulated feasible perturbation regimes.
"""
function compute_vulnerability(table::DataFrame; column::Symbol=:population_reduction)::Float64
    vals = skipmissing(table[!, column])
    finite_vals = [Float64(x) for x in vals if isfinite(Float64(x))]
    isempty(finite_vals) && return NaN
    return mean(finite_vals)
end

compute_vulnerability(values::AbstractVector{<:Real}) = mean(filter(isfinite, Float64.(values)))

"""
    run_grid(model; target, grid, t_max, recovery_steps=0, kwargs...)

Simulate a full perturbation grid and compute integrated vulnerability.

When `return_trajectories=false`, the function uses a faster internal loop that
stores only final reductions. Set `return_trajectories=true` to retain complete
trajectories for each feasible scenario.
"""
function run_grid(model_or_A;
    target::Target,
    grid::PerturbationGrid,
    t_max::Integer,
    recovery_steps::Integer=0,
    start::Integer=1,
    initial_state::Union{Nothing,AbstractVector{<:Real}}=nothing,
    normalize_by_lambda::Bool=true,
    survival_affects_fecundity::Bool=true,
    custom_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    return_trajectories::Bool=false,
    skip_infeasible::Bool=true,
    force_during_recovery::Bool=false)::GridResult

    model = _as_model(model_or_A)
    scen = scenarios(grid; skip_infeasible=skip_infeasible)
    nscen = length(scen)

    magnitudes = Vector{Float64}(undef, nscen)
    durations = Vector{Int}(undef, nscen)
    periods = Vector{Int}(undef, nscen)
    feasible = Vector{Bool}(undef, nscen)
    reductions = fill(NaN, nscen)
    final_pop = fill(NaN, nscen)
    baseline_final_pop = fill(NaN, nscen)
    trajectories = Dict{Tuple{Float64,Int,Int},SimulationResult}()

    init = if initial_state === nothing
        stable_stage_distribution(model.A)
    else
        x = Vector{Float64}(initial_state)
        sx = sum(x)
        sx > 0 || throw(ArgumentError("initial_state must have positive total abundance."))
        x ./ sx
    end

    A0 = normalize_by_lambda ? model.A ./ model.lambda : model.A

    for (idx, s) in enumerate(scen)
        magnitudes[idx] = s.magnitude
        durations[idx] = s.duration
        periods[idx] = s.period
        feasible[idx] = s.feasible

        if !s.feasible
            continue
        end

        Bpert = apply_perturbation(model, target, s.magnitude;
            survival_affects_fecundity=survival_affects_fecundity,
            custom_mask=custom_mask)
        A1 = normalize_by_lambda ? Bpert ./ model.lambda : Bpert

        if return_trajectories
            sim = simulate_dynamics(model;
                target=target,
                magnitude=s.magnitude,
                duration=s.duration,
                period=s.period,
                t_max=t_max,
                recovery_steps=recovery_steps,
                start=start,
                initial_state=init,
                normalize_by_lambda=normalize_by_lambda,
                survival_affects_fecundity=survival_affects_fecundity,
                custom_mask=custom_mask,
                return_stage_vectors=false,
                force_during_recovery=force_during_recovery)
            trajectories[(s.magnitude, s.duration, s.period)] = sim
            reductions[idx] = sim.reduction
            final_pop[idx] = sim.final_population
            baseline_final_pop[idx] = sim.baseline_final_population
        else
            red, fp, bfp = _simulate_reduction_fast(model, A0, A1, init, s.duration, s.period,
                Int(t_max), Int(recovery_steps), Int(start), force_during_recovery)
            reductions[idx] = red
            final_pop[idx] = fp
            baseline_final_pop[idx] = bfp
        end
    end

    table = DataFrame(
        magnitude=magnitudes,
        duration=durations,
        period=periods,
        feasible=feasible,
        population_reduction=reductions,
        final_population=final_pop,
        baseline_final_population=baseline_final_pop,
    )

    ϕ = compute_vulnerability(table)
    return GridResult(table, trajectories, ϕ)
end

end # module
