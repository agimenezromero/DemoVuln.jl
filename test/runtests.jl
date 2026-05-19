using Test
using DemoVuln
using Statistics: mean
using DataFrames: nrow

function simple_matrix()
    return [0.0 2.0; 0.4 0.7]
end

@testset "MatrixPopulationModel" begin
    model = MatrixPopulationModel(simple_matrix())
    @test size(model.A) == (2, 2)
    @test model.lambda > 0
    @test sum(stable_stage_distribution(model.A)) ≈ 1.0
    @test model.adult_stages == [2]
    @test model.juvenile_stages == [1]

    @test_throws ArgumentError MatrixPopulationModel(ones(2, 3))
    @test_throws ArgumentError MatrixPopulationModel([-1.0 0.0; 0.0 1.0])
end

@testset "Perturbations" begin
    A = simple_matrix()
    model = MatrixPopulationModel(A)

    B = apply_perturbation(model, :adult_survival, 0.5)
    @test B ≈ [0.0 1.0; 0.4 0.35]

    B = apply_perturbation(model, :fecundity, 0.5)
    @test B ≈ [0.0 1.0; 0.4 0.7]

    B = apply_perturbation(model, :all, 0.5)
    @test B ≈ [0.0 0.5; 0.2 0.35]
end

@testset "Simulation" begin
    model = MatrixPopulationModel(simple_matrix())

    sim0 = simulate_dynamics(model;
        target=:adult_survival,
        magnitude=0.0,
        duration=1,
        period=2,
        t_max=20,
        recovery_steps=5)
    @test abs(sim0.reduction) < 1e-8

    sim = simulate_dynamics(model;
        target=:adult_survival,
        magnitude=0.5,
        duration=1,
        period=2,
        t_max=20,
        recovery_steps=5)
    @test sim.reduction > 0
    @test length(sim.abundance) == 26

    sim_stage = simulate_dynamics(model;
        target=:juvenile_survival,
        magnitude=0.25,
        duration=1,
        period=3,
        t_max=20,
        recovery_steps=5,
        return_stage_vectors=true)
    @test size(sim_stage.stage_vectors) == (26, 2)

    @test_throws ArgumentError simulate_dynamics(model;
        target=:adult_survival,
        magnitude=0.5,
        duration=5,
        period=2,
        t_max=20)
end

@testset "Grid" begin
    model = MatrixPopulationModel(simple_matrix())
    grid = PerturbationGrid([0.0, 0.5], [1], [2, 3])

    out = run_grid(model;
        target=:adult_survival,
        grid=grid,
        t_max=20,
        recovery_steps=5)

    @test nrow(out.table) == 4
    @test out.vulnerability ≈ mean(out.table.population_reduction)

    grid2 = perturbation_grid_from_frequencies([0.1], [1], [1.0, 2.0]; generation_time=4)
    @test grid2.periods == [2, 4]

    grid3 = PerturbationGrid([0.5], [3], [2])
    out3 = run_grid(model;
        target=:adult_survival,
        grid=grid3,
        t_max=20,
        skip_infeasible=false)
    @test nrow(out3.table) == 1
    @test out3.table.feasible[1] == false
    @test isnan(out3.table.population_reduction[1])
end
