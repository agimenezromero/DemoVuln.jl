using DemoVuln

A = [0.0 2.0;
     0.4 0.7]

model = MatrixPopulationModel(A)

sim = simulate_dynamics(model;
    target=:adult_survival,
    magnitude=0.25,
    duration=1,
    period=3,
    t_max=50,
    recovery_steps=10,
)

println("Population reduction = ", sim.reduction)
println("Final abundance = ", sim.final_population)

grid = PerturbationGrid(range(0, 1, length=11), 0:3, [1, 2, 3, 5, 10])

out = run_grid(model;
    target=:adult_survival,
    grid=grid,
    t_max=50,
    recovery_steps=10,
)

println("Integrated vulnerability = ", out.vulnerability)
println(first(out.table, 5))
