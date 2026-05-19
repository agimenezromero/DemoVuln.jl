using DemoVuln

A = [0.0 2.0;
     0.4 0.7]

model = MatrixPopulationModel(A)

grid = PerturbationGrid(
    range(0, 1, length=101),
    0:10,
    1:50,
)

@time out = run_grid(model;
    target=:adult_survival,
    grid=grid,
    t_max=100,
    recovery_steps=20,
    return_trajectories=false,
)

println(out.vulnerability)
println(size(out.table))
