# DemoVuln.jl

`DemoVuln.jl` is the Julia implementation of the `demovuln` framework for simulating temporally structured demographic perturbations in matrix population models and estimating integrated population vulnerability.

The Python package is the main documented implementation:

- PyPI: https://pypi.org/project/demovuln/
- Documentation: https://demovuln.readthedocs.io/en/latest/
- GitHub: https://github.com/agimenezromero/demovuln

The Julia implementation follows the same conceptual API, but is designed for faster large-scale simulation grids.

## Installation

From Julia:

```julia
import Pkg
Pkg.add(url="https://github.com/agimenezromero/DemoVuln.jl")
```

## Basic usage

```julia
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

sim.reduction
sim.abundance
```

## Perturbation-grid analysis

```julia
using DemoVuln

A = [0.0 2.0;
     0.4 0.7]

model = MatrixPopulationModel(A)

grid = PerturbationGrid(
    range(0, 1, length=11),
    0:3,
    [1, 2, 3, 5, 10],
)

out = run_grid(model;
    target=:adult_survival,
    grid=grid,
    t_max=50,
    recovery_steps=10,
)

out.vulnerability
first(out.table, 5)
```

## Demographic targets

The package supports perturbations to:

- `:adult_survival`
- `:juvenile_survival`
- `:fecundity`
- `:all`
- `:custom`

By default, adult stages are inferred as source-stage columns with at least one fecundity entry, and juvenile stages are inferred as the remaining source-stage columns. These definitions can be specified explicitly:

```julia
model = MatrixPopulationModel(
    A;
    adult_stages=[2],
    juvenile_stages=[1],
)
```

Custom perturbation targets can be defined with Boolean masks:

```julia
custom_mask = [false false;
               true  false]

sim = simulate_dynamics(model;
    target=:custom,
    custom_mask=custom_mask,
    magnitude=0.5,
    duration=1,
    period=3,
    t_max=50,
)
```

## Performance notes

For large-scale analyses, use `run_grid(...; return_trajectories=false)`, which is the default. In this mode, the package avoids storing full trajectories and uses an internal loop that preallocates vectors and stores only final reductions.

Use `return_trajectories=true` only when full trajectories are needed for plotting or diagnostics.

## Development

From the repository root:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

## License

MIT License.

Maintainer: Àlex Giménez-Romero (<alex.gimenez@csic.es>)