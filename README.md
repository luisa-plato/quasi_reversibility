# Quasi-reversibility for the backward heat equation

Numerical experiments for the quasi-reversibility method applied to the backward heat equation. The script illustrates the logarithmic convergence rate in \(L^5\) on \((0,\pi)\) and \((0,\pi)^2\).

## Requirements

[Firedrake](https://www.firedrakeproject.org/) and matplotlib. Run the script from a Firedrake environment.

## Run

```bash
python QR_backward_diffusion.py --dim 1
python QR_backward_diffusion.py --dim 2
python QR_backward_diffusion.py            # both
```

Add `--show` to open the figures after they are saved. They are written to the working directory:

- `convergence_rates_dim_1.png`, `convergence_rates_dim_2.png`
- `exact_and_perturbed_final_values_dim_1.png`
- `exact_and_recovered_solution_at_t_0_dim_1.png`
- `exact_and_recovered_solution_at_t_0_and_t_T_dim_2.png`

The script also writes VTK time series that open in [ParaView](https://www.paraview.org/):

- `exact_dim_1.pvd`, `exact_dim_2.pvd`: the exact solution, once per dimension.
- `qr_dim_{dim}_delta_1e-{i}.pvd`: the regularized backward solve for the noise level $\delta = 10^{-i}$, with 101 snapshots from $t = 1$ down to $t = 0$.

Both use the same time values, so one time slider steps through the exact solution and the recovery together.

## Experiment

The final time is $T = 1$. The spatial discretization uses continuous piecewise linear elements on a uniform mesh of $(0,\pi)$ and bilinear ($Q_1$) elements on a uniform quadrilateral mesh of $(0,\pi)^2$, with 32 cells in each direction. The time step is $\Delta t = 0.01$. The terminal datum is the exact solution at $t = T$ interpolated onto the finite element space. Noise levels run from $10^{-1}$ to $10^{-15}$. The regularization parameter is

$$
\alpha(\delta) = \frac{T}{\log \Big(\frac{a}{\delta (\log 1/\delta)^{\nu}}\Big)}
$$

with $a = \nu = 1$. The computed $L^5$ error is compared with $c/\log(1/\delta)$, using $c = 1.2$ in one space dimension and $c = 4.6$ in two.
