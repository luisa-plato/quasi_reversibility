"""
Quasi-reversibility for the backward heat equation.

The terminal-value problem

    du/dt - Delta u = 0    in Omega x (0, T),
    u(·, T) = f            in Omega,

with homogeneous Dirichlet boundary conditions, is ill-posed. It is
regularized by the quasi-reversibility (Sobolev) equation

    d v_alpha / dt - Delta (I - alpha Delta)^{-1} v_alpha = 0,
    v_alpha(·, T) = f^delta,

and the approximation of the initial state is v_alpha(·, 0).

Logarithmic convergence of that approximation, measured in L^5, for the parameter choice

    alpha(delta) = T / log( a / (delta * log(1/delta)^nu) ),

on Omega = (0, pi) and Omega = (0, pi)^2 is demonstrated. The exact solutions are

    u(x, t)     = exp(-t) sin(x)           on (0, pi),
    u(x, y, t)  = exp(-2t) sin(x) sin(y)   on (0, pi)^2.

Requires Firedrake (https://www.firedrakeproject.org/) and matplotlib.

    python QR_backward_diffusion.py --dim 1
    python QR_backward_diffusion.py --dim 2
    python QR_backward_diffusion.py            # both
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    IntervalMesh,
    solve,
    SpatialCoordinate,
    SquareMesh,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    grad,
    inner,
    sin,
)
from firedrake.pyplot import tripcolor

#--------------------set some constants and parameters--------------------
# Plot colours. 
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
S_BLUE = "#2a78d6"
S_ORANGE = "#eb6834"
S_AQUA = "#1baf7a"
S_PURPLE = "#800080"
S_GREEN = "#008000"
CURVE_COLOURS = (S_ORANGE, S_PURPLE, S_GREEN)

# ||sin(n x)||_{L^5(0, pi)} = (16/15)^{1/5}. On the square the integral separates and the norm is the product of the 1D norms.
_L5_INTEGRAL_1D = 16.0 / 15.0

# Prefactors of the comparison curve c / log(1/delta). 
COMPARISON_CONSTANTS = {1: 1.2, 2: 4.0}

# Which noise levels are drawn, as indices into delta = 10^{-1}, ..., 10^{-15}.
# One dimension: 1e-1, 1e-2, 1e-7
# Two dimensions: 1e-1, 1e-3, 1e-15.
_PLOT_FIELDS_1D = (0, 1, 6)
_PLOT_FIELDS_2D = (0, 2, 14)

#--------------------define some helperfunctions--------------------
def add_legend(ax, **kwargs):
    """Legend with an opaque backing, so lines do not run through the text."""
    return ax.legend(
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=0.92,
        **kwargs,
    )


def lp_norm(field, p: float) -> float:
    """||field||_{L^p}. For a real field, |field|^p = (field, field)^{p/2}."""
    integrand = inner(field, field) ** (p / 2) * dx
    return float(assemble(integrand) ** (1.0 / p))


# def _eigenmode_expression(coords, dim: int, wavenumber: int):
#     """prod_{i=1}^{dim} sin(wavenumber * x_i), an eigenfunction of -Delta."""
#     expression = sin(wavenumber * coords[0])
#     for i in range(1, dim):
#         expression = expression * sin(wavenumber * coords[i])
#     return expression


#def _noise_amplitude(delta: float, dim: int) -> float:
#    """
#    Coefficient of the perturbation mode.
#
#    Dividing by the L^5 norm of that mode makes the perturbation itself
#    have L^5 norm equal to ``delta`` in one dimension. In two dimensions
#    the same normalization is used and then multiplied by 2, which is the
#    factor the published figures were computed with: the L^5 norm of the
#    two-dimensional noise is therefore ``2 * delta``.
#    """
#    if dim == 1:
#        return delta / (_L5_INTEGRAL_1D ** (1.0 / 5.0))
#    # (16/15)^2 = 256/225 = ||sin(nx) sin(ny)||_{L^5}^5
#    return 2.0 * delta / ((256.0 / 225.0) ** (1.0 / 5.0))


#--------------------define the quasi-reversibility problem--------------------
class QuasiReversibility:
    """
    Backward heat equation on (0, pi)^dim, regularized by quasi-reversibility.
    """

    def __init__(
        self,
        dim: int,
        final_time: float,
        dt: float,
        resolution: int,
        degree: int = 1,
    ):
        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2.")

        if dim == 1:
            self.mesh = IntervalMesh(resolution, 0.0, np.pi)
        else:
            self.mesh = SquareMesh(resolution, resolution, np.pi)

        self.dim = dim
        self.final_time = float(final_time)
        self.degree = degree
        self.V = FunctionSpace(self.mesh, "CG", degree)
        self.bc = DirichletBC(self.V, Constant(0.0), "on_boundary")
        self.coords = SpatialCoordinate(self.mesh)

        # Snap dt so that an integer number of steps lands exactly on T.
        self.num_steps = max(1, int(round(self.final_time / dt)))
        self.dt = self.final_time / self.num_steps

        self._trial = TrialFunction(self.V)
        self._test = TestFunction(self.V)

    def initial_state(self) -> Function:
        """
        Define initial state as a sinus function.
        """
        if self.dim == 1:
            x, = SpatialCoordinate(self.mesh)
            expression = sin(x)
        elif self.dim == 2:
            x,y = SpatialCoordinate(self.mesh)
            expression = sin(x) * sin(y)
        return Function(self.V).interpolate(expression)

    def perturb(self, terminal: Function, delta: float) -> Function:
        """
        Copy the terminal Function and add a delta perturbation.
        """
        perturbed = Function(self.V, name=f"terminal_delta_{delta:g}")
        dim = self.V.mesh().topological_dimension
        if dim == 1:
            x, = SpatialCoordinate(self.V.mesh())
            perturbed.interpolate(terminal + (delta/(16.0/15.0)**(1/5)) * sin(2*x))
        elif dim == 2:
            x,y = SpatialCoordinate(self.V.mesh())
            perturbed.interpolate(terminal + delta/(256.0/225.0)**(1/5) * sin(2*x) * sin(2*y))
        return perturbed

    def solve_forward(self, initial: Function) -> Function:
        """
        Integrate the forward heat equation from ``initial`` up to time T.
        """
        inv_dt = Constant(1.0 / self.dt)
        u_prev = Function(self.V)
        u_prev.assign(initial)
        u_next = Function(self.V)

        # (u^{n+1} - u^n)/dt - Delta u^{n+1} = 0.
        a = (
            inv_dt * inner(self._trial, self._test)
            + inner(grad(self._trial), grad(self._test))
        ) * dx
        L = inv_dt * inner(u_prev, self._test) * dx

        for _ in range(self.num_steps):
            solve(a == L, u_next, self.bc)
            u_prev.assign(u_next)
        return u_prev

    def solve_QR(self, terminal: Function, alpha: float) -> Function:
        """
        Recover an approximation of u(·, 0) from terminal data.

        Time is reversed by tau = T - t and the regularized equation is
        marched with explicit Euler. One step is the pair of weak problems

            (phi, psi) + alpha (grad phi, grad psi) = (v, psi),
            (v_new, w) = (v, w) + dt (grad phi, grad w),

        which is the weak form of the quasi-reversibility equation
        (I - alpha Delta) phi = v and v_new = v - dt Delta phi.
        """
        inv_dt = Constant(1.0 / self.dt)
        alpha_c = Constant(alpha)

        state = Function(self.V)
        state.assign(terminal)
        phi = Function(self.V)
        updated = Function(self.V)

        # (I - alpha Delta) phi = state, homogeneous Dirichlet conditions.
        a_phi = (
            inner(self._trial, self._test)
            + alpha_c * inner(grad(self._trial), grad(self._test))
        ) * dx
        L_phi = inner(state, self._test) * dx

        # Mass-matrix solve for the explicit Euler update.
        a_state = inv_dt * inner(self._trial, self._test) * dx
        L_state = (
            inner(grad(phi), grad(self._test))
            + inv_dt * inner(state, self._test)
        ) * dx

        for _ in range(self.num_steps):
            solve(a_phi == L_phi, phi, self.bc)
            solve(a_state == L_state, updated, self.bc)
            state.assign(updated)
        return state

#--------------------define plotting functions--------------------

def _sample_line(field: Function, sample: np.ndarray) -> np.ndarray:
    """Point values of a 1D field. Endpoints are kept off the boundary nodes."""
    return np.asarray([float(value) for value in field.at(sample)])


def _plot_convergence(dim, deltas, errors, comparison, constant, show: bool) -> str:
    fig, ax = plt.subplots(layout="constrained")
    ax.loglog(
        deltas,
        comparison,
        color=S_AQUA,
        linestyle="-.",
        label=rf"comparison curve  $c/\log(1/\delta)$,  $c = {constant:g}$",
    )
    ax.loglog(
        deltas,
        errors,
        color=S_BLUE,
        marker="o",
        linestyle="-",
        label=r"computed $L^5$ error",
    )
    ax.invert_xaxis()
    ax.set_xlabel(r"data noise level  $\delta$", fontsize=12)
    if dim == 1:
        domain = r"(0,\pi)"
    else:
        domain = r"((0,\pi)^2)"
    ax.set_ylabel(rf"$\|u_0 - u_\alpha^\delta\|_{{L^5{domain}}}$", fontsize=12)
    ax.set_title(f"{dim}D — logarithmic convergence", loc="center", fontsize=14)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    add_legend(ax, loc="upper right", fontsize=12)

    path = f"convergence_rates_dim_{dim}.png"
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_profiles_1d(exact_terminal, perturbed, exact_initial, recovered, deltas, show: bool) -> list[str]:
    sample = np.linspace(1e-9, np.pi - 1e-9, 400)
    paths = []

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(
        sample,
        _sample_line(exact_terminal, sample),
        label="exact final value",
        color=S_BLUE,
        linewidth=3,
        alpha=0.35,
    )
    for colour, index in zip(CURVE_COLOURS, _PLOT_FIELDS_1D):
        ax.plot(
            sample,
            _sample_line(perturbed[index], sample),
            label=rf"perturbed final value for $\delta$ = {deltas[index]}",
            color=colour,
            linestyle="--",
            linewidth=1.5,
        )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$u$", fontsize=12)
    ax.set_title("1D — exact and perturbed final values", loc="center", fontsize=14)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    add_legend(ax, loc="upper right", fontsize=9)
    path = "exact_and_perturbed_final_values_dim_1.png"
    fig.savefig(path, dpi=300)
    paths.append(path)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(
        sample,
        _sample_line(exact_initial, sample),
        label="exact initial value",
        color=S_BLUE,
        linewidth=3,
        alpha=0.35,
    )
    for colour, index in zip(CURVE_COLOURS, _PLOT_FIELDS_1D):
        ax.plot(
            sample,
            _sample_line(recovered[index], sample),
            label=rf"recovered solution at t = 0 for $\delta$ = {deltas[index]}",
            color=colour,
            linestyle="--",
            linewidth=1.5,
        )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$u$", fontsize=12)
    ax.set_title("1D — exact and recovered solution at t = 0", loc="center", fontsize=14)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    add_legend(ax, loc="upper right", fontsize=9)
    path = "exact_and_recovered_solution_at_t_0_dim_1.png"
    fig.savefig(path, dpi=300)
    paths.append(path)
    if show:
        plt.show()
    plt.close(fig)
    return paths


def _plot_fields_2d(exact_initial, exact_terminal, perturbed, recovered, show: bool) -> str:
    """
    Four rows: the exact pair (u(0), u(T)), then for three noise levels
    the recovered initial state beside the perturbed terminal datum.
    """
    colours = "inferno"
    fig, axes = plt.subplots(4, 2, figsize=(10, 20), layout="constrained")
    for ax in axes.ravel():
        ax.set_aspect("equal")

    panel = tripcolor(exact_initial, axes=axes[0][0], cmap=colours)
    axes[0][0].set_title("exact initial value t = 0.0", fontsize=12)
    fig.colorbar(panel, ax=axes[0][0])

    panel = tripcolor(exact_terminal, axes=axes[0][1], cmap=colours)
    axes[0][1].set_title("exact final value t = 1.0", fontsize=12)
    fig.colorbar(panel, ax=axes[0][1])

    delta_titles = (r"$\delta = 10^{-1}$", r"$\delta = 10^{-3}$", r"$\delta = 10^{-15}$")
    for row, index, title in zip(range(1, 4), _PLOT_FIELDS_2D, delta_titles):
        panel = tripcolor(recovered[index], axes=axes[row][0], cmap=colours)
        axes[row][0].set_title(f"recovered solution at t = 0.0, {title}", fontsize=12)
        fig.colorbar(panel, ax=axes[row][0])

        panel = tripcolor(perturbed[index], axes=axes[row][1], cmap=colours)
        axes[row][1].set_title(f"perturbed final value t = 1.0, {title}", fontsize=12)
        fig.colorbar(panel, ax=axes[row][1])

    path = "exact_and_recovered_solution_at_t_0_and_t_T_dim_2.png"
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return path

#--------------------define the main function demonstrating the convergence--------------------

def regularization_parameters(deltas, final_time: float, a: float, nu: float):
    """alpha(delta) = T / log( a / (delta * log(1/delta)^nu) )."""
    logs = np.log(1.0 / np.asarray(deltas, dtype=float))
    return final_time / np.log(a / (deltas * logs**nu))


def demonstrate_convergence(
    T: float = 1.0,
    dt: float = 0.01,
    a: float = 1.0,
    nu: float = 1.0,
    dim: int = 1,
    resolution: int = 32,
    degree: int = 1,
    norm_p: float = 5.0,
    comparison_constant: float | None = None,
    show: bool = False,
):
    """
    Sweep delta from 10^{-1} down to 10^{-15} and plot the L^p error.
    ``comparison_constant`` defaults to 1.2 in 1D and 4 in 2D.
    Figures are written in the working directory. Set ``show=True`` to
    open them as well.
    """
    if dim not in (1, 2):
        raise ValueError("dim must be 1 or 2.")
    if comparison_constant is None:
        comparison_constant = COMPARISON_CONSTANTS[dim]

    deltas = np.array([10.0 ** (-i) for i in range(1, 16)])
    alphas = regularization_parameters(deltas, T, a, nu)

    problem = QuasiReversibility(dim, T, dt, resolution, degree)
    initial = problem.initial_state()
    terminal = problem.solve_forward(initial)

    perturbed = []
    recovered = []
    errors = []
    for delta, alpha in zip(deltas, alphas):
        noisy = problem.perturb(terminal, float(delta))
        approximation = problem.solve_QR(noisy, float(alpha))
        perturbed.append(noisy)
        recovered.append(approximation)
        errors.append(lp_norm(initial - approximation, norm_p))
    errors = np.asarray(errors, dtype=float)

    comparison = comparison_constant / np.log(1.0 / deltas)

    paths = [
        _plot_convergence(dim, deltas, errors, comparison, comparison_constant, show)
    ]
    if dim == 1:
        paths.extend(
            _plot_profiles_1d(terminal, perturbed, initial, recovered, deltas, show)
        )
    else:
        paths.append(_plot_fields_2d(initial, terminal, perturbed, recovered, show))

    for path in paths:
        print(f"saved {path}")

    return {"deltas": deltas, "alphas": alphas, "errors": errors, "figures": paths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logarithmic convergence of quasi-reversibility for the backward heat equation."
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(1, 2),
        nargs="+",
        default=(1, 2),
        help="spatial dimensions to run (default: 1 and 2)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the figures after saving them",
    )
    args = parser.parse_args()
    for dim in args.dim:
        demonstrate_convergence(dim=dim, show=args.show)


if __name__ == "__main__":
    main()
