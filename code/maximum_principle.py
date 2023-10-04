"""
    FEniCSx implementation of the proximal Galerkin method for advection-diffusion introduced in [1].

    [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
       preserving finite element method for pointwise bound constraints.
       arXiv:2307.12444 [math.NA]
"""

import argparse
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import ufl
from dolfinx import fem, io, log, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, GhostMode
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import dot, dx, exp, grad, inner, cos, tanh, max_value, min_value, TrialFunction, TestFunction, ln


def rank_print(string: str, comm: MPI.Comm, rank: int = 0):
    """Helper function to print on a single rank

    :param string: String to print
    :param comm: The MPI communicator
    :param rank: Rank to print on, defaults to 0
    """
    if comm.rank == rank:
        print(string)


def allreduce_scalar(form: fem.FormMetaClass, op: MPI.Op = MPI.SUM) -> np.floating:
    """Assemble a scalar form over all processes and perform a global reduction

    :param form: Scalar form
    :param op: MPI reduction operation
    """
    comm = form.mesh.comm
    return comm.allreduce(fem.assemble_scalar(form), op=op)

def create_mesh(m: int):
    num_cells_per_direction = 2**m
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(num_cells_per_direction, num_cells_per_direction),
        cell_type=CellType.quadrilateral,
        ghost_mode=GhostMode.shared_facet,
    )
    return msh

def exact_solution(x, epsilon):
    lambda1 = pi**2 * epsilon
    r1 =  (1.0 + np.sqrt(1.0 + 4.0 * epsilon * lambda1)) / (2*epsilon)
    r2 =  (1.0 - np.sqrt(1.0 + 4.0 * epsilon * lambda1)) / (2*epsilon)
    denom = 2.0 * ( np.exp(-r2) - np.exp(-r1) )
    u_exact = ( exp( r2 * (x[0] - 1.0) ) - exp( r1 * (x[0] - 1.0) ) ) \
              * cos(pi * x[1] ) / denom + 0.5
    return u_exact

def BG_solve(
    m: int,
    polynomial_order: int,
    epsilon: float,
):
    """
    Solve the Eriksson–Johnson model problem with Bubnov–Galerkin

    """
    # Create mesh
    msh = create_mesh(m)

    # Define FE subspaces
    V = fem.FunctionSpace(msh, ("Lagrange", polynomial_order))

    # Define functions and parameters
    x = ufl.SpatialCoordinate(msh)
    f = fem.Constant(msh, 0.0)
    u_exact = exact_solution(x, epsilon)
    beta = fem.Constant(msh, ScalarType((1,0)))

    # Define BCs
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(
        u_exact, V.element.interpolation_points()))
    bc = fem.dirichletbc(value=u_bc, dofs=dofs)

    # Define bilinear form and RHS
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(epsilon*grad(u), grad(v)) * dx + dot(beta, grad(u)) * v * dx
    L = f * v * dx

    # Set-up linear problem
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
   
    # Solve linear problem
    uh = problem.solve()

    # observables
    H1_error_form = fem.form(
        epsilon * inner(grad(uh - u_exact), grad(uh - u_exact)) * dx + (uh - u_exact) ** 2 * dx
    )
    error_local = fem.assemble_scalar(H1_error_form)
    H1_error = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))
    if msh.comm.rank == 0:
        print(f"\n(Weighted) H1_error of the initial guess : {H1_error:.2e}\n")

    # Save solution
    cwd = Path.cwd()
    output_dir = cwd / "output"
    output_dir.mkdir(exist_ok=True)

    with io.VTXWriter(msh.comm, output_dir / "u_BG.bp", [uh]) as vtx:
        vtx.write(0.0)

    return uh

def PG_solve(
    m: int,
    polynomial_order: int,
    maximum_number_of_outer_loop_iterations: int,
    alpha_value: float,
    epsilon: float,
    tol_exit: float,
    GLL: bool
):
    """
    Solve the Eriksson–Johnson model problem with Proximal Galerkin

    """

    # Create mesh
    msh = create_mesh(m)

    # Define FE subspaces
    Q = ufl.FiniteElement("Lagrange", msh.ufl_cell(), polynomial_order)
    mixed_element = ufl.MixedElement([Q, Q])
    V = fem.FunctionSpace(msh, mixed_element)

    # Define functions and parameters
    alpha = fem.Constant(msh, ScalarType(1))
    alpha.value = alpha_value
    beta = fem.Constant(msh, ScalarType((1,0)))
    x = ufl.SpatialCoordinate(msh)

    f = fem.Constant(msh, 0.0)
    u_exact = exact_solution(x, epsilon)

    # Define BCs
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    V0, _ = V.sub(0).collapse()
    V1, _ = V.sub(1).collapse()
    dofs_0 = fem.locate_dofs_topological(
        (V.sub(0), V0), entity_dim=1, entities=facets)
    dofs_1 = fem.locate_dofs_topological(
        (V.sub(1), V1), entity_dim=1, entities=facets)

    u_bc = fem.Function(V0)
    u_bc.interpolate(fem.Expression(
        u_exact, V.sub(0).element.interpolation_points()))
    bc_0 = fem.dirichletbc(value=u_bc, dofs=dofs_0, V=V.sub(0))

    eps = 1e-3
    u_exact_clip = max_value(eps, min_value(u_exact, 1-eps))
    lnit_u_exact = ln(u_exact_clip/(1.0-u_exact_clip))

    psi_bc = fem.Function(V1)
    psi_bc.interpolate(fem.Expression(
        lnit_u_exact, V.sub(1).element.interpolation_points()))
    bc_1 = fem.dirichletbc(value=psi_bc, dofs=dofs_1, V=V.sub(1))

    # Define solution variables
    sol = fem.Function(V)
    sol_k = fem.Function(V)

    u, psi = ufl.split(sol)
    u_k, psi_k = ufl.split(sol_k)

    # Define non-linear residual
    if GLL:
        dx_GLL = ufl.Measure("dx", metadata = {"quadrature_rule": "GLL"})
    else:
        dx_GLL = dx
    (v, w) = ufl.TestFunctions(V)
    rho = epsilon
    F = (
        alpha/rho * inner(grad(u), grad(v)) * dx
        + psi * v * dx_GLL
        - alpha * (1 / rho - epsilon) * inner(grad(u_k), grad(v)) * dx
        + alpha * dot(beta,grad(u_k)) * v * dx
        - alpha * f * v * dx
        - psi_k * v * dx_GLL
        + u * w * dx_GLL
        - (tanh(psi/2) + 1)/2 * w * dx_GLL
    )
    J = ufl.derivative(F, sol)

    # Setup non-linear problem
    problem = NonlinearProblem(F, sol, bcs=[bc_0,bc_1], J=J)

    # Setup Newton solver
    log.set_log_level(log.LogLevel.WARNING)
    newton_solver = NewtonSolver(
        comm=msh.comm, problem=problem)

    # Observables
    L2primal_increment_form = fem.form( (u - u_k) ** 2 * dx )
    L2latent_increment_form = fem.form(((tanh(psi/2) + 1)/2 - (tanh(psi_k/2) + 1)/2) ** 2 * dx)
    H1primal_error_form = fem.form(
        epsilon * inner(grad(u - u_exact), grad(u - u_exact)) *
        dx + (u - u_exact) ** 2 * dx
    )
    L2primal_error_form = fem.form((u - u_exact) ** 2 * dx)
    L2latent_error_form = fem.form(((tanh(psi/2) + 1)/2 - u_exact) ** 2 * dx)

    # Initial guess from FEM solution
    uh = BG_solve( m, polynomial_order, epsilon )
    uh_clip = max_value(eps, min_value(uh, 1-eps))
    lnit_uh = ln(uh_clip/(1.0-uh_clip))

    sol.sub(0).interpolate(fem.Expression(
        uh, V.sub(0).element.interpolation_points()))
    sol.sub(1).interpolate(fem.Expression(
        lnit_uh, V.sub(1).element.interpolation_points()))

    # Proximal residual outer loop
    n = 0
    increment_k = 0.0
    sol_k.x.array[:] = sol.x.array[:]

    H1primal_errors = []
    L2primal_errors = []
    L2latent_errors = []
    Newton_steps = []
    step_sizes = []
    primal_increments = []
    latent_increments = []
    for k in range(maximum_number_of_outer_loop_iterations):

        rank_print(f"OUTER LOOP {k + 1} alpha: {alpha.value}", msh.comm)

        # Solve problem
        (n, converged) = newton_solver.solve(sol)
        rank_print(f"Newton steps: {n}   Converged: {converged}", msh.comm)

        # Check outer loop convergence
        primal_increment = np.sqrt(allreduce_scalar(L2primal_increment_form))
        latent_increment = np.sqrt(allreduce_scalar(L2latent_increment_form))
        H1primal_error = np.sqrt(allreduce_scalar(H1primal_error_form))
        L2primal_error = np.sqrt(allreduce_scalar(L2primal_error_form))
        L2latent_error = np.sqrt(allreduce_scalar(L2latent_error_form))

        tol_Newton = primal_increment

        rank_print(f"‖u - uₕ‖_H¹: {H1primal_error}" +
                   f"  ‖u - ũₕ‖_L² : {L2latent_error}", msh.comm)

        if increment_k > 0.0:
            rank_print(f"Increment size: {primal_increment}" +
                       f"   Ratio: {primal_increment / increment_k}", msh.comm)
        else:
            rank_print(f"Increment size: {primal_increment}", msh.comm)
        rank_print("", msh.comm)

        H1primal_errors.append(H1primal_error)
        L2primal_errors.append(L2primal_error)
        L2latent_errors.append(L2latent_error)
        Newton_steps.append(n)
        step_sizes.append(np.copy(alpha.value))
        primal_increments.append(primal_increment)
        latent_increments.append(latent_increment)

        if tol_Newton < tol_exit and n > 0:
            break

        # Reset Newton solver options
        newton_solver.atol = 1e-6
        newton_solver.rtol = tol_Newton * 1e-5

        # Update sol_k with sol_new
        sol_k.x.array[:] = sol.x.array[:]
        increment_k = primal_increment

    # Save data
    cwd = Path.cwd()
    output_dir = cwd / "output"
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame()
    df["H1 Primal errors"] = H1primal_errors
    df["L2 Primal errors"] = L2primal_errors
    df["L2 Latent errors"] = L2latent_errors
    df["Newton steps"] = Newton_steps
    df["Step sizes"] = step_sizes
    df["Primal increments"] = primal_increments
    df["Latent increments"] = latent_increments
    df["Polynomial order"] = [polynomial_order] * (k + 1)
    df["Mesh size"] = [1 / 2 ** (m - 1)] * (k + 1)
    df["dofs"] = [np.size(sol_k.x.array[:])] * (k + 1)
    filename = f"./maxppl_polyorder{polynomial_order}_m{m}.csv"
    rank_print(f"Saving data to: {str(output_dir / filename)}", msh.comm)
    df.to_csv(output_dir / filename, index=False)
    rank_print(df, msh.comm)


    # Export primal solution variable
    V_out = fem.FunctionSpace(msh, ("Lagrange", polynomial_order))
    q = fem.Function(V_out)
    q.interpolate(sol.sub(0).collapse())
    with io.VTXWriter(msh.comm, output_dir / "u.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export interpolant of exact solution u
    expr = fem.Expression(u_exact, V_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "u_exact.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export latent solution variable
    q.interpolate(sol.sub(1).collapse())
    with io.VTXWriter(msh.comm, output_dir / "psi.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export feasible discrete solution
    exp_psi = (tanh(sol.sub(1)/2) + 1)/2
    expr = fem.Expression(exp_psi, V_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "tilde_u.bp", [q]) as vtx:
        vtx.write(0.0)

# -------------------------------------------------------
# TODO
if __name__ == "__main__":
    desc = "Run advection-diffusion example from paper"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mesh-density",
        "-m",
        dest="m",
        type=int,
        default=4,
        help="MESH DENSITY",
    )
    parser.add_argument(
        "--polynomial_order",
        "-p",
        dest="polynomial_order",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Polynomial order",
    )
    parser.add_argument(
        "--max-iter",
        "-i",
        dest="maximum_number_of_outer_loop_iterations",
        type=int,
        default=100,
        help="Maximum number of outer loop iterations",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        dest="alpha",
        type=float,
        default=1,
        help="Step size",
    )
    parser.add_argument(
        "--epsilon",
        "-eps",
        dest="epsilon",
        type=float,
        default=1e-2,
        help="Diffusion parameter",
    )
    parser.add_argument(
        "--tol",
        "-t",
        dest="tol_exit",
        type=float,
        default=1e-5,
        help="Tolerance for exiting Newton iteration",
    )
    parser.add_argument(
        "--GLL-quadrature-rule",
        "-GLL",
        dest="GLL",
        action="store_true",
        help="Use GLL quadrature",
    )
    args = parser.parse_args()
    PG_solve(
        args.m,
        args.polynomial_order,
        args.maximum_number_of_outer_loop_iterations,
        args.alpha,
        args.epsilon,
        args.tol_exit,
        args.GLL
    )
