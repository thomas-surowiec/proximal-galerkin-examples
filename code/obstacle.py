'''
    FEniCSx code to reproduce obstacle problem results in [1].

    [1] B. Keith and T.M. Surowiec (2023) Proximal Galerkin: A structure-preserving
    finite element method for pointwise bound constraints
'''

from math import pi
from os import makedirs, path

import numpy as np
import pandas as pd
import ufl
from dolfinx import fem, io, log, mesh, nls
from dolfinx.mesh import CellType, GhostMode
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import conditional, div, dot, dx, exp, grad, gt, inner, lt, sin


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

# -------------------------------------------------------


# SELECT EXAMPLE
example = '3'

# SELECT MESH DENSITY (m = 2 corresponds to h_∞)
m = 5

# SELECT POLYNOMIAL ORDER
polynomial_order = 1

# ADDITIONAL PARAMETERS
maximum_number_of_outer_loop_iterations = 100
alpha_max = 1e10
tol_exit = 1e-10

if example == '1':
    C = 1.0
    r = 1.5
    q = 1.5
    step_size_rule = "double_exponential"
elif example == '2':
    C = 1.0
    r = 2
    step_size_rule = "geometric"
else:
    example = '3'
    C = 1.0
    r = 1.2
    step_size_rule = "geometric"

# -------------------------------------------------------

# Create mesh
num_cells_per_direction = 2**m
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((-1.0, -1.0), (1.0, 1.0)), n=(num_cells_per_direction, num_cells_per_direction),
                            cell_type=CellType.triangle,
                            ghost_mode=GhostMode.shared_facet)

# Define FE subspaces
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), polynomial_order)
B = ufl.FiniteElement("Bubble", msh.ufl_cell(), polynomial_order+2)
P0 = ufl.FiniteElement("DG", msh.ufl_cell(), polynomial_order-1)
P1B_P0 = ufl.MixedElement([P1+B, P0])
V = fem.FunctionSpace(msh, P1B_P0)

# Define functions and parameters
alpha = fem.Constant(msh, ScalarType(1))
x = ufl.SpatialCoordinate(msh)

if example == '1':
    '''
    This example exhibits two properties that 
    are challenging for active set solvers:
    a) The transition from inactive to active is high order
    b) There is a nontrivial biactive set
    '''
    f = conditional(gt(x[0], 0.0), -12*x[0]**2, fem.Constant(msh, 0.))
    u_exact = conditional(gt(x[0], 0.0), x[0]**4, fem.Constant(msh, 0.))
    lambda_exact = fem.Constant(msh, 0.)

elif example == '2':
    '''
    This is the non-pathological example.
    Here, we witness linear convergence with a fixed step size
    NOTE: The exact solution is not known
    '''
    f = 2.0*pi*pi*sin(pi*x[0])*sin(pi*x[1])
    u_exact = fem.Constant(msh, 0.)       # placeholder
    lambda_exact = fem.Constant(msh, 0.)  # placeholder

else:
    '''
    This is a second biactive example with a non-smooth multiplier
    '''
    z_1 = dot(x, x)
    z_2 = (x[0]**2+x[1]**2-0.25)**2
    z_3 = 7*x[0]**2+x[1]**2-0.25
    z_4 = x[0]**2+7*x[1]**2-0.25
    lambda_exact = conditional(
        lt(0.75, z_1), fem.Constant(msh, 1.), fem.Constant(msh, 0.))
    f = 256*conditional(lt(z_1, 0.25), -8*z_2*z_3-8*z_2*z_4,
                        fem.Constant(msh, 0.)) - lambda_exact
    u_exact = 256*conditional(lt(z_1, 0.25), (0.25-z_1)
                              ** 4, fem.Constant(msh, 0.))

# Define BCs
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
facets = mesh.exterior_facet_indices(msh.topology)
V0, _ = V.sub(0).collapse()
dofs = fem.locate_dofs_topological(
    (V.sub(0), V0), entity_dim=1, entities=facets)

u_bc = fem.Function(V0)
u_bc.interpolate(fem.Expression(
    u_exact, V.sub(0).element.interpolation_points()))
bcs = fem.dirichletbc(value=u_bc, dofs=dofs, V=V.sub(0))

# Define solution variables
sol = fem.Function(V)
sol_k = fem.Function(V)

u, psi = ufl.split(sol)
u_k, psi_k = ufl.split(sol_k)

# Define non-linear residual
(v, w) = ufl.TestFunctions(V)
F = alpha*inner(grad(u), grad(v))*dx + psi*v*dx + u*w*dx - \
    exp(psi)*w*dx - alpha*f*v*dx - psi_k*v*dx
residual = fem.form(F)
J = ufl.derivative(F, sol)
jacobian = fem.form(J)

# Setup non-linear problem
problem = fem.petsc.NonlinearProblem(F, sol, bcs=[bcs], J=J)

# Setup newton solver
log.set_log_level(log.LogLevel.WARNING)
NewtonSolver = nls.petsc.NewtonSolver(comm=MPI.COMM_WORLD, problem=problem)

# observables
energy_form = fem.form(0.5*inner(grad(u), grad(u)) * dx - f*u*dx)
complementarity_form = fem.form((psi_k - psi)/alpha*u*dx)
feasibility_form = fem.form(conditional(
    lt(u, 0), -u, fem.Constant(msh, 0.))*dx)
dual_feasibility_form = fem.form(conditional(
    lt(psi_k, psi), (psi - psi_k) / alpha, fem.Constant(msh, 0.))*dx)
H1increment_form = fem.form(
    inner(grad(u - u_k), grad(u - u_k)) * dx + (u - u_k)**2 * dx)
L2increment_form = fem.form((exp(u)-exp(u_k))**2 * dx)
H1primal_error_form = fem.form(
    inner(grad(u - u_exact), grad(u - u_exact)) * dx + (u - u_exact)**2 * dx)
L2primal_error_form = fem.form((u - u_exact)**2 * dx)
L2latent_error_form = fem.form((exp(psi) - u_exact)**2 * dx)
L2multiplier_error_form = fem.form(
    (lambda_exact - ((psi_k - psi) / alpha))**2 * dx)

# Proximal point outer loop
n = 0
increment_k = 0.0
sol.x.array[:] = 0.0
sol_k.x.array[:] = sol.x.array[:]
alpha_k = C

energies = []
complementarities = []
feasibilities = []
dual_feasibilities = []
H1primal_errors = []
L2primal_errors = []
L2latent_errors = []
L2multiplier_errors = []
Newton_steps = []
step_sizes = []
primal_increments = []
latent_increments = []
for k in range(maximum_number_of_outer_loop_iterations):

    # Update step size
    if step_size_rule == "constant":
        alpha.value = C
    elif step_size_rule == "double_exponential":
        alpha.value = max(C*r**(q**k) - alpha_k, C)
        alpha_k = alpha.value
        alpha.value = min(alpha.value, alpha_max)
    else:
        step_size_rule == "geometric"
        alpha.value = C*r**k
    print("OUTER LOOP ", k+1, "   alpha: ", alpha.value)

    # Solve problem
    (n, converged) = NewtonSolver.solve(sol)
    print("Newton steps: ", n, "   Converged: ", converged)

    # Check outer loop convergence
    comm = sol_k.function_space.mesh.comm
    energy = comm.allreduce(fem.assemble_scalar(energy_form), MPI.SUM)
    complementarity = np.abs(comm.allreduce(
        fem.assemble_scalar(complementarity_form), MPI.SUM))
    feasibility = comm.allreduce(
        fem.assemble_scalar(feasibility_form), MPI.SUM)
    dual_feasibility = comm.allreduce(
        fem.assemble_scalar(dual_feasibility_form), MPI.SUM)
    increment = np.sqrt(comm.allreduce(
        fem.assemble_scalar(H1increment_form), MPI.SUM))
    latent_increment = np.sqrt(comm.allreduce(
        fem.assemble_scalar(L2increment_form), MPI.SUM))
    H1primal_error = np.sqrt(comm.allreduce(
        fem.assemble_scalar(H1primal_error_form), MPI.SUM))
    L2primal_error = np.sqrt(comm.allreduce(
        fem.assemble_scalar(L2primal_error_form), MPI.SUM))
    L2latent_error = np.sqrt(comm.allreduce(
        fem.assemble_scalar(L2latent_error_form), MPI.SUM))
    L2multiplier_error = np.sqrt(comm.allreduce(
        fem.assemble_scalar(L2multiplier_error_form), MPI.SUM))

    tol_Newton = increment

    print("‖u - uₕ‖_H¹: ", H1primal_error, "  ‖u - ũₕ‖_L² : ", L2latent_error)

    if increment_k > 0.0:
        print("Increment size: ", increment,
              "   Ratio: ", increment/increment_k)
    else:
        print("Increment size: ", increment)
    print("")

    energies.append(energy)
    complementarities.append(complementarity)
    feasibilities.append(feasibility)
    dual_feasibilities.append(dual_feasibility)
    H1primal_errors.append(H1primal_error)
    L2primal_errors.append(L2primal_error)
    L2latent_errors.append(L2latent_error)
    L2multiplier_errors.append(L2multiplier_error)
    Newton_steps.append(n)
    step_sizes.append(np.copy(alpha.value))
    primal_increments.append(increment)
    latent_increments.append(latent_increment)

    if tol_Newton < tol_exit and n > 0:
        break

    # Reset Newton solver options
    NewtonSolver.atol = 1e-3
    NewtonSolver.rtol = tol_Newton*1e-4

    # Update sol_k with sol_new
    sol_k.x.array[:] = sol.x.array[:]
    increment_k = increment

# Save data
output_dir = './output/'
mkdir_p(output_dir)
df = pd.DataFrame()
df['Energy'] = energies
df['Complementarity'] = complementarities
df['Feasibility'] = feasibilities
df['Dual Feasibility'] = dual_feasibilities
df['H1 Primal errors'] = H1primal_errors
df['L2 Primal errors'] = L2primal_errors
df['L2 Latent errors'] = L2latent_errors
df['L2 Multiplier errors'] = L2multiplier_errors
df['Newton steps'] = Newton_steps
df['Step sizes'] = step_sizes
df['Primal increments'] = primal_increments
df['Latent increments'] = latent_increments
df['Polynomial order'] = [polynomial_order]*(k+1)
df['Mesh size'] = [1/2**(m-1)]*(k+1)
df['dofs'] = [np.size(sol_k.x.array[:])]*(k+1)
df['Step size rule'] = [step_size_rule]*(k+1)
filename = f'./example{example}_polyorder{polynomial_order}_m{m}.csv'
print(f"Saving data to: ", output_dir + filename)
df.to_csv(output_dir + filename, index=False)
print(df)

# Create output space for bubble function
V_out = fem.FunctionSpace(msh, ufl.FiniteElement(
    "Lagrange", msh.ufl_cell(), polynomial_order+2))
u_out = fem.Function(V_out)
u_out.interpolate(sol.sub(0).collapse())

# Export primal solution variable
# Use VTX to capture high-order dofs
with io.VTXWriter(msh.comm, output_dir + "u.bp", [u_out]) as vtx:
    vtx.write(0.)

# Export interpolant of exact solution u
V_alt = fem.FunctionSpace(msh, ("Lagrange", polynomial_order))
q = fem.Function(V_alt)
expr = fem.Expression(u_exact, V_alt.element.interpolation_points())
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "u_exact.bp", [q]) as vtx:
    vtx.write(0.)

# Export interpolant of Lagrange multiplier λ
W_out = fem.FunctionSpace(msh, ("DG", max(1, polynomial_order-1)))
q = fem.Function(W_out)
expr = fem.Expression(lambda_exact, W_out.element.interpolation_points())
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "lambda_exact.bp", [q]) as vtx:
    vtx.write(0.)

# Export latent solution variable
q = fem.Function(W_out)
expr = fem.Expression(sol.sub(1), W_out.element.interpolation_points())
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "psi.bp", [q]) as vtx:
    vtx.write(0.)

# Export feasible discrete solution
exp_psi = exp(sol.sub(1))
expr = fem.Expression(exp_psi, W_out.element.interpolation_points())
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "tilde_u_interp.bp", [q]) as vtx:
    vtx.write(0.)

# Export "Lagrange multiplier"
lam = (sol_k.sub(1) - sol.sub(1)) / alpha.value
expr = fem.Expression(lam, W_out.element.interpolation_points())
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "lambda.bp", [q]) as vtx:
    vtx.write(0.)

# Export alternative "Lagrange multiplier"
strong_laplace = -div(grad(sol.sub(0))) - f
expr = fem.Expression(strong_laplace, W_out.element.interpolation_points())
q = fem.Function(W_out)
q.interpolate(expr)
with io.VTXWriter(msh.comm, output_dir + "laplace.bp", [q]) as vtx:
    vtx.write(0.)
