
# From https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html
# DOLFIN is the user interface of the FEniCS project
import os
import matplotlib as mpl
import pyvista
import numpy as np

import ufl
from petsc4py import PETSc
from mpi4py import MPI

import meshio
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc


################################################################################
# Read the mesh from the XDMF file and capillary locations from NPZ file
xdmf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "Capillary_Locations", "Brain_Geom_A22-313_vessels_selectedregion_3.csv_region_1_mesh.xdmf")

npz_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "Capillary_Locations", "Brain_Geom_A22-313_vessels_selectedregion_3.csv_region_1_points.npz")

print(f"Loading triangle mesh from XDMF file {xdmf_file}")
with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
print('Done.')

print(f"Loading points from NPZ file {npz_file}")
data = np.load(npz_file)
source_coords = data["coords"]
capillary_centers = source_coords[:,0:1]

################################################################################
# Setup

V = fem.functionspace(domain, ("Lagrange", 1))

# Define temporal parameters
t = 0  # Start time
T = 2.0  # Final time
num_steps = 100
dt = T / num_steps  # time step size

# Create initial condition

def initial_condition(V, sigma=50.0, amplitude=1.0):
    """
    Create an initial condition as a sum of Gaussians centered at capillary_centers.

    Args:
        V: dolfinx fem.FunctionSpace
        sigma: float, standard deviation of each Gaussian
        amplitude: float, peak value of each Gaussian

    Returns:
        fem.Function defined on V
    """
    x = ufl.SpatialCoordinate(V.mesh)
    u_expr = 0

    for c in capillary_centers:
        dx = x[0] - c[0]
        dy = x[1] - c[1]
        u_expr += amplitude * ufl.exp(-(dx*dx + dy*dy)/(sigma**2))

    u0 = fem.Function(V)
    u0.interpolate(fem.Expression(u_expr, V.element.interpolation_points()))
    return u0



print("Setting up initial conditions...")
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)
print("Done.")

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)


# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)


u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

f = fem.Constant(domain, PETSc.ScalarType(0.1))

D = 0.2  # Diffusion coefficient
a = u * v * ufl.dx + dt * D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

#a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx


bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)


solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))






plotter = pyvista.Plotter()

IMGOUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../img")

plotter.open_gif(os.path.join(IMGOUTDIR,"u_time.gif"), fps=10)

grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
#                             cmap=viridis, scalar_bar_args=sargs,
#                             clim=[0, max(uh.x.array)])
renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                             cmap=viridis, scalar_bar_args=sargs,
                             clim=[np.min(uh.x.array), np.max(uh.x.array)])

plotter.view_xy()
time_text = plotter.add_text(f'Time Step: {t:.2f}', position='upper_left', font_size=20, color='black')

print("Running FEM....")
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
    # Update plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

    # Update the text label for the current time step
    time_text.set_text(position='upper_left', text=f'Time Step: {t:.4f}')


plotter.close()
xdmf.close()
print("Done!")
