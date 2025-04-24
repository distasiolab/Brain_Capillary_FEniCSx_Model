import os
import dolfinx
from dolfinx import io, fem
from dolfinx.mesh import CellType as dfx_CellType
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

import ufl
import basix
from basix import CellType as basix_CellType
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyvista as pv


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

############################################################
# MMD TMP
capillary_centers = capillary_centers[1:10,:]

############################################################

boundary_points = capillary_centers


# Define a finite element function space on the mesh
mesh = domain


cell_type_map = {
    dfx_CellType.triangle: basix_CellType.triangle,
    dfx_CellType.quadrilateral: basix_CellType.quadrilateral,
    dfx_CellType.tetrahedron: basix_CellType.tetrahedron,
    dfx_CellType.hexahedron: basix_CellType.hexahedron,
}
# Get dolfinx cell type and convert to Basix cell type
dfx_cell = mesh.topology.cell_type
basix_cell = cell_type_map[dfx_cell]

# Define element
ufl_element = basix.ufl.element(
    family="Lagrange",
    cell=basix_cell,
    degree=1,
    discontinuous=False)

# Define Function Space
V = dolfinx.fem.functionspace(mesh, ufl_element)




#--------------------------------------------------------------------------------
# Define boundary conditions (dirichelet), as time varying point sources at capillary
# locations
#--------------------------------------------------------------------------------


def point_locator(x):
    # x: shape (gdim, num_points)
    # Return a boolean array indicating which columns are close to any of your points
    return np.any([np.all(np.isclose(x.T, p, atol=1e-8), axis=1) for p in capillary_centers], axis=0)


dofs = locate_dofs_geometrical(V, point_locator)

def sinusoid_value(t, amplitude=1.0, frequency=1.0):
    return lambda x: amplitude * np.sin(2 * np.pi * frequency * t) * np.ones(x.shape[1])


# At each time step:
t = 0.0
amplitude = 1.0
frequency = 1.0

# Create a Function for the BC values
print(f"Creating boundary conditions at points...")
bc_func = fem.Function(V)
bc_func.interpolate(sinusoid_value(t, amplitude, frequency))
print("Done.")

# Create the Dirichlet BC object
bc = fem.dirichletbc(bc_func, dofs)
bcs = [bc]




#--------------------------------------------------------------------------------
# Define function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u_n = fem.Function(V)  # Previous time step

# Parameters
dt = 0.01
alpha = 1.0

# UFL constants
dt_ufl = fem.Constant(domain, dt)
alpha_ufl = fem.Constant(domain, alpha)

# Variational form for backward Euler
a = (u * v / dt_ufl + alpha_ufl * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = (u_n * v / dt_ufl) * ufl.dx

# Compile forms
a_compiled = fem.form(a)
L_compiled = fem.form(L)


print(f"Function space has {V.dofmap.index_map.size_global} degrees of freedom.")
print("Bilinear form (a) created:", a)

A = assemble_matrix(a_compiled, bcs=bcs)  # Assemble matrix
A.assemble()  # Finalize matrix

print(f"Matrix shape: {A.getSize()}")


b = assemble_vector(L_compiled)
apply_lifting(b, [a_compiled], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)




#--------------------------------------------------------------------------------
# Run the simulation

# Create the VTK mesh for PyVista
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Initial data
u_values = u_n.x.array.real.ravel()  # u_n is the solution Function
grid.point_data["u"] = u_values
grid.set_active_scalars("u")


# Create plotter
plotter = pv.Plotter()
mesh_actor = plotter.add_mesh(grid, clim=[u_values.min(), u_values.max()])
plotter.show(auto_close=False)  # Keeps the window open for updates


u_new = fem.Function(V)

import time  # For optional pause

num_steps = 100  # Set your number of time steps
dt = 0.01        # Your time step size
t = 0.0

for step in range(num_steps):

    print(f"Time: {t}")
    # Update time
    t += dt

    # Update time-dependent BCs if needed
    bc_func.interpolate(sinusoid_value(t, amplitude, frequency))
    bc = fem.dirichletbc(bc_func, dofs)
    bcs = [bc]
    # Assemble right-hand side
    b = assemble_vector(L_compiled)
    apply_lifting(b, [a_compiled], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    # Solve linear system
    solver = PETSc.KSP().create(A.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.solve(b, u_new.x.petsc_vec)
    u_new.x.scatter_forward()
    # Update previous solution for next step
    u_n.x.array[:] = u_new.x.array

    # Update PyVista plot
    u_values = u_new.x.array.real.ravel()  # Ensure 1D
    grid.point_data["u"] = u_values
    plotter.update_scalars(u_values, mesh=grid, render=True)
    plotter.add_text(f"time: {t:.3f}", font_size=12, name="timelabel")

    # Optional: slow down the animation
    time.sleep(0.05)

# Optionally, keep the plot open at the end
plotter.show(auto_close=False)
