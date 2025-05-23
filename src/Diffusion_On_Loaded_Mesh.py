import sys, os
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

import pyvista
pyvista.start_xvfb()
pyvista.OFF_SCREEN = True
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
capillary_centers = np.hstack(( source_coords[:,0:2], np.zeros((source_coords.shape[0], 1)) ))

############################################################
# MMD TMP
#random_indices = np.random.choice(capillary_centers.shape[0], size=int(np.floor(capillary_centers.shape[0]/10)), replace=False)
#capillary_centers = capillary_centers[random_indices,:]

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
from scipy.spatial import cKDTree

# Get mesh node coordinates
dof_coords = V.tabulate_dof_coordinates()

# Find closest mesh node to each capillary center
tree = cKDTree(dof_coords)
_, dof_indices = tree.query(capillary_centers)
dof_indices = np.unique(dof_indices)  # Remove duplicates

# Use these indices for Dirichlet BCs
dofs = dof_indices

print(f"Number of Dirichlet BC DOFs: {len(dofs)}")


def sinusoid_value(t, amplitude=1.0, frequency=1.0):
    return lambda x: amplitude * np.sin(2 * np.pi * frequency * t) * np.ones(x.shape[1]) + amplitude

def smooth_pulse_value(t, amplitude=1.0, frequency=1.0, flat_fraction=0.75):
    """
    Generalized smooth periodic pulse function.
    
    Parameters
    ----------
    t : float or numpy array
        Time input(s), in seconds.
    amplitude : float
        Height of the bump.
    frequency : float
        Number of cycles per second 
    flat_fraction : float
        Fraction of the period that is flat before the bump (0 < flat_fraction < 1).
    
    Returns
    -------
    f : float or numpy array
        Output of the function at time t.
    """

    period = 1 / frequency
    
    t_mod = np.mod(t, period)  # Make it periodic
    flat_duration = flat_fraction * period
    bump_duration = period - flat_duration
    
    f = np.zeros_like(t_mod)

    # Apply cosine bump for the non-flat part
    mask = t_mod >= flat_duration
    t_bump = t_mod[mask] - flat_duration
    f[mask] = amplitude * 0.5 * (1 - np.cos(2 * np.pi * t_bump / bump_duration))
    
    return f



# At each time step:
t = 0.0
amplitude = 5.0
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
dt = 0.01        # Your time step size
alpha = 10000.0
beta = 8.0  # Define sink rate (positive value)

# UFL constants
dt_ufl = fem.Constant(domain, dt)
alpha_ufl = fem.Constant(domain, alpha)
beta_ufl = fem.Constant(domain, PETSc.ScalarType(beta))


# Variational form for backward Euler
a = (u * v / dt_ufl + alpha_ufl * ufl.dot(ufl.grad(u), ufl.grad(v)) + beta_ufl * u * v) * ufl.dx
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
plotter.view_xy()
plotter.open_movie("heat_equation.mp4", framerate=20)  # Set your desired filename and FPS
plotter.show(auto_close=False)  # Keeps the window open for updates

u_new = fem.Function(V)

num_steps = 200  # Set your number of time steps
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

    plotter.write_frame()

# Optionally, keep the plot open at the end
plotter.close()

print('Complete!')
sys.exit(0)
