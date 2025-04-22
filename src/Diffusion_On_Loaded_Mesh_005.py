import os
import dolfinx
from dolfinx import io, fem
from dolfinx.mesh import CellType as dfx_CellType
from dolfinx.fem.petsc import assemble_matrix
import ufl
import basix
from basix import CellType as basix_CellType
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt

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

A = assemble_matrix(a_compiled)  # Assemble matrix
A.assemble()  # Finalize matrix

print(f"Matrix shape: {A.getSize()}")


## Define a time-dependent source term
#def source_term(t):
#    # For simplicity, let's assume a sinusoidal source term
#    return np.sin(t)
#
# Update the source term at each time step
#L = fem.Constant(domain, source_term(t)) * v * ufl.dx

#--------------------------------------------------------------------------------







