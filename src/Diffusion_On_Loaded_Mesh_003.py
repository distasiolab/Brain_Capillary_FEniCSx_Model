
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
# Read the mesh from the XDMF file
xdmf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "Capillary_Locations", "Brain_Geom_A22-313_vessels_selectedregion_3.csv_region_1.xdmf")

print(f"Loading mesh from XDMF file {xdmf_file}")
with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
print('Done.')
V = fem.functionspace(domain, ("Lagrange", 1))
################################################################################


# Define temporal parameters
t = 0  # Start time
T = 2.0  # Final time
num_steps = 100
dt = T / num_steps  # time step size

#--------------------------------------------------------------------------------
print("Processing capillary locatons...")
# Process capillary locations from mesh
points = mesh.points
field_data = mesh.field_data
# Get vertex cells (these are the point elements)
vertex_cells_list = []
vertex_tags_list = []
for i, cell_block in enumerate(mesh.cells):
    if cell_block.type == "vertex":
        vertex_cells_list.append(cell_block.data)
        vertex_tags_list.append(mesh.cell_data["gmsh:physical"][i])

if len(vertex_cells_list) == 0 or len(vertex_tags_list) == 0:
    print("No vertex cells with physical tags found. Exiting.")
    sys.exit(0)
else:
    vertex_cells = np.vstack(vertex_cells_list)
    vertex_tags = np.concatenate(vertex_tags_list)
    
    # Filter physical groups ending with '_capillary' and dimension 0
    capillary_tags = {name: tag_dim[0] for name, tag_dim in field_data.items()
                      if name.endswith("_capillary") and tag_dim[1] == 0}
    # Build tag → point index list
    tag_to_points = {name: vertex_cells[vertex_tags == int(tag)].flatten()
                     for name, tag in capillary_tags.items()}
    
    # Filter only point-based physical names that end in '_capillary'
    capillary_labels = {name: tag_dim[0] for name, tag_dim in mesh.field_data.items()
                        if name.endswith("_capillary") and tag_dim[1] == 0}
    centers_x = []
    centers_y = []
    for idx, (name, pt_indices) in enumerate(tag_to_points.items()):
        if len(pt_indices) == 0:
            continue
        points[pt_indices, 0]

print('Done')

# Create initial condition
def initial_condition(x, sigma=0.1, magnitude=0.1):
    x = np.array(x)
    centers = np.stack([centers_x, centers_y], axis=-1)
    x_diff = centers - x  # Difference from the point to each center

    # Handle element-wise sigma and amplitude
    sigma = np.asarray(sigma)
    amplitude = np.asarray(magnitude)
    if sigma.ndim == 0:
        sigma = np.full(len(centers_x), sigma)
    if amplitude.ndim == 0:
        amplitude = np.full(len(centers_x), amplitude)

    distances_squared = np.sum(x_diff**2, axis=1)
    gaussians = amplitude * np.exp(-distances_squared / (2 * sigma**2))

    return np.sum(gaussians)

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
