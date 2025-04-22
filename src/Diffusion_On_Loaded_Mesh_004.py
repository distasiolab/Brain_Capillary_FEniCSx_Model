import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, assemble_matrix, assemble_vector, apply_lifting, set_bc
import ufl
from petsc4py import PETSc
from dolfinx import fem, la
from ufl import TrialFunction, TestFunction, dx, dot, grad


import numpy as np
from dolfinx.geometry import BoundingBoxTree, compute_collisions_points
from dolfinx.geometry import compute_reference_coordinates
from dolfinx.fem import Function

class PointController:
    def __init__(self, mesh, V, control_points, tol=1e-10):
        self.mesh = mesh
        self.V = V
        self.control_points = np.array(control_points, dtype=np.float64)
        self.bb_tree = BoundingBoxTree(mesh, mesh.topology.dim)
        self.tol = tol

        # Store point-cell-refcoord-dofs-weights
        self.control_data = []

        # Loop through control points
        for pt in self.control_points:
            pt = np.array(pt, dtype=np.float64).reshape(-1, 1)

            # Step 1: find candidate cells
            cells = compute_collisions_points(self.bb_tree, pt)

            found = False
            for cell in cells:
                try:
                    # Step 2: compute reference coordinates
                    ref_coords, success = compute_reference_coordinates(
                        mesh, cell, pt, self.tol)
                    if not success:
                        continue

                    ref_coords = ref_coords[0]

                    # Get DOFs of the cell
                    dofs = self.V.dofmap.cell_dofs(cell)

                    # Get shape function weights
                    basix_el = self.V.element.basix_element
                    weights = basix_el.tabulate(0, [ref_coords])[0, 0, :]

                    self.control_data.append((pt.ravel(), cell, dofs, weights))
                    found = True
                    break
                except Exception:
                    continue

            if not found:
                raise RuntimeError(f"Control point {pt.ravel()} could not be located in any cell.")

    def apply(self, f: Function, value_fn):
        f.x.array[:] = 0.0
        for pt, cell, dofs, weights in self.control_data:
            value = value_fn(pt)
            f.x.array[dofs] += value * weights
        f.x.scatter_forward()



# --- Main solver setup ---

# Create mesh and function space
mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)
V = FunctionSpace(mesh, ("CG", 1))

# Define trial/test functions
u = TrialFunction(V)
v = TestFunction(V)

# Time step and total time
dt = 0.01
T = 2.0
num_steps = int(T / dt)

# Define functions
u_n = Function(V)       # solution at previous time
u_new = Function(V)     # solution at current time
bc_func = Function(V)   # function to hold control values

# Control points
control_points = [
    [0.3, 0.3],
    [0.6, 0.7],
    [0.2, 0.9],
]
controller = PointController(mesh, V, control_points)

# Time-varying value
def time_control_value(pt, t):
    return np.sin(2 * np.pi * t)

# Variational forms
a = form(u * v * dx + dt * dot(grad(u), grad(v)) * dx)
L = form(u_n * v * dx)

# Assemble matrix once (it's constant in time)
A = assemble_matrix(a)
A.assemble()

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.HYPRE)

# Time-stepping loop
t = 0.0
for step in range(num_steps):
    t += dt
    print(f"Time step {step+1}/{num_steps}, t = {t:.3f}")

    # Set control values in bc_func
    controller.apply(bc_func, lambda pt: time_control_value(pt, t))

    # Apply control values directly to u_new before solving
    controller.apply(u_new, lambda pt: time_control_value(pt, t))

    # Assemble RHS
    b = assemble_vector(L)

    # No classical boundary conditions, but apply lifting for consistency
    apply_lifting(b, [a], [[bc_func]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc_func])

    # Solve
    solver.solve(b, u_new.vector)
    u_new.x.scatter_forward()

    # Update previous solution
    u_n.x.array[:] = u_new.x.array

# Optional: Write result
try:
    from dolfinx.io import XDMFFile
    with XDMFFile(mesh.comm, "solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_new, t)
except ImportError:
    print("XDMF output skipped (missing dependency)")




