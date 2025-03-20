import gmsh
import numpy as np
import os
import meshio

# Initialize GMSH
gmsh.initialize()
gmsh.model.add("brain_with_holes")

mesh_size = 0.1  # Larger values produce a coarser mesh, smaller values produce a finer mesh

# Define the rectangle
x,y,z = 0.0, 0.0, 0.0
length = 10.0
width = 10.0

rectangle = gmsh.model.occ.addRectangle(x, y, z, width, length)
gmsh.model.occ.synchronize()

# Define the holes (circles)
hole_radius = 0.05

lower_bound = 0.4
upper_bound = 9.6

random_centers = np.round(np.random.rand(100, 2) * (upper_bound - lower_bound) + lower_bound, 4)

for c in random_centers:
    print(c)
    gmsh.model.occ.addPoint(c[0], c[1], z)
    h_surface = gmsh.model.occ.addDisk(c[0], c[1], z, hole_radius, hole_radius)
    gmsh.model.occ.cut([(2, rectangle)], [(2, h_surface)])
gmsh.model.occ.synchronize()

# Set mesh size
points = gmsh.model.getEntities(dim=0)  # Get points (dim=0 for points)
# Set  mesh size for each corner point
for point in points:
    gmsh.model.mesh.setSize([point], mesh_size)

gmsh.model.occ.synchronize()

# (Optional) Create the physical groups for the surfaces to be meshed
gmsh.model.addPhysicalGroup(2, [rectangle])
gmsh.model.setPhysicalName(2, 1, "Brain")


gmsh.model.mesh.generate(2)

# Save the mesh to a file
FILENAME = "MMD_Brain_Geom.msh"
gmsh.write(FILENAME)

# Finalize the GMSH model
gmsh.finalize()

# Save also as *.xdmf format
m = meshio.read(FILENAME)
meshio.write(os.path.splitext(os.path.basename(FILENAME))[0] + ".xdmf", m)
