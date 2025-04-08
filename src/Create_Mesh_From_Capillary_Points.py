import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import gmsh
import meshio

FILEPATHBASE = '/Users/mmd47/Dropbox-YaleUniversity/Marcello DiStasio/DiStasio Lab/DiStasio Lab Share/02 Analysis/Brain_Oxygen_Simulation/Brain_Capillary_FEniCSx_Model'

datadir = os.path.join(FILEPATHBASE,'data','Capillary_Locations')

vessels_files = glob.glob(os.path.join(datadir, '*vessels.csv'))


Samples = {}
for v in vessels_files:
    SampleName = os.path.split(v)[1].split('_')[0] 

    csv_in = pd.read_csv(v)
    
    Samples[SampleName] = csv_in.filter(like='Centroid').join(csv_in.filter(like='Parent'))

Region_Meshes = {}


# Define the holes (circles) in microns
hole_radius = 10

mesh_size = 1  # Larger values produce a coarser mesh, smaller values produce a finer mesh

for s in Samples.keys():

    filtered_values = Samples[s]['Parent'].str.extract(r'(Layer\d+)')[0].dropna().unique()
    unique_layers = filtered_values.tolist()

    for layer in unique_layers:

        filtered_df = Samples[s][Samples[s]['Parent'].str.match(f'{layer}.*')]
        X =  filtered_df['Centroid X µm'].tolist()
        Y =  filtered_df['Centroid Y µm'].tolist()

        # Combine X and Y into a single list of coordinates
        capillary_centers = np.array(list(zip(X, Y)))

        # Compute the Convex Hull
        hull = ConvexHull(capillary_centers)

        # Extract the boundary points (indices of points in the hull)
        boundary_points = capillary_centers[hull.vertices]

        internal_capillaries = np.array([point for i, point in enumerate(capillary_centers) if i not in hull.vertices])

        #---------------------------------------------------------------------------------
        # Create the points in GMSH (2D)
        gmsh.initialize()
    
        # Set the model name
        model_name = "boundary_model_2d"
        gmsh.model.add(model_name)
    
        point_tags = []
        for i, (x, y) in enumerate(boundary_points):
            tag = gmsh.model.occ.addPoint(x, y, 0)  # 0 for Z-coordinate (2D)
            point_tags.append(tag)
    
        # Create lines between the points to form the boundary (polygon)
        line_tags = []
        for i in range(len(boundary_points)):
            start_point = point_tags[i]
            end_point = point_tags[(i + 1) % len(boundary_points)]  # Connect last point to the first one
            line_tag = gmsh.model.occ.addLine(start_point, end_point)
            line_tags.append(line_tag)
    
        # Create a line loop to define the boundary (polygon)
        line_loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
    
    
        surface_tag = gmsh.model.occ.addPlaneSurface([line_loop_tag])
    
        gmsh.model.occ.synchronize()

        n = 1
        for c in internal_capillaries:
            print(f"Adding capillary {n} of {np.shape(internal_capillaries)[0]} for {s}, {layer}.")
            gmsh.model.occ.addPoint(c[0], c[1], 0)
            h_surface = gmsh.model.occ.addDisk(c[0], c[1], 0, hole_radius, hole_radius)
            gmsh.model.occ.cut([(2, surface_tag)], [(2, h_surface)])
            n=n+1
        gmsh.model.occ.synchronize()
    
        
        # Set mesh size
        points = gmsh.model.getEntities(dim=0)  # Get points (dim=0 for points)
        # Set  mesh size for each corner point
        for point in points:
            gmsh.model.mesh.setSize([point], mesh_size)
    
        gmsh.model.occ.synchronize()
    
        # (Optional) Create the physical groups for the surfaces to be meshed
        gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, 1, "Brain")
    
        gmsh.model.mesh.generate(2)
    
        FILEPATH = os.path.join(datadir, f"Brain_Geom_{s}_{layer}")
        gmsh.write(FILEPATH+".msh")
        # Finalize the GMSH model
        gmsh.finalize()
        print("Wrote: " + FILEPATH+".msh")
    
        # Also save as *.xdmf format
        m = meshio.read(FILEPATH+".msh")
        meshio.write(FILEPATH + ".xdmf", m)

        
