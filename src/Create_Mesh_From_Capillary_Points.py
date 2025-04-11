import sys
import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from alphashape_mmd import alphashape
import gmsh
import meshio

##################################################
# Base file path definition
##################################################
# Define the base path relative to which data is loaded and saved
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--basepath', type=str, help='Path to base directory for the project; should contain directories \'data\' and \'calc\'')
args = parser.parse_args()

if args.basepath:
    FILEPATHBASE = args.basepath
else:
    # Parent directory of this file
    FILEPATHBASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')

datadir = os.path.join(FILEPATHBASE,'data','Capillary_Locations')
vessels_files = glob.glob(os.path.join(datadir, '*vessels.csv'))


print(f"Loading capillary locations from files in directory {datadir}")
Samples = {}
for v in vessels_files:
    SampleName = os.path.split(v)[1].split('_')[0] 
    csv_in = pd.read_csv(v)
    Samples[SampleName] = csv_in.filter(like='Centroid').join(csv_in.filter(like='Parent'))

Region_Meshes = {}


# Define the size of a capillary lumen ('hole') (circles) in microns
hole_radius = 5

for s in Samples.keys():

    print(f"Loading data for sample {s} ...")
    
    filtered_values = Samples[s]['Parent'].str.extract(r'(Layer\d+)')[0].dropna().unique()
    unique_layers = filtered_values.tolist()

    Samples[s] = Samples[s][Samples[s]['Parent'].str.match(f'Layer\d+')].copy()

    X_whole = Samples[s]['Centroid X µm'].tolist()
    Y_whole = Samples[s]['Centroid Y µm'].tolist()

    capillary_centers_all = np.array(list(zip(X_whole, Y_whole)))
    
    # Compute the boundary
    boundary_alpha = 0.00792
    alpha_shape = alphashape(capillary_centers_all, alpha=boundary_alpha)

    # If the shape is a Polygon, get the exterior coords
    if isinstance(alpha_shape, Polygon):
        regions = [np.array(list(alpha_shape.exterior.coords))[1:,:]]
    elif isinstance(alpha_shape, MultiPolygon):
        regions = [np.array(list(p.exterior.coords))[1:,:] for p in alpha_shape.geoms]
    print(f'Found {len(regions)} regions.')

    region_number = 1
    for boundary_points in regions:

        #---------------------------------------------------------------------------------
        # Create the points in GMSH (2D)
        gmsh.initialize()
        
        # Set the model name
        model_name = "boundary_model_2d"
        gmsh.model.add(model_name)
        
        point_tags = []
        for i, (x, y) in enumerate(boundary_points):
            tag = gmsh.model.occ.addPoint(float(x), float(y), 0)  # 0 for Z-coordinate (2D)
            point_tags.append(tag)
        gmsh.model.occ.synchronize()

        # Create lines between the points to form the boundary (polygon)
        line_tags = []
        for i in range(len(boundary_points)):
            start_point = point_tags[i]
            end_point = point_tags[(i + 1) % len(boundary_points)]  # Connect last point to the first one
            line_tag = gmsh.model.occ.addLine(start_point, end_point)
            line_tags.append(line_tag)
            gmsh.model.occ.synchronize()
        
        # Create a line loop to define the boundary (polygon)
        line_loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
        gmsh.model.occ.synchronize()
        surface_tag = gmsh.model.occ.addPlaneSurface([line_loop_tag])
        gmsh.model.occ.synchronize()
        
        surface_physical_group_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, surface_physical_group_tag, "BrainCortex")
        gmsh.model.occ.synchronize()
        
        for layer in unique_layers:
            
            filtered_df = Samples[s][Samples[s]['Parent'].str.match(f'{layer}.*')]
            X =  filtered_df['Centroid X µm'].tolist()
            Y =  filtered_df['Centroid Y µm'].tolist()
            
            # Combine X and Y into a single list of coordinates
            capillary_centers = np.array(list(zip(X, Y)))
            internal_capillaries = np.array([point for i, point in enumerate(capillary_centers) if i not in boundary_points])
            
            # Compute the Euclidean distance between each candidate point and all boundary points and only keep those further than threshold
            distances = np.linalg.norm(boundary_points[:, np.newaxis] - internal_capillaries, axis=2)
            threshold_distance = 2*hole_radius
            is_greater_than_threshold = np.all(distances > threshold_distance, axis=0)
            internal_capillaries = internal_capillaries[is_greater_than_threshold]
            
            n = 1
            cap_points = []
            for c in internal_capillaries:
                if (n % 1000) == 0:
                    print(f"Adding capillary {n} of {np.shape(internal_capillaries)[0]} for {s}, {layer}, region {region_number}.")
                cp = gmsh.model.occ.addPoint(c[0], c[1], 0)
                cap_points.append(cp)            
                #h_surface = gmsh.model.occ.addDisk(c[0], c[1], 0, hole_radius, hole_radius)
                #gmsh.model.occ.cut([(2, surface_tag)], [(2, h_surface)])
                n=n+1
            gmsh.model.occ.synchronize()
            

            mesh_size = 1 #in microns
            #points = gmsh.model.getEntities(dim=0)  # Get points (dim=0 for points)
            ## Set  mesh size for each corner point
            for point in cap_points:
                gmsh.model.mesh.setSize([(0, point)], mesh_size)
            
            gmsh.model.occ.synchronize()
            #----------------------------------------------------------------------

            #----------------------------------------------------------------------
            # Create the physical groups for the points to be meshed
            physical_group_tag = gmsh.model.addPhysicalGroup(0, cap_points)
            gmsh.model.setPhysicalName(0, physical_group_tag, f"{layer}_capillary")
            #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # Set mesh size
        #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", (np.max(X)-np.min(X))/100)  # Maximum element size
        #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3) #in microns

        #----------------------------------------------------------------------
        # Generate mesh
        gmsh.model.mesh.generate(2)
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Output
        FILEPATH = os.path.join(datadir, f"Brain_Geom_{s}_region_{region_number}")
        gmsh.write(FILEPATH+".msh")
        # Finalize the GMSH model
        gmsh.finalize()
        print("Wrote: " + FILEPATH+".msh")
    
        # Also save as *.xdmf format
        m = meshio.read(FILEPATH+".msh")
        meshio.write(FILEPATH + ".xdmf", m)
        print("Wrote: " + FILEPATH+".xdmf")
        #----------------------------------------------------------------------

        region_number = region_number + 1
        
