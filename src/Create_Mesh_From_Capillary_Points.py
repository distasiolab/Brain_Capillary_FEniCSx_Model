import sys
import os
import glob

import pandas as pd
import numpy as np

from shapely.geometry import Polygon, MultiPolygon, Point
from alphashape_mmd import alphashape

import gmsh
import meshio

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D




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
vessels_files = glob.glob(os.path.join(datadir, '*vessels*.csv'))


print(f"Loading capillary locations from files in directory {datadir}")
Samples = {}
for v in vessels_files:
    SampleName = os.path.split(v)[1]
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
        gmsh.model.mesh.setRecombine(2, surface_tag, False)  # Explicitly forbid recombination
        gmsh.model.occ.synchronize()
        
        surface_physical_group_tag = gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, surface_physical_group_tag, "BrainCortex")
        gmsh.model.occ.synchronize()

        boundary_polygon = Polygon(boundary_points)
        
        for layer in unique_layers:
            
            filtered_df = Samples[s][Samples[s]['Parent'].str.match(f'{layer}.*')]
            X =  filtered_df['Centroid X µm'].tolist()
            Y =  filtered_df['Centroid Y µm'].tolist()
            
            # Combine X and Y into a single list of coordinates
            capillary_centers = np.array(list(zip(X, Y)))

            internal_capillaries = np.array([
                pt for pt in capillary_centers if boundary_polygon.contains(Point(pt))
            ])

            
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
            
            ## Set  mesh size for each corner point
            mesh_size = 1 #in microns
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
        gmsh.option.setNumber("Mesh.RecombineAll", 0)         # Make sure no quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", -1)  
        gmsh.option.setNumber("Mesh.Algorithm", 6)            # Frontal-Delaunay (triangle-favoring)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0) # Do not use the Blossom recombination algorithm, which results in quads, which dolfinx doesn't like
        gmsh.model.mesh.generate(2)
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Output
        FILEPATH = os.path.join(datadir, f"Brain_Geom_{s}_region_{region_number}")
        gmsh.write(FILEPATH+".msh")
        print("Wrote: " + FILEPATH+".msh")

        # Finalize the GMSH model
        gmsh.finalize()
        
        #----------------------------------------------------------------------
        # Also save as *.xdmf format
        
        mesh = meshio.read(FILEPATH+".msh")

        # Keep only triangle and vertex cells
        filtered_cells = []
        filtered_cell_data = []
        for cell_block, data in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
            if cell_block.type in ["triangle", "vertex"]:
                filtered_cells.append(cell_block)
                filtered_cell_data.append(data)
            else:
                print(f"Found a cell with type {cell_block.type}")

        mesh.cells = filtered_cells
        mesh.cell_data = {"gmsh:physical": filtered_cell_data}

        for cell_block, data in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
            if cell_block.type in ["triangle", "vertex"]:
                pass
            else:
                print(f"After filtering, found a cell with type {cell_block.type}")

                
        meshio.write(FILEPATH + ".xdmf", mesh)
        print("Wrote: " + FILEPATH+".xdmf")

        #----------------------------------------
        #And draw a PNG imge
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
            print("No vertex cells with physical tags found.")

        else:
            vertex_cells = np.vstack(vertex_cells_list)
            vertex_tags = np.concatenate(vertex_tags_list)


            # Filter physical groups ending with '_capillary' and dimension 0
            capillary_tags = {name: tag_dim[0] for name, tag_dim in field_data.items()
                              if name.endswith("_capillary") and tag_dim[1] == 0}
            # Build tag → point index list
            tag_to_points = {name: vertex_cells[vertex_tags == int(tag)].flatten()
                             for name, tag in capillary_tags.items()}
    
            # Find triangle elements (usually under 'triangle' or 'triangle3')
            #triangles = None
            #for cell_block in mesh.cells:
            #    if cell_block.type == "triangle":
            #        triangles = cell_block.data
            #        break
            #if triangles is None:
            #    raise ValueError("No triangular elements found in the mesh.")
            
            # Filter only point-based physical names that end in '_capillary'
            capillary_labels = {name: tag_dim[0] for name, tag_dim in mesh.field_data.items()
                                if name.endswith("_capillary") and tag_dim[1] == 0}

    
            # Create a color map
            cmap = plt.get_cmap("tab10", len(capillary_labels))  # or "tab20" if you have many
            
            ## Plot using matplotlib
            #mpl.rcParams['agg.path.chunksize'] = 102
            plt.figure(figsize=(15, 15))
            #plt.triplot(points[:, 0], points[:, 1], triangles, linewidth=0.5)
            
            # Plot each capillary group
            for idx, (name, pt_indices) in enumerate(tag_to_points.items()):
                if len(pt_indices) == 0:
                    continue
                color = cmap(idx)
                plt.plot(points[pt_indices, 0], points[pt_indices, 1],
                     'o', markersize=0.25, color=color, label=name)



            plt.axis('off')
            plt.legend(loc="upper right", fontsize=8)
            plt.gca().set_aspect('equal')

            # Define where to place the scale bar (adjust as needed)
            x0 = 0.05  # fraction of axis width (left-right)
            y0 = 0.05  # fraction of axis height (bottom-top)
            bar_length = 1000  # in data units
            bar_label = "1 cm"
            
            # Get axis limits to compute actual bar position
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            x_start = xlim[0] + x0 * (xlim[1] - xlim[0])
            y_start = ylim[0] + y0 * (ylim[1] - ylim[0])
            
            # Plot the scale bar
            ax.add_line(Line2D([x_start, x_start + bar_length],
                               [y_start, y_start], color='black', linewidth=2))
            
            # Add text label above or below the bar
            ax.text(x_start + bar_length / 2, y_start - 0.01 * (ylim[1] - ylim[0]),
                    bar_label, ha='center', va='top', fontsize=9)
            
            # Save to PNG
            plt.savefig(FILEPATH+".png", dpi=600, bbox_inches='tight')
            
            print("Wrote: " +  FILEPATH+".png")
            print('Done!')
        
        region_number = region_number + 1
