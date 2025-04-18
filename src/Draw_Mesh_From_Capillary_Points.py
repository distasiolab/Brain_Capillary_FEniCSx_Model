import os
import re
import glob
import meshio
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np

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

mesh_files = glob.glob(os.path.join(datadir, '*.msh'))

for v in mesh_files:
    SampleName = os.path.split(v)[1].split('.')[0]

    print(f'Drawing map for sample {SampleName} from file {v}...')
    # Load the .msh file
    mesh = meshio.read(os.path.join(datadir, v))
    points = mesh.points
    
    field_data = mesh.field_data
    
    # Get vertex cells (these are the point elements)

    vertex_cells_list = []
    vertex_tags_list = []

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == "vertex":
            #print("Vertex cells:", cell_block.data.shape)
            #print("Vertex tags:", mesh.cell_data['gmsh:physical'][i])
            vertex_cells_list.append(cell_block.data)
            vertex_tags_list.append(mesh.cell_data["gmsh:physical"][i])

    vertex_cells = np.vstack(vertex_cells_list)
    vertex_tags = np.concatenate(vertex_tags_list)

    if len(vertex_cells_list) == 0 or len(vertex_tags_list) == 0:
        raise ValueError("No vertex cells with physical tags found.")

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
    FILEPATH = os.path.join(datadir, f"{SampleName}.png")
    plt.savefig(FILEPATH, dpi=600, bbox_inches='tight')
    print('Done!')
