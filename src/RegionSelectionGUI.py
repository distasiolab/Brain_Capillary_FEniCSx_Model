import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import PolygonSelector
from shapely.geometry import Point, Polygon
from tkinter import Tk, filedialog

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

# --- Open file dialog to pick the CSV file ---
Tk().withdraw()  # Hide the root Tk window
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected.")
    exit()

# Load your CSV
df = pd.read_csv(file_path)

# Define the columns
x_col = "Centroid X µm"
y_col = "Centroid Y µm"
group_col = "Parent"

# Get X, Y values
points = df[[x_col, y_col]].values

# Assign colors to each unique value in 'Parent'
unique_groups = df[group_col].unique()
colors = plt.cm.get_cmap("tab10", len(unique_groups))  # Categorical colormap
group_to_color = {group: colors(i) for i, group in enumerate(unique_groups)}

# Create color array


# Plot
matplotlib.use("TkAgg")
fig, ax = plt.subplots()
display_df = df.iloc[::10]  # Show 1 in every 10 points
display_points = display_df[[x_col, y_col]].values
point_colors = display_df[group_col].map(group_to_color)
sc = ax.scatter(display_df[x_col], display_df[y_col], c=point_colors, s=10)
ax.set_title(f"Draw polygon to select points (colored by '{group_col}')")

# Legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=group,
               markerfacecolor=group_to_color[group], markersize=6)
    for group in unique_groups
]
ax.legend(handles=handles, title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')

# Selection
selected_points = []

def onselect(verts):
    global selected_points
    poly = Polygon(verts)
    selected_points = [
        i for i, (x, y) in enumerate(points) if poly.contains(Point(x, y))
    ]
    plt.close(fig)

selector = PolygonSelector(ax, onselect)

plt.tight_layout()
plt.show()

# Export
if selected_points:
    selected_df = df.iloc[selected_points]
    selected_df.to_csv("selected_points.csv", index=False)
    print(f"✅ Exported {len(selected_df)} selected points to 'selected_points.csv'")
else:
    print("⚠️ No points selected.")
