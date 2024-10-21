import numpy as np
import pyvista as pv
from scipy.io import loadmat
        
# Load the MAT file
mat_data = loadmat('output/wrld_info.mat')

points = mat_data['wrld']["xyz"][0][0]
colors = mat_data['wrld']["color"][0][0]

cloud = pv.PolyData(points)
cloud.point_data['colors'] = colors

# Create a plotter
plotter = pv.Plotter()

# Add the point cloud to the plotter
plotter.add_points(cloud,rgb=True,  point_size=1.5)
plotter.enable_point_picking(callback=lambda point: print(point))

# Add a triaxis in the origin of the referencial
#plotter.add_axes_at_origin(line_width=1)
#plotter.add_bounding_box()

# Show the plot
plotter.show()