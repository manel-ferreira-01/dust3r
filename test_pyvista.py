import numpy as np
import pyvista as pv
        
# Load the CSV file
data = np.loadtxt('output/out.csv', delimiter=',')

cloud = pv.PolyData( data[:, :3] )
cloud.point_data['colors'] = data[:, 3:6]

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