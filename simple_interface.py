import argparse
import numpy as np
from dust3r.utils.device import to_numpy
from dust3r.demo import get_reconstructed_scene
from dust3r.model import AsymmetricCroCo3DStereo
import os
from scipy.io import savemat

# Parse arguments
parser = argparse.ArgumentParser(description='Dust3r Inference')
parser.add_argument('--outdir', type=str, default='./output', help='Output directory e.g. ./output')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
parser.add_argument('--filelist', type=str, default='./images_in', help='Path to the filelist containing input images, e.g  ./images_in')
args = parser.parse_args()

outdir = args.outdir
device = args.device
filelist = args.filelist

# TODO: THIS NEEDS TO BE LAUNCHED AS SOON AS THE CONTAINER STARTS
model = AsymmetricCroCo3DStereo.from_pretrained("./docker/files/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").to(device)

# Get the 3D model from the scene - one function, full pipeline
scene, pts3d, rgbimgs, cams2world, confs, depths = get_reconstructed_scene(outdir, model, filelist=filelist, device=device)

# Save the output to a CSV file
mask = to_numpy(scene.get_masks())
pts3d = to_numpy(pts3d)

pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
color = np.concatenate([p[m] for p, m in zip(rgbimgs, mask)])
conf = np.concatenate([p[m] for p, m in zip(confs, mask)])

# create a dict about world info
wrld_info = {
    'wrld': {
        'xyz': pts,
        'color': color,
        'conf': conf, 
    }
}
savemat(outdir + '/wrld_info.mat', wrld_info)

cell_array = np.empty((len(cams2world), 1), dtype=object)
for i in range(0,cell_array.shape[0]):
    cell_array[i,0] = {
        'extrinsics': to_numpy(cams2world[i]),
        'intrinsics': to_numpy(scene.get_focals().cpu()[i]),
        'rgb': to_numpy(rgbimgs[i]),
        'depth': depths[i],
        'conf': confs[i]
    }
data = {
    'cams_info': cell_array  # Saving the cell array as 'myCellArray'
}
savemat(outdir + '/cam_info.mat', data)


""" np.savetxt(outdir+"/out.csv",np.hstack((pts, color, conf)), delimiter=",")
for i in range(0,len(cams2world)):
    np.savetxt(outdir+"/"+str(i)+".csv", to_numpy(cams2world[i])) """