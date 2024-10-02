import argparse
import numpy as np

from dust3r.utils.device import to_numpy

from dust3r.demo import get_reconstructed_scene
from dust3r.model import AsymmetricCroCo3DStereo


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
scene, pts3d, rgbimg, cams2world, confs = get_reconstructed_scene(outdir, model, filelist=filelist, device=device)


# Save the output to a CSV file
mask = to_numpy(scene.get_masks())
pts3d = to_numpy(pts3d)

pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
color = np.concatenate([p[m] for p, m in zip(rgbimg, mask)])
conf = np.concatenate([p[m] for p, m in zip(confs, mask)])

np.savetxt(outdir+"/out.csv",np.hstack((pts, color, conf)), delimiter=",")
for i in range(0,len(cams2world)):
    np.savetxt(outdir+"/"+str(i)+".csv", to_numpy(cams2world[i]))


