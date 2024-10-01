import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy import loadmat, savemat
import io
from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl

from dust3r.demo import get_reconstructed_scene, get_3D_model_from_scene
from dust3r.model import AsymmetricCroCo3DStereo


outdir = "output"
device = "cpu"
filelist = "./images_in"

def GRPC_Interface(grpcMessage_matFile):

    dados = loadmat(io.BytesIO(grpcMessage_matFile)) 

    ##TODO: make these the default ones inside the function
    silent = False
    image_size = 512
    schedule = "linear" # or "cosine"
    niter = 300 # number of iters
    min_conf_thr = 3
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.05
    scenegraph_type = "complete"
    winsize = 1
    refid = 0

    # TODO: THIS NEEDS TO BE LAUNCHED AS SOON AS THE CONTAINER STARTS
    model = AsymmetricCroCo3DStereo.from_pretrained("./docker/files/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").to(device)

    # Get the 3D model from the scene - one function, full pipeline
    scene, pts3d, rgbimg, cams2world, confs = get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                                as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                scenegraph_type, winsize, refid)

    ## Save the output to a CSV file
    mask = to_numpy(scene.get_masks())
    pts3d = to_numpy(pts3d)

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    color = np.concatenate([p[m] for p, m in zip(rgbimg, mask)])
    conf = np.concatenate([p[m] for p, m in zip(confs, mask)])

    np.savetxt("./output/out.csv",np.hstack((pts, color, conf)), delimiter=",")

    for i in range(0,len(cams2world)):
        np.savetxt("./output/"+str(i)+".csv", to_numpy(cams2world[i]))


