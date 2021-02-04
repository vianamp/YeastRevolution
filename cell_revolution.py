#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple script to create a 3D revolution from a 2D binary
image along its longest axis.
author: Matheus Viana, vianamp@gmail.com, 02.03.2021
"""

import argparse
import numpy as np
from scipy import spatial as scispatial
from skimage import filters as skfilters
from aicsimageio import AICSImage, writers
from sklearn import decomposition as skdec

def perfom_3d_revolution(seg, domain):
    '''
    Performs a 3D revolution of a 2D segmentation about its
    longest axis. The longest axis is calculated as the 1st
    principal component of the shape coordinates.
    
    parameters
    ----------
        seg: nd.array
        2D binary image with the shape to be revolved in 3D.
        domain: nd.array
        Empty 3D image that will be used to store the final
        revolved image.
    '''
    # PCA of 2D shape
    y, x = np.where(seg>0)
    z = 0 * x
    xm = x.mean()
    ym = y.mean()
    zm = int(0.5*domain.shape[0]) # Fix this with correct z location
    coords = np.c_[x-xm, y-ym, z]
    pca = skdec.PCA()
    pca = pca.fit(coords)
    eve = pca.components_
    # Consecutive rotations about 1st PC
    R = scispatial.transform.Rotation
    for angle in np.linspace(0,np.pi,180):
        rot_config = angle * eve[0]
        coords_rot = R.from_rotvec(rot_config).apply(coords)
        coords_rot = np.c_[
            coords_rot[:,2] + zm,
            coords_rot[:,1] + ym,
            coords_rot[:,0] + xm
        ]
        coords_rot = coords_rot.astype(np.int)
        # Assign 255 to rotated coordinates
        for z, y, x in coords_rot:
            domain[z,y,x] = 255
    # Median filter to fill in the gaps
    domain = skfilters.median(domain, selem=np.ones((3,3,3)))
    return domain

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple script for 3D revolution of 2D segmentations.')
    parser.add_argument('--bf', dest='path_bf', type=str)
    parser.add_argument('--fp', dest='path_fp', type=str)
    parser.add_argument('--save', dest='save_as', type=str)
    parser.add_argument('--export', dest='export', action='store_true', default=False)
    args = parser.parse_args()

    #Load data
    mem = AICSImage(args.path_bf).data.squeeze()
    img = AICSImage(args.path_fp).data.squeeze()
    img = (255*(img>0)).astype(np.uint8)
    mask = np.zeros(img.shape[1:], dtype=mem.dtype)
    print(mem.shape, mask.shape, img.shape)

    # Perform revolution
    mem = perfom_3d_revolution(seg=mem, domain=mask)
    img = np.concatenate([mem.reshape(1,*mem.shape),img], axis=0)
    
    # Save results
    with writers.ome_tiff_writer.OmeTiffWriter(args.save_as, overwrite_file=True) as writer:
        writer.save(
            img,
            dimension_order = 'CZYX'
        )
    if args.export:
        for ch in range(3):
            with writers.ome_tiff_writer.OmeTiffWriter(f'{args.save_as}-{ch}.tif', overwrite_file=True) as writer:
                writer.save(
                    img[ch],
                    dimension_order = 'ZYX'
                )