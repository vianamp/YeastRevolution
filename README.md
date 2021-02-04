# YeastRevolution
Simple script to create a 3D revolution from a 2D object about its longest axis

# To run:

`python cell_revolution.py --bf 01_002b-1_BF.tif --fp 01_002b-1_GFP+DAPI.tif --save merged3d.tif`

where:

`-bf` specifies the path to the bright filed image that was segmented and contains the binary object.

`-fp` specifies the path to the two-channels fp image that contains nucleus segmentation and structure segmentation in this order.
