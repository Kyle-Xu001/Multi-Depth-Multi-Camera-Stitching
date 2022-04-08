# Multi-Depth-Multi-Camera-Stitching

## Project Description

This submission consists of various methods for image stitching from multi-cameras.

<div align=center>
<img src="https://github.com/Kyle-Xu001/Multi-Depth-Multi-Camera-Stitching/blob/main/Result/panaroma.gif" controls="controls" muted="muted"/>
</div>

## Files Description
    .
    ├── Result                    # Folder for Video and Image Demonstrations
    │ 
    ├── paranoma_test.py          # Generate paranoma image for one frame
    ├── ImageStitch.py            # Define the Image and Stitch class
    ├── stitch_custom.py          # Script about Real-time Video Stitching
    ├── .gitignore
    ├── LICENSE
    └── README.md

- `paranoma_test.py` - Stitch the new image with input stitching combination, using Perspective Transform to stitch the left whole area and the right whole area
- `ImageStitch.py` - Define the `Image` class which combines properties and functions for feature processing on one image, and `Stitch` class which combines properties and functions for matches and features on a pair of images
- `stitch_custom.py` - Given the distortion videos of multiple cameras, utilize the estimated homography parameters generated from `paranoma_test.py` to stitch the image of every frame to create paranoma video
