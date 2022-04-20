# Multi-Depth-Multi-Camera-Stitching

## Project Description

This submission consists of various methods for video stitching from multi-cameras to generate a real-time overview panorama video. 

<div align=center>
<img src="https://github.com/Kyle-Xu001/Multi-Depth-Multi-Camera-Stitching/blob/main/Result/panaroma.gif" controls="controls" muted="muted"/>
</div>

## Files Description
    .
    ├── Result                    # Folder for Animation and Image Demonstrations
    │ 
    ├── panorama_test.py          # Generate panorama image for one frame
    ├── ImageStitch.py            # Define the Image and Stitch class
    ├── PositioningSystem.py      # Transform the Detection Information from Local to Global
    ├── stitch_custom.py          # Script about Real-time Video Stitching
    ├── utils.py                  # Basic functions for stitching
    ├── .gitignore
    ├── LICENSE
    └── README.md

- `panorama_test.py` - Stitch the new image with input stitching combination, using Perspective Transform to stitch the left whole area and the right whole area
- `ImageStitch.py` - Define the `Image` class which combines properties and functions for feature processing on one image, and `Stitch` class which combines properties and functions for matches and features on a pair of images
- `stitch_custom.py` - Given the distortion videos of multiple cameras, utilize the estimated homography parameters generated from `panorama_test.py` to stitch the image of every frame to create panorama video

## Usage
- Video Stitching Test: Stitch the input videos to generate a panorama video:
```
    $ python stitch_custom.py -ivid /PATH/TO/VIDEO/GROUP -pst /PATH/TO/PARAMS/FILE
```
- Image Stitching Test: Stitch the images at the same frame from all cameras to generate a panorama image:
```
    $ python panorama_test.py
```