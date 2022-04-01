# Multi-Depth-Multi-Camera-Stitching

## Project Description

This submission consists of various methods for image stitching from multi-cameras.

## Files Description
    .
    ├── paranoma_test.py          # Generate paranoma image for one frame
    ├── ImageStitch.py            # Define the Image and Stitch class
    ├── .gitignore
    ├── LICENSE
    └── README.md
- `paranoma_test.py` - Stitch the new image with input stitching combination, using Perspective Transform to stitch the left whole area and the right whole area
- `ImageStitch.py` - Define the `Image` class which combines properties and functions for feature processing on one image, and `Stitch` class which combines properties and functions for matches and features on a pair of images