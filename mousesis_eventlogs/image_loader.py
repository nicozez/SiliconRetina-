#!/usr/bin/env python3
"""
Image Loader

This script loads image data from HDF5 files and provides an interface
to load images in a time range. It uses efficient HDF5 data access for fast loading.
"""

from typing import cast
import h5py
import numpy as np
import numpy.typing as npt

class ImageLoader:
    def __init__(self, h5_file: h5py.File):
        self.h5_file = h5_file

        self.images = cast(h5py.Dataset, self.h5_file['images'])
        self.height = self.images.shape[1]
        self.width = self.images.shape[2]

    def query_frame(self, index: int) -> npt.NDArray[np.uint8]:
        """
        Query a frame by index.
        
        Args:
            index: Frame index to retrieve
            
        Returns:
            Image array with shape (720, 1280, 3) representing height, width, channels
        """
        image = self.images[index]
        return image

    def print_metadata(self):
        print(f"File: {self.h5_file.filename}")
        print(f"Keys: {self.h5_file.keys()}")
        print(f"Number of images: {self.images.shape[0]}")
        print(f"Image dimensions: {self.images.shape[1:]} (height, width, channels)")

    def get_num_images(self) -> int:
        return self.images.shape[0]