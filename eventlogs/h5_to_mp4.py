#!/usr/bin/env python3
"""
H5 to MP4 Converter

This script converts an array of images from an HDF5 file to an MP4 video file.
Uses the ImageLoader class to efficiently load images from the HDF5 file.

Example usage:
python3 h5_to_mp4.py /path/to/input.h5 /path/to/output.mp4 --fps=30
"""

from image_loader import ImageLoader
import argparse
import h5py
import sys
import cv2
from tqdm import tqdm

def h5_to_mp4(h5_file_path: str, output_path: str, fps: int):
    """
    Convert images from HDF5 file to MP4 video.
    
    Args:
        h5_file_path: Path to the HDF5 file containing images
        output_path: Path for the output MP4 file
        fps: Frames per second for the video
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            loader = ImageLoader(f)
            
            # Get video dimensions from first image
            first_frame = loader.query_frame(0)
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process all images
            num_images = loader.get_num_images()
            print(f"Converting {num_images} images to MP4...")
            
            for i in tqdm(range(num_images), desc="Converting frames"):
                frame = loader.query_frame(i)
                out.write(frame)
            
            out.release()
            print(f"Video saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert HDF5 images to MP4 video')
    parser.add_argument('h5_file', help='Path to HDF5 file containing images')
    parser.add_argument('output_file', help='Path for the output MP4 file')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    
    args = parser.parse_args()
    
    h5_to_mp4(args.h5_file, args.output_file, args.fps)
