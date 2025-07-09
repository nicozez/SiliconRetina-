from image_loader import ImageLoader
import argparse
import h5py
import sys
import cv2

if __name__ == "__main__":
    """Main function to handle command line arguments and run visualization."""
    
    parser = argparse.ArgumentParser(description='Interactive 3D Event Data Visualization')
    parser.add_argument('h5_file', help='Path to HDF5 file containing event data')
    parser.add_argument('--index', '-i', type=int, default=0,
                       help='Index of the frame to load')
    
    args = parser.parse_args()
    
    try:
        with h5py.File(args.h5_file, 'r') as f:
            loader = ImageLoader(f)
            
            index = 0
            end_index = loader.get_num_images()

            while index < end_index:
                frame = loader.query_frame(index)
                cv2.imshow('Frame', frame)
                index += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except Exception as e:
        print(f"Error: Cannot open HDF5 file '{args.h5_file}': {e}")
        sys.exit(1)

