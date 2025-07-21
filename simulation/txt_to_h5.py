#!/usr/bin/env python3
"""
Text to HDF5 Eventlog Converter (txt_to_h5.py)

This script converts text eventlog files to HDF5 format for compatibility
with the existing event_loader.py system.

Usage:
    python txt_to_h5.py --input <eventlog.txt> --output <eventlog.h5>
"""

import argparse
import h5py
import numpy as np
import os
import re
import sys
import time
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EVENT_DTYPES = {
    'x': np.uint64,
    'y': np.uint64, 
    't': np.uint32,
    'p': np.uint8
}

class TextEventlogParser:
    """Parser for text eventlog files."""
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.height = None
        self.width = None
        
    def parse_header(self) -> bool:
        """Parse header information from the text file."""
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip comment lines
                    if line.startswith('#'):
                        # Extract image dimensions
                        dim_match = re.search(r'Image dimensions: (\d+)x(\d+)', line)
                        if dim_match:
                            self.width = int(dim_match.group(1))
                            self.height = int(dim_match.group(2))
                            logger.info(f"Found image dimensions: {self.width}x{self.height}")
                        continue
                    
                    # If we reach a non-comment line, we're done with header
                    break
                    
            return self.height is not None and self.width is not None
            
        except Exception as e:
            logger.error(f"Error parsing header: {e}")
            return False
    
    def count_events(self) -> int:
        """Count the number of events in the file."""
        try:
            count = 0
            with open(self.input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting events: {e}")
            return 0
    
    def parse_events(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse all events from the text file."""
        try:
            # Count events first
            total_events = self.count_events()
            logger.info(f"Found {total_events} events to process")
            
            if total_events == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Pre-allocate arrays
            x_coords = np.zeros(total_events, dtype=np.uint64)
            y_coords = np.zeros(total_events, dtype=np.uint64)
            timestamps = np.zeros(total_events, dtype=np.uint32)
            polarities = np.zeros(total_events, dtype=np.uint8)
            
            # Parse events
            event_idx = 0
            with open(self.input_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse "x y t p" format
                        parts = line.split()
                        if len(parts) != 4:
                            logger.warning(f"Invalid event format at line {line_num}: {line}")
                            continue
                        
                        x_coords[event_idx] = int(parts[0])
                        y_coords[event_idx] = int(parts[1])
                        timestamps[event_idx] = int(parts[2])
                        polarities[event_idx] = int(parts[3])
                        
                        event_idx += 1
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing event at line {line_num}: {e}")
                        continue
            
            # Trim arrays to actual number of events
            if event_idx < total_events:
                x_coords = x_coords[:event_idx]
                y_coords = y_coords[:event_idx]
                timestamps = timestamps[:event_idx]
                polarities = polarities[:event_idx]
            
            logger.info(f"Successfully parsed {len(x_coords)} events")
            return x_coords, y_coords, timestamps, polarities
            
        except Exception as e:
            logger.error(f"Error parsing events: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

class HDF5EventlogWriter:
    """Writer for HDF5 eventlog files."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        
    def write_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, 
                    height: int, width: int):
        """Write events to HDF5 file in the format expected by event_loader.py."""
        try:
            with h5py.File(self.output_file, 'w') as h5_file:
                # Create datasets
                h5_file.create_dataset('x', data=x, dtype=EVENT_DTYPES['x'])
                h5_file.create_dataset('y', data=y, dtype=EVENT_DTYPES['y'])
                h5_file.create_dataset('t', data=t, dtype=EVENT_DTYPES['t'])
                h5_file.create_dataset('p', data=p, dtype=EVENT_DTYPES['p'])
                
                # Store metadata
                h5_file.attrs['height'] = height
                h5_file.attrs['width'] = width
                h5_file.attrs['created'] = time.time()
                h5_file.attrs['converted_from'] = 'text_eventlog'
                
                logger.info(f"Successfully wrote {len(x)} events to {self.output_file}")
                logger.info(f"File size: {os.path.getsize(self.output_file) / (1024*1024):.2f} MB")
                
        except Exception as e:
            logger.error(f"Error writing HDF5 file: {e}")
            raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert text eventlog to HDF5 format')
    parser.add_argument('input', help='Input text eventlog file (.txt)')
    parser.add_argument('output', help='Output HDF5 eventlog file (.h5)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse text file
    logger.info(f"Parsing text eventlog: {args.input}")
    parser = TextEventlogParser(args.input)
    
    # Parse header
    if not parser.parse_header():
        logger.error("Failed to parse header information")
        sys.exit(1)
    
    # Parse events
    x, y, t, p = parser.parse_events()
    
    if len(x) == 0:
        logger.warning("No events found in input file")
        # Still create the HDF5 file with empty datasets
        x = np.array([], dtype=np.uint64)
        y = np.array([], dtype=np.uint64)
        t = np.array([], dtype=np.uint32)
        p = np.array([], dtype=np.uint8)
    
    # Write HDF5 file
    logger.info(f"Writing HDF5 eventlog: {args.output}")
    writer = HDF5EventlogWriter(args.output)
    writer.write_events(x, y, t, p, parser.height, parser.width)
  
    logger.info("Conversion completed successfully!")

if __name__ == "__main__":
    main() 