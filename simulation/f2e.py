#!/usr/bin/env python3
"""
Frame to Event Converter

This script watches a folder for newly created folders containing image frames, 
consumes all frames in each new folder, and appends events to a text eventlog file. 
Each event represents a brightness level change with x, y, t coordinates and polarity (ON or OFF).

Usage:
    python f2e.py <base_path> <folder_prefix> <output_dir> <heartbeat_port> <ON_events_display_id> <OFF_events_display_id>
"""

import argparse
import cv2
import numpy as np
import os
import re
import sys
import time
import glob

from f2e_utils import SupervisorClient, EventLogger, FolderWatcher, FileWatcher
from queue import Queue, Empty
from typing import Optional, Tuple

def find_latest_folder(base_dir: str, folder_id: str) -> Optional[str]:
    """Find the latest folder with the given ID."""
    pattern = os.path.join(base_dir, f"{folder_id}_*")
    matching_folders = glob.glob(pattern)

    if not matching_folders:
        return None

    directories = [f for f in matching_folders if os.path.isdir(f)]

    if not directories:
        return None

    directories.sort(key=lambda x: os.path.getctime(x), reverse=True)

    return directories[0]

def extract_events_from_image(image_path: str, time_delta: int, display_id: str, polarity: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filename = os.path.basename(image_path)
    pattern = rf"{re.escape(display_id)}_(\d+)"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match is None:
        raise Exception()

    frame_number = int(match.group(1))
    base_time = frame_number * time_delta
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception()
                
    # Find white pixels (events)
    # Assuming white pixels (255) represent events, black pixels (0) represent no events
    event_pixels = np.where(image > 128)  # Threshold for white pixels
    
    y_coords = event_pixels[0].astype(np.uint64)
    x_coords = event_pixels[1].astype(np.uint64)
    
    timestamps = np.full(len(x_coords), base_time, dtype=np.uint32)
    polarities = np.full(len(x_coords), polarity, dtype=np.uint8)
    
    print(f"Extracted {len(x_coords)} events from frame {frame_number} (polarity: {polarity}, timestamp: {base_time})")
    
    return x_coords, y_coords, timestamps, polarities


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Frame to Event Converter')
    parser.add_argument('base_dir', help='Base directory to search for dynamic folders')
    parser.add_argument('folder_prefix', help='Folder prefix for dynamic folder detection (e.g., "rec")')
    parser.add_argument('output_dir', help='Output directory for eventlog files')
    parser.add_argument('heartbeat_port', type=int, help='Port for heartbeat communication')
    parser.add_argument('ON_events_display_id', type=str, help='Display ID from which to extract ON events')
    parser.add_argument('OFF_events_display_id', type=str, help='Display ID from which to extract OFF events')
    parser.add_argument('--time_delta', type=int, default=300, help='Discrete time delta between frames in milliseconds')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing (default: 10)')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Base directory does not exist: {args.base_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    supervisor_client = SupervisorClient(args.heartbeat_port)
    supervisor_client.start()

    folder_queue = Queue()
    folder_watcher = FolderWatcher(folder_queue, args.base_dir, args.folder_prefix)
    folder_watcher.start()
    print(f"Watching for new folders in directory: {args.base_dir}")
    
    try:
        while True:
            try:
                folder_path = folder_queue.get(timeout=1.0)
                if folder_path is None:
                    time.sleep(0.1)
                    continue

                eventlog_name = os.path.basename(folder_path).replace('\\', '/')
                event_logger = EventLogger(args.output_dir, eventlog_name, args.height, args.width)

                file_queue = Queue()
                file_watcher = FileWatcher(file_queue, folder_path, args.ON_events_display_id, args.OFF_events_display_id)
                file_watcher.start()
                print(f"Watching for new files in directory: {folder_path}")

                time.sleep(1) # Wait for the file watcher to capture the first file
                
                try:
                    while folder_queue.empty() and not file_queue.empty():
                        file_path = file_queue.get(timeout=1.0)
                        if file_path is None:
                            time.sleep(0.1)
                            continue

                        filename = os.path.basename(file_path)
                        display_id = args.ON_events_display_id if filename.startswith(f"{args.ON_events_display_id}_") else args.OFF_events_display_id
                        polarity = 1 if filename.startswith(f"{args.ON_events_display_id}_") else 0

                        x, y, t, p = extract_events_from_image(file_path, args.time_delta, display_id, polarity)
                        event_logger.append_events(x, y, t, p)

                        # Delete the file after processing
                        os.remove(file_path)
                finally:
                    file_watcher.stop()
                    event_logger.close()

            except Empty:
                # No new folders, continue waiting
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing folder: {folder_path}")
                print(e)
            finally:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    finally:
        print("Shutting down gracefully...")
        supervisor_client.stop()
        folder_watcher.stop()
        print("Shutdown complete")

if __name__ == "__main__":
    main()
