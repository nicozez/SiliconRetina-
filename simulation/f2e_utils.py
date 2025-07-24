import numpy as np
import os
import socket
import threading
import time
from pathlib import Path
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Set
import glob

class SupervisorClient:
    """Client for communicating with the supervisor (heartbeat only)."""
    
    def __init__(self, heartbeat_port: int):
        self.heartbeat_port = heartbeat_port
        self.heartbeat_socket = None
        self.running = False
        self.heartbeat_thread = None
        
    def start(self):
        """Start heartbeat thread."""
        self.running = True
        
        self.heartbeat_thread = threading.Thread(target=self.__heartbeat_loop__, daemon=True)
        self.heartbeat_thread.start()
        print(f"Heartbeat client started on port {self.heartbeat_port}")
        
    def stop(self):
        """Stop heartbeat thread."""
        self.running = False
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
            
    def send_heartbeat(self):
        """Send a single heartbeat message."""
        try:
            if not self.heartbeat_socket:
                self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.heartbeat_socket.connect(('localhost', self.heartbeat_port))
                print(f"Connected to supervisor heartbeat on port {self.heartbeat_port}")
            
            # Send heartbeat message
            heartbeat_msg = f"f2e_heartbeat:{time.time()}"
            self.heartbeat_socket.send(heartbeat_msg.encode())
            
        except (socket.error, ConnectionRefusedError) as e:
            print(f"Heartbeat connection failed: {e}")
            if self.heartbeat_socket:
                self.heartbeat_socket.close()
                self.heartbeat_socket = None
        except Exception as e:
            print(f"Heartbeat error: {e}")
            
    def __heartbeat_loop__(self):
        """Main heartbeat loop."""
        while self.running:
            try:
                self.send_heartbeat()
                time.sleep(0.5)  # Send heartbeat every 500 milliseconds
                
            except Exception as e:
                print(f"Heartbeat loop error: {e}")
                time.sleep(0.5)

class EventLogger:
    """Manages text eventlog file operations for fast writing."""
    
    def __init__(self, output_dir: str, eventlog_name: str, height: int, width: int):
        self.output_dir = output_dir
        self.eventlog_name = eventlog_name
        self.eventlog_filepath = os.path.join(self.output_dir, f"{self.eventlog_name}.txt")
        self.image_height = height
        self.image_width = width

        try:
            self.file_handle = open(self.eventlog_filepath, 'a')

            if os.path.getsize(self.eventlog_filepath) == 0:
                header =  f"# Event log file\n"
                header += f"# Image dimensions: {width}x{height}\n"
                header += f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"#\n"
                
                self.file_handle.write(header)
                self.file_handle.flush()

            print(f"Initialized eventlog file: {self.eventlog_filepath}")
        except Exception as e:
            print(f"Error initializing eventlog file: {e}")
            raise
    
    def append_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray):
        """Append events to the text file."""
        if len(x) == 0:
            return
            
        try:
            for i in range(len(x)):
                line = f"{x[i]} {y[i]} {t[i]} {p[i]}\n"
                self.file_handle.write(line)
            
            self.file_handle.flush()
                
        except Exception as e:
            print(f"Error appending events: {e}")
    
    def close(self):
        if self.file_handle:
            self.file_handle.close()
            print(f"Closed eventlog file: {self.eventlog_filepath}")

class FileWatcher(FileSystemEventHandler):
    """Watches for new image files in the designated directory."""
    
    def __init__(self, file_queue: Queue, watch_dir: str, display_id: int):
        self.file_queue = file_queue
        self.processed_files = set()
        self.display_id = display_id

        self.observer = Observer()
        self.observer.schedule(self, watch_dir, recursive=False)
        
    def start(self):
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self.__is_valid_image_file__(file_path):
            print(f"New image file detected: {file_path} (display {self.display_id})")
            self.file_queue.put(file_path)
    
    def on_moved(self, event):
        """Handle file move events (some systems trigger this instead of created)."""
        if event.is_directory:
            return
            
        file_path = event.dest_path
        if self.__is_valid_image_file__(file_path):
            print(f"New image file moved: {file_path} (display {self.display_id})")
            self.file_queue.put(file_path)
    
    def __is_valid_image_file__(self, file_path: str) -> bool:
        """Check if file is a valid image file to process."""
        if file_path in self.processed_files:
            return False
            
        file_ext = Path(file_path).suffix.lower()
        if file_ext != '.bmp':
            return False

        if not file_path.startswith(f"{self.display_id}_"):
            return False
            
        if not os.path.exists(file_path):
            return False

        # Wait a bit to ensure file is fully written
        time.sleep(0.05)

        # Check if file is fully written (not being written)
        try:
            with open(file_path, 'rb') as f:
                pass
                
            self.processed_files.add(file_path)
            return True
            
        except Exception:
            return False


class FolderWatcher(FileSystemEventHandler):
    """Watches for new folders in the designated directory."""
    
    def __init__(self, folder_queue: Queue, base_dir: str, folder_prefix: str):
        self.folder_queue = folder_queue
        self.base_dir = base_dir
        self.folder_prefix = folder_prefix
        self.processed_folders: Set[str] = set()
        
        self._scan_existing_folders()
        
        self.observer = Observer()
        self.observer.schedule(self, base_dir, recursive=False)
        
    def _scan_existing_folders(self):
        """Scan for existing folders and mark them as processed."""
        pattern = os.path.join(self.base_dir, f"{self.folder_prefix}_*")
        existing_folders = glob.glob(pattern)
        
        for folder in existing_folders:
            if os.path.isdir(folder):
                self.processed_folders.add(folder)
                print(f"Found existing folder (will skip): {folder}")
        
    def start(self):
        self.observer.start()
        print(f"Watching for new folders in: {self.base_dir}")

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def on_created(self, event):
        """Handle folder creation events."""
        if not event.is_directory:
            return
            
        folder_path = event.src_path
        if self._is_valid_target_folder(folder_path):
            print(f"New target folder detected: {folder_path}")
            self.folder_queue.put(folder_path)
    
    def on_moved(self, event):
        """Handle folder move events."""
        if not event.is_directory:
            return
            
        folder_path = event.dest_path
        if self._is_valid_target_folder(folder_path):
            print(f"New target folder moved: {folder_path}")
            self.folder_queue.put(folder_path)
    
    def _is_valid_target_folder(self, folder_path: str) -> bool:
        """Check if folder is a valid target folder to process."""
        if folder_path in self.processed_folders:
            return False
            
        folder_name = os.path.basename(folder_path)
        if not folder_name.startswith(f"{self.folder_prefix}_"):
            return False
            
        time.sleep(0.1)
        
        if not os.path.exists(folder_path):
            return False
            
        self.processed_folders.add(folder_path)
        return True
