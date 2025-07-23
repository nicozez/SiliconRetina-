import numpy as np
import os
import socket
import threading
import time
from pathlib import Path
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SupervisorClient:
    """Client for communicating with the supervisor (heartbeat + commands)."""
    
    def __init__(self, heartbeat_port: int, command_port: int):
        self.heartbeat_port = heartbeat_port
        self.command_port = command_port

        self.heartbeat_socket = None
        self.command_socket = None

        self.running = False
        self.heartbeat_thread = None
        self.command_thread = None
        self.stop_callback = None
        
    def set_stop_callback(self, callback):
        """Set callback function to be called when stop command is received."""
        self.stop_callback = callback
        
    def start(self):
        """Start both heartbeat and command threads."""
        self.running = True
        
        self.heartbeat_thread = threading.Thread(target=self.__heartbeat_loop__, daemon=True)
        self.heartbeat_thread.start()
        print(f"Heartbeat client started on port {self.heartbeat_port}")
        
        self.command_thread = threading.Thread(target=self.__command_loop__, daemon=True)
        self.command_thread.start()
        print(f"Command listener started on port {self.command_port}")
        
    def stop(self):
        """Stop both heartbeat and command threads."""
        self.running = False
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
            
        if self.command_socket:
            self.command_socket.close()
        if self.command_thread:
            self.command_thread.join(timeout=1.0)
            
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
    
    def __command_loop__(self):
        """Main command listening loop."""
        try:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.command_socket.bind(('localhost', self.command_port))
            self.command_socket.listen(1)
            self.command_socket.settimeout(1.0)  # 1 second timeout for accept
            
            print(f"Command listener listening on port {self.command_port}")
            
            while self.running:
                try:
                    # Accept connection from supervisor
                    client_socket, addr = self.command_socket.accept()
                    print(f"Command connection from {addr}")
                    
                    # Handle commands
                    self.__handle_commands__(client_socket)
                    
                except socket.timeout:
                    # Timeout is expected, continue listening
                    continue
                except (socket.error, ConnectionRefusedError) as e:
                    print(f"Command connection error: {e}")
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Command listener error: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            print(f"Failed to start command listener: {e}")
        finally:
            if self.command_socket:
                self.command_socket.close()
    
    def __handle_commands__(self, client_socket):
        try:
            client_socket.settimeout(1.0)  # 1 second timeout for commands
            
            while self.running:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    
                    command = data.decode().strip()
                    print(f"Received command: {command}")
                    
                    if command == "stop":
                        self.stop_callback()
                        break
                    else:
                        print(f"Unknown command: {command}")
                        
                except socket.timeout:
                    # Timeout is expected, continue listening for commands
                    continue
                    
        except Exception as e:
            print(f"Error handling commands: {e}")
        finally:
            client_socket.close()

class EventLogger:
    """Manages text eventlog file operations for fast writing."""
    
    def __init__(self, output_file: str, height: int, width: int):
        self.output_file = output_file
        self.image_height = height
        self.image_width = width

        try:
            self.file_handle = open(self.output_file, 'a')

            if os.path.getsize(self.output_file) == 0:
                header =  f"# Event log file\n"
                header += f"# Image dimensions: {width}x{height}\n"
                header += f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"#\n"
                
                self.file_handle.write(header)
                self.file_handle.flush()

            print(f"Initialized eventlog file: {self.output_file}")
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
            print(f"Closed eventlog file: {self.output_file}")

class FileWatcher(FileSystemEventHandler):
    """Watches for new image files in the designated directory."""
    
    def __init__(self, file_queue: Queue, watch_dir: str):
        self.file_queue = file_queue
        self.processed_files = set()

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
            print(f"New image file detected: {file_path}")
            self.file_queue.put(file_path)
    
    def on_moved(self, event):
        """Handle file move events (some systems trigger this instead of created)."""
        if event.is_directory:
            return
            
        file_path = event.dest_path
        if self.__is_valid_image_file__(file_path):
            print(f"New image file moved: {file_path}")
            self.file_queue.put(file_path)
    
    def __is_valid_image_file__(self, file_path: str) -> bool:
        """Check if file is a valid image file to process."""
        if file_path in self.processed_files:
            return False
            
        file_ext = Path(file_path).suffix.lower()
        if file_ext != '.bmp':
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