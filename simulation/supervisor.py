import subprocess
import threading
import socket
import time
import sys
import os
import json
import psutil

CLIENT_HEARTBEAT_TIMEOUT_SECONDS = 2  # seconds

class ProcessSupervisor:
    def __init__(self, config_path):
        """
        Initialize ProcessSupervisor with configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config = self.load_config(config_path)
        
        # Extract configuration values
        self.video_source_path = self.config['video_source_path']
        self.server_config_path = self.config['server_config_path']
        self.server_executable_path = self.config['server_executable_path']
        self.host_config_path = self.config['host_config_path']
        self.host_executable_path = self.config['host_executable_path']
        self.client_executable_path = self.config['client_executable_path']
        self.output_directory = self.config['output_directory']
        self.heartbeat_port = int(self.config['heartbeat_port'])

        self.server_process = None
        self.host_process = None
        self.client_process = None

        self.last_heartbeat_timestamp = time.time()
        self.shutdown_signal = threading.Event()
        self.client_restart_count = 0

    def load_config(self, config_path):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            KeyError: If required configuration keys are missing
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required keys
            required_keys = [
                'video_source_path',
                'server_config_path', 
                'server_executable_path',
                'host_config_path',
                'host_executable_path',
                'client_executable_path',
                'output_directory',
                'heartbeat_port'
            ]
            
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise KeyError(f"Missing required configuration keys: {missing_keys}")
            
            return config
            
        except FileNotFoundError:
            print(f"[!] Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"[!] Invalid JSON in configuration file: {e}")
            raise
        except KeyError as e:
            print(f"[!] Configuration error: {e}")
            raise

    def update_server_configuration(self):
        try:
            print(f"[+] Updating {self.server_config_path} with video_path: {self.video_source_path}")

            with open(self.server_config_path, "r+") as f:
                server_config_data = json.load(f)
                
                video_path_json = self.video_source_path.replace('\\', '/')
                server_config_data['source_path'] = video_path_json
                
                f.seek(0)
                f.truncate()
                
                json.dump(server_config_data, f, indent=4)
            
            print(f"[+] Server configuration updated successfully")
            
        except Exception as e:
            print(f"[!] Error updating server configuration: {e}")
            raise

    def update_host_configuration(self):
        try:
            print(f"[+] Updating {self.host_config_path} with output_folder: {self.output_directory}")
            
            with open(self.host_config_path, 'r+') as f:
                host_config_data = json.load(f)
                
                output_folder_json = self.output_directory.replace('\\', '/')
                host_config_data['record_dir'] = output_folder_json
                
                f.seek(0)
                f.truncate()
                
                json.dump(host_config_data, f, indent=4)
            
            print(f"[+] Host configuration updated successfully")
            
        except Exception as e:
            print(f"[!] Error updating host configuration: {e}")
            raise

    def update_configuration_files(self):
        self.update_server_configuration()
        self.update_host_configuration()
        print("[+] All configuration files updated successfully")

    def launch_server(self):
        """Launch the server process without capturing output to prevent blocking"""
        print("[+] Starting server...")
        
        working_directory = os.path.dirname(os.path.abspath(self.server_executable_path))
        print(f"[+] Working directory for server: {working_directory}")
        
        command = [self.server_executable_path]
        
        subprocess_kwargs = {
            'cwd': working_directory
        }
        
        return subprocess.Popen(command, **subprocess_kwargs)

    def launch_host(self):
        """Launch the host process without capturing output"""
        print("[+] Starting host...")
        
        working_directory = os.path.dirname(os.path.abspath(self.host_executable_path))
        print(f"[+] Working directory for host: {working_directory}")
        
        command = [self.host_executable_path]
        
        subprocess_kwargs = {'cwd': working_directory}
        
        return subprocess.Popen(command, **subprocess_kwargs)

    def launch_client(self):
        """Launch the client process with heartbeat port argument"""
        print("[+] Starting client...")
        
        working_directory = os.path.dirname(os.path.abspath(self.client_executable_path))
        print(f"[+] Working directory for client: {working_directory}")
        
        command = [self.client_executable_path, str(self.heartbeat_port)]
        
        subprocess_kwargs = {
            'cwd': working_directory,
            'creationflags': subprocess.CREATE_NEW_CONSOLE
        }
        
        return subprocess.Popen(command, **subprocess_kwargs)

    def restart_client_process(self):
        self.client_restart_count += 1
        print(f"[!] Restarting client (restart #{self.client_restart_count})...")
        
        if self.client_process:
            try:
                client_process_handle = psutil.Process(self.client_process.pid)
                
                if not client_process_handle.is_running():
                    print("[+] Client process already terminated")
                    return
                
                # Graceful termination
                print("[+] Terminating client...")
                client_process_handle.terminate()
                client_process_handle.wait(timeout=3)
                print("[+] Client terminated gracefully")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                print("[+] Client process already terminated")
            except Exception as e:
                print(f"[!] Error during process termination: {e}")
        
        # Wait for server to detect client disconnection
        print("[+] Waiting for server to detect client disconnection...")
        time.sleep(1)
           
        # Start new client
        print(f"[+] Starting new client (restart #{self.client_restart_count})...")
        time.sleep(0.5)
        
        self.last_heartbeat_timestamp = time.time()
        self.client_process = self.launch_client()
        
        if self.client_process:
            print(f"[+] New client started successfully (restart #{self.client_restart_count}, PID: {self.client_process.pid})")
        else:
            print(f"[!] Failed to start new client (restart #{self.client_restart_count})")

    def start_heartbeat_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as heartbeat_socket:
            heartbeat_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            heartbeat_socket.bind(('localhost', self.heartbeat_port))
            heartbeat_socket.listen(1)
            heartbeat_socket.settimeout(1.0)
            print(f"[+] Heartbeat listener ready on port {self.heartbeat_port}")

            while not self.shutdown_signal.is_set():
                try:
                    client_connection, client_address = heartbeat_socket.accept()
                    print(f"[+] Heartbeat connection from {client_address} (restart #{self.client_restart_count})")
                    client_connection.settimeout(1.0)
                    
                    with client_connection:
                        while not self.shutdown_signal.is_set():
                            try:
                                heartbeat_data = client_connection.recv(1024)
                                if not heartbeat_data:
                                    print(f"[!] Client disconnected (restart #{self.client_restart_count})")
                                    break
                                self.last_heartbeat_timestamp = time.time()
                            except socket.timeout:
                                continue
                            except Exception as e:
                                print(f"[!] Heartbeat receive error: {e}")
                                break
                    
                    print("[+] Heartbeat connection closed, waiting for new connection...")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.shutdown_signal.is_set():
                        print(f"[!] Heartbeat listener error: {e}")
                    break

    def monitor_client_heartbeat(self):
        while not self.shutdown_signal.is_set():
            time.sleep(1)
            time_since_last_heartbeat = time.time() - self.last_heartbeat_timestamp
            if time_since_last_heartbeat > CLIENT_HEARTBEAT_TIMEOUT_SECONDS:
                print(f"[!] Client heartbeat timeout after {time_since_last_heartbeat:.1f}s (limit: {CLIENT_HEARTBEAT_TIMEOUT_SECONDS}s). Restarting client...")
                self.restart_client_process()
                self.last_heartbeat_timestamp = time.time()

    def run(self):
        try:
            self.update_configuration_files()
            
            os.makedirs(self.output_directory, exist_ok=True)
            self.server_process = self.launch_server()
            
            if self.server_process.poll() is not None:
                print(f"[!] Server process failed to start or exited immediately")
                return
            
            print(f"[+] Server process started with PID: {self.server_process.pid}")
            print("[+] Waiting for server to initialize...")
            time.sleep(2)
            
            self.host_process = self.launch_host()
            self.client_process = self.launch_client()

            listener_thread = threading.Thread(target=self.start_heartbeat_listener, daemon=True)
            listener_thread.start()

            self.monitor_client_heartbeat()

        except KeyboardInterrupt:
            print("\n[!] Supervisor shutting down...")
        finally:
            self.shutdown_signal.set()
            for process_name, process_handle in [("server", self.server_process), ("host", self.host_process), ("client", self.client_process)]:
                if process_handle:
                    try:
                        print(f"[+] Terminating {process_name} process...")
                        process_handle.terminate()
                        process_handle.wait(timeout=3)
                        print(f"[+] {process_name} process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"[!] {process_name} process didn't terminate gracefully, force killing...")
                        process_handle.kill()
                    except Exception as e:
                        print(f"[!] Error terminating {process_name} process: {e}")
            print("[+] Clean shutdown complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python supervisor.py config.json")
        print("Example config.json structure:")
        print("""
                {
                    "video_source_path": "path/to/video.mp4",
                    "server_config_path": "path/to/server_config.json",
                    "server_executable_path": "path/to/server.exe",
                    "host_config_path": "path/to/host_config.json", 
                    "host_executable_path": "path/to/host.exe",
                    "client_executable_path": "path/to/client.exe",
                    "output_directory": "path/to/output",
                    "heartbeat_port": 8080
                }
        """)
        sys.exit(1)

    config_path = sys.argv[1]
    
    try:
        sup = ProcessSupervisor(config_path)
        sup.run()
    except Exception as e:
        print(f"[!] Failed to start supervisor: {e}")
        sys.exit(1)
