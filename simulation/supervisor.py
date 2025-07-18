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
    def __init__(self, video_source_path, server_config_path, server_executable_path, host_config_path, host_executable_path, client_executable_path, output_directory_path, heartbeat_port):
        self.video_source_path = video_source_path
        self.server_config_path = server_config_path
        self.server_executable_path = server_executable_path
        self.host_config_path = host_config_path
        self.host_executable_path = host_executable_path
        self.client_executable_path = client_executable_path
        self.output_directory_path = output_directory_path
        self.heartbeat_port = int(heartbeat_port)

        self.server_process = None
        self.host_process = None
        self.client_process = None

        self.last_heartbeat_timestamp = time.time()
        self.shutdown_signal = threading.Event()
        self.client_restart_count = 0

    def update_configuration_files(self):
        """Update server and host configuration files with provided parameters"""
        try:
            # Update server_config.json
            print(f"[+] Updating {self.server_config_path} with video_path: {self.video_source_path}")
            with open(self.server_config_path, 'r') as f:
                server_config_data = json.load(f)
            
            video_path_json = self.video_source_path.replace('\\', '/')
            server_config_data['source_path'] = video_path_json
            
            with open(self.server_config_path, 'w') as f:
                json.dump(server_config_data, f, indent=4)
            
            # Update host_config.json
            print(f"[+] Updating {self.host_config_path} with output_folder: {self.output_directory_path}")
            
            with open(self.host_config_path, 'r') as f:
                content = f.read()
            
            content = content.replace('\\', '\\\\')
            host_config_data = json.loads(content)
            
            output_folder_json = self.output_directory_path.replace('\\', '/')
            host_config_data['record_dir'] = output_folder_json
            
            with open(self.host_config_path, 'w') as f:
                json.dump(host_config_data, f, indent=4)
                
            print("[+] Configuration files updated successfully")
            
        except Exception as e:
            print(f"[!] Error updating configuration files: {e}")
            raise

    def launch_process(self, path, name, args=None, use_new_terminal=False, capture_output=True, use_text_output=False):
        """
        Launch a subprocess with specified configuration.
        
        Args:
            path: Path to the executable
            name: Name of the process for logging
            args: Optional list of arguments
            use_new_terminal: Whether to open in new terminal (Windows only)
            capture_output: Whether to capture stdout/stderr
            use_text_output: Whether to use text mode for output (implies capture_output=True)
        """
        print(f"[+] Starting {name}...")
        
        working_directory = os.path.dirname(os.path.abspath(path))
        print(f"[+] Working directory for {name}: {working_directory}")
        
        command = [path]
        if args:
            command.extend(args)
        
        subprocess_kwargs = {'cwd': working_directory}
        
        if use_new_terminal:
            subprocess_kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        elif capture_output or use_text_output:
            subprocess_kwargs['stdout'] = subprocess.PIPE
            subprocess_kwargs['stderr'] = subprocess.STDOUT
            if use_text_output:
                subprocess_kwargs['text'] = True
                subprocess_kwargs['bufsize'] = 1
        
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
        self.client_process = self.launch_process(self.client_executable_path, "client", [str(self.heartbeat_port)], use_new_terminal=True)
        
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
            
            os.makedirs(self.output_directory_path, exist_ok=True)
            self.server_process = self.launch_process(self.server_executable_path, "server", capture_output=True, use_text_output=True)
            
            if self.server_process.poll() is not None:
                print(f"[!] Server process failed to start or exited immediately")
                return
            
            print(f"[+] Server process started with PID: {self.server_process.pid}")
            print("[+] Waiting for server to initialize...")
            time.sleep(2)
            
            self.host_process = self.launch_process(self.host_executable_path, "host", capture_output=False)
            self.client_process = self.launch_process(self.client_executable_path, "client", [str(self.heartbeat_port)], use_new_terminal=True)

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
    if len(sys.argv) != 9:
        print("Usage: python supervisor.py video_path server_config.json server.exe host_config.json host.exe client.exe output_folder port")
        sys.exit(1)

    sup = ProcessSupervisor(
        video_source_path=sys.argv[1],
        server_config_path=sys.argv[2],
        server_executable_path=sys.argv[3],
        host_config_path=sys.argv[4],
        host_executable_path=sys.argv[5],
        client_executable_path=sys.argv[6],
        output_directory_path=sys.argv[7],
        heartbeat_port=sys.argv[8],
    )
    sup.run()
