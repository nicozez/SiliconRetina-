import subprocess
import threading
import socket
import time
import sys
import os
import json
import psutil

PING_TIMEOUT = 5  # seconds

class Supervisor:
    def __init__(self, video_path, server_config, server_path, host_config, host_path, client_path, output_folder, ping_port):
        self.video_path = video_path
        self.server_config = server_config
        self.server_path = server_path
        self.host_config = host_config
        self.host_path = host_path
        self.client_path = client_path
        self.output_folder = output_folder
        self.ping_port = int(ping_port)

        self.server_proc = None
        self.host_proc = None
        self.client_proc = None

        self.last_ping = time.time()
        self.shutdown_event = threading.Event()
        self.restart_count = 0  # Track number of restarts

    def update_config_files(self):
        """Update server_config.json and host_config.json with provided parameters"""
        try:
            # Update server_config.json
            print(f"[+] Updating {self.server_config} with video_path: {self.video_path}")
            with open(self.server_config, 'r') as f:
                server_config_data = json.load(f)
            
            # Convert Windows backslashes to forward slashes for JSON compatibility
            video_path_json = self.video_path.replace('\\', '/')
            server_config_data['source_path'] = video_path_json
            
            with open(self.server_config, 'w') as f:
                json.dump(server_config_data, f, indent=4)
            
            # Update host_config.json
            print(f"[+] Updating {self.host_config} with output_folder: {self.output_folder}")
            
            # Read the file as text first to fix any existing escape sequence issues
            with open(self.host_config, 'r') as f:
                content = f.read()
            
            # Fix any existing backslash escape issues by replacing single backslashes with double backslashes
            # This handles cases where the JSON file already has malformed paths
            content = content.replace('\\', '\\\\')
            
            # Parse the fixed content
            host_config_data = json.loads(content)
            
            # Convert Windows backslashes to forward slashes for JSON compatibility
            output_folder_json = self.output_folder.replace('\\', '/')
            host_config_data['record_dir'] = output_folder_json
            
            with open(self.host_config, 'w') as f:
                json.dump(host_config_data, f, indent=4)
                
            print("[+] Configuration files updated successfully")
            
        except Exception as e:
            print(f"[!] Error updating configuration files: {e}")
            raise

    def start_server_with_monitoring(self):
        """Start server process with output monitoring in background thread"""
        print(f"[+] Starting server...")
        
        # Set working directory to the directory containing the executable
        working_dir = os.path.dirname(os.path.abspath(self.server_path))
        print(f"[+] Working directory for server: {working_dir}")
        
        # Start server with output capture for monitoring
        cmd = [self.server_path]
        server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     cwd=working_dir, text=True, bufsize=1)
        
        # Start background thread to read server output
        def monitor_server_output():
            try:
                if server_proc.stdout:
                    for line in iter(server_proc.stdout.readline, ''):
                        if line:
                            print(f"[SERVER] {line.rstrip()}")
            except Exception as e:
                print(f"[!] Server output monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_server_output, daemon=True)
        monitor_thread.start()
        
        return server_proc

    def start_process(self, path, name, args=None, new_terminal=False):
        print(f"[+] Starting {name}...")
        
        # Set working directory to the directory containing the executable
        working_dir = os.path.dirname(os.path.abspath(path))
        print(f"[+] Working directory for {name}: {working_dir}")
        
        if new_terminal:
            # Start in a new terminal window (Windows) - use CREATE_NEW_CONSOLE
            cmd = [path]
            if args:
                cmd.extend(args)
            return subprocess.Popen(cmd, cwd=working_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # Start in the same process - but don't capture output for server to prevent blocking
            cmd = [path]
            if args:
                cmd.extend(args)
            # For server and host, don't capture output to prevent blocking
            if name in ["server", "host"]:
                return subprocess.Popen(cmd, cwd=working_dir)
            else:
                return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=working_dir)

    def restart_client(self):
        self.restart_count += 1
        print(f"[!] Restarting client (restart #{self.restart_count})...")
        
        if self.client_proc:
            try:
                # Get the process tree
                parent = psutil.Process(self.client_proc.pid)
                children = parent.children(recursive=True)
                all_processes = [parent] + children
                
                print(f"[+] Process tree PIDs: {[p.pid for p in all_processes]}")
                print(f"[+] Parent process: {parent.name()} (PID: {parent.pid})")
                
                # Check if process is still running before attempting termination
                if not parent.is_running():
                    print("[+] Client process already terminated")
                    return
                
                # Use graceful termination first - don't interfere with network connections
                print("[+] Attempting graceful termination...")
                try:
                    # Send SIGTERM (graceful shutdown)
                    parent.terminate()
                    parent.wait(timeout=5)
                    print("[+] Client terminated gracefully")
                except psutil.TimeoutExpired:
                    print("[!] Graceful termination timeout, trying CTRL+C...")
                    try:
                        parent.send_signal(2)  # SIGINT (CTRL+C)
                        parent.wait(timeout=5)
                        print("[+] Client terminated via CTRL+C")
                    except psutil.TimeoutExpired:
                        print("[!] CTRL+C timeout, force killing...")
                        # Only use force kill as last resort
                        parent.kill()
                        parent.wait(timeout=2)
                        print("[+] Client force killed")
                except Exception as e:
                    print(f"[!] Error during graceful termination: {e}")
                    print("[!] Attempting force kill...")
                    parent.kill()
                    parent.wait(timeout=2)
                    print("[+] Client force killed")
                
                # Clean up any remaining child processes
                for child in children:
                    try:
                        if child.is_running():
                            child.terminate()
                            child.wait(timeout=2)
                    except:
                        pass
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                print("[+] Client process already terminated")
            except Exception as e:
                print(f"[!] Error during process termination: {e}")
        
        # Wait for server to detect client disconnection
        print("[+] Waiting for server to detect client disconnection...")
        time.sleep(3)  # Reduced wait time
        
        # Check if server is still responsive before starting new client
        print("[+] Checking if server is still responsive...")
        server_ready = False
        for attempt in range(3):  # Try up to 3 times
            try:
                # Try to connect to server port to see if it's accepting connections
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(2.0)
                print(f"[+] Attempting to connect to server port 9000 (attempt {attempt + 1})...")
                result = test_sock.connect_ex(('127.0.0.1', 9000))
                
                if result == 0:  # Connection successful
                    print(f"[+] Server port 9000 is accepting connections!")
                    test_sock.close()
                    server_ready = True
                    break
                else:
                    print(f"[!] Server port 9000 connection failed with error code: {result}")
                    test_sock.close()
                    time.sleep(2)
            except Exception as e:
                print(f"[!] Error checking server port: {e}")
                try:
                    test_sock.close()
                except:
                    pass
                time.sleep(2)
        
        if not server_ready:
            print("[!] Warning: Server port appears to be unresponsive!")
            print("[!] This might indicate the server has stopped responding")
            # Don't start a new client if server is unresponsive
            return
        
        print("[+] Server appears to be ready for new client connections")
        
        # Additional wait before starting new client
        print(f"[+] Starting new client (restart #{self.restart_count})...")
        time.sleep(2)
        
        # Reset ping timer before starting new client
        self.last_ping = time.time()
        print(f"[+] Reset ping timer for restart #{self.restart_count}")
        
        self.client_proc = self.start_process(self.client_path, "client", [str(self.ping_port)], new_terminal=True)
        
        if self.client_proc:
            print(f"[+] New client started successfully (restart #{self.restart_count}, PID: {self.client_proc.pid})")
        else:
            print(f"[!] Failed to start new client (restart #{self.restart_count})")

    def ping_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(('localhost', self.ping_port))
            server_sock.listen(1)
            server_sock.settimeout(1.0)  # Set timeout for accept() calls
            print(f"[+] Ping listener ready on port {self.ping_port}")

            while not self.shutdown_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    print(f"[+] Ping connection from {addr} (restart #{self.restart_count})")
                    conn.settimeout(1.0)  # Set timeout for recv() calls
                    
                    with conn:
                        while not self.shutdown_event.is_set():
                            try:
                                data = conn.recv(1024)
                                if not data:
                                    print(f"[!] Client disconnected (restart #{self.restart_count})")
                                    break
                                self.last_ping = time.time()
                                # Debug: Log successful pings occasionally
                                if int(time.time()) % 30 == 0:  # Every 30 seconds
                                    print(f"[DEBUG] Ping received from restart #{self.restart_count} client")
                            except socket.timeout:
                                continue  # Keep trying to receive
                            except Exception as e:
                                print(f"[!] Ping receive error: {e}")
                                break
                    
                    print("[+] Ping connection closed, waiting for new connection...")
                    
                except socket.timeout:
                    continue  # Keep trying to accept new connections
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        print(f"[!] Ping listener error: {e}")
                    break

    def monitor_loop(self):
        while not self.shutdown_event.is_set():
            time.sleep(1)
            time_since_last_ping = time.time() - self.last_ping
            if time_since_last_ping > PING_TIMEOUT:
                print(f"[!] Client ping timeout after {time_since_last_ping:.1f}s (limit: {PING_TIMEOUT}s). Restarting client...")
                self.restart_client()
                self.last_ping = time.time()
            else:
                # Debug: Show ping status every 10 seconds
                if int(time_since_last_ping) % 10 == 0 and int(time_since_last_ping) > 0:
                    print(f"[DEBUG] Time since last ping: {time_since_last_ping:.1f}s")

    def run(self):
        try:
            # Update configuration files first
            self.update_config_files()
            
            os.makedirs(self.output_folder, exist_ok=True)
            self.server_proc = self.start_server_with_monitoring()
            
            # Check if server started successfully
            if self.server_proc.poll() is not None:
                print(f"[!] Server process failed to start or exited immediately")
                # Since we're monitoring output in background, we can't get error details here
                return
            
            print(f"[+] Server process started with PID: {self.server_proc.pid}")
            
            # Wait a bit for server to fully initialize
            print("[+] Waiting for server to initialize...")
            time.sleep(3)
            
            self.host_proc = self.start_process(self.host_path, "host")
            self.client_proc = self.start_process(self.client_path, "client", [str(self.ping_port)], new_terminal=True)

            listener_thread = threading.Thread(target=self.ping_listener, daemon=True)
            listener_thread.start()

            self.monitor_loop()

        except KeyboardInterrupt:
            print("\n[!] Supervisor shutting down...")
        finally:
            self.shutdown_event.set()
            # Graceful shutdown of all processes
            for proc_name, proc in [("server", self.server_proc), ("host", self.host_proc), ("client", self.client_proc)]:
                if proc:
                    try:
                        print(f"[+] Terminating {proc_name} process...")
                        proc.terminate()
                        proc.wait(timeout=5)
                        print(f"[+] {proc_name} process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"[!] {proc_name} process didn't terminate gracefully, force killing...")
                        proc.kill()
                    except Exception as e:
                        print(f"[!] Error terminating {proc_name} process: {e}")
            print("[+] Clean shutdown complete.")

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python supervisor.py video_path server_config.json server.exe host_config.json host.exe client.exe output_folder port")
        sys.exit(1)

    sup = Supervisor(
        video_path=sys.argv[1],
        server_config=sys.argv[2],
        server_path=sys.argv[3],
        host_config=sys.argv[4],
        host_path=sys.argv[5],
        client_path=sys.argv[6],
        output_folder=sys.argv[7],
        ping_port=sys.argv[8],
    )
    sup.run()
