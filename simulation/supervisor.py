"""
Example config.json structure:
{
    "video_source_path": "path/to/video.mp4",
    "output_directory_prefix": "prefix",
    "display_id": "display_id",
    "eventlog_output_dir": "path/to/eventlog_output_dir",

    "server_config_path": "path/to/server_config.json",
    "server_executable_path": "path/to/server.exe",

    "host_config_path": "path/to/host_config.json", 
    "host_executable_path": "path/to/host.exe",

    "client_executable_path": "path/to/client.exe",
    "client_heartbeat_port": 8080,

    "f2e_executable_path": "path/to/f2e.exe",
    "f2e_heartbeat_port": 8081,
}
"""

import argparse
import subprocess
import threading
import socket
import time
import sys
import os
import json
import psutil

CLIENT_HEARTBEAT_TIMEOUT_SECONDS = 2  # seconds
F2E_HEARTBEAT_TIMEOUT_SECONDS = 2  # seconds

def launch_process(executable_path, args=[], newConsole=False):
    working_directory = os.path.dirname(os.path.abspath(executable_path))        
    command = [executable_path] + args

    subprocess_kwargs = {'cwd': working_directory}
    if newConsole:
        subprocess_kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE

    return subprocess.Popen(command, **subprocess_kwargs)

def stop_process(name, process):
    try:
        process_handle = psutil.Process(process.pid)
        
        if not process_handle.is_running():
            print(f"[+] {name} already terminated")
            return

        print(f"[+] Terminating {name}...")
        process_handle.terminate()
        process_handle.wait(timeout=3)
        print(f"[+] {name} terminated gracefully")

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        print(f"[+] {name} process already terminated")
    except Exception as e:
        print(f"[!] Error during process termination: {e}")

def update_config_file(config_path, key, value):
    try:
        with open(config_path, 'r+') as f:
            config_data = json.load(f)
            config_data[key] = value
            f.seek(0)
            f.truncate()
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"[!] Error updating config file: {e}")
        raise

def load_supervisor_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"[!] Error loading config file: {e}")
        raise

class ProcessSupervisor:
    def __init__(self, name, launch_process, heartbeat_port, HEARTBEAT_TIMEOUT_SECONDS):
        self.server_process = None
        self.host_process = None
        self.process = None
        self.launch_process = launch_process
        self.heartbeat_port = heartbeat_port
        self.name = name
        self.HEARTBEAT_TIMEOUT_SECONDS = HEARTBEAT_TIMEOUT_SECONDS

        self.last_heartbeat_timestamp = time.time()
        self.shutdown_signal = threading.Event()
        self.restart_count = 0

        self.listener_thread = None
        self.monitor_thread = None

    def restart_process(self):
        self.restart_count += 1
        print(f"[!] Restarting {self.name} (restart #{self.restart_count})...")
        
        if self.process:
            stop_process(self.name, self.process)
        
        self.last_heartbeat_timestamp = time.time()
        self.process = self.launch_process()
        
        if self.process:
            print(f"[+] New {self.name} started successfully (restart #{self.restart_count}, PID: {self.process.pid})")
        else:
            print(f"[!] Failed to start new {self.name} (restart #{self.restart_count})")

    def start_heartbeat_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as heartbeat_socket:
            heartbeat_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            heartbeat_socket.bind(('localhost', self.heartbeat_port))
            heartbeat_socket.listen(1)
            heartbeat_socket.settimeout(1.0)
            print(f"[+] Heartbeat listener ready on port {self.heartbeat_port}")

            while not self.shutdown_signal.is_set():
                try:
                    connection, address = heartbeat_socket.accept()
                    print(f"[+] Heartbeat connection from {address} (restart #{self.restart_count})")
                    connection.settimeout(1.0)
                    
                    while not self.shutdown_signal.is_set():
                        try:
                            heartbeat_data = connection.recv(1024)
                            if not heartbeat_data:
                                print(f"[!] {self.name} disconnected (restart #{self.restart_count})")
                                break
                            self.last_heartbeat_timestamp = time.time()
                            print(f"[+] {self.name} heartbeat received")
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"[!] Error during heartbeat: {e}")
                            break
                    
                    connection.close()
                    
                except socket.timeout:
                    continue

    def monitor_heartbeat(self):
        while not self.shutdown_signal.is_set():
            time.sleep(1)
            time_since_last_heartbeat = time.time() - self.last_heartbeat_timestamp
            self.last_heartbeat_timestamp = time.time()

            if time_since_last_heartbeat > self.HEARTBEAT_TIMEOUT_SECONDS:
                print(f"[!] {self.name} heartbeat timeout after {time_since_last_heartbeat:.1f}s (limit: {self.HEARTBEAT_TIMEOUT_SECONDS}s). Restarting {self.name}...")
                self.restart_process()
                self.last_heartbeat_timestamp = time.time()

    def start(self):
        self.process = self.launch_process()
        time.sleep(0.25)

        self.listener_thread = threading.Thread(target=self.start_heartbeat_listener, daemon=True)
        self.listener_thread.start()

        self.monitor_thread = threading.Thread(target=self.monitor_heartbeat, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.shutdown_signal.set()
        stop_process(self.name, self.process)
        self.listener_thread.join()
        self.monitor_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Supervisor')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config = load_supervisor_config(args.config_path)
        
    video_source_path = config['video_source_path']
    output_directory_prefix = config['output_directory_prefix']
    display_id = config['display_id']
    eventlog_output_dir = config['eventlog_output_dir']

    server_config_path = config['server_config_path']
    server_executable_path = config['server_executable_path']

    host_config_path = config['host_config_path']
    host_executable_path = config['host_executable_path']

    client_executable_path = config['client_executable_path']
    client_heartbeat_port = int(config['client_heartbeat_port'])

    f2e_executable_path = config['f2e_executable_path']
    f2e_heartbeat_port = int(config['f2e_heartbeat_port'])

    update_config_file(server_config_path, 'source_path', video_source_path.replace('\\', '/'))
    update_config_file(host_config_path, 'record_dir', output_directory_prefix.replace('\\', '/'))

    server_process = launch_process(server_executable_path)
    host_process = launch_process(host_executable_path)

    def launch_client_process():
        args = [str(client_heartbeat_port)]
        return launch_process(client_executable_path, args, newConsole=True)

    def launch_f2e_process():
        # f2e.py expects: base-dir, folder-prefix, output-dir, heartbeat-port, display-id
        args = [
            os.path.dirname(host_executable_path),
            output_directory_prefix, 
            eventlog_output_dir,
            str(f2e_heartbeat_port),
            str(display_id)
        ]
        return launch_process("python3", [f2e_executable_path] + args, newConsole=True)

    client_supervisor = ProcessSupervisor("client", launch_client_process, client_heartbeat_port, CLIENT_HEARTBEAT_TIMEOUT_SECONDS)
    client_supervisor.start()

    f2e_supervisor = ProcessSupervisor("f2e", launch_f2e_process, f2e_heartbeat_port, F2E_HEARTBEAT_TIMEOUT_SECONDS)
    f2e_supervisor.start()

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            client_supervisor.stop()
            f2e_supervisor.stop()
            stop_process("host", host_process)
            stop_process("server", server_process)
            sys.exit(1)
        time.sleep(1)
