import os
import socket
import time

def start_server_on_inherited_socket(fd):
    """This function runs in the child process."""
    print(f"[Child PID {os.getpid()}] Inherited file descriptor: {fd}")
    
    # Re-create the socket object from the file descriptor
    server_socket = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
    
    # Get the address it's already bound to
    host, port = server_socket.getsockname()
    print(f"[Child PID {os.getpid()}] Socket is bound to {host}:{port}")

    # Now the child can use the socket
    server_socket.listen(1)
    print("[Child PID {os.getpid()}] Server is listening...")
    # ... accept connections, etc. ...
    time.sleep(5) # Simulate server work
    server_socket.close()
    print("[Child PID {os.getpid()}] Server finished.")


# --- Parent Process Logic ---
if __name__ == "__main__":
    # 1. Parent creates and binds the socket
    parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    parent_socket.bind(('', 0)) # Bind to a free port
    host, port = parent_socket.getsockname()
    fd = parent_socket.fileno() # Get the integer file descriptor

    print(f"[Parent PID {os.getpid()}] Bound to {host}:{port}, file descriptor: {fd}")

    start_server_on_inherited_socket(fd)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        host, port = server_socket.getsockname()
        print(f"[Parent PID {os.getpid()}] Bound to {host}:{port}")
        server_socket.listen(1)
        print("[Parent PID {os.getpid()}] Server is listening...")
        time.sleep(5) # Simulate server work
        server_socket.close()
        print("[Parent PID {os.getpid()}] Server finished.")