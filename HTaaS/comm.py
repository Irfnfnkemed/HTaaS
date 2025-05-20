import json
import socket
import subprocess
from typing import Any, Tuple

"""
This module provides classes and functions for communication between servers and clients using sockets."""
""""""


def get_ip() -> str:
    """
    Get the current machine's public IP address.
    If the connection fails, returns '127.0.0.1' as a fallback.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = "127.0.0.1"
    s.close()
    return ip


def get_free_port(ip: str = "", local=False) -> int:
    """
    Get a free port number on the specified machine.
    Returns the port number as an integer, or 0 if an error occurs.
    """
    if ip == "" or ip == "127.0.0.1" or ip == "localhost" or local:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    else:
        code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
                """
        cmd = ["ssh", ip, "python3", "-c", f"{code}"]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            return int(result.stdout)
        else:
            return 0


class ServerInstance:
    def __init__(self, conn: socket.socket, ip: str):
        """
        Initialize a ServerInstance instance.

        Parameters:
            conn (socket.socket): The socket connection used for communication.
            ip (str): The IP address associated with the connection.
        """
        self._conn = conn
        self._ip = ip

    def send(self, cmd: str, data: Any):
        """
        Send a command and data to the client.

        Parameters:
            cmd (str): The command string to be sent.
            data (Any): The data to be sent.
        """
        self._conn.send(json.dumps({"cmd": cmd, "data": data}).encode())

    def recv(self) -> Tuple[str, Any]:
        """
        Receive a command and data from the client.

        Returns:
            Tuple[str, Any]: A tuple of command string and data received from the client.
        """
        request = self._conn.recv(4096)
        received_data = json.loads(request.decode())
        return received_data["cmd"], received_data["data"]

    def close(self):
        """
        Close the socket connection.
        """
        self._conn.close()


class Server:

    def __init__(self):
        """
        Creates a socket object for communication with clients.
        """
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def serve(self, ip: str, port: int):
        """
        Start serving on the specified IP address and port number.

        Parameters:
            ip (str): The IP address to bind to.
            port (int): The port number to listen on.
        """
        self._server.bind((ip, port))
        self._server.listen(5)

    def accept(self) -> ServerInstance:
        """
        Accept a new connection and return a ServerInstance instance.

        Returns:
            ServerInstance: A ServerInstance instance associated with the new connection.
        """
        conn, ip = self._server.accept()
        return ServerInstance(conn, ip)

    def close(self):
        """
        Close the server socket, stopping it from accepting further connections.
        """
        self._server.close()

    def get_ip(self) -> str:
        """
        Retrieve the IP address the server socket is currently bound to.

        Returns:
            str: The IP address as a string.
        """
        return self._server.getsockname()[0]

    def get_port(self) -> int:
        """
        Retrieve the port number the server socket is currently bound to.

        Returns:
            int: The port number as an integer.
        """
        return self._server.getsockname()[1]


class ClientInstance:
    def __init__(self):
        """
        Initialize a ClientInstance instance.
        """
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, host: str, port: int):
        """
        Establish a connection to a server at the specified host and port.

        Parameters:
            host (str): The hostname or IP address to connect to.
            port (int): The port number to connect to.
        """
        self._conn.connect((host, port))

    def send(self, cmd: str, data: Any):
        """
        Send a command and data to the server.

        Parameters:
            cmd (str): The command string to be sent.
            data (Any): The data to be sent.
        """
        self._conn.send(json.dumps({"cmd": cmd, "data": data}).encode())

    def recv(self) -> Tuple[str, Any]:
        """
        Receive a command and data from the server.

        Returns:
            Tuple[str, Any]: A tuple containing the command string and associated data received from the server.
        """
        request = self._conn.recv(4096)
        received_data = json.loads(request.decode())
        return received_data["cmd"], received_data["data"]

    def close(self):
        """
        Close the socket connection.
        """
        self._conn.close()
