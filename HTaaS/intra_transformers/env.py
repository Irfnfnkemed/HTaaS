import os


def rank() -> int:
    return int(os.getenv("RANK", "0"))

def local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))

def world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))

def proxy_host() -> str:
    return os.getenv("PROXY_HOST", "0.0.0.0")

def proxy_port() -> int:
    return int(os.getenv("PROXY_PORT", "0"))