import os


def rank() -> int:
    return int(os.getenv("RANK", "0"))

def local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))

def world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))

def processor_host() -> str:
    return os.getenv("PROCESSOR_HOST", "0.0.0.0")

def processor_port() -> int:
    return int(os.getenv("PROCESSOR_PORT", "0"))
