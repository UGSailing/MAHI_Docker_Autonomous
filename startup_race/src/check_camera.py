import subprocess
import time

PING_IPS = ["192.168.77.100", "192.168.77.99"]
DEFAULT_TIMEOUT = 30


def ping_host(ip: str, timeout: int) -> bool:
    print(f"Pinging {ip} with timeout {timeout} seconds...")
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout), ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 2,
        )
        print(f"Ping result for {ip}: returncode={result.returncode}")
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def check(timeout: int = DEFAULT_TIMEOUT) -> bool:
    start = time.time()
    for ip in PING_IPS:
        elapsed = time.time() - start
        remaining = max(1, int(timeout - elapsed))
        if remaining <= 0 or not ping_host(ip, remaining):
            return False
    return True

def mock_check() -> bool:
    print("Not implemented yet but the code continues in order to check the statemachine")
    print("waiting 1 second")
    time.sleep(1)
    return True

