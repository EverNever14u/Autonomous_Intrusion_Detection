import time

# ANSI color codes
class Color:
    GREEN = '\033[92m'
    RESET = '\033[0m'

# Function to format time in green
def get_duration(start):
    duration_s = (time.time() - start)
    return f"{Color.GREEN}({duration_s:.2f}s){Color.RESET}"