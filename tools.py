import time

# ANSI color codes
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    
    RESET = '\033[0m'

    
# Function to format time in green
def get_duration(start):
    duration_s = (time.time() - start)
    if duration_s < 1:
        return f"{Color.GREEN}({duration_s:.2f}s){Color.RESET}"
    elif duration_s < 60:
        return f"{Color.YELLOW}({duration_s:.2f}s){Color.RESET}"
    else :
        return f"{Color.RED}({duration_s:.2f}s){Color.RESET}"

# Parsing bools from env
def str_to_bool(val):
    return str(val).lower() in ("true", "1", "yes")