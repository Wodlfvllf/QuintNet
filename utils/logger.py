import sys
import os
import torch.distributed as dist

class Logger(object):
    """
    Logger that writes to both a file and the original stdout/stderr.
    """
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_rank_logging(log_dir="logs"):
    """
    Sets up per-rank logging.
    Redirects stdout and stderr to a file named 'rank_{rank}.log' in the specified directory.
    """
    rank = dist.get_rank()
    
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass # Race condition in multi-process creation
            
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    
    # Redirect stdout and stderr
    sys.stdout = Logger(log_file, sys.stdout)
    sys.stderr = Logger(log_file, sys.stderr)
    
    print(f"[Rank {rank}] Logging initialized. Writing to {log_file}", flush=True)
