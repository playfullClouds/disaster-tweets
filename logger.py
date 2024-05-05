import logging
import os
import sys
from datetime import datetime


project_root = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    # filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(module)s - %(message)s",
    # level = logging.INFO,
    level=logging.NOTSET,
    
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)


log = logging.getLogger(project_root)
log.setLevel(logging.INFO)  

if __name__ == '__main__':
    log.info("Logging has started")