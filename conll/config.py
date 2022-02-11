import sys
import json

from transformers import AutoTokenizer

#importing the module 
import logging 

from datetime import datetime
from pathlib import Path

now = datetime.now()
date_time_am = now.strftime("%Y_%b_%d_%H_%M%p")

logsfolder = "logs/"
Path(logsfolder).mkdir(parents=True, exist_ok=True)

#now we will Create and configure logger 
logging.basicConfig(filename=f"{logsfolder}/log_{date_time_am}.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

logger_file = f"log_{date_time_am}.log"

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

#some messages to test
logger.info("config loaded") 

with open("config_settings/conll.json","r") as fh:
    params = json.load(fh)

def check_half_p(params):
    if params["CUDA"]==False:
        if params["HALF_PRECISION"]==True: return False
    return True

assert check_half_p(params), "Half precision training cannot start in CPU mode. Set the flag `HALF_PRECISION` to `false` if using in CPU mode"


RANDOM_STATE = 42 #Use this for consistent shuffle/splits.


TOKENIZER = AutoTokenizer.from_pretrained(params["TOKENIZER_PATH"])
