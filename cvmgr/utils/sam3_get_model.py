
import logging
import fiftyone.zoo 
from huggingface_hub import login
import pathlib
import yaml

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)


def sam3_get_model():
    
    login(secrets_yaml["huggingface"]["token"])
    
    if "https://github.com/harpreetsahota204/sam3_images" in fiftyone.zoo.list_zoo_model_sources():
        logging.info("SAM3 model source already registered.")
    else:
        fiftyone.zoo.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")
        logging.info("SAM3 model source registered.")