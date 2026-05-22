
import fiftyone.zoo 
from huggingface_hub import login
import pathlib
import yaml
from .logging_check import util_log

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)


@util_log("sam3_get_model", success_text=lambda result, args, kwargs: "source_registered OR already_present")
def sam3_get_model():
    
    login(secrets_yaml["huggingface"]["token"])


    if "https://github.com/harpreetsahota204/sam3_images" in fiftyone.zoo.list_zoo_model_sources():
        return True

    fiftyone.zoo.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")
    return "https://github.com/harpreetsahota204/sam3_images" in fiftyone.zoo.list_zoo_model_sources()