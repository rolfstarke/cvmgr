import fiftyone
from .logging_check import util_log

@util_log("fiftyone_replace", success_text=lambda result, args, kwargs: "dataset_absent OR deleted")
def fiftyone_replace(dataset_name: str, replace: bool = False): 
    try:
        dataset_exists = fiftyone.dataset_exists(dataset_name)

        if dataset_exists:
            if replace:
               
                fiftyone.delete_dataset(dataset_name)

            else:
                return False
    except Exception as e:
        return False
        
    return True