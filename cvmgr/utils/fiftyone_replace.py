import fiftyone
import logging
logger = logging.getLogger('cvmgr')

def fiftyone_replace(dataset_name: str, replace: bool = False): 
    try:
        dataset_exists = fiftyone.dataset_exists(dataset_name)

        if dataset_exists:
            if replace:
               
                fiftyone.delete_dataset(dataset_name)

            else:
                logger.info(f"Dataset {dataset_name} already exists. Skipping download/import.")
                return False
    except Exception as e:
     
        logger.error(f"Error checking/deleting dataset {dataset_name}: {e}")
        return False
        
    return True