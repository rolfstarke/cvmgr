import argparse
import fiftyone
import fiftyone.utils.random
import yaml
import pathlib
from cvmgr import fiftyone_download



# create an ArgumentParser object and a mutually exclusive group. add command line arguments and parse the arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument()