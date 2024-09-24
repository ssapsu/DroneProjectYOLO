#!pip install roboflow

import os
from dotenv import load_dotenv

load_dotenv()

from roboflow import Roboflow
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("percepdrone").project("parcel-collection")
version = project.version(1)
dataset = version.download("yolov8")
