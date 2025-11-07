

from roboflow import Roboflow
rf = Roboflow(api_key="GJHIraxqb0eIV9nna3Uo")
project = rf.workspace("foot-detection-smjrs").project("human-foot-object-detction-we9gi")
version = project.version(6)
dataset = version.download("coco")
                