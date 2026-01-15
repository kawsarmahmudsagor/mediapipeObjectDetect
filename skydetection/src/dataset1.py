from roboflow import Roboflow
rf = Roboflow(api_key="GJHIraxqb0eIV9nna3Uo")
project = rf.workspace("fajr-zafar").project("wall-segmentation-yvsfp")
version = project.version(5)
dataset = version.download("coco")
                