from roboflow import Roboflow
rf = Roboflow(api_key="V7gbNKr6N2UZG7FvzG6H")
project = rf.workspace("xr23").project("logo_detection-9lwbt")
version = project.version(2)
dataset = version.download("coco")
                
                
                