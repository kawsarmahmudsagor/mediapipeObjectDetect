from roboflow import Roboflow
rf = Roboflow(api_key="V7gbNKr6N2UZG7FvzG6H")
project = rf.workspace("xr23").project("sky-detection-ehdlv")
version = project.version(4)
dataset = version.download("coco")
                
                
                
                
                
                
                
                
                