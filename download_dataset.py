
from roboflow import Roboflow
rf = Roboflow(api_key="0drstMeKLPaoMjsFlgKz")
project = rf.workspace("fakhaars-workspace").project("skardu_road_hazards")
version = project.version(1)
dataset = version.download("yolov11")


print("Dataset downloaded successfully!")