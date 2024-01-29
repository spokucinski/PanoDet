import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)

session.wait()

i = 4

#pgrep mongod
#kill -9 [ID]