from azureml.core import Model
from azureml.core import Workspace

ws = Workspace.from_config()
model = Model.register(
    workspace = ws,
    model_path = 'src/cust_seg_model.joblib',
    model_name = 'cust_seg_model'
)
