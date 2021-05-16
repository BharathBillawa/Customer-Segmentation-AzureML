from azureml.core import Workspace

ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(
    src_dir = './data/cust-seg',
    target_path = 'cust-seg',
    overwrite = True
)
