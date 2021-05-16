from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'cust-seg'))

    experiment = Experiment(workspace=ws, name='cust-seg')
    config = ScriptRunConfig(
        source_directory = './src',
        script = 'cust-seg.py',
        compute_target = 'Standard10',
        arguments = [
            '--data_path', dataset.as_named_input('input').as_mount()
        ]
    )

    env = ws.environments['AzureML-Tutorial']
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)
