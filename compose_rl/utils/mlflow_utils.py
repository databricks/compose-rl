import os
from typing import Any, Optional

import mlflow
import torch.distributed as dist

from composer.utils import dist as composer_dist


def get_mlflow_run_id() -> Optional[str]:
    return os.environ.get('MLFLOW_RUN_ID', None)


def get_mlflow_relative_path_for_save_folder(save_folder: str) -> str:
    """Returns the relative path for the given save folder"""
    return save_folder.lstrip('/')


def get_mlflow_absolute_path_for_save_folder(save_folder: str) -> str:
    """Returns the mlflow artifact path for the given save folder"""
    mlflow_prefix = 'dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}'
    mlflow_artifact_path = os.path.join(mlflow_prefix, 'artifacts', get_mlflow_relative_path_for_save_folder(save_folder))
    return mlflow_artifact_path


def validate_save_folder(save_folder: str) -> None:
    """Validates the save folder"""
    if save_folder.startswith("dbfs:/"):
        raise ValueError(f"Using dbfs save_folder ({save_folder}) to store checkpoints is not supported. Please use a local save_folder.")


def artifact_exists_on_mlflow(artifact_path: str) -> bool:
    """Return True if artifact_path exists (file or directory) for the run.

    Artifact path needs to be a relative path to the save folder.
    """
    client = mlflow.MlflowClient()
    run_id = get_mlflow_run_id()
    assert run_id is not None, "Run ID must be set"

    # Walk down the path parts level-by-level
    parent = ""
    if artifact_path:
        parts = artifact_path.split("/")
        for i, part in enumerate(parts):
            entries = {os.path.basename(fi.path): fi for fi in client.list_artifacts(run_id, parent)}
            if part not in entries:
                return False
            fi = entries[part]
            is_last = (i == len(parts) - 1)
            if not is_last and not fi.is_dir:
                # trying to descend into a file
                return False
            parent = fi.path  # descend

    # If we got here, the path exists (root or found item).
    return True


def get_valid_mlflow_experiment_name(config: Any) -> str:
    """Fixes the experiment name to be an absolute path for mlflow.

    MLflow requires the experiment name to be an absolute path.
    If the experiment name is not an absolute path, we turn it
    into an absolute path.
    """
    mlflow_experiment_name = config.loggers.mlflow.experiment_name
    if mlflow_experiment_name.startswith('/'):
        return mlflow_experiment_name
    else:
        from databricks.sdk import WorkspaceClient
        return f'/Users/{WorkspaceClient().current_user.me().user_name}/{mlflow_experiment_name}'


def get_mlflow_run_name(config: Any) -> str:
    """Gets the mlflow run name from the config.

    If the run name is not set in the config, it will return the COMPOSER_RUN_NAME environment variable
    as this is set for interactive mode as well.
    """
    try:
        return config.loggers.mlflow.run_name
    except:
        return os.environ['COMPOSER_RUN_NAME']

# NOTE: This doesn't work yet for a few reasons:
# 1. Downloading nested mlflow artifacts doesn't work correctly due to the MlflowObjectStore
# having issues. For instance, https://github.com/mosaicml/composer/blob/4ae29b1afec56ce2d54f6fa07a7f9578a0d364b0/composer/utils/object_store/mlflow_object_store.py#L465-L476
# requires `tmp_path = os.path.join(tmp_dir, os.path.basename(artifact_path))` instead of what it currently
# does. By doing that, the symlink can be loaded correctly.
# 2. If save_folder is an absolute path (e.g. /tmp/checkpoints), the symlink will be created using this
# absolute path. This is not a valid symlink in mlflow so we need to do some os.path gymnastics to
# support absolute paths for save_folder.
# 3. We also need to support save_folder being a dbfs path eventually.
# Proposed Approach
# - Create an MlflowCheckpointActor (allowing us to set WORLD_SIZE=1)
# and create functions within that are based on MlflowObjectStore.
# that safely handle dbfs paths and absolute paths.
def get_file(path: str, destination: str, overwrite: bool = True):
    """
    A helper function to get a file from mlflow. The existing mlflow utils code
    uses dist operations which isn't supported in the RolloutAgent so this approach
    works around that limitation.
    """
    from composer.utils.file_helpers import parse_uri, get_file as composer_get_file
    from composer.utils.object_store import MLFlowObjectStore
    backend, _, path = parse_uri(path)
    assert backend == 'dbfs', "Only dbfs paths are supported"
    object_store = MLFlowObjectStore(path)
    composer_get_file(path, destination, object_store, overwrite)


def setup_mlflow(config: Any):
    """
    Sets up mlflow for the current process.

    This function should be called before any other mlflow functions are called.
    It will set the mlflow experiment and run. It will create both if they don't exist.
    It will set all environment variables needed for mlflow.
    """
    dist.init_process_group(backend='gloo')
    mlflow.set_tracking_uri('databricks')

    mlflow_experiment_name = get_valid_mlflow_experiment_name(config)
    setattr(config.loggers.mlflow, 'experiment_name', mlflow_experiment_name)
    mlflow_run_name = get_mlflow_run_name(config)
    setattr(config.loggers.mlflow, 'run_name', mlflow_run_name)

    # get mlflow experiment if it exists, otherwise create it and set it to all ranks.
    experiment_id = None
    if composer_dist.get_global_rank() == 0:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)
        else:
            experiment_id = experiment.experiment_id
    experiment_id_broadcast_list = [experiment_id]
    composer_dist.broadcast_object_list(experiment_id_broadcast_list, src=0)
    experiment_id = experiment_id_broadcast_list[0]

    mlflow.set_experiment(experiment_id=experiment_id)

    # get mlflow run if it exists and we are autoresuming, otherwise create it and set it to all ranks.
    run_id = None
    if composer_dist.get_global_rank() == 0:
        existing_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags.run_name = "{mlflow_run_name}"',
            output_format='list',
        ) if config.autoresume else []
        if len(existing_runs) > 0:
            run_id = existing_runs[0].info.run_id
            print(f'Resuming mlflow run with run id: {run_id}')
        else:
            run_id = mlflow.start_run(run_name=mlflow_run_name).info.run_id
            print(f'Creating new mlflow run with run id: {run_id}')
    run_id_broadcast_list = [run_id]
    composer_dist.broadcast_object_list(run_id_broadcast_list, src=0)
    run_id = run_id_broadcast_list[0]

    # set all the right enviornment variables
    assert run_id is not None and experiment_id is not None, "Run ID and experiment ID must be set"
    os.environ['MLFLOW_RUN_ID'] = run_id
    os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'

    dist.destroy_process_group()
