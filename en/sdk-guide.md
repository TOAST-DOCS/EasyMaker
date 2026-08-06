<!-- pre-align:aligned sig=348e352651f6 -->

<a id="ai.easymaker.sdk.guide"></a>
## Machine Learning > AI EasyMaker > SDK User Guide { #ai.easymaker.sdk.guide }

<a id="sdk.settings"></a>
## SDK Settings { #sdk.settings }

<a id="sdk.settings.sdk.install"></a>
### Install AI EasyMaker Python SDK { #sdk.settings.sdk.install }

```bash
python -m pip install easymaker
```

- AI EasyMaker is installed in the notebook by default.

<a id="sdk.settings.sdk.init"></a>
### Initialize AI EasyMaker SDK { #sdk.settings.sdk.init }

You can find the AppKey in the **URL & Appkey** menu at the right top on the console.
You can learn more about Access Tokens in [User Access Key Token](https://docs.nhncloud.com/en/nhncloud/en/public-api/user-access-key-token/).
Enter the AppKey, AccessToken, and region information of enabled AI EasyMaker.
Intialization code is required to use the AI EasyMaker SDK.

```python
import easymaker

easymaker.init(
    appkey='EASYMAKER_APPKEY',
    region='kr1',
    access_token='EASYMAKER_ACCESS_TOKEN',
    experiment_id="EXPERIMENT_ID", # Optional
)
```

<a id="experiment"></a>
## Experiment { #experiment }

<a id="experiment.create"></a>
### Create Experiment { #experiment.create }

Before creating a training, you must create an experiment to sort trainings.

[Parameter]

| Name                     | Type      | Required | Default value | Valid range          | Description                                |
|------------------------|---------|-------|---------------|----------------------|--------------------------------------------|
| experiment_name        | String  | Required    | None          | Up to 50 characters  | Experiment name                            |
| description | String  | Optional    | None          | Up to 255 characters | Description for experiment                 |
| wait                   | Boolean | Optional    | True          | True, False          | True: return after creation is complete, False: return upon creation request  |

```python
experiment = easymaker.Experiment().create(
    experiment_name='experiment_name',
    description='experiment_description',
    # wait=False,
)
```

<a id="experiment.list"></a>
### List Experiments { #experiment.list }

```python
experiment_list = easymaker.Experiment.get_list()
for experiment in experiment_list:
    experiment.print_info()
```

<a id="experiment.delete"></a>
### Delete Experiment { #experiment.delete }

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | Required    | None   | Up to 36 characters | Experiment ID |

```python
easymaker.Experiment(experiment_id).delete()
```

<a id="training"></a>
## Training { #training }

<a id="training.image.list"></a>
### List Images { #training.image.list }

```python
image_list = easymaker.Training.get_image_list()
for image in image_list:
    image.print_info()
```

<a id="training.instance.list"></a>
### List Instances { #training.instance.list }

```python
instance_type_list = easymaker.Training.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="training.create"></a>
### Create Training { #training.create }

[Parameter]

| Name                                     | Type      | Required                                             | Default value | Valid range                             | Description                                                                                     |
|------------------------------------------|---------|------------------------------------------------------|---------------|-----------------------------------------|-------------------------------------------------------------------------------------------------|
| experiment_id                            | String  | Required if not entered in easymaker.init            | None          |  Up to 36 characters                                    | Experiment ID                                                                                   |
| training_name                            | String  | Required                                             | None          | Up to 50 characters                     | Training name                                                                                   |
| description                     | String  | Optional                                             | None          | Up to 255 characters                    | Description for training                                                                        |
| image_name                         | String  | Required                                             | None          | None                                    | Image name to be used for training (Inquiry available with CLI)                                 |
| instance_type_name                      | String  | Required                                             | None          | None                                    | Instance type name (Inquiry available with CLI)                                                 |
| distributed_node_count                   | Integer | Optional                                             | 1          | 1~10                                    | Number of nodes to apply distributed training to                                               |
| use_torchrun                             | Boolean | Optional                                             | False         | True, False                             | Whether torchrun is enabled, only available for Pytorch images                                  |
| nproc_per_node                           | Integer | Required when use_torchrun is True                   | 1             | 1 to (number of CPUs or number of GPUs) | Number of processes per node, value that must be set if use_torchrun is enabled                 |
| data_storage_size                        | Integer | Required when using Object Storage                   | None          | 300~10000                               | Storage size to download data for training (unit: GB), unnecessary when using NAS               |
| algorithm_name                           | String  | Required when using algorithms provided by NHN Cloud | None          | Up to 64 characters                     | Algorithm name (Inquiry available with CLI)                                                     |
| source_dir_uri                           | String  | Required when using own algorithm                    | None          | Up to 255 characters                    | Path of files required for training (NHN Cloud Object Storage or NHN Cloud NAS)                 |
| entry_point                              | String  | Required when using own algorithm                    | None          | Up to 255 characters                    | Information of Python files to be executed initially in source_dir_uri                          |
| model_upload_uri                         | String  | Required                                             | None          | Up to 255 characters                    | Path to upload the model completed with training (NHN Cloud Object Storage or NHN Cloud NAS)    |
| check_point_input_uri                    | String  | Optional                                             | None          | Up to 255 characters                    | Input checkpoint file path (NHN Cloud Object Storage or NHN Cloud NAS)                          |
| check_point_upload_uri                   | String  | Optional                                             | None          | Up to 255 characters                    | The path where the checkpoint file will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS) |
| timeout_hours                            | Integer | Optional                                             | 720           | 1~720                                   | Max training time (unit: hour)                                                                  |
| hyperparameter_list                      | easymaker.Parameter Array   | Optional                                             | None          | Max 100                                 | Information of hyperparameters (consists of parameterName/parameterValue)                       |
| hyperparameter_list[0].parameter_name      | String  | Optional                                             | None          | Up to 255 characters                    | hyperparameter key                                                                                  |
| hyperparameter_list[0].parameter_value    | String  | Optional                                             | None          | Up to 1000 characters                   | Parameter value                                                                                 |
| dataset_list                             | easymaker.Dataset Array   | Required                                             | None          | Max 10                                  | Information of dataset to be used for training (consists of dataset_name/data_uri)                |
| dataset_list[0].dataset_name              | String  | Required                                             | None          | Up to 36 characters                     | Data name                                                                                  |
| dataset_list[0].data_uri               | String  | Required                                             | None          | Up to 255 characters                    | Data path                                                                                  |
| use_log                                  | Boolean | Optional                                             | False         | True, False                             | Whether to leave logs in the Log & Crash Search service                                         |
| wait                                     | Boolean | Optional                                             | True          | True, False                             | True: return after creation is complete, False: return upon creation request                    |

```python
training = easymaker.Training().run(
    experiment_id=experiment.experiment_id, # Optional if already set in init
    training_name='training_name',
    description='training_description',
    image_name='Ubuntu 18.04 CPU TensorFlow Training',
    instance_type_name='m2.c4m8',
    distributed_node_count=1,
    data_storage_size=300,  # minimum size : 300GB
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_list=[
        easymaker.Parameter(
            parameter_name= "epochs",
            parameter_value= "10",
        ),
        easymaker.Parameter(
            parameter_name= "batch-size",
            parameter_value= "30",
        ),
    ],
    timeout_hours=100,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_input_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_input_path}',
    check_point_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
     easymaker.Dataset(
            dataset_name= "train",
            data_uri= "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_path}",
        ),
        easymaker.Dataset(
            dataset_name= "test",
            data_uri= "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_path}",
        ),
    ],
    use_log=True,
    # wait=False,
)
```

<a id="training.list"></a>
### List of Training { #training.list }

```python
training_list = easymaker.Training.get_list()
for training in training_list:
    training.print_info()
```

<a id="training.delete"></a>
### Delete Training { #training.delete }

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | Required    | None   | Up to 36 characters | Training ID |

```python
easymaker.Training(training_id).delete()
```

<a id="hyperparameter.tuning"></a>
## Hyperparameter Tuning { #hyperparameter.tuning }

<a id="hyperparameter.tuning.image.list"></a>
### List Images { #hyperparameter.tuning.image.list }

```python
image_list = easymaker.HyperparameterTuning.get_image_list()
for image in image_list:
    image.print_info()
```

<a id="hyperparameter.tuning.instance.list"></a>
### List Instances { #hyperparameter.tuning.instance.list }

```python
instance_type_list = easymaker.HyperparameterTuning.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="hyperparameter.tuning.create"></a>
### Create Hyperparameter Tuning { #hyperparameter.tuning.create }

[Parameter]

| Name                                                             | Type             | Required                                                            | Default value   | Valid range                                                                   | Description                                                                                                                  |
|----------------------------------------------------------------|----------------|---------------------------------------------------------------------|-------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| experiment_id                                                  | String         |  Required if not entered in easymaker.init                                     | None    | Up to 36 characters                                                                         | Experiment ID                                                                                  |
| hyperparameter_tuning_name                                     | String         | Required                                                            | None    | Up to 50 characters                                                           | Hyperparameter Tuning Name                                                                                  |
| description                              | String         | Optional                                                            | None    | Up to 255 characters                                                          | Description of hyperparameter tuning                                          |
| image_name                                                     | String         | Required                                                            | None    | None                                                                          | Image name to be used for hyperparameter tuning (Inquiry available with CLI)                                                    |
| instance_type_name                                                  | String         | Required                                                            | None    | None                                                                          | Instance type name (Inquiry available with CLI)                                                                            |
| distributed_node_count                                         | Integer        | Required                                                            | 1      | The product of distributed_node_count and parallel_trial_count is 10 or less. | Number of nodes to apply distributed learning per each training in hyperparameter tuning                                           |
| parallel_trial_count                                           | Integer        | Required                                                            | 1      | The product of distributed_node_count and parallel_trial_count is 10 or less. | Number of trainings to run in parallel in hyperparameter tuning                                                              |
| use_torchrun                                                   | Boolean        | Optioanl                                                            | False  | True, False                                                                   | Use torchrun or not, Only available in Pytorch images                                                                        |
| nproc_per_node                                                 | Integer        | Required when use_torchrun is True                                  | 1      | 1~(Number of CPUs or GPUs)                                                    | Number of processes per node, Required when use_torchrun is used                                                             |
| data_storage_size                                              | Integer        | Required when using Object Storage                                  | None    | 300~10000                                                                     | Size of storage space to download data required for hyperparameter tuning (unit: GB), not required when using NAS            |
| algorithm_name                                                 | String         | Required when using algorithms provided by NHN Cloud                | None    | Up to 64 characters                                                           | Algorithm name (Inquiry available with CLI)                                                                                  |
| source_dir_uri                                                 | String         | Required when using own algorithm                                   | None    | Up to 255 characters                                                          | Path containing files required for hyperparameter tuning (NHN Cloud Object Storage or NHN Cloud NAS)                         |
| entry_point                                                    | String         | Required when using own algorithm                                   | None    | Up to 255 characters                                                          | Information of Python files to be executed initially in source_dir_uri                                                       |
| model_upload_uri                                               | String         | Required                                                            | None    | Up to 255 characters                                                          | The path where the trained model in hyperparameter tuning will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS)       |
| check_point_input_uri                                          | String         | Optional                                                            | None    | Up to 255 characters                                                          | Input checkpoint file path (NHN Cloud Object Storage or NHN Cloud NAS)                                                       |
| check_point_upload_uri                                         | String         | Optional                                                            | None    | Up to 255 characters                                                          | The path where the checkpoint file will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS)                              |
| timeout_hours                                                  | Integer        | Optional                                                            | 720   | 1~720                                                                         | Maximum hyperparameter tuning time (unit: hours)                                                                             |
| hyperparameter_spec_list                                       | easymaker.HyperparameterSpec Array          | Optional                                                            | None    | Up to 100                                                                     | Hyperparameter specification information                                                                        |
| hyperparameter_spec_list[0].<br>hyperparameter_name             | String         | Optional                                                            | None    | Up to 255 characters                                                          | Hyperparameter name                                                                                |
| hyperparameter_spec_list[0].<br>hyperparameter_type_code         | easymaker.HyperparameterTypeCode         | Optional                                                            | None    | INT, DOUBLE, DISCRETE, CATEGORICAL                                            | Hyperparameter Type                                                                              |
| hyperparameter_spec_list[0].<br>hyperparameter_min_value         | String | Required if hyperparameterTypeCode is INT, DOUBLE(Enter a number as a string type)                   | None    | None                                                                          | Hyperparameter minimum value                                                                                |
| hyperparameter_spec_list[0].<br>hyperparameter_max_value         | String | Required if hyperparameterTypeCode is INT, DOUBLE(Enter a number as a string type)                 | None    | None                                                                          | Hyperparameter maximum value                                                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameter_step             | String | Required if hyperparameterTypeCode is INT, DOUBLE and GRID strategy | None    | None                                                                          | Magnitude of change in hyperparameter values when using the "Grid" tuning strategy                                           |
| hyperparameter_spec_list[0].<br>hyperparameter_specified_values  | String         | Required if hyperparameterTypeCode is DISCRETE or CATEGORICAL       | None    | Up to 3000 characters                                                         | A list of defined hyperparameters (strings or numbers separated by `,`)                 |
| dataset_list                                                   | easymaker.Dataset Array          | Required                                                            | None    | Max 10                                                                        | Dataset information to be used for hyperparameter tuning (configured as dataset_name/data_uri)                                 |
| dataset_list[0].dataset_name                                    | String         | Required                                                            | None    | Up to 36 characters                                                           | Data name                                                                                 |
| dataset_list[0].dataset_uri                                     | String         | Required                                                            | None    | Up to 255 characters                                                          | Data path                                                                               |
| metric_list                                                    | easymaker.Metric          | Required when using own algorithm                                   | None    | Up to 10 (string list of indicator names)                                     | Define which metrics to collect from logs output by the training code.                                                       |
| metric_list[0].name                                              | String                             | Required when using own algorithm                                             | None    | None                                                     | Metric name                                                                      |
| metric_regex                                                   | String         | Select when using own algorithm                                     | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?)                             | Up to 255 characters                                                                                                         | Enter a regular expression to use to collect metrics. The training algorithm should output metrics to match the regular expression.                                                          |
| objective_metric_name                                          | String         | Required when using own algorithm                                   | None    | Up to 36 characters, one of metric_list                                       | Choose which metrics you want to optimize for.                                                                               |
| objective_type_code                                            | easymaker.ObjectiveTypeCode         | Required when using own algorithm                                   | None    | MINIMIZE, MAXIMIZE                                                            | Choose a target metric optimization type.                                                                                    |
| objective_goal                                                 | Double         | Optional                                                            | None    | None                                                                          | The tuning job ends when the target metric reaches this value.                                                               |
| max_failed_trial_count                                         | Integer        | Optional                                                            | None    | None                                                                          | Define the maximum number of failed lessons. When the number of failed trainings reaches this value, tuning ends in failure. |
| max_trial_count                                                | Integer        | Optional                                                            | None    | None                                                                          | Defines the maximum number of lessons. Tuning runs until the number of auto-run training reaches this value.                 |
| tuning_strategy_name                                           | easymaker.TuningStrategy         | Required                                                            | None    | None                                                                          | Choose which strategy to use to find the optimal hyperparameters.                                                            |
| tuning_strategy_random_state                                   | Integer        | Optional                                                            | None    | None                                                                          | Determine random number generation. Specify a fixed value for reproducible results.                                          |
| early_stopping_algorithm                                       | easymaker.EarlyStoppingAlgorithm         | Required                                                            | None    | EarlyStoppingAlgorithm.<br>MEDIAN                                           | Stop training early if the model is no longer good even though training continues.                                           |
| early_stopping_min_trial_count                                 | Integer        | Optional                                                            | 3     | None                                                                          | Define how many trainings the target metric value will be taken from when calculating the median.                            |
| early_stopping_start_step                                      | Integer        | Optional                                                            | 4     | None                                                                          | Set the training step from which to apply early stop.                                                                        |
| use_log                                                        | Boolean        | Optional                                                            | False | True, False                                                                   | Whether to leave logs in the Log & Crash Search service                                                                      |
| wait                                                           | Boolean        | Optional                                                            | True  | True, False                                                                   | True: return after creation is complete, False: return upon creation request                                                                                   |

```python
hyperparameter_tuning = easymaker.HyperparameterTuning().run(
    experiment_id=experiment.experiment_id, # Optional if already set in init
    hyperparameter_tuning_name='hyperparameter_tuning_name',
    description='hyperparameter_tuning_description',
    image_name='Ubuntu 18.04 CPU TensorFlow Training',
    instance_type_name='m2.c8m16',
    distributed_node_count=1,
    parallel_trial_count=1,
    data_storage_size=300,
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_spec_list=[
        easymaker.HyperparameterSpec(
            hyperparameter_name="learning_rate",
            hyperparameter_type_code=easymaker.HyperparameterTypeCode.DOUBLE,
            hyperparameter_min_value="0.01",
            hyperparameter_max_value="0.05",
        ),
        easymaker.HyperparameterSpec(
            hyperparameter_name="epochs",
            hyperparameter_type_code=easymaker.HyperparameterTypeCode.INT,
            hyperparameter_min_value="100",
            hyperparameter_max_value="1000",
        )
    ],
    timeout_hours=10,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_input_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_input_path}',
    check_point_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        easymaker.Dataset(
            dataset_name="train",
            data_uri= "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_path}"
        ),
        easymaker.Dataset(
            dataset_name="test",
            data_uri="obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_path}"
        )
    ],
    metric_list=[
        easymaker.Metric(name="loss"),
        easymaker.Metric(name="accuracy"),
        easymaker.Metric(name="val_loss"),
    ],
    metric_regex='([\w|-]+)\s*:\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?)',
    objective_metric_name="val_loss",
    objective_type_code=easymaker.ObjectiveTypeCode.MINIMIZE,
    objective_goal=0.01,
    max_failed_trial_count=3,
    max_trial_count=10,
    tuning_strategy_name=easymaker.TuningStrategy.BAYESIAN_OPTIMIZATION,
    tuning_strategy_random_state=1,
    early_stopping_algorithm=easymaker.EarlyStoppingAlgorithm.MEDIAN,
    early_stopping_min_trial_count=3,
    early_stopping_start_step=4,
    use_log=True,
    # wait=False,
)
```

<a id="hyperparameter.tuning.list"></a>
### List Hyperparameter Tuning { #hyperparameter.tuning.list }

```python
hyperparameter_tuning_list = easymaker.HyperparameterTuning.get_list()
for hyperparameter_tuning in hyperparameter_tuning_list:
    hyperparameter_tuning.print_info()
```

<a id="hyperparameter.tuning.delete"></a>
### Delete Hyperparameter Tuning { #hyperparameter.tuning.delete }

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description           |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | Required    | None   | Up to 36 characters | Hyperparameter Tuning ID |

```python
easymaker.HyperparameterTuning(hyperparameter_tuning_id).delete()
```

<a id="fine.tuning"></a>
## Fine Tuning { #fine.tuning }

A feature that specializes model performance by performing additional training on a pre-trained large language model using a dataset tailored to a specific domain or task.

<a id="fine.tuning.model.preset.list"></a>
### List Base Models { #fine.tuning.model.preset.list }

Retrieves a list of base models available for fine tuning.

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| ----------------- | ------ | ----- | --- | ----- | ------------------------------- |
| model_preset_name | String | Optional | None | None | Base model name (filter by name; retrieves all if not entered) |

```python
base_model_list = easymaker.FineTuning.get_base_model_list()
for base_model in base_model_list:
    base_model.print_info()

# Select one of the retrieved base models
base_model = base_model_list[0]
base_model_preset_id = base_model.model_preset_id
```

<a id="fine.tuning.instance.list"></a>
### List Instance Types { #fine.tuning.instance.list }

Retrieves a list of instance types available for the selected base model preset (`model_preset_id`).

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| --------------- | ------ | ----- | --- | ------ | ------------- |
| model_preset_id | String | Optional | None | Up to 36 characters | Base model preset ID |

```python
instance_type_list = easymaker.FineTuning.get_instance_type_list(model_preset_id=base_model_preset_id)
for instance in instance_type_list:
    instance.print_info()
```

<a id="fine.tuning.parameter.spec.list"></a>
### Retrieve Hyperparameter Specifications { #fine.tuning.parameter.spec.list }

Retrieves the fine tuning hyperparameter specifications for the selected base model. The retrieved specifications can be used to build a hyperparameter list with default values.

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| -------------------- | ------ | ----- | --- | ------ | ------------- |
| base_model_preset_id | String | Required | None | Up to 36 characters | Base model preset ID |

```python
parameter_spec_list = easymaker.FineTuning.get_parameter_spec_list(
    base_model_preset_id=base_model_preset_id,
)
for spec in parameter_spec_list:
    spec.print_info()

# Build hyperparameter list using default values for parameters that have defaults (modify values if needed)
hyperparameter_list = [
    easymaker.Parameter(parameter_name=spec.parameter_name, parameter_value=spec.default_value)
    for spec in parameter_spec_list
    if spec.default_value is not None
]
```

<a id="fine.tuning.create"></a>
### Create Fine Tuning { #fine.tuning.create }

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| -------------------------------------- | --------------------------- | ------------------------- | ----- |---------------------------|-----------------------------------------------------------------|
| experiment_id | String | Required if not entered in easymaker.init | None | Up to 36 characters | Experiment ID |
| experiment_name | String | Optional | None | Up to 50 characters | New experiment name (used when creating an experiment at the same time) |
| experiment_description | String | Optional | None | Up to 255 characters | Description for the new experiment |
| fine_tuning_name | String | Required | None | Up to 50 characters | Fine tuning name |
| description | String | Optional | None | Up to 255 characters | Description for the fine tuning |
| flavor_name | String | Required | None | None | Instance type name (can be retrieved) |
| instance_count | Integer | Optional | 1 | 1–10 | Number of training instances |
| base_model_preset_id | String | Required | None | Up to 36 characters | Base model preset ID |
| model_upload_uri | String | Required | None | Up to 255 characters | Path where the completed fine tuning model will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS) |
| timeout_hours | Integer | Optional | 720 | 1–720 | Maximum fine tuning duration (unit: hours) |
| hyperparameter_list | easymaker.Parameter Array | Optional | None | Up to 100 | Hyperparameter information (consists of parameter_name/parameter_value) |
| hyperparameter_list[0].parameter_name | String | Optional | None | Up to 255 characters | Hyperparameter key |
| hyperparameter_list[0].parameter_value | String | Optional | None | Up to 1,000 characters | Hyperparameter value |
| dataset_list | easymaker.Dataset Array | Required | None | Up to 10 | Dataset information to be used for fine tuning |
| dataset_list[0].dataset_name | String | Required | None | Up to 36 characters | Data name |
| dataset_list[0].data_uri | String | Required | None | Up to 255 characters | Data path |
| dataset_list[0].dataset_format_code | easymaker.DatasetFormatCode | Required | None | CHAT_TEMPLATE, COMPLETION | Dataset format |
| dataset_list[0].dataset_split_code | easymaker.DatasetSplitCode | Required | TRAIN | TRAIN, VALIDATION | Dataset split (training/validation); at least 1 TRAIN required |
| data_storage_size | Integer | Required when using Object Storage | None | 300–10,000 | Storage size for downloading data required for fine tuning (unit: GB); not required when using NAS |
| validation_split_percent | Integer | Optional | 0 | 0–100 | Percentage (%) of training data to split for validation. If 0, no split is performed. If a VALIDATION dataset is specified, that dataset is used for validation and this value is ignored. |
| use_log | Boolean | Optional | False | True, False | Whether to save logs to the Log & Crash Search service |
| wait | Boolean | Optional | True | True, False | True: Returns after creation is complete; False: Returns immediately after the creation request |

```python
fine_tuning = easymaker.FineTuning().run(
    experiment_id=experiment.experiment_id, # Optional if already set in init
    fine_tuning_name='fine_tuning_name',
    description='fine_tuning_description',
    flavor_name='g4.c92m1800',
    instance_count=1,
    base_model_preset_id=base_model_preset_id,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    timeout_hours=24,
    validation_split_percent=10,  # Use 10% of training data for validation
    hyperparameter_list=[
        easymaker.Parameter(
            parameter_name="epoch",
            parameter_value="1",
        ),
        easymaker.Parameter(
            parameter_name="learning_rate",
            parameter_value="0.0002",
        ),
        easymaker.Parameter(
            parameter_name="batch_size",
            parameter_value="1",
        ),
    ],
    dataset_list=[
        easymaker.Dataset(
            dataset_name="train-dataset",
            data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_path}',
            dataset_format_code=easymaker.DatasetFormatCode.CHAT_TEMPLATE,
            dataset_split_code=easymaker.DatasetSplitCode.TRAIN,
        ),
    ],
    data_storage_size=300,
    use_log=False,
    # wait=False,
)
```

<a id="fine.tuning.list"></a>
### List Fine Tunings { #fine.tuning.list }

```python
fine_tuning_list = easymaker.FineTuning.get_list()
for fine_tuning in fine_tuning_list:
    fine_tuning.print_info()
```

<a id="fine.tuning.stop"></a>
### Stop Fine Tuning { #fine.tuning.stop }

Stops a fine tuning that is in the RUNNING state.

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| -------------- | ------ | ----- | --- | ------ | -------- |
| fine_tuning_id | String | Required | None | Up to 36 characters | Fine tuning ID |

```python
easymaker.FineTuning(fine_tuning_id).stop()
```

<a id="fine.tuning.delete"></a>
### Delete Fine Tuning { #fine.tuning.delete }

[Parameters]

| Name | Type | Required | Default | Valid range | Description |
| -------------- | ------ | ----- | --- | ------ | -------- |
| fine_tuning_id | String | Required | None | Up to 36 characters | Fine tuning ID |

```python
easymaker.FineTuning(fine_tuning_id).delete()
```


<a id="model"></a>
## Model { #model }

<a id="model.create"></a>
### Create Model { #model.create }

Request to create a model with the training, hyperparameter, and fine tuning ID.
The model is used when creating endpoints.

[Parameter]

| Name                       | Type     | Required                              | Default value | Valid range   | Description                                  |
|----------------------------------|---------------------------|--------------------------|-----|--------------------------------------------------------------|-------------------------------------------|
| model_format_code                | easymaker.ModelFormatCode | Required if fine_tuning_id is not entered | None | TENSORFLOW, PYTORCH, SKLEARN, HUGGING_FACE, TRITON, SAPEON | Model format information used for inference serving |
| training_id                      | String                    | Optional | None | None | Training ID to create as a model |
| hyperparameter_tuning_id         | String                    | Optional | None | None | Hyperparameter tuning ID to create as a model (created from the best training) |
| fine_tuning_id                   | String                    | Optional | None | None | Fine tuning ID to create as a model |
| model_name                       | String                    | Required | None | Up to 50 characters | Model name |
| description                      | String                    | Optional | None | Up to 255 characters | Description for the model |
| parameter_list                   | Array                     | Optional | None | Up to 10 | Parameter information (consists of parameterName/parameterValue) |
| parameter_list[0].parameterName  | String                    | Optional | None | Up to 64 characters | Parameter name |
| parameter_list[0].parameterValue | String                    | Optional | None | Up to 255 characters | Parameter value |

```python
model = easymaker.Model().create(
    model_name='model_name',
    training_id=training.training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning.hyperparameter_tuning_id,
    model_format_code=easymaker.ModelFormatCode.PYTORCH,
    description='model_description',
)
```

```python
model = easymaker.Model().create(
    model_name='model_name',
    fine_tuning_id=fine_tuning.fine_tuning_id,
    description='model_description',
)
```

Even if there is no training, hyperparameter tuning, and fine tuning ID, you can create a model by entering the path information for the model and framework type.

[Parameter]

| Name                   | Type     | Required | Default value | Valid range                                   | Description                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| model_format_code | easymaker.ModelFormatCode | Required | None | TENSORFLOW, PYTORCH, SKLEARN, HUGGING_FACE, TRITON, SAPEON | Model format information used for inference serving |
| model_upload_uri            | String | Required    | None  | Up to 255 characters                                 | Path for model file (NHN Cloud Object Storage or NHN Cloud NAS) |
| model_name           | String | Required    | None  | Up to 50 characters                                  | Model name                                               |
| description    | String | Optional    | None  | Up to 255 characters                                 | Description for model                                           |
| parameter_list                   | Array  | Optional    | None  | Max 10                                  | Information of parameters (consists of parameterName/parameterValue)         |
| parameter_list[0].parameterName  | String | Optional    | None  | Up to 64 characters                     | Parameter name                                              |
| parameter_list[0].parameterValue | String | Optional    | None  | Up to 255 characters                    | Parameter value                                             |

```python
# TensorFlow Model
model = easymaker.Model().create(
    model_format_code=easymaker.ModelFormatCode.TENSORFLOW,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    description='model_description',
)
# HuggingFace Model
model = easymaker.Model().create_hugging_face_model(
    model_name='model_name',
    description='model_description',
    parameter_list=[
        {
            'parameterName': 'model_id',
            'parameterValue': 'huggingface_model_id',
        }
    ],
)
```

<a id="model.list"></a>
### List Models { #model.list }

```python
model_list = easymaker.Model.get_list()
for model in model_list:
    model.print_info()
```

<a id="model.delete"></a>
### Delete Model { #model.delete }

[Parameter]

| Name                        | Type      | Required | Default value  | Valid range  | Description    |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | Required    | None   | Up to 36 characters | Model ID |

```python
easymaker.Model(model_id).delete()
```

<a id="model.evaluation"></a>
## Evaluate Models { #model.evaluation }

<a id="model.evaluation.instance.list"></a>
### List Instances { #model.evaluation.instance.list }

```python
instance_type_list = easymaker.ModelEvaluation.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="model.evaluation.create"></a>
### Create a model evaluation { #model.evaluation.create }

Create a model evaluation that measures the performance metrics of the model. Batch inference is run with the selected model and the evaluation metrics are saved.

[Parameter]

| Name                                        | Type      | Required | Default value   | Valid range                                          | Description                                                              |
|-------------------------------------------|---------|-------|-------|------------------------------------------------|-----------------------------------------------------------------|
| model_evaluation_name                     | String  | Required    | None    | Up to 50 characters                                         | Model evaluation name                                                        |
| description                               | String  | Optional    | None    | Up to 255 characters                                        | Description for model evaluation                                                    |
| model_id                                  | String  | Required    | None    | Up to 36 characters                                         | Model to evaluate ID                                                       |
| objective_code                            | String  | Required    | None    | "CLASSIFICATION", "REGRESSION" | Evaluation objectives                                                           |
| class_names                               | String  | Optional    | None    | 1~5000                                         | List of possible classes resulting from the classification model (strings or numbers separated by `,`)                     |
| instance_type_name                             | String  | Required    | None    | None                                             | Instance type name (Inquiry available with CLI)                                          |
| input_data_uri                            | String  | Required    | None    | Up to 255 characters                                        | Input data file path (NHN Cloud Object Storage or NHN Cloud NAS)         |
| input_data_type_code                      | String  | Required    | None    | "CSV", "JSONL"                 | Input data type                                                       |
| target_field_name                         | String  | Required    | None    | Up to 255 characters                                        | Field name of the ground truth label                                     |
| timeout_hours                             | Integer | Optional    | 720    | 1~720                                          | Maximum model evaluation time (in hours)                                             |
| batch_inference_instance_type_name             | String  | Required    | None    | None                                             | Instance type name (Inquiry available with CLI)                                          |
| batch_inference_instance_count            | Integer | Required    | None    | 1~10                                           | Number of instances to use for batch inference                                              |
| batch_inference_pod_count                 | Integer | Required    | None    | 1~100                                          | Number of pods to apply distributed inference to                                                 |
| batch_inference_output_upload_uri         | String  | Required    | None    | Up to 255 characters                                        | Path where batch inference result files will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS) |
| batch_inference_max_batch_size            | Integer | Required    | None    | 1~1000                                         | Number of data samples processed simultaneously                                              |
| batch_inference_inference_timeout_seconds | Integer | Required    | None    | 1~1200                                         | Maximum allowed time for a single inference request                                              |
| use_log                                   | Boolean | Optional    | False | True, False                                    | Whether to leave logs in the Log & Crash Search service                              |
| wait                                      | Boolean | Optional    | True  | True, False                                    | True: return after creation is complete, False: return upon creation request                       |

```python
# Create Regression Model Evaluation
regression_model_evaluation  = easymaker.ModelEvaluation().create(
    model_evaluation_name="regression_model_evaluation",
    description="regression model evaluation sample",
    model_id=regression_model.model_id,
    objective_code="REGRESSION",
    instance_type_name="m2.c4m8",
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type_code="CSV",
    target_field_name="target_field_name",
    timeout_hours=1,
    batch_inference_instance_type_name="m2.c4m8",
    batch_inference_instance_count=1,
    batch_inference_pod_count=1,
    batch_inference_output_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    batch_inference_max_batch_size=100,
    batch_inference_inference_timeout_seconds=1200,
    use_log=False,
    wait=True,
)
# Create Classification Model Evaluation
classification_model_evaluation  = easymaker.ModelEvaluation().create(
    model_evaluation_name="classification_model_evaluation",
    description="classification model evaluation sample",
    model_id=classification_model.model_id,
    objective_code="CLASSIFICATION",
    class_names="classA,classB,classC",
    instance_type_name="m2.c4m8",
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type_code="CSV",
    target_field_name="target_field_name",
    timeout_hours=1,
    batch_inference_instance_type_name="m2.c4m8",
    batch_inference_instance_count=1,
    batch_inference_pod_count=1,
    batch_inference_output_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    batch_inference_max_batch_size=100,
    batch_inference_inference_timeout_seconds=1200,
    use_log=False,
    wait=True,
)
```

<a id="model.evaluation.list"></a>
### List Model Evaluations { #model.evaluation.list }

```python
model_evaluation_list = easymaker.ModelEvaluation.get_list()
for model_evaluation in model_evaluation_list:
    model_evaluation.print_info()
```

<a id="model.evaluation.delete"></a>
### Delete a model evaluation { #model.evaluation.delete }

[Parameter]

| Name                                        | Type      | Required | Default value   | Valid range                                          | Description                                                              |
|---------------------------|---------|-------|------|--------|----------|
| model_evaluation_id | String  | Required    | None   | Up to 36 characters | Model evaluation ID |

```python
easymaker.ModelEvaluation(model_evaluation_id).delete()
```

<a id="endpoint"></a>
## Endpoint { #endpoint }

<a id="endpoint.instance.list"></a>
### List Instances { #endpoint.instance.list }

```python
instance_type_list = easymaker.Endpoint.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="endpoint.create"></a>
### Create Endpoint { #endpoint.create }

When creating an endpoint, the default stage is created.

[Parameter]

| Name                                                        | Type      | Required | Default value | Valid range                      | Description                                             |
|-------------------------------------------------------------|---------------------------------------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                                               | String                                | Required    | None    | Up to 50 characters                     | Endpoint name                                                               |
| description                                                 | String                                | Optional    | None    | Up to 255 characters                    | Description for endpoint                                                           |
| instance_type_name                                          | String                                | Required    | None    | None                         | Instance type name to be used for endpoint                                                   |
| instance_count                                              | Integer                               | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                      |
| endpoint_model_resource_list                                | easymaker.EndpointModelResource Array | Required    | None    | Up to 10                     | Resource information to be used on the stage                                                 |
| endpoint_model_resource_list[0].model_id                   | String                                | Required    | None    | None                       | Model ID to be created as a stage resource                                   |
| endpoint_model_resource_list[0].resource_option_detail        | easymaker.ResourceOptionDetail        | Required    | None    |                                  | Details of stage resource                  |
| endpoint_model_resource_list[0].resource_option_detail.cpu    | String                                | Required    | None    | 0.0~                             | CPU to be used for stage resource                |
| endpoint_model_resource_list[0].resource_option_detail.memory | String                                | Required    | None    | 1Mi~                             | Memory to be used for stage resource             |
| endpoint_model_resource_list[0].pod_auto_scale_enable          | Boolean                               | Optional    | False   | True, False                      | Pod autoscaler to be used for stage resource |
| endpoint_model_resource_list[0].scale_metric_code             | easymaker.ScaleMetricCode           | Optional    | None    | CPU_UTILIZATION, MEMORY_UTILIZATION | Scaling unit to be used for stage resource          |
| endpoint_model_resource_list[0].scale_metric_target           | Integer                               | Optional    | None    | 1~                               | Scaling threshold to be used for stage resource     |
| endpoint_model_resource_list[0].description                 | String                                | Optional    | None    | 최대 255자                  | Description of stage resource                                       |
| use_log                                                     | Boolean                               | Optional    | False | True, False                | Whether to leave logs in the Log & Crash Search service                                             |
| wait                                                        | Boolean                               | Optional    | True   | True, False | True: return after creation is complete, False: return upon creation request |

```python
endpoint = easymaker.Endpoint().create(
    endpoint_name='endpoint_name',
    description='endpoint_description',
    instance_type_name='c2.c16m16',
    instance_count=1,
    endpoint_model_resource_list=[
        easymaker.EndpointModelResource(
            model_id=model.model_id,
            resource_option_detail=easymaker.ResourceOptionDetail(
                cpu="15",
                memory="15Gi",
            ),
            pod_auto_scale_enable=True,
            scale_metric_code=easymaker.ScaleMetricCode.CPU_UTILIZATION,
            scale_metric_target=50,
        )
    ],
    use_log=True,
    # wait=False,
)
```

Use the created endpoint

```python
endpoint = easymaker.Endpoint(endpoint_id)
```

<a id="endpoint.stage.create"></a>
### Add Stage { #endpoint.stage.create }

You can add a new stage to existing endpoints.

[Parameter]

| Name                                                        | Type      | Required | Default value | Valid range                      | Description                                                                 |
|-------------------------------------------------------------|---------------------------------------|-------|-------|----------------------------|--------------------------------------------------------------------|
| endpoint_id                                                 | String                                | Required    | None   | Up to 36 characters                      | Endpoint ID                                                            |
| stage_name                                                  | String                                | Required    | None    | Up to 50 characters                     | Stage name                                                            |
| description                                                 | String                                | Optional    | None    | Up to 255 characters                    | Description for stage                                                        |
| instance_type_name                                          | String                                | Required    | None    | None                         | Instance type name to be used for endpoint                                              |
| instance_count                                              | Integer                               | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                  |
| endpoint_model_resource_list                                | easymaker.EndpointModelResource Array | Required    | None    | up to 10                     | Resource information to be used on the stage                                                 |
| endpoint_model_resource_list[0].model_id                   | String                                | Required    | None    | None                       | Model ID to be created as a stage resource                                   |
| endpoint_model_resource_list[0].resource_option_detail        | easymaker.ResourceOptionDetail        | Required    | None    |                                  | Details of stage resource                 |
| endpoint_model_resource_list[0].resource_option_detail.cpu    | String                                | Required    | None    | 0.0~                             | CPU to be used for stage resource                |
| endpoint_model_resource_list[0].resource_option_detail.memory | String                                | Required    | None    | 1Mi~                             | Memory to be used for stage resource             |
| endpoint_model_resource_list[0].pod_auto_scale_enable          | Boolean                               | Optional    | False   | True, False                      | Pod autoscaler to be used for stage resource |
| endpoint_model_resource_list[0].scale_metric_code             | easymaker.ScaleMetricCode           | Optional    | None    | CPU_UTILIZATION, MEMORY_UTILIZATION | Scaling unit to be used for stage resource          |
| endpoint_model_resource_list[0].scale_metric_target           | Integer                               | Optional    | None    | 1~                               | Scaling threshold to be used for stage resource     |
| endpoint_model_resource_list[0].description                 | String                                | Optional    | None    | Up to 255 characters                  | Description of stage resource                                       |
| use_log                                                     | Boolean                               | Optional    | False | True, False                | Whether to leave logs in the Log & Crash Search service                                         |
| wait                                                        | Boolean                               | Optional    | True   | True, False | True: return after creation is complete, False: return upon creation request |

```python
endpoint_stage = easymaker.EndpointStage().create(
    endpoint_id=endpoint.endpoint_id,
    stage_name='stage01',  # lowercase/number within 30 characters
    description='test endpoint',
    instance_type_name='c2.c16m16',
    instance_count=1,
    endpoint_model_resource_list=[
        easymaker.EndpointModelResource(
            model_id=model.model_id,
            resource_option_detail=easymaker.ResourceOptionDetail(
                cpu="15",
                memory="15Gi",
            ),
            pod_auto_scale_enable=True,
            scale_metric_code=easymaker.ScaleMetricCode.CPU_UTILIZATION,
            scale_metric_target=50,
            description='stage_resource_description'
        )
    ],
    use_log=True,
    # wait=False,
)
```

<a id="endpoint.stage.list"></a>
### Retrieve Stages { #endpoint.stage.list }

Retrieves endpoint stages.

```python
endpoint_stage_list = easymaker.Endpoint(endpoint_id).get_stage_list()
```

<a id="endpoint.inference.request"></a>
### Endpoint Inference { #endpoint.inference.request }

Inference to the default stage

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.Endpoint('endpoint_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

Inference by specifying a specific stage

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.EndpointStage('endpoint_stage_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

<a id="endpoint.list"></a>
### List Endpoints { #endpoint.list }

```python
endpoint_list = easymaker.Endpoint.get_list()
for endpoint in endpoint_list:
    endpoint.print_info()
```

<a id="endpoint.delete"></a>
### Delete Endpoint { #endpoint.delete }

[Parameter]

| Name            | Type      | Required | Default value  | Valid range  | Description       |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | Required    | None   | Up to 36 characters | Endpoint ID |

```python
easymaker.Endpoint(endpoint_id).delete()
```

<a id="endpoint.stage.delete"></a>
### Delete Endpoint Stage { #endpoint.stage.delete }

[Parameter]

| Name         | Type      | Required | Default value  | Valid range  | Description      |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | Required    | None   | Up to 36 characters | Stage ID |

```python
easymaker.EndpointStage(stage_id).delete()
```

<a id="batch.inference"></a>
## Batch Inference { #batch.inference }

<a id="batch.inference.instance.list"></a>
### List Instances { #batch.inference.instance.list }

```python
instance_type_list = easymaker.BatchInference.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="batch.inference.create"></a>
### Create Batch Inference { #batch.inference.create }

[Parameter]

| Name                      | Type    | Required | Default value | Valid range   | Description                                                                                 |
| ------------------------- | ------- | --------- | ------ | ----------- |-----------------------------------------------------------------|
| batch_inference_name      | String  | Required      | None   | Up to 50 characters   | Batch inference name                                                        |
| instance_count            | Integer | Optional      | 1   | 1~10        | Number of instances to use for batch inference                                               |
| timeout_hours             | Integer | Required      | 720    | 1~720       | Maximum batch inference time (in hours)                                             |
| instance_type_name             | String  | Required      | None   | None        | Instance type name (Inquiry available with CLI)                                          |
| model_id                | String  | Required      | None   | None        | Model ID                                                            |
| pod_count                 | Integer | Optional      | 1   | 1~100       | Number of pods to apply distributed inference to                                                 |
| batch_size                | Integer | Required      | None   | 1~1000      | Number of data samples processed simultaneously                                              |
| inference_timeout_seconds | Integer | Required      | None   | 1~1200      | Maximum allowable time for a single inference request                                              |
| input_data_uri            | String  | Required      | None   | Up to 255  | Path for input data file (NHN Cloud Object Storage or NHN Cloud NAS)         |
| input_data_type           | String  | Required      | None   | "JSON", "JSONL" | Input data type                                                      |
| include_glob_pattern      | String  | Optional      | None   | Up to 255  | Glob pattern to include a set of files in the input data                                     |
| exclude_glob_pattern      | String  | Optional      | None   | Up to 255  | Glob pattern to exclude a set of files in the input data                                     |
| output_upload_uri         | String  | Required      | None   | Up to 255  | The path where the batch inference result file will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS) |
| data_storage_size         | Integer | Required      | None   | 300~10000   | Storage size to download data for batch inference (unit: GB)                          |
| description               | String  | Optional      | None   | Up to 255  | Explanation of batch inference                                                    |
| use_log                   | Boolean | Optional      | False  | True, False | Whether to leave logs with the Log & Crash Search service                              |
| wait                      | Boolean | Optional      | True   | True, False | True: return after creation is complete, False: return upon creation request                       |

```python
batch_inference = easymaker.BatchInference().run(
    batch_inference_name='batch_inference_name',
    instance_count=1,
    timeout_hours=100,
    instance_type_name='m2.c4m8',
    model_id=model.model_id,
    pod_count=1,
    batch_size=32,
    inference_timeout_seconds=120,
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type='JSONL',
    include_glob_pattern=None,
    exclude_glob_pattern=None,
    output_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{output_upload_path}',
    data_storage_size=300, # minimum size : 300GB
    description='description',
    use_log=True,
    # wait=False,
)
```

<a id="batch.inference.list"></a>
### List of Batch Inference { #batch.inference.list }

```python
batch_inference_list = easymaker.BatchInference.get_list()
for batch_inference in batch_inference_list:
    batch_inference.print_info()
```

<a id="batch.inference.delete"></a>
### Delete Batch Inference { #batch.inference.delete }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- | ------------ |
| batch_inference_id | String | Required      | None   | Up to 36 characters | Batch Inference ID |

```python
easymaker.BatchInference(batch_inference_id).delete()
```

<a id="pipeline"></a>
## Pipeline { #pipeline }

<a id="pipeline.create"></a>
### Create Pipeline { #pipeline.create }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
|-----------------------------|---------| --------- | ------ | --------- |-------------------------------------------|
| pipeline_name               | String  | Required      | None   | Max 50 characters   | Pipeline name                                  |
| pipeline_spec_manifest_path | String  | Required      | None   | none      | Pipeline file path to upload                          |
| description                 | String  | Optional  | None   | Max 255 characters  | Description for pipeline                              |
| wait                        | Boolean | Optional    | True   | True, False | True: return after creation is complete, False: return immediately after creation request |

```python
pipeline = easymaker.Pipeline().upload(
    pipeline_name='pipeline_01',
    pipeline_spec_manifest_path='./sample-pipeline.yaml',
    description='test',
    # wait=False,
)
```

<a id="pipeline.list"></a>
### List of Pipeline { #pipeline.list }

```python
pipeline_list = easymaker.Pipeline.get_list()
for pipeline in pipeline_list:
    pipeline.print_info()
```

<a id="pipeline.delete"></a>
### Delete Pipeline { #pipeline.delete }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |----------|
| pipeline_id | String | Required | None | Up to 36 characters | Pipeline ID

```python
easymaker.Pipeline(pipeline_id).delete()
```

<a id="pipeline.instance.list"></a>
### List Instances { #pipeline.instance.list }

```python
instance_type_list = easymaker.PipelineRun.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

<a id="pipeline.run.create"></a>
### Create Pipeline Run { #pipeline.run.create }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
|-----------------------------------|---------------------------|---------------------------| ------ |-------------|------------------------------------------|
| pipeline_run_name                 | String                    | Required                        | None   | Up to 50 Characters      | Pipeline run name                              |
| pipeline_id                       | String                    | Required                        | None   | Up to 36 Characters      | Pipeline schedule name                              |
| experiment_id                     | String                    | Required if not entered in easymaker.init  | None    | Up to 36 Characters      | Experiment ID                                    |
| description                       | String                    | Optional                        | None   | Up to 255 Characters     | Description of pipeline execution                          |
| instance_type_name                | String                    | Required                        | None   | None          | Instance type name (Inquiry available with CLI)                   |
| instance_count                    | Integer                   | Optional                        | 1   | 1~10        | Number of instances to use                               |
| boot_storage_size                 | Integer                   | Required                        | None   | 50~         | The boot storage size (in GB) of the instance that will run the pipeline.      |
| parameter_list                    | easymaker.Parameter Array | Optional                        | None   | None          | Parameter information to pass to the pipeline                       |
| parameter_list[0].parameter_name  | String                    | Optional                        | None   | Up to 255 Characters     | Parameter key                                   |
| parameter_list[0].parameter_value | String                    | Optional                        | None   | Up to 1000 Characters    | Parameter value                                   |
| nas_list                          | easymaker.Nas Array       | Optional                        | None   | Up to 10      | NAS information                                   |
| nas_list[0].mount_dir_name        | String                    | Optional                        | None   | Up to 64 Characters      | Directory name to be mounted on instances                       |
| nas_list[0].nas_uri               | String                    | Optional                        | None   | Up to 255 Characters    | The path to the NAS in the format `nas://{NAS ID}:/{path}`      |
| wait                              | Boolean                   | Optional                        | True   | True, False | True: return after creation is complete, False: return immediately after creation request |

```python
pipeline_run = easymaker.PipelineRun().create(
    pipeline_run_name='pipeline_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_type_name='m2.c4m8',
    instance_count=1,
    boot_storage_size=50,
    parameter_list=[
        easymaker.Parameter(parameter_name="experiment_name", parameter_value="pipeline_experiment"),
    ],
    nas_list=[
        easymaker.Nas(mount_dir_name="user_nas", nas_uri="nas://{NAS ID}:/{path}"),
    ],
    # wait=False,
)
```

<a id="pipeline.run.list"></a>
### List of Pipeline Run { #pipeline.run.list }

```python
pipeline_run_list = easymaker.PipelineRun.get_list()
for pipeline_run in pipeline_run_list:
    pipeline_run.print_info()
```

<a id="pipeline.run.delete"></a>
### Delete Pipeline Run { #pipeline.run.delete }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_run_id | String | Required      | None   | Up to 36 characters | Pipeline run ID |

```python
easymaker.PipelineRun(pipeline_run_id).delete()
```

<a id="pipeline.recurring.run.create"></a>
### Create Pipeline Schedule { #pipeline.recurring.run.create }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
|----------------------------------|---------|------------------------------------| ------ |-------------|------------------------------------------------|
| pipeline_recurring_run_name      | String  | Required                                 | None   | Max 50 characters      | Pipeline schedule name                                 |
| pipeline_id                      | String  | Required                                 | None   | Max 36 character      | Pipeline schedule name                                |
| experiment_id                    | String  | Required if not entered in easymaker.init     | None    | Max 36 character      | Experiment ID                                          |
| description                      | String  | Optional                                 | None   | Max 255 character| Description of pipeline schedules                                |
| instance_type_name                    | String  | Required                                 | None   | None          | Instance type name (Inquiry available with CLI)                         |
| instance_count                   | Integer | Optional                                 | 1   | 1~10        | Number of instances to use                                    |
| boot_storage_size                | Integer | Required                                 | None   | 50~         | The boot storage size (in GB) of the instance that will run the pipeline.            |
| schedule_periodic_minutes        | String  | schedule_cron_expression 미입력시 Required  | None   | None          | Set a time interval to run the pipeline repeatedly                        |
| schedule_cron_expression         | String  | schedule_periodic_minutes 미입력시 Required | None   | None          | Set up a Cron expression to run the pipeline repeatedly                 |
| max_concurrency_count            | Integer  | Optional                                 | 1   | 1~10          | Limit the number of concurrent runs by specifying a maximum number of parallel runs             |
| schedule_start_datetime          | String  | Optional                                 | None   | None          | Set a start time for the pipeline schedule, which will run the pipeline at the set interval if not entered. |
| schedule_end_datetime            | String  | Optional                                 | None   | None          | Set an end time for a pipeline schedule, creating a pipeline run until it stops if no input is received. |
| use_catchup                      | Boolean | Optional                                 | None   | None          | Missed run catch-up: Whether to catch up when pipeline runs fall behind schedule. |
| parameter_list                    | easymaker.Parameter Array | Optional                        | None   | None          | Parameter information to pass to the pipeline                       |
| parameter_list[0].parameter_name  | String                    | Optional                        | None   | Up to 255 character     | Parameter key                                   |
| parameter_list[0].parameter_value | String                    | Optional                        | None   | Up to 1000 character    | Parameter value                                   |
| nas_list                          | easymaker.Nas Array       | Optional                        | None   | Up to 10      | NAS information                                   |
| nas_list[0].mount_dir_name        | String                    | Optional                        | None   | Up to 64 character     | Directory name to be mounted on instances                       |
| nas_list[0].nas_uri               | String                    | Optional                        | None   | Up to 255 character     | The path to the NAS in the format `nas://{NAS ID}:/{path}`      |
| wait                             | Boolean | Optional                                 | True   | True, False | True: return after creation is complete, False: return immediately after creation request     |

```python
pipeline_recurring_run = easymaker.PipelineRecurringRun().create(
    pipeline_recurring_run_name='pipeline_recurring_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_type_name='m2.c4m8',
    parameter_list=[
        easymaker.Parameter(parameter_name="experiment_name", parameter_value="pipeline_experiment"),
    ],
    nas_list=[
        easymaker.Nas(mount_dir_name="user_nas", nas_uri="nas://{NAS ID}:/{path}"),
    ],
    boot_storage_size=50,
    schedule_cron_expression='0 0 * * * ?',
    max_concurrency_count=1,
    schedule_start_datetime='2025-01-01T00:00:00+09:00'
    # wait=False,
)
```

<a id="pipeline.recurring.run.stop.start"></a>
### Stop/Restart Pipeline Schedule { #pipeline.recurring.run.stop.start }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | Required      | None   | Max 36 characters | Pipeline Schedule ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).stop()
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).start()

```

<a id="pipeline.recurring.run.list"></a>
### List of Pipeline Schedule { #pipeline.recurring.run.list }

```python
pipeline_recurring_run_list = easymaker.PipelineRecurringRun.get_list()
for pipeline_recurring_run in pipeline_recurring_run_list:
    pipeline_recurring_run.print_info()
```

<a id="pipeline.recurring.run.delete"></a>
### Delete Pipeline Schedule { #pipeline.recurring.run.delete }

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | Required      | None   | Max 36 characters | Pipeline Schedule ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).delete()
```

<a id="feature"></a>
## Other Features { #feature }

<a id="feature.lncs.log.send"></a>
### NHN Cloud - Log & Crash Search Log Sending Feature { #feature.lncs.log.send }

```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default: INFO
                      project_version='2.0.0',  # default: 1.0.0
                      parameters={'serviceType': 'EasyMakerSample'})  # Add custom parameters
```

<a id="feature.object.storage.file.send"></a>
### NHN Cloud - Object Storage File Sending Feature { #feature.object.storage.file.send }

Provide a feature to upload and download files with Object Storage.

```python
easymaker.upload(
    easymaker_obs_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{upload_path}',
    local_path='./local_dir',
    username='userId@nhn.com',
    password='nhn_object_storage_api_password'
)

easymaker.download(
    easymaker_obs_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_00000000000000000000000000000000/SDK/sample/source_dir',
    download_dir_path='./download_dir',
    username='userId@nhn.com',
    password='nhn_object_storage_api_password'
)
```
