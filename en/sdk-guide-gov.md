
## Machine Learning > AI EasyMaker > SDK User Guide

## SDK Settings

### Install AI EasyMaker Python SDK

python -m pip install easymaker

- AI EasyMaker is installed in the notebook by default.

### Initialize AI EasyMaker SDK

You can find the AppKey in the **URL & Appkey** menu at the right top on the console.
You can learn more about Access Tokens in [API Call and Authentication](https://docs.nhncloud.com/en/nhncloud/en/public-api/api-authentication/).
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

## Experiment

### Create Experiment

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

### List Experiments

```python
experiment_list = easymaker.Experiment.get_list()
for experiment in experiment_list:
    experiment.print_info()
```

### Delete Experiment

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | Required    | None   | Up to 36 characters | Experiment ID |

```python
easymaker.Experiment(experiment_id).delete()
```

## Training

### List Images

```python
image_list = easymaker.Training.get_image_list()
for image in image_list:
    image.print_info()
```

### List Instances

```python
instance_type_list = easymaker.Training.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create Training

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
| dataset_list                             | easymaker.Dataset Array   | Optional                                             | None          | Max 10                                  | Information of dataset to be used for training (consists of dataset_name/data_uri)                |
| dataset_list[0].dataset_name              | String  | Optional                                             | None          | Up to 36 characters                     | Data name                                                                                  |
| dataset_list[0].data_uri               | String  | Optional                                             | None          | Up to 255 characters                    | Data path                                                                                  |
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

### List of Training

```python
training_list = easymaker.Training.get_list()
for training in training_list:
    training.print_info()
```

### Delete Training

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | Required    | None   | Up to 36 characters | Training ID |

```python
easymaker.Training(training_id).delete()
```

## Hyperparameter Tuning

### List Images

```python
image_list = easymaker.HyperparameterTuning.get_image_list()
for image in image_list:
    image.print_info()
```

### List Instances

```python
instance_type_list = easymaker.HyperparameterTuning.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create Hyperparameter Tuning

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
| hyperparameter_spec_list[0].<br>hyperparameter_type_code         | easymaker.HYPERPARAMETER_TYPE_CODE         | Optional                                                            | None    | INT, DOUBLE, DISCRETE, CATEGORICAL                                            | Hyperparameter Type                                                                              |
| hyperparameter_spec_list[0].<br>hyperparameter_min_value         | String | Required if hyperparameterTypeCode is INT, DOUBLE(Enter a number as a string type)                   | None    | None                                                                          | Hyperparameter minimum value                                                                                |
| hyperparameter_spec_list[0].<br>hyperparameter_max_value         | String | Required if hyperparameterTypeCode is INT, DOUBLE(Enter a number as a string type)                 | None    | None                                                                          | Hyperparameter maximum value                                                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameter_step             | String | Required if hyperparameterTypeCode is INT, DOUBLE and GRID strategy | None    | None                                                                          | Magnitude of change in hyperparameter values when using the "Grid" tuning strategy                                           |
| hyperparameter_spec_list[0].<br>hyperparameter_specified_values  | String         | Required if hyperparameterTypeCode is DISCRETE or CATEGORICAL       | None    | Up to 3000 characters                                                         | A list of defined hyperparameters (strings or numbers separated by `,`)                 |
| dataset_list                                                   | easymaker.Dataset Array          | Optional                                                            | None    | Max 10                                                                        | Dataset information to be used for hyperparameter tuning (configured as dataset_name/data_uri)                                 |
| dataset_list[0].dataset_name                                    | String         | Optional                                                            | None    | Up to 36 characters                                                           | Data name                                                                                 |
| dataset_list[0].dataset_uri                                     | String         | Optional                                                            | None    | Up to 255 characters                                                          | Data path                                                                               |
| metric_list                                                    | easymaker.Metric          | Required when using own algorithm                                   | None    | Up to 10 (string list of indicator names)                                     | Define which metrics to collect from logs output by the training code.                                                       |
| metric_list[0].name                                              | String                             | Required when using own algorithm                                             | None    | None                                                     | Metric name                                                                      |
| metric_regex                                                   | String         | Select when using own algorithm                                     | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?)                             | Up to 255 characters                                                                                                         | Enter a regular expression to use to collect metrics. The training algorithm should output metrics to match the regular expression.                                                          |
| objective_metric_name                                          | String         | Required when using own algorithm                                   | None    | Up to 36 characters, one of metric_list                                       | Choose which metrics you want to optimize for.                                                                               |
| objective_type_code                                            | easymaker.OBJECTIVE_TYPE_CODE         | Required when using own algorithm                                   | None    | MINIMIZE, MAXIMIZE                                                            | Choose a target metric optimization type.                                                                                    |
| objective_goal                                                 | Double         | Optional                                                            | None    | None                                                                          | The tuning job ends when the target metric reaches this value.                                                               |
| max_failed_trial_count                                         | Integer        | Optional                                                            | None    | None                                                                          | Define the maximum number of failed lessons. When the number of failed trainings reaches this value, tuning ends in failure. |
| max_trial_count                                                | Integer        | Optional                                                            | None    | None                                                                          | Defines the maximum number of lessons. Tuning runs until the number of auto-run training reaches this value.                 |
| tuning_strategy_name                                           | easymaker.TUNING_STRATEGY         | Required                                                            | None    | None                                                                          | Choose which strategy to use to find the optimal hyperparameters.                                                            |
| tuning_strategy_random_state                                   | Integer        | Optional                                                            | None    | None                                                                          | Determine random number generation. Specify a fixed value for reproducible results.                                          |
| early_stopping_algorithm                                       | easymaker.EARLY_STOPPING_ALGORITHM         | Required                                                            | None    | EARLY_STOPPING_ALGORITHM.<br>MEDIAN                                           | Stop training early if the model is no longer good even though training continues.                                           |
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
            hyperparameter_type_code=easymaker.HYPERPARAMETER_TYPE_CODE.DOUBLE,
            hyperparameter_min_value="0.01",
            hyperparameter_max_value="0.05",
        ),
        easymaker.HyperparameterSpec(
            hyperparameter_name="epochs",
            hyperparameter_type_code=easymaker.HYPERPARAMETER_TYPE_CODE.INT,
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
    objective_type_code=easymaker.OBJECTIVE_TYPE_CODE.MINIMIZE,
    objective_goal=0.01,
    max_failed_trial_count=3,
    max_trial_count=10,
    tuning_strategy_name=easymaker.TUNING_STRATEGY.BAYESIAN_OPTIMIZATION,
    tuning_strategy_random_state=1,
    early_stopping_algorithm=easymaker.EARLY_STOPPING_ALGORITHM.MEDIAN,
    early_stopping_min_trial_count=3,
    early_stopping_start_step=4,
    use_log=True,
    # wait=False,
)
```

### List Hyperparameter Tuning

```python
hyperparameter_tuning_list = easymaker.HyperparameterTuning.get_list()
for hyperparameter_tuning in hyperparameter_tuning_list:
    hyperparameter_tuning.print_info()
```

### Delete Hyperparameter Tuning

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description           |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | Required    | None   | Up to 36 characters | Hyperparameter Tuning ID |

```python
easymaker.HyperparameterTuning(hyperparameter_tuning_id).delete()
```

## Model

### Create Model

Request to create a model with the training ID.
The model is used when creating endpoints.

[Parameter]

| Name                       | Type     | Required                              | Default value | Valid range   | Description                                  |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | Required if hyperparameter_tuning_id does not exist | None  | None      | Training ID to create a model                       |
| hyperparameter_tuning_id | String | Required if training_id is not present              | None  | None      | Hyperparameter tuning ID to be created by model (created by best learning) |
| model_name               | String | Required                                 | None  | Up to 50 characters  | Model name                               |
| description        | String | Optional                                 | None  | Up to 255 characters | Description for model                           |
| parameter_list                   | Array  | Optional    | None  | Up to 10                                  | Parameter information (consist of parameterName/parameterValue)         |
| parameter_list[0].parameterName  | String | Optional    | None  | Up to 64 characters                                  | Parameter name                                              |
| parameter_list[0].parameterValue | String | Optional    | None  | Up to 255 characters                                 | Parameter value                                                |

```python
model = easymaker.Model().create(
    training_id=training.training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning.hyperparameter_tuning_id,
    model_name='model_name',
    description='model_description',
)
```

Even if there is no training ID, you can create a model by entering the path information for the model and framework type.

[Parameter]

| Name                   | Type     | Required | Default value | Valid range                                   | Description                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| model_type_code       | Enum   | Required    | None  | easymaker.TENSORFLOW, easymaker.PYTORCH | Framework information used for training                                    |
| model_upload_uri            | String | Required    | None  | Up to 255 characters                                 | Path for model file (NHN Cloud Object Storage or NHN Cloud NAS) |
| model_name           | String | Required    | None  | Up to 50 characters                                  | Model name                                               |
| description    | String | Optional    | None  | Up to 255 characters                                 | Description for model                                           |
| parameter_list                   | Array  | Optional    | None  | Max 10                                  | Information of parameters (consists of parameterName/parameterValue)         |
| parameter_list[0].parameterName  | String | Optional    | None  | Up to 64 characters                     | Parameter name                                              |
| parameter_list[0].parameterValue | String | Optional    | None  | Up to 255 characters                    | Parameter value                                             |

```python
# TensorFlow Model
model = easymaker.Model().create_by_model_upload_uri(
    model_type_code=easymaker.TENSORFLOW,
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

### List Models

```python
model_list = easymaker.Model.get_list()
for model in model_list:
    model.print_info()
```

### Delete Model

[Parameter]

| Name                        | Type      | Required | Default value  | Valid range  | Description    |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | Required    | None   | Up to 36 characters | Model ID |

```python
easymaker.Model(model_id).delete()
```

## Evaluate Models

### List Instances

```python
instance_type_list = easymaker.ModelEvaluation.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create a model evaluation

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

### List Model Evaluations

```python
model_evaluation_list = easymaker.ModelEvaluation.get_list()
for model_evaluation in model_evaluation_list:
    model_evaluation.print_info()
```

### Delete a model evaluation

[Parameter]

| Name                                        | Type      | Required | Default value   | Valid range                                          | Description                                                              |
|---------------------------|---------|-------|------|--------|----------|
| model_evaluation_id | String  | Required    | None   | Up to 36 characters | Model evaluation ID |

```python
easymaker.ModelEvaluation(model_evaluation_id).delete()
```

## Endpoint

### List Instances

```python
instance_type_list = easymaker.Endpoint.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create Endpoint

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
| endpoint_model_resource_list[0].scale_metric_code             | easymaker.SCALE_METRIC_CODE           | Optional    | None    | CPU_UTILIZATION, MEMORY_UTILIZATION | Scaling unit to be used for stage resource          |
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
            scale_metric_code=easymaker.SCALE_METRIC_CODE.CPU_UTILIZATION,
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

### Add Stage

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
| endpoint_model_resource_list[0].scale_metric_code             | easymaker.SCALE_METRIC_CODE           | Optional    | None    | CPU_UTILIZATION, MEMORY_UTILIZATION | Scaling unit to be used for stage resource          |
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
            scale_metric_code=easymaker.SCALE_METRIC_CODE.CPU_UTILIZATION,
            scale_metric_target=50,
            description='stage_resource_description'
        )
    ],
    use_log=True,
    # wait=False,
)
```

### Retrieve Stages

Retrieves endpoint stages.

```python
endpoint_stage_list = easymaker.Endpoint(endpoint_id).get_stage_list()
```

### Endpoint Inference

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

### List Endpoints

```python
endpoint_list = easymaker.Endpoint.get_list()
for endpoint in endpoint_list:
    endpoint.print_info()
```

### Delete Endpoint

[Parameter]

| Name            | Type      | Required | Default value  | Valid range  | Description       |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | Required    | None   | Up to 36 characters | Endpoint ID |

```python
easymaker.Endpoint(endpoint_id).delete()
```

### Delete Endpoint Stage

[Parameter]

| Name         | Type      | Required | Default value  | Valid range  | Description      |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | Required    | None   | Up to 36 characters | Stage ID |

```python
easymaker.EndpointStage(stage_id).delete()
```

## Batch Inference

### List Instances

```python
instance_type_list = easymaker.BatchInference.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create Batch Inference

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

### List of Batch Inference

```python
batch_inference_list = easymaker.BatchInference.get_list()
for batch_inference in batch_inference_list:
    batch_inference.print_info()
```

### Delete Batch Inference

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- | ------------ |
| batch_inference_id | String | Required      | None   | Up to 36 characters | Batch Inference ID |

```python
easymaker.BatchInference(batch_inference_id).delete()
```

## Pipeline

### Create Pipeline

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

### List of Pipeline

```python
pipeline_list = easymaker.Pipeline.get_list()
for pipeline in pipeline_list:
    pipeline.print_info()
```

### Delete Pipeline

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |----------|
| pipeline_id | String | Required | None | Up to 36 characters | Pipeline ID

```python
easymaker.Pipeline(pipeline_id).delete()
```

### List Instances

```python
instance_type_list = easymaker.PipelineRun.get_instance_type_list()
for instance in instance_type_list:
    instance.print_info()
```

### Create Pipeline Run

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

### List of Pipeline Run

```python
pipeline_run_list = easymaker.PipelineRun.get_list()
for pipeline_run in pipeline_run_list:
    pipeline_run.print_info()
```

### Delete Pipeline Run

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_run_id | String | Required      | None   | Up to 36 characters | Pipeline run ID |

```python
easymaker.PipelineRun(pipeline_run_id).delete()
```

### Create Pipeline Schedule

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

### Stop/Restart Pipeline Schedule

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | Required      | None   | Max 36 characters | Pipeline Schedule ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).stop()
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).start()

```

### List of Pipeline Schedule

```python
pipeline_recurring_run_list = easymaker.PipelineRecurringRun.get_list()
for pipeline_recurring_run in pipeline_recurring_run_list:
    pipeline_recurring_run.print_info()
```

### Delete Pipeline Schedule

[Parameter]

| Name               | Type   | Required | Default value | Valid range | Description         |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | Required      | None   | Max 36 characters | Pipeline Schedule ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).delete()
```

## Other Features

### NHN Cloud - Log & Crash Search Log Sending Feature

```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default: INFO
                      project_version='2.0.0',  # default: 1.0.0
                      parameters={'serviceType': 'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storage File Sending Feature

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
