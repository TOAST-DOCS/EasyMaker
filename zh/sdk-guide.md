## NHN Cloud > SDK User Guide > AI EasyMaker

## Development Guide

### Install AI EasyMaker Python SDK

python -m pip install easymaker

* AI EasyMaker is installed in the notebook by default.


### Initialize AI EasyMaker SDK
You can find AppKey and Secret key in the **URL & Appkey** menu at the right top on the console.
Enter the AppKey, SecretKey, and region information of enabled AI EasyMaker.
Intialization code is required to use the AI EasyMaker SDK.
```python
import easymaker

easymaker.init(
    appkey='EASYMAKER_APPKEY',
    region='kr1',
    secret_key='EASYMAKER_SECRET_KEY',
)
```

### Create Experiment
Before creating a training, you must create an experiment to sort trainings.

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range       | Description                                                         |
|------------------------|---------|-------|------|-------------|------------------------------------------------------------|
| experiment_name        | String  | Required    | None   | Up to 50 characters      | Experiment name                                                      |
| experiment_description | String  | Optional    | None   | Up to 255 characters     | Description for experiment                                                  |
| wait                   | Boolean | Optional    | True | True, False | True: Return the experiment ID after creating the experiment, 
False: Return the experiment ID immediately after request to create |

```python
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### Delete Experiment

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | Required    | None   | Up to 36 characters | Experiment ID |

```python
easymaker.Experiment().delete(experiment_id)
```

### Create Training

[Parameter]

| Name                                         | Type      | Required                     | Default value   | Valid range       | Description                                                              |
|--------------------------------------------|---------|---------------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | Required                        | None    | None          | Experiment ID                                                           |
| training_name                              | String  | Required                        | None    | Up to 50 characters      | Training name                                                           |
| training_description                       | String  | Optional                        | None    | Up to 255 characters     | Description for training                                                       |
| train_image_name                           | String  | Required                        | None    | None          | Image name to be used for training (Inquiry available with CLI)                                      |
| train_instance_name                        | String  | Required                        | None    | None          | Instance flavor name (Inquiry available with CLI)                                          |
| distributed_training_count                 | Integer | Required                        | None    | 1~10         | Number of distributed trainings to apply for training                                                 |
| data_storage_size                          | Integer | Required when using Object Storage    | None    | 300~10000   | Storage size to download data for training (unit: GB), unnecessary when using NAS               |
| algorithm_name                             | String  | Required when using algorithms provided by NHN Cloud | None    | Up to 64 characters      | Algorithm name (Inquiry available with CLI)                                             |
| source_dir_uri                             | String  | Required when using own algorithm           | None    | Up to 255 characters     | Path of files required for training (NHN Cloud Object Storage or NHN Cloud NAS) |
| entry_point                                | String  | Required when using own algorithm           | None    | Up to 255 characters     | Information of Python files to be executed initially in source_dir_uri                             |
| model_upload_uri                           | String  | Required                        | None    | Up to 255 characters     | Path to upload the model completed with training (NHN Cloud Object Storage or NHN Cloud NAS)   |
| check_point_input_uri                      | String  | Optional                        | None    | Up to 255 characters     | Input checkpoint file path (NHN Cloud Object Storage or NHN Cloud NAS)                 |
| check_point_upload_uri                     | String  | Optional                        | None    | Up to 255 characters     | The path where the checkpoint file will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS)   |
| timeout_hours                              | Integer | Optional                        | 720   | 1~720       | Max training time (unit: hour)                                                |
| hyperparameter_list                        | Array   | Optional                        | None    | Max 100     | Information of hyperparameters (consists of hyperparameterKey/hyperparameterValue)           |
| hyperparameter_list[0].hyperparameterKey   | String  | Optional                        | None    | Up to 255 characters     | Hyperparameter key                                                       |
| hyperparameter_list[0].hyperparameterValue | String  | Optional                        | None    | Up to 1000 characters    | Hyperparameter value                                                       |
| dataset_list                               | Array   | Optional                        | None    | Max 10      | Information of dataset to be used for training (consists of datasetName/dataUri)                      |
| dataset_list[0].datasetName                | String  | Optional                        | None    | Up to 36 characters      | Data name                                                          |
| dataset_list[0].datasetUri                 | String  | Optional                        | None    | Up to 255 characters     | Data pah                                                          |
| tag_list                                   | Array   | Optional                        | None    | Max 10      | Tag information                                                           |
| tag_list[0].tagKey                         | String  | Optional                        | None    | Up to 64 characters      | Tag key                                                            |
| tag_list[0].tagValue                       | String  | Optional                        | None    | Up to 255 characters     | Tag value                                                            |
| use_log                                    | Boolean | Optional                        | False | True, False | Whether to leave logs in Log & Crash product                                      |
| wait                                       | Boolean | Optional                        | True  | True, False | True: Return the training ID after creating training, 
False: Return the training ID immediately after requesting to create      |

```python
training_id = easymaker.Training().run(
    experiment_id=experiment_id,
    training_name='training_name',
    training_description='training_description',
    train_image_name='Ubuntu 18.04 CPU TensorFlow Training',
    train_instance_name='m2.c4m8',
    distributed_training_count=1,
    data_storage_size=300,  # minimum size : 300GB
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_list=[
        {
            "hyperparameterKey": "epochs",
            "hyperparameterValue": "10",
        },
        {
            "hyperparameterKey": "batch-size",
            "hyperparameterValue": "30",
        }
    ],
    timeout_hours=100,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_input_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_input_path}',
    check_point_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        {
            "datasetName": "train",
            "dataUri": "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_download_path}"
        },
        {
            "datasetName": "test",
            "dataUri": "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_download_path}"
        }
    ],
    tag_list=[
        {
            "tagKey": "tag1",
            "tagValue": "test_tag_1",
        },
        {
            "tagKey": "tag2",
            "tagValue": "test_tag_2",
        }
    ],
    use_log=True,
    # wait=False,
)
```

### Delete Training

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description    |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | Required    | None   | Up to 36 characters | Training ID |

```python
easymaker.Training().delete(training_id)
```

### Create Hyperparameter Tuning

[Parameter]

| Name                                                             | Type             | Required                                                 | Default value   | Valid range                                        | Description                                                                         |
|----------------------------------------------------------------|----------------|-------------------------------------------------------|-------|----------------------------------------------|----------------------------------------------------------------------------|
| experiment_id                                                  | String         | Required                                                    | None    | None                                           | Experiment ID                                                                      |
| hyperparameter_tuning_name                                     | String         | Required                                                    | None    | Up to 50 characters                                       | Hyperparameter Tuning Name                                                              |
| hyperparameter_tuning_description                              | String         | Optional                                                    | None    | Up to 255 characters                                      | Description of hyperparameter tuning                                                          |
| image_name                                                     | String         | Required                                                    | None    | None                                           | Image name to be used for hyperparameter tuning (can be queried with CLI)                                         |
| instance_name                                                  | String         | Required                                                    | None    | None                                           | Instance flavor name (Inquiry available with CLI)                                                     |
| distributed_training_count                                     | Integer        | Required                                                    | 1      | The product of distributed_training_count and parallel_trial_count is 10 or less. | Number of distributed training to apply for each learning in hyperparameter tuning                                                      |
| parallel_trial_count                                           | Integer        | Required                                                    | 1      | The product of distributed_training_count and parallel_trial_count is 10 or less. | Number of trainings to run in parallel in hyperparameter tuning                                                      |
| data_storage_size                                              | Integer        | Required when using Object Storage                                | None    | 300~10000                                    | Size of storage space to download data required for hyperparameter tuning (unit: GB), not required when using NAS                  |
| algorithm_name                                                 | String         | Required when using algorithms provided by NHN Cloud                             | None    | Up to 64 characters                                       | Algorithm name (Inquiry available with CLI)                                                        |
| source_dir_uri                                                 | String         | Required when using own algorithm                                       | None    | Up to 255 characters                                      | Path containing files required for hyperparameter tuning (NHN Cloud Object Storage or NHN Cloud NAS)    |
| entry_point                                                    | String         | Required when using own algorithm                                       | None    | Up to 255 characters                                      | Information of Python files to be executed initially in source_dir_uri                                        |
| model_upload_uri                                               | String         | Required                                                    | None    | Up to 255 characters                                      | The path where the trained model in hyperparameter tuning will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS) |
| check_point_input_uri                                          | String         | Optional                                                    | None    | Up to 255 characters                                      | Input checkpoint file path (NHN Cloud Object Storage or NHN Cloud NAS)                 |
| check_point_upload_uri                                         | String         | Optional                                                    | None    | Up to 255 characters                                      | The path where the checkpoint file will be uploaded (NHN Cloud Object Storage or NHN Cloud NAS)              |
| timeout_hours                                                  | Integer        | Optional                                                    | 720   | 1~720                                        | Maximum hyperparameter tuning time (unit: hours)                                                   |
| hyperparameter_spec_list                                       | Array          | Optional                                                    | None    | Max 100                                      | Hyperparameter specification information                                                              |
| hyperparameter_spec_list[0].<br>hyperparameterName             | String         | Optional                                                    | None    | Up to 255 characters                                      | Hyperparameter Name                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterTypeCode         | String         | Optional                                                    | None    | INT, DOUBLE, DISCRETE, CATEGORICAL           | Hyperparameter Type                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterMinValue         | Integer/Double | Required if hyperparameterTypeCode is INT, DOUBLE            | None    | None                                           | Hyperparameter minimum value                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterMaxValue         | Integer/Double | Required if hyperparameterTypeCode is INT, DOUBLE            | None    | None                                           | Hyperparameter maximum value                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterStep             | Integer/Double | Required if hyperparameterTypeCode is INT, DOUBLE and GRID strategy | None    | None                                           | Magnitude of change in hyperparameter values when using the "Grid" tuning strategy                                       |
| hyperparameter_spec_list[0].<br>hyperparameterSpecifiedValues  | String         | Required if hyperparameterTypeCode is DISCRETE or CATEGORICAL   | None    | Up to 3000 characters                                       | A list of defined hyperparameters (strings or numbers separated by ,)                                          |
| dataset_list                                                   | Array          | Optional                                                    | None    | Max 10                                       | Dataset information to be used for hyperparameter tuning (configured as datasetName/dataUri)                         |
| dataset_list[0].datasetName                                    | String         | Optional                                                    | None    | Up to 36 characters                                       | Data name                                                                     |
| dataset_list[0].datasetUri                                     | String         | Optional                                                    | None    | Up to 255 characters                                      | Data pah                                                                     |
| metric_list                                                    | Array          | Required when using own algorithm                                       | None    | Up to 10 (string list of indicator names)                    | Define which metrics to collect from logs output by the training code.                                       |
| metric_regex                                                   | String         | Select when using own algorithm                                       | ([\\w\\ | -]+)\\s\*=\\s*([+-]?\\d*(.\\d+)?([Ee][+-]?\\d+)?) | Up to 255 characters                                                                    | Enter a regular expression to use to collect metrics. The learning algorithm should output metrics to match the regular expression.                                                          |
| objective_metric_name                                          | String         | Required when using own algorithm                                       | None    | Up to 36 characters, one of metric_list                     | Choose which metrics you want to optimize for.                                                 |
| objective_type_code                                            | String         | Required when using own algorithm                                       | None    | MINIMIZE, MAXIMIZE                           | Choose a target metric optimization type.                                                       |
| objective_goal                                                 | Double         | Optional                                                    | None    | None                                           | The tuning job ends when the target metric reaches this value.                                             |
| max_failed_trial_count                                         | Integer        | Optional                                                    | None    | None                                           | Define the maximum number of failed lessons. When the number of failed trainings reaches this value, tuning ends in failure.                 |
| max_trial_count                                                | Integer        | Optional                                                    | None    | None                                           | Defines the maximum number of lessons. Tuning runs until the number of auto-run training reaches this value.                     |
| tuning_strategy_name                                           | String         | Required                                                    | None    | None                                           | Choose which strategy to use to find the optimal hyperparameters.                                        |
| tuning_strategy_random_state                                   | Integer        | Optional                                                    | None    | None                                           | Determine random number generation. Specify a fixed value for reproducible results.                                 |
| early_stopping_algorithm                                       | String         | Required                                                    | None    | EARLY_STOPPING_ALGORITHM.<br>MEDIAN          | Stop training early if the model is no longer good even though training continues.                              |
| early_stopping_min_trial_count                                 | Integer        | Required                                                    | 3     | None                                           | Define how many trainings the target metric value will be taken from when calculating the median.                                |
| early_stopping_start_step                                      | Integer        | Required                                                    | 4     | None                                           | Set the training step from which to apply early stop.                                            |
| tag_list                                                       | Array          | Optional                                                    | None    | Max 10                                       | Tag information                                                                      |
| tag_list[0].tagKey                                             | String         | Optional                                                    | None    | Up to 64 characters                                       | Tag key                                                                       |
| tag_list[0].tagValue                                           | String         | Optional                                                    | None    | Up to 255 characters                                      | Tag value                                                                       |
| use_log                                                        | Boolean        | Optional                                                    | False | True, False                                  | Whether to leave logs in Log & Crash product                                                 |
| wait                                                           | Boolean        | Optional                                                    | True  | True, False                                  | True: returns hyperparameter tuning ID after creation of hyperparameter tuning is complete, False: returns training ID immediately after creation request |

```python
hyperparameter_tuning_id = easymaker.HyperparameterTuning().run(
    experiment_id=experiment_id,
    hyperparameter_tuning_name='hyperparameter_tuning_name',
    hyperparameter_tuning_description='hyperparameter_tuning_description',
    image_name='Ubuntu 18.04 CPU TensorFlow Training',
    instance_name='m2.c8m16',
    distributed_training_count=1,
    parallel_trial_count=1,
    data_storage_size=300,
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_spec_list=[
        {
            "hyperparameterName": "learning_rate",
            "hyperparameterTypeCode": easymaker.HYPERPARAMETER_TYPE_CODE.DOUBLE,
            "hyperparameterMinValue": "0.01",
            "hyperparameterMaxValue": "0.05",
        },
         {
            "hyperparameterName": "epochs",
            "hyperparameterTypeCode": easymaker.HYPERPARAMETER_TYPE_CODE.INT,
            "hyperparameterMinValue": "100",
            "hyperparameterMaxValue": "1000",
        }
    ],
    timeout_hours=10,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_input_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_input_path}',
    check_point_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        {
            "datasetName": "train",
            "dataUri": "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_download_path}"
        },
        {
            "datasetName": "test",
            "dataUri": "obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_download_path}"
        }
    ],
    metric_list=["val_loss", "loss", "accuracy"],
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
    tag_list=[
        {
            "tagKey": "tag1",
            "tagValue": "test_tag_1",
        }
    ],
    use_log=True,
    # wait=False,
)
```

### Delete Hyperparameter Tuning

[Parameter]

| Name                     | Type      | Required | Default value  | Valid range  | Description           |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | Required    | None   | Up to 36 characters | Hyperparameter Tuning ID |

```python
easymaker.HyperparameterTuning().delete(hyperparameter_tuning_id)
```

### Create Model
Request to create a model with the training ID.
The model is used when creating endpoints.

[Parameter]

| Name                       | Type     | Required                              | Default value | Valid range   | Description                                  |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | Required if hyperparameter_tuning_id does not exist | None  | None      | Training ID to create a model                       |
| hyperparameter_tuning_id | String | Required if training_id is not present              | None  | None      | Hyperparameter tuning ID to be created by model (created by best learning) |
| model_name               | String | Required                                 | None  | Up to 50 characters  | Model name                               |
| model_description        | String | Optional                                 | None  | Up to 255 characters | Description for model                           |
| tag_list                 | Array  | Optional                                 | None  | Max 10  | Tag information                               |
| tag_list[0].tagKey       | String | Optional                                 | None  | Up to 64 characters  | Tag key                                |
| tag_list[0].tagValue     | String | Optional                                 | None  | Up to 255 characters | Tag value                                |


```python
model_id = easymaker.Model().create(
    training_id=training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning_id,
    model_name='model_name',
    model_description='model_description',
)
```

Even if there is no training ID, you can create a model by entering the path information for the model and framework type.

[Parameter]

| Name                   | Type     | Required | Default value | Valid range                                   | Description                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| framework_code       | Enum   | Required    | None  | easymaker.TENSORFLOW, easymaker.PYTORCH | Framework information used for training                                    |
| model_uri            | String | Required    | None  | Up to 255 characters                                 | Path for model file (NHN Cloud Object Storage or NHN Cloud NAS) |
| model_name           | String | Required    | None  | Up to 50 characters                                  | Model name                                               |
| model_description    | String | Optional    | None  | Up to 255 characters                                 | Description for model                                           |
| tag_list             | Array  | Optional    | None  | Max 10                                  | Tag information                                               |
| tag_list[0].tagKey   | String | Optional    | None  | Up to 64 characters                                  | Tag key                                                |
| tag_list[0].tagValue | String | Optional    | None  | Up to 255 characters                                 | Tag value                                                |


```python
model_id = easymaker.Model().create_by_model_uri(
    framework_code=easymaker.TENSORFLOW,
    model_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### Delete Model

[Parameter]

| Name                        | Type      | Required | Default value  | Valid range  | Description    |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | Required    | None   | Up to 36 characters | Model ID |

```python
easymaker.Model().delete(model_id)
```

### Create Endpoint

When creating an endpoint, the default stage is created.

[Parameter]

| Name                                    | Type      | Required | Default value   | Valid range                      | Description                                                                     |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                         | String  | Required    | None    | Up to 50 characters                     | Endpoint name                                                               |
| endpoint_description                  | String  | Optional    | None    | Up to 255 characters                    | Description for endpoint                                                           |
| endpoint_instance_name                | String  | Required    | None    | None                         | Instance flavor name to be used for endpoint                                                  |
| endpoint_instance_count               | Integer | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                      |
| endpoint_model_resource_list          | Array   | Required    | None    | Max 10                     | Resource information to be used on the stage                                                 |
| endpoint_model_resource_list[0].modelId           | String   | Required    | None    | None                       | Model ID to be created as a stage resource                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | Required    | None    | Up to 255 characters                  | Path for API Gateway resource starting with /                             |
| endpoint_model_resource_list[0].podCount          | Integer  | Required    | None    | 1~100                     | Number of pods to be used for stage resources                                    |
| endpoint_model_resource_list[0].description       | String   | Optional    | None    | Up to 255 characters                  | Description of stage resource                                       |
| tag_list                              | Array   | Optional    | None    | Max 10                     | Tag information                                                                  |
| tag_list[0].tagKey                    | String  | Optional    | None    | Up to 64 characters                     | Tag key                                                                   |
| tag_list[0].tagValue                  | String  | Optional    | None    | Up to 255 characters                    | Tag value                                                                   |
| use_log                               | Boolean | Optional    | False | True, False                | Whether to leave logs in Log & Crash product                                             |
| wait                                  | Boolean | Optional    | True  | True, False                | True: Return the endpoint ID after creating endpoint, 
False: Return the endpoint ID immediately after requesting endpoint |

```python
endpoint = easymaker.Endpoint()
endpoint_id = endpoint.create(
    endpoint_name='endpoint_name',
    endpoint_description='endpoint_description',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1,
    endpoint_model_resource_list=[
        {
            'modelId': model_id,
            'apigwResourceUri': '/predict',
            'podCount': 1,
            'description': 'stage_resource_description'
        }
    ],
    use_log=True,
    # wait=False,
)
```

Use the created endpoint

```python
endpoint = easymaker.Endpoint()
```

### Add Stage

You can add a new stage to existing endpoints.

[Parameter]

| Name                                    | Type      | Required | Default value   | Valid range                      | Description                                                                 |
|---------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| stage_name                            | String  | Required    | None    | Up to 50 characters                     | Stage name                                                            |
| stage_description                     | String  | Optional    | None    | Up to 255 characters                    | Description for stage                                                        |
| endpoint_instance_name                | String  | Required    | None    | None                         | Instance flavor name to be used for endpoint                                              |
| endpoint_instance_count               | Integer | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                  |
| endpoint_model_resource_list          | Array   | Required    | None    | Max 10                     | Resource information to be used on the stage                                                 |
| endpoint_model_resource_list[0].modelId           | String   | Required    | None    | None                       | Model ID to be created as a stage resource                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | Required    | None    | Up to 255 characters                  | Path for API Gateway resource starting with /                             |
| endpoint_model_resource_list[0].podCount          | Integer  | Required    | None    | 1~100                     | Number of pods to be used for stage resources                                    |
| endpoint_model_resource_list[0].description       | String   | Optional    | None    | Up to 255 characters                  | Description of stage resource                                       |
| tag_list                              | Array   | Optional    | None    | Max 10                     | Tag information                                                              |
| tag_list[0].tagKey                    | String  | Optional    | None    | Up to 64 characters                     | Tag key                                                               |
| tag_list[0].tagValue                  | String  | Optional    | None    | Up to 255 characters                    | Tag value                                                               |
| use_log                               | Boolean | Optional    | False | True, False                | Whether to leave logs in Log & Crash product                                         |
| wait                                  | Boolean | Optional    | True  | True, False                | True: Return the stage ID after creating stage,
False: Return the stage ID immediately after requesting stage |
```python
stage_id = endpoint.create_stage(
    stage_name='stage01', # Within 30 lowercase letters/numbers
    stage_description='test endpoint',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1;
    endpoint_model_resource_list=[
        {
            'modelId': model_id,
            'apigwResourceUri': '/predict',
            'podCount': 1;
            'description': 'stage_resource_description'
        }
    ],
    use_log=True;
    #wait=False,
)
```

### Endpoint Inference

Inference to the default stage

```python
# Check basic stage information
endpoint_stage_info = endpoint.get_default_endpoint_stage()
print(f'endpoint_stage_info : {endpoint_stage_info}')

# Request inference by specifying a stage
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

Inference by specifying a specific stage

```python
# Check stage information
endpoint_stage_info = endpoint.get_endpoint_stage_by_id(endpoint_stage_id=stage_id)
print(f'endpoint_stage_info : {endpoint_stage_info}')

# Request inference by specifying a stage
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

### Delete Endpoint

[Parameter]

| Name            | Type      | Required | Default value  | Valid range  | Description       |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | Required    | None   | Up to 36 characters | Endpoint ID |

```python
endpoint.Endpoint().delete_endpoint(endpoint_id)
```

### Delete Endpoint Stage

[Parameter]

| Name         | Type      | Required | Default value  | Valid range  | Description      |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | Required    | None   | Up to 36 characters | Stage ID |

```python
endpoint.Endpoint().delete_endpoint_stage(stage_id)
```

### NHN Cloud - Log & Crash Log Sending Feature
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

## CLI Command
If you know the app key, secret key, and region information, you can check various information through Python CLI without accessing the console.

| Feature                          | Command                                                                                        |
|-----------------------------|--------------------------------------------------------------------------------------------|
| Query instance type list         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| Query image list                 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| Query algorithm list             | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -algorithm  |
| Query experiment list            | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| Query training list              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| Query hyperparameter tuning list | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -tuning     |
| Query model list                 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| Query endpoint list              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |
