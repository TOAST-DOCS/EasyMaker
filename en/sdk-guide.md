## NHN Cloud > SDK User Guide > AI EasyMaker

## Development Guide

### Install AI EasyMaker Python SDK

python -m pip install easymaker

* AI EasyMaker is installed in the notebook by default.


### Initialize AI EasyMaker SDK
You can find AppKey and SecretKey in the **URL & Appkey** menu at the right top on the console.
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
| wait                   | Boolean | Optional    | True | True, False | True: Return the experiment ID after creating the experiment, False: Return the experiment ID immediately after request to create |

```python
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### Create Training

[Parameter]

| Name                                         | Type      | Required                 | Default value   | Valid range       | Description                                                              |
|--------------------------------------------|---------|-----------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | Required                    | None    | None          | Experiment ID                                                           |
| training_name                              | String  | Required                    | None    | Up to 50 characters      | Training name                                                           |
| training_description                       | String  | Optional                    | None    | Up to 255 characters     | Description for training                                                       |
| train_image_name                           | String  | Required                    | None    | None          | Image name to be used for training (Inquiry available with CLI)                                      |
| train_instance_name                        | String  | Required                    | None    | None          | Instance flavor name (Inquiry available with CLI)                                          |
| train_instance_count                       | Integer | Required                    | None    | 1~10        | Number of instances to be used for training                                                  |
| data_storage_size                          | Integer | Required when using Object Storage | None    | 300~10000   | Storage size to download data for training (unit: GB), unnecessary when using NAS                |
| source_dir_uri                             | String  | Required                    | None    | Up to 255 characters     | Path of files required for training (NHN Cloud Object Storage or NHN Cloud NAS) |
| model_upload_uri                           | String  | Required                    | None    | Up to 255 characters     | Path to upload the model completed with training (NHN Cloud Object Storage or NHN Cloud NAS)  |
| check_point_upload_uri                     | String  | Optional                    | None    | Up to 255 characters     | Path to upload checkpoint files (NHN Cloud Object Storage or NHN Cloud NAS)   |
| entry_point                                | String  | Required                    | None    | Up to 255 characters     | Information of Python files to be executed initially in source_dir_uri                             |
| timeout_hours                              | Integer | Optional                    | 720   | 1~720       | Max training time (unit: hour)                                                |
| hyperparameter_list                        | Array   | Optional                    | None    | Max 100     | Information of hyperparameters (consists of hyperparameterKey/hyperparameterValue)           |
| hyperparameter_list[0].hyperparameterKey   | String  | Optional                    | None    | Up to 255 characters     | Hyperparameter key                                                       |
| hyperparameter_list[0].hyperparameterValue | String  | Optional                    | None    | Up to 1000 characters    | Hyperparameter value                                                       |
| dataset_list                               | Array   | Optional                    | None    | Max 10      | Information of dataset to be used for training (consists of datasetName/dataUri)                      |
| dataset_list[0].datasetName                | String  | Optional                    | None    | Up to 36 characters      | Data name                                                          |
| dataset_list[0].datasetUri                 | String  | Optional                    | None    | Up to 255 characters     | Data pah                                                          |
| tag_list                                   | Array   | Optional                    | None    | Max 10      | Tag information                                                           |
| tag_list[0].tagKey                         | String  | Optional                    | None    | Up to 64 characters      | Tag key                                                            |
| tag_list[0].tagValue                       | String  | Optional                    | None    | Up to 255 characters     | Tag value                                                            |
| use_log                                    | Boolean | Optional                    | False | True, False | Whether to leave logs in Log & Crash product                                      |
| wait                                       | Boolean | Optional                    | True  | True, False | True: Return the training ID after creating training, False: Return the training ID immediately after requesting to create      |

```python
training_id = easymaker.Training().run(
    experiment_id=experiment_id,
    training_name='training_name',
    training_description='training_description',
    train_image_name='Ubuntu 18.04 CPU TensorFlow Training',
    train_instance_name='m2.c4m8',
    train_instance_count=1,
    data_storage_size=300,  # minimum size : 300GB
    source_dir_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
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
    model_upload_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_upload_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        {
            "datasetName": "train",
            "dataUri": "obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_download_path}"
        },
        {
            "datasetName": "test",
            "dataUri": "obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_download_path}"
        }
    ],
    tag_list=[
        {
            "tagKey": "tag_num",
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

### Create Model
Request to create a model with the training ID.
The model is used when creating endpoints.

[Parameter]

| Name                   | Type     | Required | Default value | Valid range   | Description            |
|----------------------|--------|-------|-----|---------|---------------|
| training_id          | String | Required    | None  | None      | Training ID to create a model |
| model_name           | String | Required    | None  | Up to 50 characters  | Model name         |
| model_description    | String | Optional    | None  | Up to 255 characters | Description for model     |
| tag_list             | Array  | Optional    | None  | Max 10  | Tag information         |
| tag_list[0].tagKey   | String | Optional    | None  | Up to 64 characters  | Tag key          |
| tag_list[0].tagValue | String | Optional    | None  | Up to 255 characters | Tag value          |


```python
model_id = easymaker.Model().create(
    training_id=training_id,
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
    model_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### Create Endpoint

When creating an endpoint, the default stage is created.

[Parameter]

| Name                                    | Type      | Required | Default value   | Valid range                      | Description                                                                     |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| model_id                              | String  | Required    | None    | None                         | Model ID to be created with endpoint                                                       |
| endpoint_name                         | String  | Required    | None    | Up to 50 characters                     | Endpoint name                                                               |
| endpoint_description                  | String  | Optional    | None    | Up to 255 characters                    | Description for endpoint                                                           |
| endpoint_instance_name                | String  | Required    | None    | None                         | Instance flavor name to be used for endpoint                                                  |
| endpoint_instance_count               | Integer | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                      |
| apigw_resource_uri                    | String  | Required    | None    | Up to 255 characters                    | Path for API Gateway resource starting with /                                             |
| tag_list                              | Array   | Optional    | None    | Max 10                     | Tag information                                                                  |
| tag_list[0].tagKey                    | String  | Optional    | None    | Up to 64 characters                     | Tag key                                                                   |
| tag_list[0].tagValue                  | String  | Optional    | None    | Up to 255 characters                    | Tag value                                                                   |
| use_log                               | Boolean | Optional    | False | True, False                | Whether to leave logs in Log & Crash product                                             |        
| wait                                  | Boolean | Optional    | True  | True, False                | True: Return the endpoint ID after creating endpoint, False: Return the endpoint ID immediately after requesting endpoint |

```python
endpoint = easymaker.Endpoint()
endpoint_id = endpoint.create(
    model_id=model_id,
    endpoint_name='endpoint_name',
    endpoint_description='endpoint_description',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1
    apigw_resource_uri='/predict',
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
| model_id                              | String  | Required    | None    | None                         | Model ID to be created with endpoint                                                   |
| stage_name                            | String  | Required    | None    | Up to 50 characters                     |                                                             |
|                      | String  | Optional    | None    | Up to 255 characters                    | Description for stage                                                        |
| endpoint_instance_name                | String  | Required    | None    | None                         | Instance flavor name to be used for endpoint                                              |
| endpoint_instance_count               | Integer | Optional    | 1     | 1~10                       | Instance count to be used for endpoint                                                  |
| tag_list                              | Array   | Optional    | None    | Max 10                     | Tag information                                                              |
| tag_list[0].tagKey                    | String  | Optional    | None    | Up to 64 characters                     | Tag key                                                               |
| tag_list[0].tagValue                  | String  | Optional    | None    | Up to 255 characters                    | Tag value                                                               |
| use_log                               | Boolean | Optional    | False | True, False                | Whether to leave logs in Log & Crash product                                         |        
| wait                                  | Boolean | Optional    | True  | True, False                | True: Return the stage ID after creating stage, False: Return the stage ID immediately after requesting stage |
```python
stage_id = endpoint.create_stage(
    model_id=model_id,
    stage_name='stage01',  # Lowercase letters within 30 characters and numbers
    stage_description='test endpoint',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1,
    use_log=True,
    # wait=False,
)
```

### Endpoint Inference

Inference to the default stage

```python
input_data = [6.8, 2.8, 4.8, 1.4]
endpoint.predict(json={'instances': [input_data]})
```

Inference by specifying a specific stage

```python
# Query stage information
endpoint_stage_info_list = endpoint.get_endpoint_stage_info_list()
for endpoint_stage_info in endpoint_stage_info_list:
    print(f'endpoint_stage_info : {endpoint_stage_info}')
    
# Request inference by specifying a stage
input_data = [6.0, 3.4, 4.5, 1.6]
for endpoint_stage_info in endpoint_stage_info_list:
    if endpoint_stage_info['stage_name'] == 'stage01':
        endpoint.predict(json={'instances': [input_data]},
                         endpoint_stage_info=endpoint_stage_info)
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
    easymaker_obs_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{upload_path}',
    src_dir_path='./local_dir',
    username='userId@nhn.com',
    password='nhn_object_storage_api_password'
)

easymaker.download(
    easymaker_obs_uri='obs://api-storage.cloud.toast.com/v1/AUTH_00000000000000000000000000000000/SDK/sample/source_dir',
    download_dir_path='./download_dir',
    username='userId@nhn.com',
    password='nhn_object_storage_api_password'
)
```

## CLI Command
If you have AppKey, SecretKey, and region information, you can check various information through Python CLI without accessing the console.

| Feature                  | Command                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------|
| Query instance type list | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| Query image list         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| Query experiment list    | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| Query training list      | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| Query model list         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| Query endpoint list      | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |