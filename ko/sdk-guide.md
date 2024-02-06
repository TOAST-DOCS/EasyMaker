## NHN Cloud > SDK 사용 가이드 > AI EasyMaker

## 개발 가이드

### AI EasyMaker 파이썬 SDK 설치

python -m pip install easymaker

* AI EasyMaker 노트북에는 기본적으로 설치되어 있습니다.


### AI EasyMaker SDK 초기화
앱키(Appkey)와 비밀 키(Secret key)는 콘솔 오른쪽 상단의 **URL & Appkey** 메뉴에서 확인할 수 있습니다.
활성화한 AI EasyMaker 상품의 앱키, 비밀 키, 리전 정보를 입력합니다.
AI EasyMaker SDK를 사용하기 위해서는 초기화 코드가 필요합니다.
```python
import easymaker

easymaker.init(
    appkey='EASYMAKER_APPKEY',
    region='kr1',
    secret_key='EASYMAKER_SECRET_KEY',
)
```

### 실험 생성
학습을 생성하기 전에 학습을 분류할 수 있는 실험 생성이 필요합니다.

[Parameter]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위       | 설명                                                         |
|------------------------|---------|-------|------|-------------|------------------------------------------------------------|
| experiment_name        | String  | 필수    | 없음   | 최대 50자      | 실험 이름                                                      |
| experiment_description | String  | 선택    | 없음   | 최대 255자     | 실험에 대한 설명                                                  |
| wait                   | Boolean | 선택    | True | True, False | True: 실험 생성이 완료된 이후 실험 ID를 반환, False: 생성 요청 후 즉시 실험 ID를 반환 |

```python
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### 실험 삭제

[Parameter]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | 필수    | 없음   | 최대 36자 | 실험 ID |

```python
easymaker.Experiment().delete(experiment_id)
```

### 학습 생성

[Parameter]

| 이름                                         | 타입      | 필수 여부                     | 기본값   | 유효 범위       | 설명                                                              |
|--------------------------------------------|---------|---------------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | 필수                        | 없음    | 없음          | 실험 ID                                                           |
| training_name                              | String  | 필수                        | 없음    | 최대 50자      | 학습 이름                                                           |
| training_description                       | String  | 선택                        | 없음    | 최대 255자     | 학습에 대한 설명                                                       |
| train_image_name                           | String  | 필수                        | 없음    | 없음          | 학습에 사용될 이미지 이름(CLI로 조회 가능)                                      |
| train_instance_name                        | String  | 필수                        | 없음    | 없음          | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| distributed_training_count                 | Integer | 필수                        | 없음    | 1~10         | 학습에 적용할 분산 학습 수                                                 |
| data_storage_size                          | Integer | Obejct Storage 사용 시 필수    | 없음    | 300~10000   | 학습에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB), NAS 사용 시 불필요               |
| algorithm_name                             | String  | NHN Cloud 제공 알고리즘 사용 시 필수 | 없음    | 최대 64자      | 알고리즘 이름(CLI로 조회 가능)                                             |
| source_dir_uri                             | String  | 자체 알고리즘 사용 시 필수           | 없음    | 최대 255자     | 학습에 필요한 파일들이 들어 있는 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| entry_point                                | String  | 자체 알고리즘 사용 시 필수           | 없음    | 최대 255자     | source_dir_uri 안에서 최초 실행될 파이썬 파일 정보                             |
| model_upload_uri                           | String  | 필수                        | 없음    | 최대 255자     | 학습 완료된 모델이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)   |
| check_point_input_uri                      | String  | 선택                        | 없음    | 최대 255자     | 입력 체크 포인트 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)                 |
| check_point_upload_uri                     | String  | 선택                        | 없음    | 최대 255자     | 체크 포인트 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)   |
| timeout_hours                              | Integer | 선택                        | 720   | 1~720       | 최대 학습 시간(단위: 시간)                                                |
| hyperparameter_list                        | Array   | 선택                        | 없음    | 최대 100개     | 하이퍼파라미터 정보(hyperparameterKey/hyperparameterValue로 구성)           |
| hyperparameter_list[0].hyperparameterKey   | String  | 선택                        | 없음    | 최대 255자     | 하이퍼파라미터 키                                                       |
| hyperparameter_list[0].hyperparameterValue | String  | 선택                        | 없음    | 최대 1000자    | 하이퍼파라미터 값                                                       |
| dataset_list                               | Array   | 선택                        | 없음    | 최대 10개      | 학습에 사용될 데이터 세트 정보(datasetName/dataUri로 구성)                      |
| dataset_list[0].datasetName                | String  | 선택                        | 없음    | 최대 36자      | 데이터 이름                                                          |
| dataset_list[0].datasetUri                 | String  | 선택                        | 없음    | 최대 255자     | 데이터 경로                                                          |
| tag_list                                   | Array   | 선택                        | 없음    | 최대 10개      | 태그 정보                                                           |
| tag_list[0].tagKey                         | String  | 선택                        | 없음    | 최대 64자      | 태그 키                                                            |
| tag_list[0].tagValue                       | String  | 선택                        | 없음    | 최대 255자     | 태그 값                                                            |
| use_log                                    | Boolean | 선택                        | False | True, False | Log & Crash 상품에 로그를 남길지 여부                                      |
| wait                                       | Boolean | 선택                        | True  | True, False | True: 학습 생성이 완료된 이후 학습 ID를 반환, False: 생성 요청 후 즉시 학습 ID를 반환      |

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

### 학습 삭제

[Parameter]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | 필수    | 없음   | 최대 36자 | 학습 ID |

```python
easymaker.Training().delete(training_id)
```

### 하이퍼파라미터 튜닝 생성

[Parameter]

| 이름                                                             | 타입             | 필수 여부                                                 | 기본값   | 유효 범위                                        | 설명                                                                         |
|----------------------------------------------------------------|----------------|-------------------------------------------------------|-------|----------------------------------------------|----------------------------------------------------------------------------|
| experiment_id                                                  | String         | 필수                                                    | 없음    | 없음                                           | 실험 ID                                                                      |
| hyperparameter_tuning_name                                     | String         | 필수                                                    | 없음    | 최대 50자                                       | 하이퍼파라미터 튜닝 이름                                                              |
| hyperparameter_tuning_description                              | String         | 선택                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에 대한 설명                                                          |
| image_name                                                     | String         | 필수                                                    | 없음    | 없음                                           | 하이퍼파라미터 튜닝에 사용될 이미지 이름(CLI로 조회 가능)                                         |
| instance_name                                                  | String         | 필수                                                    | 없음    | 없음                                           | 인스턴스 타입 이름(CLI로 조회 가능)                                                     |
| distributed_training_count                                     | Integer        | 필수                                                    | 1      | distributed_training_count와 parallel_trial_count의 곱이 10 이하 | 하이퍼파라미터 튜닝에서 각 학습당 적용할 분산 학습 수                                                      |
| parallel_trial_count                                           | Integer        | 필수                                                    | 1      | distributed_training_count와 parallel_trial_count의 곱이 10 이하 | 하이퍼파라미터 튜닝에서 병렬로 실행할 학습 수                                                      |
| data_storage_size                                              | Integer        | Obejct Storage 사용 시 필수                                | 없음    | 300~10000                                    | 하이퍼파라미터 튜닝에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB), NAS 사용 시 불필요                  |
| algorithm_name                                                 | String         | NHN Cloud 제공 알고리즘 사용 시 필수                             | 없음    | 최대 64자                                       | 알고리즘 이름(CLI로 조회 가능)                                                        |
| source_dir_uri                                                 | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에 필요한 파일들이 들어있는 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)    |
| entry_point                                                    | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 255자                                      | source_dir_uri 안에서 최초 실행될 파이썬 파일 정보                                        |
| model_upload_uri                                               | String         | 필수                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에서 학습 완료된 모델이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| check_point_input_uri                                          | String         | 선택                                                    | 없음    | 최대 255자                                      | 입력 체크 포인트 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)                 |
| check_point_upload_uri                                         | String         | 선택                                                    | 없음    | 최대 255자                                      | 체크 포인트 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)              |
| timeout_hours                                                  | Integer        | 선택                                                    | 720   | 1~720                                        | 최대 하이퍼파라미터 튜닝 시간(단위: 시간)                                                   |
| hyperparameter_spec_list                                       | Array          | 선택                                                    | 없음    | 최대 100개                                      | 하이퍼파라미터 스펙 정보                                                              |
| hyperparameter_spec_list[0].<br>hyperparameterName             | String         | 선택                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 이름                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterTypeCode         | String         | 선택                                                    | 없음    | INT, DOUBLE, DISCRETE, CATEGORICAL           | 하이퍼파라미터 타입                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterMinValue         | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE인 경우 필수            | 없음    | 없음                                           | 하이퍼파라미터 최솟값                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterMaxValue         | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE인 경우 필수            | 없음    | 없음                                           | 하이퍼파라미터 최댓값                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterStep             | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE이면서 GRID 전략인 경우 필수 | 없음    | 없음                                           | "Grid" 튜닝 전략을 사용할 때 하이퍼파라미터 값의 변화 크기                                       |
| hyperparameter_spec_list[0].<br>hyperparameterSpecifiedValues  | String         | hyperparameterTypeCode가 DISCRETE, CATEGORICAL 경우 필수   | 없음    | 최대 3000자                                       | 정해진 하이퍼파라미터 목록(`,`로 구분된 문자열이나 숫자)                                          |
| dataset_list                                                   | Array          | 선택                                                    | 없음    | 최대 10개                                       | 하이퍼파라미터 튜닝에 사용될 데이터 세트 정보(datasetName/dataUri로 구성)                         |
| dataset_list[0].datasetName                                    | String         | 선택                                                    | 없음    | 최대 36자                                       | 데이터 이름                                                                     |
| dataset_list[0].datasetUri                                     | String         | 선택                                                    | 없음    | 최대 255자                                      | 데이터 경로                                                                     |
| metric_list                                                    | Array          | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 10개(지표 이름들로 된 문자열 리스트)                    | 학습 코드가 출력하는 로그 중에 어떤 지표를 수집할지 정의합니다.                                       |
| metric_regex                                                   | String         | 자체 알고리즘 사용 시 선택                                       | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?) | 최대 255자                                                                    | 지표를 수집하는 데 사용할 정규 표현식을 입력합니다. 학습 알고리즘이 정규 표현식에 맞게 지표를 출력해야 합니다.                                                          |
| objective_metric_name                                          | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 36자, metric_list 중 하나                     | 어떤 지표를 최적화하는 게 목표인지 선택합니다.                                                 |
| objective_type_code                                            | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | MINIMIZE, MAXIMIZE                           | 목표 지표 최적화 유형을 선택합니다.                                                       |
| objective_goal                                                 | Double         | 선택                                                    | 없음    | 없음                                           | 목표 지표가 이 값에 도달하면 튜닝 작업이 종료됩니다.                                             |
| max_failed_trial_count                                         | Integer        | 선택                                                    | 없음    | 없음                                           | 실패한 학습의 최대 개수를 정의합니다. 실패한 학습의 개수가 이 값에 도달하면 튜닝이 실패로 종료됩니다.                 |
| max_trial_count                                                | Integer        | 선택                                                    | 없음    | 없음                                           | 최대 학습 수를 정의합니다. 자동 실행된 학습의 개수가 이 값에 도달할 때까지 튜닝이 실행됩니다.                     |
| tuning_strategy_name                                           | String         | 필수                                                    | 없음    | 없음                                           | 어떤 전략을 사용해서 최적의 하이퍼파라미터를 찾을지 선택합니다.                                        |
| tuning_strategy_random_state                                   | Integer        | 선택                                                    | 없음    | 없음                                           | 난수 생성을 결정합니다. 재현 가능한 결과를 위해 고정된 값으로 지정합니다.                                 |
| early_stopping_algorithm                                       | String         | 필수                                                    | 없음    | EARLY_STOPPING_ALGORITHM.<br>MEDIAN          | 학습이 계속 진행되어도 모델이 더 이상 좋아지지 않으면 학습을 조기에 종료합니다.                              |
| early_stopping_min_trial_count                                 | Integer        | 필수                                                    | 3     | 없음                                           | 중간값을 계산할 때 몇 개의 학습으로부터 목표 지표 값을 가져올지 정의합니다.                                |
| early_stopping_start_step                                      | Integer        | 필수                                                    | 4     | 없음                                           | 몇 번째 학습 단계부터 조기 중지를 적용할지 설정합니다.                                            |
| tag_list                                                       | Array          | 선택                                                    | 없음    | 최대 10개                                       | 태그 정보                                                                      |
| tag_list[0].tagKey                                             | String         | 선택                                                    | 없음    | 최대 64자                                       | 태그 키                                                                       |
| tag_list[0].tagValue                                           | String         | 선택                                                    | 없음    | 최대 255자                                      | 태그 값                                                                       |
| use_log                                                        | Boolean        | 선택                                                    | False | True, False                                  | Log & Crash 상품에 로그를 남길지 여부                                                 |
| wait                                                           | Boolean        | 선택                                                    | True  | True, False                                  | True: 하이퍼파라미터 튜닝 생성이 완료된 이후 하이퍼파라미터 튜닝 ID를 반환, False: 생성 요청 후 즉시 학습 ID를 반환 |

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

### 하이퍼파라미터 튜닝 삭제

[Parameter]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명           |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | 필수    | 없음   | 최대 36자 | 하이퍼파라미터 튜닝 ID |

```python
easymaker.HyperparameterTuning().delete(hyperparameter_tuning_id)
```

### 모델 생성
학습 ID 값으로 모델 생성을 요청할 수 있습니다.
모델은 엔드포인트 생성 시 사용됩니다.

[Parameter]

| 이름                       | 타입     | 필수 여부                              | 기본값 | 유효 범위   | 설명                                  |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | hyperparameter_tuning_id가 없는 경우 필수 | 없음  | 없음      | 모델로 생성할 학습 ID                       |
| hyperparameter_tuning_id | String | training_id가 없는 경우 필수              | 없음  | 없음      | 모델로 생성할 하이퍼파라미터 튜닝 ID(최고 학습으로 생성됨) |
| model_name               | String | 필수                                 | 없음  | 최대 50자  | 모델 이름                               |
| model_description        | String | 선택                                 | 없음  | 최대 255자 | 모델에 대한 설명                           |
| tag_list                 | Array  | 선택                                 | 없음  | 최대 10개  | 태그 정보                               |
| tag_list[0].tagKey       | String | 선택                                 | 없음  | 최대 64자  | 태그 키                                |
| tag_list[0].tagValue     | String | 선택                                 | 없음  | 최대 255자 | 태그 값                                |


```python
model_id = easymaker.Model().create(
    training_id=training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning_id,
    model_name='model_name',
    model_description='model_description',
)
```

학습 ID가 없더라도, 모델이 저장된 경로 정보와 프레임워크 종류를 입력하여 모델을 생성할 수 있습니다.

[Parameter]

| 이름                   | 타입     | 필수 여부 | 기본값 | 유효 범위                                   | 설명                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| framework_code       | Enum   | 필수    | 없음  | easymaker.TENSORFLOW, easymaker.PYTORCH | 학습에 사용된 프레임워크 정보                                    |
| model_uri            | String | 필수    | 없음  | 최대 255자                                 | 모델 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| model_name           | String | 필수    | 없음  | 최대 50자                                  | 모델 이름                                               |
| model_description    | String | 선택    | 없음  | 최대 255자                                 | 모델에 대한 설명                                           |
| tag_list             | Array  | 선택    | 없음  | 최대 10개                                  | 태그 정보                                               |
| tag_list[0].tagKey   | String | 선택    | 없음  | 최대 64자                                  | 태그 키                                                |
| tag_list[0].tagValue | String | 선택    | 없음  | 최대 255자                                 | 태그 값                                                |


```python
model_id = easymaker.Model().create_by_model_uri(
    framework_code=easymaker.TENSORFLOW,
    model_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### 모델 삭제

[Parameter]

| 이름                        | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | 필수    | 없음   | 최대 36자 | 모델 ID |

```python
easymaker.Model().delete(model_id)
```

### 엔드포인트 생성

엔드포인트 생성 시 기본 스테이지가 생성됩니다.

[Parameter]

| 이름                                    | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                     |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                         | String  | 필수    | 없음    | 최대 50자                     | 엔드포인트 이름                                                               |
| endpoint_description                  | String  | 선택    | 없음    | 최대 255자                    | 엔드포인트에 대한 설명                                                           |
| endpoint_instance_name                | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                                  |
| endpoint_instance_count               | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                      |
| endpoint_model_resource_list          | Array   | 필수    | 없음    | 최대 10개                     | 스테이지에 사용될 리소스 정보                                                 |
| endpoint_model_resource_list[0].modelId           | String   | 필수    | 없음    | 없음                       | 스테이지 리소스로 생성할 모델 ID                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | 필수    | 없음    | 최대 255자                  | /로 시작하는 API Gateway 리소스 경로                             |
| endpoint_model_resource_list[0].podCount          | Integer  | 필수    | 없음    | 1~100                     | 스테이지 리소스에 사용될 파드 수                                    |
| endpoint_model_resource_list[0].description       | String   | 선택    | 없음    | 최대 255자                  | 스테이지 리소스에 대한 설명                                       |
| tag_list                              | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                                  |
| tag_list[0].tagKey                    | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                                   |
| tag_list[0].tagValue                  | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                                   |
| use_log                               | Boolean | 선택    | False | True, False                | Log & Crash 상품에 로그를 남길지 여부                                             |
| wait                                  | Boolean | 선택    | True  | True, False                | True: 엔드포인트 생성이 완료된 이후 엔드포인트 ID를 반환, False: 엔드포인트 요청 후 즉시 엔드포인트 ID를 반환 |

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

생성해둔 엔드포인트 사용

```python
endpoint = easymaker.Endpoint()
```

### 스테이지 추가

기존 엔드포인트에 신규 스테이지를 추가할 수 있습니다.

[Parameter]

| 이름                                    | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                 |
|---------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| stage_name                            | String  | 필수    | 없음    | 최대 50자                     | 스테이지 이름                                                            |
| stage_description                     | String  | 선택    | 없음    | 최대 255자                    | 스테이지에 대한 설명                                                        |
| endpoint_instance_name                | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                              |
| endpoint_instance_count               | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                  |
| endpoint_model_resource_list          | Array   | 필수    | 없음    | 최대 10개                     | 스테이지에 사용될 리소스 정보                                                 |
| endpoint_model_resource_list[0].modelId           | String   | 필수    | 없음    | 없음                       | 스테이지 리소스로 생성할 모델 ID                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | 필수    | 없음    | 최대 255자                  | /로 시작하는 API Gateway 리소스 경로                             |
| endpoint_model_resource_list[0].podCount          | Integer  | 필수    | 없음    | 1~100                     | 스테이지 리소스에 사용될 파드 수                                    |
| endpoint_model_resource_list[0].description       | String   | 선택    | 없음    | 최대 255자                  | 스테이지 리소스에 대한 설명                                       |
| tag_list                              | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                              |
| tag_list[0].tagKey                    | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                               |
| tag_list[0].tagValue                  | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                               |
| use_log                               | Boolean | 선택    | False | True, False                | Log & Crash 상품에 로그를 남길지 여부                                         |
| wait                                  | Boolean | 선택    | True  | True, False                | True: 스테이지 생성이 완료된 이후 스테이지 ID를 반환, False: 스테이지 요청 후 즉시 스테이지 ID를 반환 |
```python
stage_id = endpoint.create_stage(
    stage_name='stage01',  # 30자 이내 소문자/숫자
    stage_description='test endpoint',
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

### 엔드포인트 인퍼런스

기본 스테이지에 인퍼런스

```python
# 기본 스테이지 정보 조회
endpoint_stage_info = endpoint.get_default_endpoint_stage()
print(f'endpoint_stage_info : {endpoint_stage_info}')

# 스테이지를 지정하여 인퍼런스 요청
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

특정 스테이지 지정하여 인퍼런스

```python
# 스테이지 정보 조회
endpoint_stage_info = endpoint.get_endpoint_stage_by_id(endpoint_stage_id=stage_id)
print(f'endpoint_stage_info : {endpoint_stage_info}')

# 스테이지를 지정하여 인퍼런스 요청
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

### 엔드포인트 삭제

[Parameter]

| 이름            | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명       |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | 필수    | 없음   | 최대 36자 | 엔드포인트 ID |

```python
endpoint.Endpoint().delete_endpoint(endpoint_id)
```

### 엔드포인트 스테이지 삭제

[Parameter]

| 이름         | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명      |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | 필수    | 없음   | 최대 36자 | 스테이지 ID |

```python
endpoint.Endpoint().delete_endpoint_stage(stage_id)
```

### NHN Cloud - Log & Crash 로그 전송 기능
```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default: INFO
                      project_version='2.0.0',  # default: 1.0.0
                      parameters={'serviceType': 'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storage 파일 전송 기능
Object Storage 상품으로 파일을 업로드하고 다운로드하는 기능을 제공합니다.
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
앱키, 비밀 키, 리전 정보를 알고 있다면, 콘솔에 접근하지 않고도 파이썬 CLI를 통해 여러 정보를 확인할 수 있습니다.

| 기능                          | 명령어                                                                                        |
|-----------------------------|--------------------------------------------------------------------------------------------|
| Instance type 목록 조회         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| Image 목록 조회                 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| Algorithm 목록 조회             | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -algorithm  |
| Experiment 목록 조회            | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| Training 목록 조회              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| Hyperparameter tuning 목록 조회 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -tuning     |
| Model 목록 조회                 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| Endpoint 목록 조회              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |
