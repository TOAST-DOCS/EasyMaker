## Machine Learning > AI EasyMaker > SDK 사용 가이드

## SDK 설정

### AI EasyMaker 파이썬 SDK 설치

python -m pip install easymaker

- AI EasyMaker 노트북에는 기본적으로 설치되어 있습니다.

### AI EasyMaker SDK 초기화

앱키(Appkey)는 콘솔 오른쪽 상단의 **URL & Appkey** 메뉴에서 확인할 수 있습니다.
인증 토큰(Access token)에 대한 내용은 [API 호출 및 인증](https://docs.nhncloud.com/ko/nhncloud/ko/public-api/api-authentication/)에서 확인할 수 있습니다.
활성화한 AI EasyMaker 상품의 앱키, 인증 토큰, 리전 정보를 입력합니다.
AI EasyMaker SDK를 사용하기 위해서는 초기화 코드가 필요합니다.

```python
import easymaker

easymaker.init(
    appkey='EASYMAKER_APPKEY',
    region='kr1',
    access_token='EASYMAKER_ACCESS_TOKEN',
    experiment_id="EXPERIMENT_ID", # Optional
)
```

## CLI Command

앱키, 인증 토큰, 리전 정보를 알고 있다면, 콘솔에 접근하지 않고도 파이썬 CLI를 통해 여러 정보를 확인할 수 있습니다.

| 기능                      | 명령어                                                                                        |
|-------------------------|--------------------------------------------------------------------------------------------|
| 인스턴스 타입 목록 조회           | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -instance   |
| 학습 이미지 목록 조회            | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -image      |
| NHN Cloud 제공 알고리즘 목록 조회 | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -algorithm  |
| 실험 목록 조회                | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -experiment |
| 학습 목록 조회                | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -training   |
| 하이퍼파라미터 튜닝 목록 조회        | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -tuning     |
| 모델 목록 조회                | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -model      |
| 엔드포인트 목록 조회             | python -m easymaker --region kr1 --appkey EM_APPKEY --access_token EM_ACCESS_TOKEN -endpoint   |

## 실험

### 실험 생성

학습을 생성하기 전에 학습을 분류할 수 있는 실험 생성이 필요합니다.

[파라미터]

| 이름                       | 타입       | 필수 여부 | 기본값  | 유효 범위       | 설명                                                         |
|--------------------------|----------|-------|------|-------------|------------------------------------------------------------|
| experiment_name          | String   | 필수    | 없음   | 최대 50자      | 실험 이름                                                      |
| description   | String   | 선택    | 없음   | 최대 255자     | 실험에 대한 설명                                                  |
| wait                     | Boolean  | 선택    | True | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

```python
experiment = easymaker.Experiment().create(
    experiment_name='experiment_name',
    description='experiment_description',
    # wait=False,
)
```

### 실험 삭제

[파라미터]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | 필수    | 없음   | 최대 36자 | 실험 ID |

```python
easymaker.Experiment(experiment_id).delete()
```

## 학습

### 학습 생성

[파라미터]

| 이름                                    | 타입      | 필수 여부                     | 기본값   | 유효 범위       | 설명                                                              |
|---------------------------------------|---------|---------------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                         | String  | easymaker.init에서 미입력 시 필수 | 없음    | 최대 36자          | 실험 ID                                                           |
| training_name                         | String  | 필수                        | 없음    | 최대 50자      | 학습 이름                                                           |
| description                  | String  | 선택                        | 없음    | 최대 255자     | 학습에 대한 설명                                                       |
| image_name                      | String  | 필수                        | 없음    | 없음          | 학습에 사용될 이미지 이름(CLI로 조회 가능)                                      |
| instance_type_name                   | String  | 필수                        | 없음    | 없음          | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| distributed_node_count                | Integer | 필수                        | 없음    | 1~10         | 분산 학습을 적용할 노드 수                                                 |
| use_torchrun                          | Boolean | 선택                        | False  | True, False | torchrun 사용 여부, Pytorch 이미지에서만 사용 가능                            |
| nproc_per_node                        | Integer | use_torchrun True 시 필수    | 1      | 1~(CPU 개수 또는 GPU 개수) | 노드당 프로세스 개수, use_torchrun을 사용할 경우 반드시 설정해야 하는 값       |
| data_storage_size                     | Integer | Obejct Storage 사용 시 필수    | 없음    | 300~10000   | 학습에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB), NAS 사용 시 불필요               |
| algorithm_name                        | String  | NHN Cloud 제공 알고리즘 사용 시 필수 | 없음    | 최대 64자      | 알고리즘 이름(CLI로 조회 가능)                                             |
| source_dir_uri                        | String  | 자체 알고리즘 사용 시 필수           | 없음    | 최대 255자     | 학습에 필요한 파일들이 들어 있는 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| entry_point                           | String  | 자체 알고리즘 사용 시 필수           | 없음    | 최대 255자     | source_dir_uri 안에서 최초 실행될 파이썬 파일 정보                             |
| model_upload_uri                      | String  | 필수                        | 없음    | 최대 255자     | 학습 완료된 모델이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)   |
| check_point_input_uri                 | String  | 선택                        | 없음    | 최대 255자     | 입력 체크 포인트 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)                 |
| check_point_upload_uri                | String  | 선택                        | 없음    | 최대 255자     | 체크 포인트 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)   |
| timeout_hours                         | Integer | 선택                        | 720   | 1~720       | 최대 학습 시간(단위: 시간)                                                |
| hyperparameter_list                   | Array   | 선택                        | 없음    | 최대 100개     | 하이퍼파라미터 정보(parameterName/parameterValue로 구성)           |
| hyperparameter_list[0].parameterName  | String  | 선택                        | 없음    | 최대 255자     | 하이퍼파라미터 키                                                       |
| hyperparameter_list[0].parameterValue | String  | 선택                        | 없음    | 최대 1000자    | 하이퍼파라미터 값                                                       |
| dataset_list                          | Array   | 선택                        | 없음    | 최대 10개      | 학습에 사용될 데이터 세트 정보(datasetName/dataUri로 구성)                      |
| dataset_list[0].datasetName           | String  | 선택                        | 없음    | 최대 36자      | 데이터 이름                                                          |
| dataset_list[0].datasetUri            | String  | 선택                        | 없음    | 최대 255자     | 데이터 경로                                                          |
| tag_list                              | Array   | 선택                        | 없음    | 최대 10개      | 태그 정보                                                           |
| tag_list[0].tagKey                    | String  | 선택                        | 없음    | 최대 64자      | 태그 키                                                            |
| tag_list[0].tagValue                  | String  | 선택                        | 없음    | 최대 255자     | 태그 값                                                            |
| use_log                               | Boolean | 선택                        | False | True, False | Log & Crash Search 서비스에 로그를 남길지 여부                                      |
| wait                                  | Boolean | 선택                        | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

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
        {
            "parameterName": "epochs",
            "parameterValue": "10",
        },
        {
            "parameterName": "batch-size",
            "parameterValue": "30",
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

[파라미터]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | 필수    | 없음   | 최대 36자 | 학습 ID |

```python
easymaker.Training(training_id).delete()
```

## 하이퍼파라미터 튜닝

### 하이퍼파라미터 튜닝 생성

[파라미터]

| 이름                                                            | 타입             | 필수 여부                                                 | 기본값   | 유효 범위                                        | 설명                                                                         |
|---------------------------------------------------------------|----------------|-------------------------------------------------------|-------|----------------------------------------------|----------------------------------------------------------------------------|
| experiment_id                                                 | String         | easymaker.init에서 미입력 시 필수                             | 없음    | 최대 36자                                           | 실험 ID                                                                      |
| hyperparameter_tuning_name                                    | String         | 필수                                                    | 없음    | 최대 50자                                       | 하이퍼파라미터 튜닝 이름                                                              |
| description                             | String         | 선택                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에 대한 설명                                                          |
| image_name                                                    | String         | 필수                                                    | 없음    | 없음                                           | 하이퍼파라미터 튜닝에 사용될 이미지 이름(CLI로 조회 가능)                                         |
| instance_type_name                                                 | String         | 필수                                                    | 없음    | 없음                                           | 인스턴스 타입 이름(CLI로 조회 가능)                                                     |
| distributed_node_count                                        | Integer        | 필수                                                    | 1      | distributed_node_count와 parallel_trial_count의 곱이 10 이하 | 하이퍼파라미터 튜닝에서 각 학습당 분산 학습을 적용할 노드 수                                                      |
| parallel_trial_count                                          | Integer        | 필수                                                    | 1      | distributed_node_count와 parallel_trial_count의 곱이 10 이하 | 하이퍼파라미터 튜닝에서 병렬로 실행할 학습 수                                                          |
| use_torchrun                                                  | Boolean        | 선택                                                    | False  | True, False | torchrun 사용 여부, Pytorch 이미지에서만 사용 가능                                                 |
| nproc_per_node                                                | Integer        | use_torchrun True 시 필수                                | 1      | 1~(CPU 개수 또는 GPU 개수) | 노드 당 프로세스 개수, use_torchrun을 사용할 경우 반드시 설정해야 하는 값                                         |
| data_storage_size                                             | Integer        | Obejct Storage 사용 시 필수                                | 없음    | 300~10000                                    | 하이퍼파라미터 튜닝에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB), NAS 사용 시 불필요                  |
| algorithm_name                                                | String         | NHN Cloud 제공 알고리즘 사용 시 필수                             | 없음    | 최대 64자                                       | 알고리즘 이름(CLI로 조회 가능)                                                        |
| source_dir_uri                                                | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에 필요한 파일들이 들어있는 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)    |
| entry_point                                                   | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 255자                                      | source_dir_uri 안에서 최초 실행될 파이썬 파일 정보                                        |
| model_upload_uri                                              | String         | 필수                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 튜닝에서 학습 완료된 모델이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| check_point_input_uri                                         | String         | 선택                                                    | 없음    | 최대 255자                                      | 입력 체크 포인트 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)                 |
| check_point_upload_uri                                        | String         | 선택                                                    | 없음    | 최대 255자                                      | 체크 포인트 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)              |
| timeout_hours                                                 | Integer        | 선택                                                    | 720   | 1~720                                        | 최대 하이퍼파라미터 튜닝 시간(단위: 시간)                                                   |
| hyperparameter_spec_list                                      | Array          | 선택                                                    | 없음    | 최대 100개                                      | 하이퍼파라미터 스펙 정보                                                              |
| hyperparameter_spec_list[0].<br>hyperparameterName            | String         | 선택                                                    | 없음    | 최대 255자                                      | 하이퍼파라미터 이름                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterTypeCode        | String         | 선택                                                    | 없음    | INT, DOUBLE, DISCRETE, CATEGORICAL           | 하이퍼파라미터 타입                                                                 |
| hyperparameter_spec_list[0].<br>hyperparameterMinValue        | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE인 경우 필수            | 없음    | 없음                                           | 하이퍼파라미터 최솟값                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterMaxValue        | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE인 경우 필수            | 없음    | 없음                                           | 하이퍼파라미터 최댓값                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterStep            | Integer/Double | hyperparameterTypeCode가 INT, DOUBLE이면서 GRID 전략인 경우 필수 | 없음    | 없음                                           | "Grid" 튜닝 전략을 사용할 때 하이퍼파라미터 값의 변화 크기                                       |
| hyperparameter_spec_list[0].<br>hyperparameterSpecifiedValues | String         | hyperparameterTypeCode가 DISCRETE, CATEGORICAL 경우 필수   | 없음    | 최대 3000자                                       | 정해진 하이퍼파라미터 목록(`,`로 구분된 문자열이나 숫자)                                          |
| dataset_list                                                  | Array          | 선택                                                    | 없음    | 최대 10개                                       | 하이퍼파라미터 튜닝에 사용될 데이터 세트 정보(datasetName/dataUri로 구성)                         |
| dataset_list[0].datasetName                                   | String         | 선택                                                    | 없음    | 최대 36자                                       | 데이터 이름                                                                     |
| dataset_list[0].datasetUri                                    | String         | 선택                                                    | 없음    | 최대 255자                                      | 데이터 경로                                                                     |
| metric_list                                                   | Array          | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 10개(지표 이름들로 된 문자열 리스트)                    | 학습 코드가 출력하는 로그 중에 어떤 지표를 수집할지 정의                                       |
| metric_regex                                                  | String         | 자체 알고리즘 사용 시 선택                                       | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?) | 최대 255자                                                                    | 지표를 수집하는 데 사용할 정규 표현식을 입력. 학습 알고리즘이 정규 표현식에 맞게 지표를 출력해야 함.                                                        |
| objective_metric_name                                         | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | 최대 36자, metric_list 중 하나                     | 어떤 지표를 최적화하는 게 목표인지 선택                                                 |
| objective_type_code                                           | String         | 자체 알고리즘 사용 시 필수                                       | 없음    | MINIMIZE, MAXIMIZE                           | 목표 지표 최적화 유형 선택                                                       |
| objective_goal                                                | Double         | 선택                                                    | 없음    | 없음                                           | 목표 지표가 이 값에 도달하면 튜닝 작업이 종료됨                                             |
| max_failed_trial_count                                        | Integer        | 선택                                                    | 없음    | 없음                                           | 실패한 학습의 최대 개수를 정의. 실패한 학습의 개수가 이 값에 도달하면 튜닝이 실패로 종료됨.                 |
| max_trial_count                                               | Integer        | 선택                                                    | 없음    | 없음                                           | 최대 학습 수를 정의. 자동 실행된 학습의 개수가 이 값에 도달할 때까지 튜닝이 실행됨.                     |
| tuning_strategy_name                                          | String         | 필수                                                    | 없음    | 없음                                           | 어떤 전략을 사용해서 최적의 하이퍼파라미터를 찾을지 선택                                        |
| tuning_strategy_random_state                                  | Integer        | 선택                                                    | 없음    | 없음                                           | 난수 생성을 결정. 재현 가능한 결과를 위해 고정된 값으로 지정함.                                 |
| early_stopping_algorithm                                      | String         | 필수                                                    | 없음    | EARLY_STOPPING_ALGORITHM.<br>MEDIAN          | 학습이 계속 진행되어도 모델이 더 이상 좋아지지 않으면 학습을 조기에 종료                              |
| early_stopping_min_trial_count                                | Integer        | 필수                                                    | 3     | 없음                                           | 중간값을 계산할 때 몇 개의 학습으로부터 목표 지표 값을 가져올지 정의                                |
| early_stopping_start_step                                     | Integer        | 필수                                                    | 4     | 없음                                           | 몇 번째 학습 단계부터 조기 중지를 적용할지 설정합니다.                                            |
| tag_list                                                      | Array          | 선택                                                    | 없음    | 최대 10개                                       | 태그 정보                                                                      |
| tag_list[0].tagKey                                            | String         | 선택                                                    | 없음    | 최대 64자                                       | 태그 키                                                                       |
| tag_list[0].tagValue                                          | String         | 선택                                                    | 없음    | 최대 255자                                      | 태그 값                                                                       |
| use_log                                                       | Boolean        | 선택                                                    | False | True, False                                  | Log & Crash Search 서비스에 로그를 남길지 여부                                                 |
| wait                                                          | Boolean        | 선택                                                    | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

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

[파라미터]

| 이름                     | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명           |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | 필수    | 없음   | 최대 36자 | 하이퍼파라미터 튜닝 ID |

```python
easymaker.HyperparameterTuning(hyperparameter_tuning_id).delete()
```

## 모델

### 모델 생성

학습 ID 값으로 모델 생성을 요청할 수 있습니다.
모델은 엔드포인트 생성 시 사용됩니다.

[파라미터]

| 이름                       | 타입     | 필수 여부                              | 기본값 | 유효 범위   | 설명                                  |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | hyperparameter_tuning_id가 없는 경우 필수 | 없음  | 없음      | 모델로 생성할 학습 ID                       |
| hyperparameter_tuning_id | String | training_id가 없는 경우 필수              | 없음  | 없음      | 모델로 생성할 하이퍼파라미터 튜닝 ID(최고 학습으로 생성됨) |
| model_name               | String | 필수                                 | 없음  | 최대 50자  | 모델 이름                               |
| description        | String | 선택                                 | 없음  | 최대 255자 | 모델에 대한 설명                           |
| tag_list                 | Array  | 선택                                 | 없음  | 최대 10개  | 태그 정보                               |
| tag_list[0].tagKey       | String | 선택                                 | 없음  | 최대 64자  | 태그 키                                |
| tag_list[0].tagValue     | String | 선택                                 | 없음  | 최대 255자 | 태그 값                                |

```python
model = easymaker.Model().create(
    training_id=training.training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning.hyperparameter_tuning_id,
    model_name='model_name',
    description='model_description',
)
```

학습 ID가 없더라도, 모델이 저장된 경로 정보와 프레임워크 종류를 입력하여 모델을 생성할 수 있습니다.

[파라미터]

| 이름                   | 타입     | 필수 여부 | 기본값 | 유효 범위                                   | 설명                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| model_type_code       | Enum   | 필수    | 없음  | easymaker.TENSORFLOW, easymaker.PYTORCH | 학습에 사용된 프레임워크 정보                                    |
| model_upload_uri            | String | 필수    | 없음  | 최대 255자                                 | 모델 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| model_name           | String | 필수    | 없음  | 최대 50자                                  | 모델 이름                                               |
| description    | String | 선택    | 없음  | 최대 255자                                 | 모델에 대한 설명                                           |
| parameter_list                   | Array  | 선택    | 없음  | 최대 10개                                  | 파라미터 정보(parameterName/parameterValue로 구성)         |
| parameter_list[0].parameterName  | String | 선택    | 없음  | 최대 64자                                  | 파라미터 이름                                              |
| parameter_list[0].parameterValue | String | 선택    | 없음  | 최대 255자                                 | 파라미터 값                                                |
| tag_list             | Array  | 선택    | 없음  | 최대 10개                                  | 태그 정보                                               |
| tag_list[0].tagKey   | String | 선택    | 없음  | 최대 64자                                  | 태그 키                                                |
| tag_list[0].tagValue | String | 선택    | 없음  | 최대 255자                                 | 태그 값                                                |

```python
# TensorFlow 모델
model = easymaker.Model().create_by_model_upload_uri(
    model_type_code=easymaker.TENSORFLOW,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    description='model_description',
)
# HuggingFace 모델
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

### 모델 삭제

[파라미터]

| 이름                        | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명    |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | 필수    | 없음   | 최대 36자 | 모델 ID |

```python
easymaker.Model(model_id).delete()
```

## 모델 평가

### 모델 평가 생성

모델의 성능 지표를 측정하는 모델 평가를 생성합니다. 선택한 모델로 배치 추론이 실행되며 평가 지표가 저장됩니다.

[파라미터]

| 이름                                        | 타입      | 필수 여부 | 기본값   | 유효 범위                                          | 설명                                                              |
|-------------------------------------------|---------|-------|-------|------------------------------------------------|-----------------------------------------------------------------|
| model_evaluation_name                     | String  | 필수    | 없음    | 최대 50자                                         | 모델 평가 이름                                                        |
| description                               | String  | 선택    | 없음    | 최대 255자                                        | 모델 평가에 대한 설명                                                    |
| model_id                                  | String  | 필수    | 없음    | 최대 36자                                         | 평가할 모델 ID                                                       |
| objective_code                            | String  | 필수    | 없음    | easymaker.CLASSIFICATION, easymaker.REGRESSION | 평가 목표                                                           |
| class_names                               | String  | 선택    | 없음    | 1~5000                                         | 분류 모델에서 결과로 가능한 class 목록(`,`로 구분된 문자열이나 숫자)                     |
| instance_type_name                             | String  | 필수    | 없음    | 없음                                             | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| input_data_uri                            | String  | 필수    | 없음    | 최대 255자                                        | 입력 데이터 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)         |
| input_data_type_code                      | String  | 필수    | 없음    | easymaker.CSV, easymaker.JSONL                 | 입력 데이터 타입                                                       |
| target_field_name                         | String  | 필수    | 없음    | 최대 255자                                        | 정답(Ground truth) 레이블의 필드 이름                                     |
| timeout_hours                             | Integer | 선택    | 720    | 1~720                                          | 최대 모델 평가 시간(단위: 시간)                                             |
| batch_inference_instance_type_name             | String  | 필수    | 없음    | 없음                                             | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| batch_inference_instance_count            | Integer | 필수    | 없음    | 1~10                                           | 배치 추론에 사용할 인스턴스 수                                               |
| batch_inference_pod_count                 | Integer | 필수    | 없음    | 1~100                                          | 분산 추론을 적용할 파드 수                                                 |
| batch_inference_output_upload_uri         | String  | 필수    | 없음    | 최대 255자                                        | 배치 추론 결과 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| batch_inference_max_batch_size            | Integer | 필수    | 없음    | 1~1000                                         | 동시에 처리되는 데이터 샘플의 수                                              |
| batch_inference_inference_timeout_seconds | Integer | 필수    | 없음    | 1~1200                                         | 단일 추론 요청의 최대 허용 시간                                              |
| use_log                                   | Boolean | 선택    | False | True, False                                    | Log & Crash Search 서비스에 로그를 남길지 여부                              |
| wait                                      | Boolean | 선택    | True  | True, False                                    | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환                       |

```python
model_evaluation  = easymaker.ModelEvaluation().create(
    model_evaluation_name="model_evaluation_name",
    description="description",
    model_id=model.model_id,
    objective_code=easymaker.REGRESSION,
    class_names="class_a, class_b",
    generate_feature_attributions=False,
    instance_type_name="m2.c4m8",
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type_code=easymaker.CSV,
    target_field_name="target_field_name",
    boot_storage_size=50,
    data_storage_size=300,
    timeout_hours=1,
    batch_inference_instance_type_name="m2.c4m8",
    batch_inference_instance_count=1,
    batch_inference_pod_count=1,
    batch_inference_output_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{output_upload_uri}',
    batch_inference_max_batch_size=32,
    batch_inference_inference_timeout_seconds=120,
    use_log=True,
    wait=True,
)
# 회귀 모델 평가 생성
regression_model_evaluation  = easymaker.ModelEvaluation().create(
    model_evaluation_name="regression_model_evaluation",
    description="regression model evaluation sample",
    model_id=regression_model.model_id,
    objective_code=easymaker.REGRESSION,
    generate_feature_attributions=False,
    instance_type_name="m2.c4m8",
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type_code=easymaker.CSV,
    target_field_name="target_field_name",
    boot_storage_size=50,
    data_storage_size=300,
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
# 분류 모델 평가 생성
classification_model_evaluation  = easymaker.ModelEvaluation().create(
    model_evaluation_name="classification_model_evaluation",
    description="classification model evaluation sample",
    model_id=classification_model.model_id,
    objective_code=easymaker.CLASSIFICATION,
    class_names="classA,classB,classC",
    generate_feature_attributions=False,
    instance_type_name="m2.c4m8",
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type_code=easymaker.CSV,
    target_field_name="target_field_name",
    boot_storage_size=50,
    data_storage_size=300,
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

### 모델 평가 삭제

[파라미터]

| 이름                        | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명       |
|---------------------------|---------|-------|------|--------|----------|
| model_evaluation_id | String  | 필수    | 없음   | 최대 36자 | 모델 평가 ID |

```python
easymaker.ModelEvaluation(model_evaluation_id).delete()
```

## 엔드포인트

### 엔드포인트 생성

엔드포인트 생성 시 기본 스테이지가 생성됩니다.

[파라미터]

| 이름                                                          | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                     |
|-------------------------------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                                               | String  | 필수    | 없음    | 최대 50자                     | 엔드포인트 이름                                                               |
| description                                        | String  | 선택    | 없음    | 최대 255자                    | 엔드포인트에 대한 설명                                                           |
| instance_type_name                                      | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                                  |
| instance_count                                     | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                      |
| endpoint_model_resource_list                                | Array   | 필수    | 없음    | 최대 10개                     | 스테이지에 사용될 리소스 정보                                                 |
| endpoint_model_resource_list[0].modelId                     | String   | 필수    | 없음    | 없음                       | 스테이지 리소스로 생성할 모델 ID                                   |
| endpoint_model_resource_list[0].resourceOptionDetail        | Object   | 필수    | 없음    |                                  | 스테이지 리소스의 상세 정보                 |
| endpoint_model_resource_list[0].resourceOptionDetail.cpu    | Double   | 필수    | 없음    | 0.0~                             | 스테이지 리소스에 사용될 CPU                |
| endpoint_model_resource_list[0].resourceOptionDetail.memory | Object   | 필수    | 없음    | 1Mi~                             | 스테이지 리소스에 사용될 메모리             |
| endpoint_model_resource_list[0].podAutoScaleEnable          | Boolean  | 선택    | False   | True, False                      | 스테이지 리소스에 사용될 파드 오토 스케일러 |
| endpoint_model_resource_list[0].scaleMetricCode             | String   | 선택    | 없음    | CPU_UTILIZATION, MEMORY_UTILIZATION | 스테이지 리소스에 사용될 증설 단위          |
| endpoint_model_resource_list[0].scaleMetricTarget           | Integer  | 선택    | 없음    | 1~                               | 스테이지 리소스에 사용될 증설 임계치 값     |
| endpoint_model_resource_list[0].description                 | String   | 선택    | 없음    | 최대 255자                  | 스테이지 리소스에 대한 설명                                       |
| tag_list                                                    | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                                  |
| tag_list[0].tagKey                                          | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                                   |
| tag_list[0].tagValue                                        | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                                   |
| use_log                                                     | Boolean | 선택    | False | True, False                | Log & Crash Search 서비스에 로그를 남길지 여부                                             |
| wait                                                        | Boolean | 선택    | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

```python
endpoint = easymaker.Endpoint().create(
    endpoint_name='endpoint_name',
    description='endpoint_description',
    instance_type_name='c2.c16m16',
    instance_count=1,
    endpoint_model_resource_list=[
        {
            'modelId': model.model_id,
            'resourceOptionDetail': {
                'cpu': '15',
                'memory': '15Gi'
            },
            'description': 'stage_resource_description'
        }
    ],
    use_log=True,
    # wait=False,
)
```

### 스테이지 추가

기존 엔드포인트에 신규 스테이지를 추가할 수 있습니다.

[파라미터]

| 이름                                                          | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                 |
|-------------------------------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| endpoint_id                                                 | String  | 필수    | 없음   | 최대 36자                      | 엔드포인트 ID                                                            |
| stage_name                                                  | String  | 필수    | 없음    | 최대 50자                     | 스테이지 이름                                                            |
| description                                           | String  | 선택    | 없음    | 최대 255자                    | 스테이지에 대한 설명                                                        |
| instance_type_name                                      | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                              |
| instance_count                                     | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                  |
| endpoint_model_resource_list                                | Array   | 필수    | 없음    | 최대 10개                     | 스테이지에 사용될 리소스 정보                                                 |
| endpoint_model_resource_list[0].modelId                     | String   | 필수    | 없음    | 없음                       | 스테이지 리소스로 생성할 모델 ID                                   |
| endpoint_model_resource_list[0].resourceOptionDetail        | Object   | 필수    | 없음    |                                  | 스테이지 리소스의 상세 정보                 |
| endpoint_model_resource_list[0].resourceOptionDetail.cpu    | Double   | 필수    | 없음    | 0.0~                             | 스테이지 리소스에 사용될 CPU                |
| endpoint_model_resource_list[0].resourceOptionDetail.memory | Object   | 필수    | 없음    | 1Mi~                             | 스테이지 리소스에 사용될 메모리             |
| endpoint_model_resource_list[0].podAutoScaleEnable          | Boolean  | 선택    | False   | True, False                      | 스테이지 리소스에 사용될 파드 오토 스케일러 |
| endpoint_model_resource_list[0].scaleMetricCode             | String   | 선택    | 없음    | CPU_UTILIZATION, MEMORY_UTILIZATION | 스테이지 리소스에 사용될 증설 단위          |
| endpoint_model_resource_list[0].scaleMetricTarget           | Integer  | 선택    | 없음    | 1~                               | 스테이지 리소스에 사용될 증설 임계치 값     |
| endpoint_model_resource_list[0].description                 | String   | 선택    | 없음    | 최대 255자                  | 스테이지 리소스에 대한 설명                                       |
| tag_list                                                    | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                              |
| tag_list[0].tagKey                                          | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                               |
| tag_list[0].tagValue                                        | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                               |
| use_log                                                     | Boolean | 선택    | False | True, False                | Log & Crash Search 서비스에 로그를 남길지 여부                                         |
| wait                                                        | Boolean | 선택    | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

```python
endpoint_stage = easymaker.EndpointStage().create(
    endpoint_id=endpoint.endpoint_id,
    stage_name='stage01',  # 30자 이내 소문자/숫자
    description='test endpoint',
    instance_type_name='c2.c16m16',
    instance_count=1,
    endpoint_model_resource_list=[
        {
            'modelId': model.model_id,
            'resourceOptionDetail': {
                'cpu': '15',
                'memory': '15Gi'
            },
            'description': 'stage_resource_description'
        }
    ],
    use_log=True,
    # wait=False,
)
```

### 스테이지 목록 조회

엔드포인트 스테이지 목록을 조회합니다.

```python
endpoint_stage_list = easymaker.Endpoint(endpoint_id).get_stage_list()
```

### 엔드포인트 인퍼런스

기본 스테이지에 인퍼런스

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.Endpoint('endpoint_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

특정 스테이지 지정하여 인퍼런스

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.EndpointStage('endpoint_stage_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

### 엔드포인트 삭제

[파라미터]

| 이름            | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명       |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | 필수    | 없음   | 최대 36자 | 엔드포인트 ID |

```python
easymaker.Endpoint(endpoint_id).delete()
```

### 엔드포인트 스테이지 삭제

[파라미터]

| 이름         | 타입      | 필수 여부 | 기본값  | 유효 범위  | 설명      |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | 필수    | 없음   | 최대 36자 | 스테이지 ID |

```python
easymaker.EndpointStage(stage_id).delete()
```

## 배치 추론

### 배치 추론 생성

[파라미터]

| 이름                      | 타입    | 필수 여부 | 기본값 | 유효 범위   | 설명                                                              |
| ------------------------- | ------- | --------- | ------ | ----------- |-----------------------------------------------------------------|
| batch_inference_name      | String  | 필수      | 없음   | 최대 50자   | 배치 추론 이름                                                        |
| instance_count            | Integer | 필수      | 없음   | 1~10        | 배치 추론에 사용할 인스턴스 수                                               |
| timeout_hours             | Integer | 선택      | 720    | 1~720       | 최대 배치 추론 시간(단위: 시간)                                             |
| instance_type_name             | String  | 필수      | 없음   | 없음        | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| model_name                | String  | 필수      | 없음   | 없음        | 모델 이름(CLI로 조회 가능)                                               |
| pod_count                 | Integer | 필수      | 없음   | 1~100       | 분산 추론을 적용할 파드 수                                                 |
| batch_size                | Integer | 필수      | 없음   | 1~1000      | 동시에 처리되는 데이터 샘플의 수                                              |
| inference_timeout_seconds | Integer | 필수      | 없음   | 1~1200      | 단일 추론 요청의 최대 허용 시간                                              |
| input_data_uri            | String  | 필수      | 없음   | 최대 255자  | 입력 데이터 파일 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS)         |
| input_data_type           | String  | 필수      | 없음   | JSON, JSONL | 입력 데이터의 유형                                                      |
| include_glob_pattern      | String  | 선택      | 없음   | 최대 255자  | 파일 집합을 입력 데이터에서 포함할 Glob 패턴                                     |
| exclude_glob_pattern      | String  | 선택      | 없음   | 최대 255자  | 파일 집합을 입력 데이터에서 제외할 Glob 패턴                                     |
| output_upload_uri         | String  | 필수      | 없음   | 최대 255자  | 배치 추론 결과 파일이 업로드될 경로(NHN Cloud Object Storage 또는 NHN Cloud NAS) |
| data_storage_size         | Integer | 필수      | 없음   | 300~10000   | 배치 추론에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB)                          |
| description               | String  | 선택      | 없음   | 최대 255자  | 배치 추론에 대한 설명                                                    |
| tag_list                  | Array   | 선택      | 없음   | 최대 10개   | 태그 정보                                                           |
| tag_list[0].tagKey        | String  | 선택      | 없음   | 최대 64자   | 태그 키                                                            |
| tag_list[0].tagValue      | String  | 선택      | 없음   | 최대 255자  | 태그 값                                                            |
| use_log                   | Boolean | 선택      | False  | True, False | Log & Crash Search 서비스에 로그를 남길지 여부                              |
| wait                      | Boolean | 선택      | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환                       |

```python
batch_inference = easymaker.BatchInference().run(
    batch_inference_name='batch_inference_name',
    instance_count=1,
    timeout_hours=100,
    instance_type_name='m2.c4m8',
    model_name='model_name',
    pod_count=1,
    batch_size=32,
    inference_timeout_seconds=120,
    input_data_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{input_data_path}',
    input_data_type='JSONL',
    include_glob_pattern=None,
    exclude_glob_pattern=None,
    output_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{output_upload_path}',
    data_storage_size=300,  # minimum size : 300GB
    description='description',
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

### 배치 추론 삭제

[파라미터]

| 이름               | 타입   | 필수 여부 | 기본값 | 유효 범위 | 설명         |
| ------------------ | ------ | --------- | ------ | --------- | ------------ |
| batch_inference_id | String | 필수      | 없음   | 최대 36자 | 배치 추론 ID |

```python
easymaker.BatchInference(batch_inference_id).delete()
```

## 파이프라인

### 파이프라인 생성

[파라미터]

| 이름                          | 타입      | 필수 여부 | 기본값 | 유효 범위   | 설명                                        |
|-----------------------------|---------| --------- | ------ | --------- |-------------------------------------------|
| pipeline_name               | String  | 필수      | 없음   | 최대 50자   | 파이프라인 이름                                  |
| pipeline_spec_manifest_path | String  | 필수      | 없음   | 1~10      | 업로드할 파이프라인 파일 경로                          |
| description                 | String  | 선택      | 없음   | 최대 255자  | 파이프라인에 대한 설명                              |
| tag_list                    | Array   | 선택      | 없음   | 최대 10개   | 태그 정보                                     |
| tag_list[0].tagKey          | String  | 선택      | 없음   | 최대 64자   | 태그 키                                      |
| tag_list[0].tagValue        | String  | 선택      | 없음   | 최대 255자  | 태그 값                                      |
| wait                        | Boolean | 선택      | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

```python
pipeline = easymaker.Pipeline().upload(
    pipeline_name='pipeline_01',
    pipeline_spec_manifest_path='./sample-pipeline.yaml',
    description='test',
    tag_list=[],
    # wait=False,
)
```

### 파이프라인 삭제

[파라미터]

| 이름               | 타입   | 필수 여부 | 기본값 | 유효 범위 | 설명       |
| ------------------ | ------ | --------- | ------ | --------- |----------|
| pipeline_id | String | 필수      | 없음   | 최대 36자 | 파이프라인 ID |

```python
easymaker.Pipeline(pipeline_id).delete()
```

### 파이프라인 실행 생성

[파라미터]

| 이름                               | 타입      | 필수 여부                     | 기본값 | 유효 범위       | 설명                                       |
|----------------------------------|---------|---------------------------| ------ |-------------|------------------------------------------|
| pipeline_run_name                | String  | 필수                        | 없음   | 최대 50자      | 파이프라인 실행 이름                              |
| pipeline_id                      | String  | 필수                        | 없음   | 최대 36자      | 파이프라인 일정 이름                              |
| experiment_id                    | String  | easymaker.init에서 미입력 시 필수 | 없음    | 최대 36자      | 실험 ID                                    |
| description                      | String  | 선택                        | 없음   | 최대 255자     | 파이프라인 실행에 대한 설명                          |
| instance_type_name                    | String  | 필수                        | 없음   | 없음          | 인스턴스 타입 이름(CLI로 조회 가능)                   |
| instance_count                   | Integer | 필수                        | 없음   | 1~10        | 사용할 인스턴스 수                               |
| boot_storage_size                | Integer | 필수                        | 없음   | 50~         | 파이프라인을 실행할 인스턴스의 부트 스토리지 크기(단위: GB)      |
| parameter_list                   | Array   | 선택                        | 없음   | 없음          | 파이프라인에 전달할 파라미터 정보                       |
| parameter_list[0].parameterKey   | String  | 선택                        | 없음   | 최대 255자     | 파라미터 키                                   |
| parameter_list[0].parameterValue | String  | 선택                        | 없음   | 최대 1000자    | 파라미터 값                                   |
| nas_list                         | Array   | 선택                        | 없음   | 최대 10개      | NAS 정보                                   |
| nas_list[0].mountDirName         | String  | 선택                        | 없음   | 최대 64자      | 인스턴스에 마운트할 디렉터리 이름                       |
| nas_list[0].nasUri               | String  | 선택                        | 없음   | 최대 255자     | `nas://{NAS ID}:/{path}` 형식의 NAS 경로      |
| tag_list                         | Array   | 선택                        | 없음   | 최대 10개      | 태그 정보                                    |
| tag_list[0].tagKey               | String  | 선택                        | 없음   | 최대 64자      | 태그 키                                     |
| tag_list[0].tagValue             | String  | 선택                        | 없음   | 최대 255자     | 태그 값                                     |
| wait                             | Boolean | 선택                        | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환 |

```python
pipeline_run = easymaker.PipelineRun().create(
    pipeline_run_name='pipeline_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_type_name='m2.c4m8',
    instance_count=1,
    boot_storage_size=50,
    # wait=False,
)
```

### 파이프라인 실행 삭제

[파라미터]

| 이름               | 타입   | 필수 여부 | 기본값 | 유효 범위 | 설명          |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_run_id | String | 필수      | 없음   | 최대 36자 | 파이프라인 실행 ID |

```python
easymaker.PipelineRun(pipeline_run_id).delete()
```

### 파이프라인 일정 생성

[파라미터]

| 이름                               | 타입      | 필수 여부                              | 기본값 | 유효 범위       | 설명                                             |
|----------------------------------|---------|------------------------------------| ------ |-------------|------------------------------------------------|
| pipeline_recurring_run_name      | String  | 필수                                 | 없음   | 최대 50자      | 파이프라인 일정 이름                                    |
| pipeline_id                      | String  | 필수                                 | 없음   | 최대 36자      | 파이프라인 일정 이름                                    |
| experiment_id                    | String  | easymaker.init에서 미입력 시 필수          | 없음    | 최대 36자      | 실험 ID                                          |
| description                      | String  | 선택                                 | 없음   | 최대 255자     | 파이프라인 일정에 대한 설명                                |
| instance_type_name                    | String  | 필수                                 | 없음   | 없음          | 인스턴스 타입 이름(CLI로 조회 가능)                         |
| instance_count                   | Integer | 필수                                 | 없음   | 1~10        | 사용할 인스턴스 수                                     |
| boot_storage_size                | Integer | 필수                                 | 없음   | 50~         | 파이프라인을 실행할 인스턴스의 부트 스토리지 크기(단위: GB)            |
| schedule_periodic_minutes        | String  | schedule_cron_expression 미입력시 필수  | 없음   | 없음          | 파이프라인을 반복 실행할 시간 주기 설정                         |
| schedule_cron_expression         | String  | schedule_periodic_minutes 미입력시 필수 | 없음   | 없음          | 파이프라인을 반복 실행할 Cron 표현식 설정                      |
| max_concurrency_count            | String  | 선택                                 | 없음   | 없음          | 동시 실행 최대 개수를 지정하여 병렬로 실행되는 개수를 제한             |
| schedule_start_datetime          | String  | 선택                                 | 없음   | 없음          | 파이프라인 일정의 시작 시간을 설정, 미입력 시 설정한 주기에 맞춰 파이프라인 실행 |
| schedule_end_datetime            | String  | 선택                                 | 없음   | 없음          | 파이프라인 일정의 종료 시간을 설정, 미입력 시 중지 전까지 파이프라인 실행을 생성 |
| use_catchup                      | Boolean | 선택                                 | 없음   | 없음          | 누락 실행 캐치업: 파이프라인 실행이 일정에 뒤처질 경우 따라잡을지 여부를 선택 |
| parameter_list                   | Array   | 선택                                 | 없음   | 없음          | 파이프라인에 전달할 파라미터 정보                             |
| parameter_list[0].parameterKey   | String  | 선택                                 | 없음   | 최대 255자     | 파라미터 키                                         |
| parameter_list[0].parameterValue | String  | 선택                                 | 없음   | 최대 1000자    | 파라미터 값                                         |
| nas_list                         | Array   | 선택                                 | 없음   | 최대 10개      | NAS 정보                                         |
| nas_list[0].mountDirName         | String  | 선택                                 | 없음   | 최대 64자      | 인스턴스에 마운트할 디렉터리 이름                             |
| nas_list[0].nasUri               | String  | 선택                                 | 없음   | 최대 255자     | `nas://{NAS ID}:/{path}` 형식의 NAS 경로            |
| tag_list                         | Array   | 선택                                 | 없음   | 최대 10개      | 태그 정보                                          |
| tag_list[0].tagKey               | String  | 선택                                 | 없음   | 최대 64자      | 태그 키                                           |
| tag_list[0].tagValue             | String  | 선택                                 | 없음   | 최대 255자     | 태그 값                                           |
| wait                             | Boolean | 선택                                 | True   | True, False | True: 생성이 완료된 이후 반환, False: 생성 요청 후 즉시 반환      |

```python
pipeline_recurring_run = easymaker.PipelineRecurringRun().create(
    pipeline_recurring_run_name='pipeline_recurring_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_type_name='m2.c4m8',
    boot_storage_size=50,
    schedule_cron_expression='0 0 * * * ?',
    max_concurrency_count=1,
    schedule_start_datetime='2025-01-01T00:00:00+09:00'
    # wait=False,
)
```

### 파이프라인 일정 중지/재시작

[파라미터]

| 이름               | 타입   | 필수 여부 | 기본값 | 유효 범위 | 설명          |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | 필수      | 없음   | 최대 36자 | 파이프라인 일정 ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).stop()
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).start()

```

### 파이프라인 일정 삭제

[파라미터]

| 이름               | 타입   | 필수 여부 | 기본값 | 유효 범위 | 설명          |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | 필수      | 없음   | 최대 36자 | 파이프라인 일정 ID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).delete()
```

## 기타 기능

### NHN Cloud - Log & Crash Search 로그 전송

```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default: INFO
                      project_version='2.0.0',  # default: 1.0.0
                      parameters={'serviceType': 'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storage 파일 전송

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
