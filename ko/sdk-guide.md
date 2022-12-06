## NHN Cloud > SDK 사용 가이드 > AI EasyMaker

## 개발 가이드

### AI EasyMaker Python SDK 설치

python -m pip install easymaker

* AI EasyMaker 노트북에는 기본적으로 설치되어 있습니다.


### AI EasyMaker SDK 초기화
앱 키(Appkey)와 시크릿키(SecretKey)는 콘솔 오른쪽 위의 **URL & Appkey** 메뉴에서 확인할 수 있습니다.
활성화한 AI EasyMaker 상품의 앱키, 시크릿키, 리전 정보를 입력합니다.
AI EasyMaker SDK를 사용하기 위해선 초기화 코드가 필요합니다.
```
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

```
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### 학습 생성

[Parameter]

| 이름                                         | 타입      | 필수 여부                 | 기본값   | 유효 범위       | 설명                                                              |
|--------------------------------------------|---------|-----------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | 필수                    | 없음    | 없음          | 실험 ID                                                           |
| training_name                              | String  | 필수                    | 없음    | 최대 50자      | 학습 이름                                                           |
| training_description                       | String  | 선택                    | 없음    | 최대 255자     | 학습에 대한 설명                                                       |
| train_image_name                           | String  | 필수                    | 없음    | 없음          | 학습에 사용될 이미지 이름(CLI로 조회 가능)                                      |
| train_instance_name                        | String  | 필수                    | 없음    | 없음          | 인스턴스 타입 이름(CLI로 조회 가능)                                          |
| train_instance_count                       | Integer | 필수                    | 없음    | 1~10        | 학습에 사용될 인스턴스 수                                                  |
| data_storage_size                          | Integer | Obejct Storage 사용시 필수 | 없음    | 300~10000   | 학습에 필요한 데이터를 다운로드할 저장 공간 크기(단위: GB), NAS 사용시 불필요                |
| source_dir_uri                             | String  | 필수                    | 없음    | 최대 255자     | 학습에 필요한 파일들이 들어있는 경로(NHN Cloud Object Storage or NHN Cloud NAS) |
| model_upload_uri                           | String  | 필수                    | 없음    | 최대 255자     | 학습 완료된 모델이 업로드 될 경로(NHN Cloud Object Storage or NHN Cloud NAS)  |
| check_point_upload_uri                     | String  | 선택                    | 없음    | 최대 255자     | 체크포인트 파일이 업로드 될 경로(NHN Cloud Object Storage or NHN Cloud NAS)   |
| entry_point                                | String  | 필수                    | 없음    | 최대 255자     | source_dir_uri 안에서 최초 실행될 파이썬 파일 정보                             |
| timeout_hours                              | Integer | 선택                    | 720   | 1~720       | 최대 학습 시간(단위: 시간)                                                |
| hyperparameter_list                        | Array   | 선택                    | 없음    | 최대 100개     | 하이퍼파라미터 정보(hyperparameterKey/hyperparameterValue로 구성)           |
| hyperparameter_list[0].hyperparameterKey   | String  | 선택                    | 없음    | 최대 255자     | 하이퍼파라미터 키                                                       |
| hyperparameter_list[0].hyperparameterValue | String  | 선택                    | 없음    | 최대 1000자    | 하이퍼파라미터 값                                                       |
| dataset_list                               | Array   | 선택                    | 없음    | 최대 10개      | 학습에 사용될 데이터 세트 정보(datasetName/dataUri로 구성)                      |
| dataset_list[0].datasetName                | String  | 선택                    | 없음    | 최대 36자      | 데이터 이름                                                          |
| dataset_list[0].datasetUri                 | String  | 선택                    | 없음    | 최대 255자     | 데이터 경로                                                          |
| tag_list                                   | Array   | 선택                    | 없음    | 최대 10개      | 태그 정보                                                           |
| tag_list[0].tagKey                         | String  | 선택                    | 없음    | 최대 64자      | 태그 키                                                            |
| tag_list[0].tagValue                       | String  | 선택                    | 없음    | 최대 255자     | 태그 값                                                            |
| use_log                                    | Boolean | 선택                    | False | True, False | Log & Crash 상품에 로그를 남길지 여부                                      |
| wait                                       | Boolean | 선택                    | True  | True, False | True: 학습 생성이 완료된 이후 학습 ID를 반환, False: 생성 요청 후 즉시 학습 ID를 반환      |

```
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

### 모델 생성
학습 ID값으로 모델 생성을 요청할 수 있습니다.
모델은 엔드포인트 생성시 사용됩니다.

[Parameter]

| 이름                   | 타입     | 필수 여부 | 기본값 | 유효 범위   | 설명            |
|----------------------|--------|-------|-----|---------|---------------|
| training_id          | String | 필수    | 없음  | 없음      | 모델로 생성할 학습 ID |
| model_name           | String | 필수    | 없음  | 최대 50자  | 모델 이름         |
| model_description    | String | 선택    | 없음  | 최대 255자 | 모델에 대한 설명     |
| tag_list             | Array  | 선택    | 없음  | 최대 10개  | 태그 정보         |
| tag_list[0].tagKey   | String | 선택    | 없음  | 최대 64자  | 태그 키          |
| tag_list[0].tagValue | String | 선택    | 없음  | 최대 255자 | 태그 값          |


```
model_id = easymaker.Model().create(
    training_id=training_id,
    model_name='model_name',
    model_description='model_description',
)
```

학습 ID가 없더라도, 모델이 저장된 경로 정보와 프레임워크 종류를 입력하여 모델을 생성 할 수 있습니다.

[Parameter]

| 이름                   | 타입     | 필수 여부 | 기본값 | 유효 범위                                   | 설명                                                  |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| framework_code       | Enum   | 필수    | 없음  | easymaker.TENSORFLOW, easymaker.PYTORCH | 학습에 사용된 프레임워크 정보                                    |
| model_uri            | String | 필수    | 없음  | 최대 255자                                 | 모델 파일 경로(NHN Cloud Object Storage or NHN Cloud NAS) |
| model_name           | String | 필수    | 없음  | 최대 50자                                  | 모델 이름                                               |
| model_description    | String | 선택    | 없음  | 최대 255자                                 | 모델에 대한 설명                                           |
| tag_list             | Array  | 선택    | 없음  | 최대 10개                                  | 태그 정보                                               |
| tag_list[0].tagKey   | String | 선택    | 없음  | 최대 64자                                  | 태그 키                                                |
| tag_list[0].tagValue | String | 선택    | 없음  | 최대 255자                                 | 태그 값                                                |


```
model_id = easymaker.Model().create_by_model_uri(
    framework_code=easymaker.TENSORFLOW,
    model_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### 엔드포인트 생성

엔드포인트 생성시 기본 스테이지가 생성됩니다.

[Parameter]

| 이름                                    | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                     |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| model_id                              | String  | 필수    | 없음    | 없음                         | 엔드포인트로 생성할 모델 ID                                                       |
| endpoint_name                         | String  | 필수    | 없음    | 최대 50자                     | 엔드포인트 이름                                                               |
| endpoint_description                  | String  | 선택    | 없음    | 최대 255자                    | 엔드포인트에 대한 설명                                                           |
| endpoint_instance_name                | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                                  |
| endpoint_instance_count               | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                      |
| apigw_resource_uri                    | String  | 필수    | 없음    | 최대 255자                    | /로 시작하는 API Gateway 리소스 경로                                             |
| tag_list                              | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                                  |
| tag_list[0].tagKey                    | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                                   |
| tag_list[0].tagValue                  | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                                   |
| use_log                               | Boolean | 선택    | False | True, False                | Log & Crash 상품에 로그를 남길지 여부                                             |        
| wait                                  | Boolean | 선택    | True  | True, False                | True: 엔드포인트 생성이 완료된 이후 엔드포인트 ID를 반환, False: 엔드포인트 요청 후 즉시 엔드포인트 ID를 반환 |

```
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

생성해둔 엔드포인트 사용

```
endpoint = easymaker.Endpoint()
```

### 스테이지 추가

기존 엔드포인트에 신규 스테이지를 추가할 수 있습니다.

[Parameter]

| 이름                                    | 타입      | 필수 여부 | 기본값   | 유효 범위                      | 설명                                                                 |
|---------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| model_id                              | String  | 필수    | 없음    | 없음                         | 엔드포인트로 생성할 모델 ID                                                   |
| stage_name                            | String  | 필수    | 없음    | 최대 50자                     | 스테이지 이름                                                            |
| stage_description                     | String  | 선택    | 없음    | 최대 255자                    | 스테이지에 대한 설명                                                        |
| endpoint_instance_name                | String  | 필수    | 없음    | 없음                         | 엔드포인트에 사용될 인스턴스 타입 이름                                              |
| endpoint_instance_count               | Integer | 선택    | 1     | 1~10                       | 엔드포인트에 사용될 인스턴스 수                                                  |
| tag_list                              | Array   | 선택    | 없음    | 최대 10개                     | 태그 정보                                                              |
| tag_list[0].tagKey                    | String  | 선택    | 없음    | 최대 64자                     | 태그 키                                                               |
| tag_list[0].tagValue                  | String  | 선택    | 없음    | 최대 255자                    | 태그 값                                                               |
| use_log                               | Boolean | 선택    | False | True, False                | Log & Crash 상품에 로그를 남길지 여부                                         |        
| wait                                  | Boolean | 선택    | True  | True, False                | True: 스테이지 생성이 완료된 이후 스테이지 ID를 반환, False: 스테이지 요청 후 즉시 스테이지 ID를 반환 |
```
stage_id = endpoint.create_stage(
    model_id=model_id,
    stage_name='stage01',  # 30자 이내 소문자/숫자
    stage_description='test endpoint',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1,
    use_log=True,
    # wait=False,
)
```

### 엔드포인트 인퍼런스

기본 스테이지에 인퍼런스

```
input_data = [6.8, 2.8, 4.8, 1.4]
endpoint.predict(json={'instances': [input_data]})
```

특정 스테이지 지정하여 인퍼런스

```
# 스테이지 정보 조회
endpoint_stage_info_list = endpoint.get_endpoint_stage_info_list()
for endpoint_stage_info in endpoint_stage_info_list:
    print(f'endpoint_stage_info : {endpoint_stage_info}')
    
# 스테이지를 지정하여 인퍼런스 요청
input_data = [6.0, 3.4, 4.5, 1.6]
for endpoint_stage_info in endpoint_stage_info_list:
    if endpoint_stage_info['stage_name'] == 'stage01':
        endpoint.predict(json={'instances': [input_data]},
                         endpoint_stage_info=endpoint_stage_info)
```

### NHN Cloud - Log & Crash 로그 전송 기능
```
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default: INFO
                      project_version='2.0.0',  # default: 1.0.0 
                      parameters={'serviceType': 'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storage 파일 전송 기능
Object Storage 상품으로 파일을 업로드하고 다운로드하는 기능을 제공합니다.
```
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
콘솔에 접근하지 않고도 앱키, 시크릿키, 리전 정보가 있다면, Python CLI를 통해 여러 정보를 확인할 수 있습니다.

| 기능                  | 명령어                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------|
| instance type 목록 조회 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| image 목록 조회         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| experiment 목록 조회    | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| training 목록 조회      | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| model 목록 조회         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| endpoint 목록 조회      | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |