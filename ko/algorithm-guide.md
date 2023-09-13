## Machine Learning > AI EasyMaker > NHN Cloud 제공 알고리즘 가이드
NHN Cloud AI EasyMaker에서 제공하는 알고리즘을 소개합니다.
기본 알고리즘을 활용하면 데이터 세트만 준비하면 별도로 학습 코드를 작성하지 않아도 머신 러닝 모델을 생성할 수 있습니다.

## Image Classification 
이미지의 종류를 분류하는 알고리즘(ResNet-50)입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value | 범위      | 설명 |
| --- | --- | --- | -- |---------| --- |
| input_size | False | int | 28 | [4~∞)   | 출력 이미지의 해상도  |
| learning_rate | False | float | 0.1 | [0.0~∞) | AdamW 옵티마이저의 초기 learning rate 값 |
| per_device_train_batch_size | False | int | 16 | [2~∞)   | GPU/TPU core/CPU당 training 배치 크기  |
| per_device_eval_batch_size | False | int | 16 | [1~∞)   |GPU/TPU core/CPU당  evaluation 배치 크기 |
| num_train_epochs | False | int | 3 | [1~∞)   | 전체 training을 수행하는 총횟수  |
| save_steps  | False | int | 500 | [1~∞)   | 체크 포인트를 저장하는 step 주기 |
| logging_steps  | False | int | 10 | [1~∞)   | 로그를 출력하는 step 주기 |


### 데이터 세트
train, validation, test 데이터 세트를 준비합니다.

#### train(필수)
훈련을 위한 데이터 세트입니다. 데이터 세트는 다음과 같이 정의된 디렉터리 구조로 준비해야 합니다. 
```
folder/train/{lable}/image_file.png
```
이미지 종류의 레이블({lable}) 디렉터리를 생성하고, 하위 디렉터리에 이미지 파일을 저장합니다.

[예시] Cat-Dog 분류 train 데이터 세트 
```
folder/train/cat/bengal.png
folder/train/cat/main_coon.png
folder/train/dog/chihuahua.png
folder/train/dog/golden_retriever.png
...
```

#### validation(필수)
검증을 위한 데이터 세트입니다. 데이터 세트는 다음과 같이 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/validation/{lable}/image_file.png
```

이미지 종류의 레이블({lable}) 디렉터리를 생성하고, 하위 디렉터리에 이미지 파일을 저장합니다.

[예시] Cat-Dog 분류 validation 데이터 세트 
```
folder/validation/cat/abyssinian.png
folder/validation/cat/aegean.png
folder/validation/dog/billy.png
folder/validation/dog/calupoh.png
...
```


#### test(선택)
테스트를 위한 데이터 세트입니다. 데이터 세트는 다음과 같이 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/test/{lable}/image_file.png
```

이미지 종류의 레이블({lable}) 디렉터리를 생성하고, 하위 디렉터리에 이미지 파일을 저장합니다.

[예시] Cat-Dog 분류 test 데이터 세트 
```
folder/test/cat/arabian_mau.png
folder/test/cat/american_curl.png
folder/test/dog/boerboel.png
folder/test/dog/cretan_hound.png
...
```

### 지표 

Image Classification 알고리즘은 다음의 지표를 생성합니다.
학습 중 생성된 지표는 **학습 > 텐서보드 바로가기**를 통해 확인할 수 있습니다.  


| 지표 이름 | 설명 |
| --- | --- |
| Accuracy | 모델이 올바르게 예측한 데이터 수/실제 데이터 수 |
| Precision | 각 클래스 별(모델이 올바르게 예측한 데이터 수/실제 해당 클래스의 데이터 수)의 평균 |
| Recall | 각 클래스 별(모델이 올바르게 예측한 데이터 수/모델이 해당 클래스로 예측한 데이터 수)의 평균 |
| F1-Score | Precision과 Recall의 조화 평균 |


### 추론  
학습된 모델로 엔드포인트를 생성하고 추론을 요청하려면 [엔드포인트 생성과 추론 요청](./algorithm-guide/#_15) 문서를 참고해 주세요.

#### 응답 형식 
이미지 종류(label)별 score 값이 응답됩니다.

[예시] Cat-Dog 분류의 추론 API 응답 본문 

``` json
[
    {
        "score": 0.9992493987083435,
        "label": "dog"
    },
    {
        "score": 0.0007505337707698345,
        "label": "cat"
    }
]
```


## Semantic Segmentation
이미지 내의 모든 픽셀 영역의 레이블을 예측하는 알고리즘(SegFormer-B3)입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value | 유효 범위 | 설명 | 
| --- | --- | --- |---------------| --- | --- |
| learning_rate | False | float | 2e-4          | [0.0~∞) | AdamW 옵티마이저의 초기 learning rate 값 |
| per_device_train_batch_size | False | int | 4             | [0~∞) |GPU/TPU core/CPU당 training 배치 크기  |
| num_train_epochs | False | float | 3.0           | [0.0~∞) | 전체 training을 수행하는 총횟수  |
| save_steps  | False | int | 500           | [1~∞) | 체크 포인트를 저장하는 step 주기 |
| logging_steps  | False | int | 10            | [1~∞)   | 로그를 출력하는 step 주기 |


### 데이터 세트
train, validation, resources, test 데이터 세트를 준비합니다.

#### train(필수)
훈련을 위한 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉터리 구조로 준비해야 합니다. 
```

folder/train/train.json

folder/train/images/0001.png
folder/train/images/0002.png
folder/train/images/0003.png
...

folder/train/annotations/0001.png
folder/train/annotations/0002.png
folder/train/annotations/0003.png
...

```

* train.json  
  image와 segmentation map의 매핑 파일을 작성합니다. 

```
[
    {
        "image": "images/0001.png",
        "seg_map": "annotations/0001.png"
    },
    {
        "image": "images/0002.png",
        "seg_map": "annotations/0002.png"
    },
    {
        "image": "images/0003.png",
        "seg_map": "annotations/0003.png"
    }
]
```

* image: 이미지 파일 경로를 작성합니다.
* seg_map: segmentation map 파일 경로를 작성합니다. 

#### validation(필수)
검증을 위한 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/validation/validation.json

folder/validation/images/0001.png
folder/validation/images/0002.png
folder/validation/images/0003.png
...

folder/validation/annotations/0001.png
folder/validation/annotations/0002.png
folder/validation/annotations/0003.png
...

```

* validation.json   
  image와 segmentation map의 매핑 파일을 작성합니다. 

```
[
    {
        "image": "images/0001.png",
        "seg_map": "annotations/0001.png"
    },
    {
        "image": "images/0002.png",
        "seg_map": "annotations/0002.png"
    },
    {
        "image": "images/0003.png",
        "seg_map": "annotations/0003.png"
    }
]
```

* image: 이미지 파일 경로를 작성합니다.
* seg_map: segmentation map 파일 경로를 작성합니다. 


#### resources(필수)
모델을 설정할 때 필요한 레이블 클래스에 레이블 ID를 매핑하기 위한 Key-Value 형식의 Dictionary를 작성합니다.

```
folder/resources/id2lable.json
```

* id2lable.json

```json
{
    "0": "unlabeled",
    "1": "flat-road",
    "2": "flat-sidewalk",
    "3": "flat-crosswalk",
    "...": "..."
}
```

#### test(선택)
테스트를 위한 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/test/train.json

folder/test/images/0001.png
folder/test/images/0002.png
folder/test/images/0003.png
...

folder/test/annotations/0001.png
folder/test/annotations/0002.png
folder/test/annotations/0003.png
...

```


* test.json 
image와 segmentation map의 매핑 파일을 작성합니다. 

```json
[
    {
        "image": "images/0001.png",
        "seg_map": "annotations/0001.png"
    },
    {
        "image": "images/0002.png",
        "seg_map": "annotations/0002.png"
    },
    {
        "image": "images/0003.png",
        "seg_map": "annotations/0003.png"
    }
]
```

* image: 이미지 파일 경로를 작성합니다.
* seg_map: segmentation map 파일 경로를 작성합니다. 

### 지표 

Semantic Segmentation 알고리즘은 다음의 지표를 생성합니다.  
학습 중 생성된 지표는 **학습 > 텐서보드 바로가기**를 통해 확인할 수 있습니다.  

| 지표 이름 | 설명 |
|--|--|
| mean_iou | 모델이 예측한 영역과 정답 영역의 겹치는 비율의 클래스 평균 |
| mean_accuracy| 모델이 예측한 값과 정답이 같은 비율의 클래스 평균 |
| overall_accuracy | 모델이 예측한 값과 정답이 같은 비율의 모든 이미지 평균 |
| per_category_accuracy | 클래스 별 모델이 예측한 값과 정답이 같은 비율 |
| per_category_iou | 클래스 별 모델이 예측한 영역과 정답 영역의 겹치는 비율 |


### 추론  
학습된 모델로 엔드포인트를 생성하고 추론을 요청하려면 [엔드포인트 생성과 추론 요청](./algorithm-guide/#_15) 문서를 참고해 주세요.

#### 응답 형식
요청 이미지를 512 X 512 크기로 조정한 후, 각 이미지의 픽셀마다 label 값이 배열 형태로 응답됩니다.

```json
{
    "predictions": [
        [
            [
                1, 1, 27, 27, ... 
            ],
            [
                27, 27, 1, 11, ... 
            ]
            ...
        ]
    ]
}
```

## Object Detection

이미지 내 존재하는 모든 객체의 위치(bbox) 및 종류(class)를 예측하는 알고리즘(detr-resnet-50)입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value | 유효 범위     | 설명 | 
| --- | --- | --- | -- |-----------| --- |
| learning_rate | False | float | 2e-4 | [0.0~∞)   | AdamW 옵티마이저의 초기 learning rate 값 |
| per_device_train_batch_size | False | int | 4 | [1~∞)     | GPU/TPU core/CPU당 training 배치 크기  |
| per_device_eval_batch_size | False | int | 4 | [1~∞)     | GPU/TPU core/CPU당  evaluation 배치 크기 |
| num_train_epochs | False | float | 3.0 | [0.0~∞)   | 전체 training을 수행하는 총횟수 |
| threshold | False | float | 0.5 | [0.0~1.0] | 추론 Threshold | 
| save_steps  | False | int | 500 | [1~∞)     | 체크 포인트를 저장하는 step 주기 |
| logging_steps  | False | int | 10 | [1~∞)   | 로그를 출력하는 step 주기 |


### 데이터 세트
train, test 데이터 세트를 준비합니다.

#### train(필수)
훈련을 위한 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/train/_annotations.coco.json

folder/train/0001.png
folder/train/0002.png
folder/train/0003.png
...
```

* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성합니다. 
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format과 Object Detection 내용을 참고합니다.

[예시] Balloon Object Detection 예시
``` json
{
    "info": {
        "year": "2022",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "",
        "url": "https://public.roboflow.ai/object-detection/undefined",
        "date_created": "2022-08-23T09:36:56+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "none",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "balloon",
            "supercategory": "balloon"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.png",
            "height": 416,
            "width": 416,
            "date_captured": "2022-08-23T09:36:56+00:00"
        },
        {
            "id": 1,
            "license": 1,
            "file_name": "0002.png",
            "height": 416,
            "width": 416,
            "date_captured": "2022-08-23T09:36:56+00:00"
        },
        {
            "id": 2,
            "license": 1,
            "file_name": "0003.png",
            "height": 416,
            "width": 416,
            "date_captured": "2022-08-23T09:36:56+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": [
                201,
                166,
                93.5,
                144.5
            ],
            "area": 13510.75,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [
                17,
                20,
                217.5,
                329
            ],
            "area": 71557.5,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "bbox": [
                26,
                248,
                162.5,
                117
            ],
            "area": 19012.5,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
```

#### validation(필수)
검증을 위한 데이터 세트입니다. 데이터 세트는 다음과 같이 정의된 디렉터리 구조로 준비해야 합니다.

```
folder/validation/_annotations.coco.json

folder/validation/0001.png
folder/validation/0002.png
folder/validation/0003.png
...
```

* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성합니다.
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format과 Object Detection 내용을 참고합니다.


#### test(필수)
test를 위한 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉터리 구조로 준비해야 합니다. 

```
folder/test/_annotations.coco.json

folder/test/0001.png
folder/test/0002.png
folder/test/0003.png
...
```

* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성합니다. 
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format과 Object Detection 내용을 참고합니다.


### 추론  
학습된 모델로 엔드포인트를 생성하고 추론을 요청하려면 [엔드포인트 생성과 추론 요청](./algorithm-guide/#_15) 문서를 참고해 주세요.

#### 응답 형식
detection된 object의 bbox(xmin, ymin, xmax, ymax) 목록을 반환합니다.

``` json
{
   "predictions": [
      [
         {
            "balloon": {
               "xmin": 293,
               "ymin": 325,
               "xmax": 361,
               "ymax": 375
            }
         },
         {
            "balloon": {
               "xmin": 322,
               "ymin": 157,
               "xmax": 404,
               "ymax": 273
            }
         }
      ]
   ]
}
```

## 엔드포인트 생성과 추론 요청

학습이 완료된 모델로 엔드포인트를 생성하고 추론을 하려면 다음의 가이드를 참고해 주세요.
1. 완료된 학습을 선택합니다.
2. **모델 생성**버튼을 클릭한 후, 모델 이름을 작성하고 모델 생성 버튼을 클릭하여 모델을 생성합니다.
3. (2) 에서 생성한 모델로 **엔드포인트 생성** 을 클릭합니다. 엔드포인트 설정 정보를 입력한 후 엔드포인트를 생성합니다.
4. 생성 완료된 엔드포인트 이름을 클릭하고, 스테이지를 선택합니다.
5. 스테이지 엔드포인트 URL을 통해 실시간 추론 API를 요청할 수 있습니다.

### 요청 

* Request URI: POST https://kr1-{apigwSeviceId}.api.nhncloudservice.com/inference
* Request Body

```json
{
    "instances": [
        {
            "data": "image_to_bytes_array"
        }
    ]
}
```

* image_to_bytes_array 값은 이미지를 Base64 Byte Array 변환한 값입니다. [참고] 이미지 바이트 배열 변환 파이썬 코드 내용을 참고해 주세요.


### [참고] 이미지 바이트 배열 변환 파이썬 코드

``` python
import base64
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="converts image to bytes array",
                    type=str)
args = parser.parse_args()

image = open(args.filename, 'rb')  # open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')
request = {
  "instances": [
    {
      "data": bytes_array
    }
  ]
}

with open('input.json', 'w') as outfile:
    json.dump(request, outfile, indent=4, sort_keys=True)
```
