## Machine Learning > AI EasyMaker > 알고리즘 가이드
NHN Cloud AI EasyMaker에서 제공하는 알고리즘을 소개합니다.
기본 알고리즘을 활용하면 학습 코드를 생성하지 않고 준비한 데이터세트로 머신러닝 모델을 만들 수 있습니다. 

## Image Classification 
ResNet-50 모델로 이미지의 종류를 분류하는 알고리즘입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value |  범위  | 설명 |
| --- | --- | --- | --- | --- | --- |
| input\_size | False | int | 28 | [1 \~ ∞)  | 출력 이미지의 해상도  |
| learning\_rate | False | float | 0.1 | [0.0 \~ ∞) | AdamW 옵티마이저의 초기 learning rate 값 |
| per\_device\_train\_batch\_size | False | int | 16 | [2 \~ ∞) | GPU/TPU core/CPU 당 training 배치 크기  |
| per\_device\_eval\_batch\_size | False | int | 16 | [1 \~ ∞) |GPU/TPU core/CPU 당  evaluation 배치 크기 |
| num\_train\_epochs | False | int | 3 | [1 \~ ∞) | 전체 training을 수행하는 총 횟수  |
| save_steps  | False | int | 500 | [1 \~ ∞) | 체크포인트를 저장 step 주기 |


### 데이터 세트
train, validation 데이터 세트가 필요하며, 각 데이터 세트는 정의된 디렉토리 구조로 준비해야합니다.

#### train 
training용 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉토리 구조로 준비해야합니다.  
```
folder/train/{lable}/image_file.png
```
이미지 종류마다 {label} 디렉토리를 생성하고, 하위에 이미지 파일을 추가합니다.


[예시] Cat/Dog traing 데이터 세트 > 디렉토리 구조 
```
folder/train/dog/golden_retriever.png
folder/train/dog/chihuahua.png
folder/train/cat/main_coon.png
folder/train/cat/bengal.png
...
```

#### validation 
evaludation용 데이터 세트입니다. 디렉토리는 다음과 같은 구조로 준비해야합니다.  

```
folder/validation/{lable}/image_file.png
```
이미지 종류마다 {label} 디렉토리를 생성하고, 하위에 이미지 파일을 추가합니다.

[예시] Cat/Dog validation 데이터 세트 > 디렉토리 구조 

```
folder/validation/dog/german_shepherd.png
folder/validation/cat/birman.png
...
```

### 지표 

Image Claasification 알고리즘은 다음의 지표를 생성합니다.
학습 중 생성된 지표는 학습 > 텐서보드 바로가기를 통해 확인할 수 있습니다.  


| 지표 이름 | 설명 |
| --- | --- |
| Accuracy | 모델이 올바르게 예측한 데이터 수 / 실제 데이터 수 |
| Precision | 각 클래스 별 (모델이 올바르게 예측한 데이터 수 / 실제 해당 클래스의 데이터 수)의 평균 |
| Recall | 각 클래스 별 (모델이 올바르게 예측한 데이터 수 / 모델이 해당 클래스로 예측한 데이터 수)의 평균 |
| F1-Score | Precision과 Recall의 조화 평균 |


### 엔드포인트 



#### 응답 
이미지 종류(label)별 score 값이 응답됩니다.


[예시] Cat / Dog 
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
이미지내의 모든 픽셀영역의 레이블을 예측하는 모델(SegFormer-B3) 입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value | 유효 범위 | 설명 | 
| --- | --- | --- | --- | --- | --- |
| learning\_rate | False | float | 2e-4 | [0.0 \~ ∞) | AdamW 옵티마이저의 초기 learning rate 값 |
| per\_device\_train\_batch\_size | False | int | 4 | [0 \~ ∞) |GPU/TPU core/CPU 당 training 배치 크기  |
| num\_train\_epochs | False | float | 3.0 | [0.0 \~ ∞) | 전체 training을 수행하는 총 횟수  |
| save_steps  | False | int | 500 | [1 \~ ∞) | 체크포인트를 저장 step 주기 |


### 데이터 세트
train, test, validation, resources 데이터 세트가 필요하며, 각 데이터 세트는 정의된 디렉토리 구조로 준비해야합니다.

#### train 

training용 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉토리 구조로 준비해야합니다.  

* train 데이터 세트의 디렉토리 구조 
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

#### validation
validation 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉토리 구조로 준비해야합니다.  

* validation 데이터 세트의 디렉토리 구조 
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


#### test 
test 데이터 세트입니다. 데이터 세트는 다음과 같은 정의된 디렉토리 구조로 준비해야합니다.  

* test 데이터 세트의 디렉토리 구조 

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

#### resources
모델을 설정할 때 필요한 레이블 클래스에 레이블 ID를 매핑하기 위한 Key-Value 형식의 Dictionary를 작성합니다.

* resources 데이터 세트의 디렉토리 구조 
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

### 지표 

Semantic Segmentation 알고리즘은 다음의 지표를 생성합니다.  
학습 중 생성된 지표는 학습 > 텐서보드 바로가기를 통해 확인할 수 있습니다.  

| 지표 이름 | 설명 |
|--|--|
| mean_iou | 모델이 예측한 영역과 정답 영역의 겹치는 비율의 클래스 평균 |
| mean_accuracy| 모델이 예측한 값과 정답이 같은 비율의 클래스 평균 |
| overall_accuracy | 모델이 예측한 값과 정답이 같은 비율의 모든 이미지 평균 |
| per_category_accuracy | 클래스 별 모델이 예측한 값과 정답이 같은 비율 |
| per_category_iou | 클래스 별 모델이 예측한 영역과 정답 영역의 겹치는 비율 |



## Object Detection 

이미지 내 존재하는 모든 객체의 위치(bbox) 및 종류(class)를 예측하는 모델(detr-resnet-50)입니다.

### 하이퍼파라미터 

| 하이퍼파라미터 이름 | 필수 여부 | Value Type | Default Value | 유효 범위 | 설명 | 
| --- | --- | --- | --- | --- | --- |
| learning\_rate | False | float | 2e-4 | [0.0 \~ ∞) | AdamW 옵티마이저의 초기 learning rate 값 |
| per\_device\_train\_batch\_size | False | int | 4 | [0 \~ ∞) | GPU/TPU core/CPU 당 training 배치 크기  |
| per\_device\_eval\_batch\_size | False | int | 4 | [0 \~ ∞) | GPU/TPU core/CPU 당  evaluation 배치 크기 |
| num\_train\_epochs | False | float | 3.0 | [0.0 \~ ∞) | 전체 training을 수행하는 총 횟수 |
| threshold | False | float | 0.5 | [0.0 \~ 1.0] |  | 
| save_steps  | False | int | 500 | [1 \~ ∞) | 체크포인트를 저장 step 주기 |


### 데이터 세트
train, test, validation 데이터 세트가 필요하며, 각 데이터 세트는 정의된 디렉토리 구조로 준비해야합니다.



#### train 

```
folder/train/_annotations.coco.json

folder/train/0001.png
folder/train/0002.png
folder/train/0003.png
...
```

* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성을 합니다. 
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format와 Object Detection 내용을 참고합니다.

[예시] Bolloon Object Detection 예시
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


#### test 

```
folder/test/_annotations.coco.json

folder/test/0001.png
folder/test/0002.png
folder/test/0003.png
...
```

* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성을 합니다. 
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format와 Object Detection 내용을 참고합니다.


#### validation


```
folder/validation/_annotations.coco.json

folder/validation/0001.png
folder/validation/0002.png
folder/validation/0003.png
...
```
* _annotations.coco.json 파일  
COCO Dataset의 형식으로 작성을 합니다. 
자세한 형식은 [COCO Dataset의 format-data](https://cocodataset.org/#format-data) 문서의 Data format와 Object Detection 내용을 참고합니다.


### 실시간 추론  

#### 요청 

* Request URI: POST https://kr1-{apigwSeviceId}.api.nhncloudservice.com/inference
* Request Body: 자세한 내용은 [부록: 이미지 바이트 배열 변환 파이썬 코드]()를 참고해주세요.

``` json
{
    "instances": [
        {
        "data": "image_to_bytes_array"
        }
    ]
}
```

#### 응답 
이미지 종류(label)별 score 값이 응답됩니다.

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




## 엔드포인트 생성과 실시간 추론 요청

1. 완료된 학습을 선택합니다.
2. [모델 생성] 버튼을 클릭 한 후, 모델 이름을 작성하고 모델 생성 버튼을 클릭하여 모델을 생성합니다.
3. (2) 에서 생성한 모델로 [엔드포인트 생성] 을 클릭합니다. 엔드포인트 설정 정보를 입력 한 후 엔드포인트를 생성합니다.
4. 생성 완료된 엔드포인트 이름을 클릭하고, 스테이지를 선택합니다.
5. 스테이지 엔드포인트 URL을 통해 실시간 추론 API를 요청할 수 있습니다.

#### 요청 

* Request URI: POST https://kr1-{apigwSeviceId}.api.nhncloudservice.com/inference
* Request Body: 자세한 내용은 [부록: 이미지 바이트 배열 변환 파이썬 코드]()를 참고해주세요.

``` json
{
    "instances": [
        {
          "data": "image_to_bytes_array"
        }
    ]
}
```


## 부록: 이미지 바이트 배열 변환 파이썬 코드

이미지를 Base64 Byte Array 변환하고 추론 API 요청 본문의 형식의 json으로 변환하는 파이썬 코드입니다.


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
