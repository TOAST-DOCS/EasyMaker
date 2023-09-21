## Machine Learning > AI EasyMaker > NHN Cloud Algorithms Guide
This document describes algorithms provided by NHN Cloud AI EasyMaker.
By using the underlying algorithms, you can create a machine learning model by preparing a data set without writing any training code.

## Image Classification 
It is an algorithm (ResNet-50) that classifies types of images.

### Hyperparameter 

| Hyperparameter Name | Required | Value Type | Default Value | Range      | Description |
| --- | --- | --- | -- |---------| --- |
| input_size | False | int | 28 | [4~∞)   | Resolution of the output image  |
| learning_rate | False | float | 0.1 | [0.0~∞) | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 16 | [2~∞)   | Training batch size per GPU/TPU core/CPU  |
| per_device_eval_batch_size | False | int | 16 | [1~∞)   |evaluation batch size per GPU/TPU core/CPU |
| num_train_epochs | False | int | 3 | [1~∞)   | The total number of times the entire training is performed  |
| save_steps  | False | int | 500 | [1~∞)   | Step cycle to store checkpoints |
| logging_steps  | False | int | 10 | [1~∞)   | Step cycle to output logs |


### Data Set
Prepare train, validation, and test data sets.

#### Train (required)
A data set for training. Data sets should be prepared in a directory structure defined as follows. 
```
folder/train/{lable}/image_file.png
```
Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification train data set 
```
folder/train/cat/bengal.png
folder/train/cat/main_coon.png
folder/train/dog/chihuahua.png
folder/train/dog/golden_retriever.png
...
```

#### Validation (required)
This is the data set for validation. Data sets should be prepared in a directory structure defined as follows. 

```
folder/validation/{lable}/image_file.png
```

Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification validation data set 
```
folder/validation/cat/abyssinian.png
folder/validation/cat/aegean.png
folder/validation/dog/billy.png
folder/validation/dog/calupoh.png
...
```


#### Test (optional)
This is the data set for testing. Data sets should be prepared in a directory structure defined as follows. 

```
folder/test/{lable}/image_file.png
```

Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification test data set 
```
folder/test/cat/arabian_mau.png
folder/test/cat/american_curl.png
folder/test/dog/boerboel.png
folder/test/dog/cretan_hound.png
...
```

### Indicators 

The Image Classification algorithm produces the following metrics.
Indicators generated during training can be checked through **Training > Go to TensorBoard**.  


| Indicator name | Description |
| --- | --- |
| Accuracy | Number of data correctly predicted by the model / Number of actual data |
| Precision | Average for each class (the number of data correctly predicted by the model / the number of data in the actual corresponding class) |
| Recall | Average for each class (number of data correctly predicted by model/number of data predicted by model for that class) |
| F1-Score | Harmonic Average of Precision and Recall |


### Inference  
To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](./algorithm-guide/#create-endpoint-and-request-inference).

#### Response Format 
The score value for each image type (label) is answered.

[Example] Inference API response body of Cat-Dog classification 

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
An algorithm (SegFormer-B3) that predicts the label of every pixel region within an image.

### Hyperparameter 

| Hyperparameter Name | Required | Value Type | Default Value | Valid range | Description | 
| --- | --- | --- |---------------| --- | --- |
| learning_rate | False | float | 2e-4          | [0.0~∞) | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 4             | [0~∞) |Training batch size per GPU/TPU core/CPU  |
| num_train_epochs | False | float | 3.0           | [0.0~∞) | The total number of times the entire training is performed  |
| save_steps  | False | int | 500           | [1~∞) | Step cycle to store checkpoints |
| logging_steps  | False | int | 10            | [1~∞)   | Step cycle to output logs |


### Data Set
Prepare train, validation, resources, and test data sets.

#### Train (required)
A data set for training. Datasets should be prepared in a defined directory structure like this: 
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
  Create a mapping file of image and segmentation map. 

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

* image: Write the image file path.
* seg_map: Write the segmentation map file path. 

#### Validation (required)
This is the data set for validation. Datasets should be prepared in a defined directory structure like this: 

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
  Create a mapping file of image and segmentation map. 

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

* image: Write the image file path.
* seg_map: Write the segmentation map file path. 


#### Resources (required)
Create a dictionary in key-value format to map label IDs to label classes required when setting up the model.

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

#### Test (optional)
This is the data set for testing. Datasets should be prepared in a defined directory structure like this: 

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
Create a mapping file of image and segmentation map. 

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

* image: Write the image file path.
* seg_map: Write the segmentation map file path. 

### Indicators 

The Semantic Segmentation algorithm generates the following metrics.  
Indicators generated during training can be checked through **Training > Go to TensorBoard**.  

| Indicator name | Description |
|--|--|
| mean_iou | The class average of the percentage of overlap between the area predicted by the model and the correct area |
| mean_accuracy| The class mean of the proportion of correct answers equal to the value predicted by the model |
| overall_accuracy | Average of all images with the same proportion of correct answers as the value predicted by the model |
| per_category_accuracy | Percentage of correct answers equal to the value predicted by the model for each class |
| per_category_iou | The overlapping ratio between the area predicted by the model for each class and the correct area |


### Inference  
To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](./#create-endpoint-and-request-inference).

#### Response Format
After resizing the requested image to 512 X 512, the label value for each pixel of each image is returned in the form of an array.

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

An algorithm (detr-resnet-50) that predicts the position (bbox) and class (class) of all objects present in an image.

### Hyperparameter 

| Hyperparameter Name | Required | Value Type | Default Value | Valid range     | Description | 
| --- | --- | --- | -- |-----------| --- |
| learning_rate | False | float | 2e-4 | [0.0~∞)   | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 4 | [1~∞)     | Training batch size per GPU/TPU core/CPU  |
| per_device_eval_batch_size | False | int | 4 | [1~∞)     | evaluation batch size per GPU/TPU core/CPU |
| num_train_epochs | False | float | 3.0 | [0.0~∞)   | The total number of times the entire training is performed |
| threshold | False | float | 0.5 | [0.0~1.0] | Inference Threshold | 
| save_steps  | False | int | 500 | [1~∞)     | Step cycle to store checkpoints |
| logging_steps  | False | int | 10 | [1~∞)   | Step cycle to output logs |


### Data Set
Prepare the train and test data sets.

#### Train (required)
A data set for training. Datasets should be prepared in a defined directory structure like this: 

```
folder/train/_annotations.coco.json

folder/train/0001.png
folder/train/0002.png
folder/train/0003.png
...
```

* \_annotations.coco.json file  
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).

[Example] Example of Balloon Object Detection
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

#### Validation (required)
This is the data set for validation. Data sets should be prepared in a directory structure defined as follows.

```
folder/validation/_annotations.coco.json

folder/validation/0001.png
folder/validation/0002.png
folder/validation/0003.png
...
```

* \_annotations.coco.json file  
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).


#### Test (required)
This is the data set for test. Datasets should be prepared in a defined directory structure like this: 

```
folder/test/_annotations.coco.json

folder/test/0001.png
folder/test/0002.png
folder/test/0003.png
...
```

* \_annotations.coco.json file  
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).


### Inference  
To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](./algorithm-guide/#create-endpoint-and-request-inference).

#### Response Format
Returns a list of bboxes (xmin, ymin, xmax, ymax) of detected objects.

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

## Create Endpoint and Request Inference

Please refer to the following guide to create an endpoint and perform inference with a model that has been trained.
1. Select a completed training.
2. After clicking the **Create Model** button, write a model name and click the Create Model button to create a model.
3. (2) Click **Create Endpoint** with the model created in . After entering the endpoint setup information, create the endpoint.
4. Click the created endpoint name and select a stage.
5. You can request the real-time inference API through the stage endpoint URL.

### Request 

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

* The image_to_bytes_array value is a value obtained by converting the image to a Base64 Byte Array. [Note] Please refer to the image byte array conversion python code.


### [Note] Image byte array conversion python code

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
