<!-- pre-align:aligned sig=e090d93d817c -->

<a id="ai.easymaker.algorithm.guide"></a>
## Machine Learning > AI EasyMaker > NHN Cloud Algorithms Guide { #ai.easymaker.algorithm.guide }

This document describes algorithms provided by NHN Cloud AI EasyMaker.
By using the underlying algorithms, you can create a machine learning model by preparing a data set without writing any training code.

<a id="image.classification"></a>
## Image Classification { #image.classification }

It is an algorithm (ResNet-50) that classifies types of images.

<a id="image.classification.hyperparameter"></a>
### Hyperparameter { #image.classification.hyperparameter }

| Hyperparameter Name | Required | Value Type | Default Value | Range      | Description |
| --- | --- | --- | -- |---------| --- |
| input_size | False | int | 28 | [4~∞)   | Resolution of the output image  |
| learning_rate | False | float | 0.1 | [0.0~∞) | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 16 | [2~∞)   | Training batch size per GPU/TPU core/CPU  |
| per_device_eval_batch_size | False | int | 16 | [1~∞)   |evaluation batch size per GPU/TPU core/CPU |
| num_train_epochs | False | int | 3 | [1~∞)   | The total number of times the entire training is performed  |
| logging_steps  | False | int | 500 | [500~∞)   | Step cycle to output logs |

<a id="image.classification.data.set"></a>
### Data Set { #image.classification.data.set }

Prepare train, validation, and test data sets.

<a id="image.classification.data.set.train"></a>
#### Train (required)

A data set for training. Data sets should be prepared in a directory structure defined as follows.

```text
folder/train/{lable}/image_file.png
```

Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification train data set

```text
folder/train/cat/bengal.png
folder/train/cat/main_coon.png
folder/train/dog/chihuahua.png
folder/train/dog/golden_retriever.png
...
```

<a id="image.classification.data.set.validation"></a>
#### Validation (required)

This is the data set for validation. Data sets should be prepared in a directory structure defined as follows.

```text
folder/validation/{lable}/image_file.png
```

Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification validation data set

```text
folder/validation/cat/abyssinian.png
folder/validation/cat/aegean.png
folder/validation/dog/billy.png
folder/validation/dog/calupoh.png
...
```

<a id="image.classification.data.set.test"></a>
#### Test (optional)

This is the data set for testing. Data sets should be prepared in a directory structure defined as follows.

```text
folder/test/{lable}/image_file.png
```

Creates a label ({label}) directory for image types, and stores image files in subdirectories.

[Example] Cat-Dog classification test data set

```text
folder/test/cat/arabian_mau.png
folder/test/cat/american_curl.png
folder/test/dog/boerboel.png
folder/test/dog/cretan_hound.png
...
```

<a id="image.classification.metric"></a>
### Indicators { #image.classification.metric }

The Image Classification algorithm produces the following metrics.
Indicators generated during training can be checked in **Training > Go to TensorBoard**.

| Indicator name | Description |
| --- | --- |
| Accuracy | Number of data correctly predicted by the model / Number of actual data |
| Precision | Average for each class (the number of data correctly predicted by the model / the number of data in the actual corresponding class) |
| Recall | Average for each class (number of data correctly predicted by model/number of data predicted by model for that class) |
| F1-Score | Harmonic Average of Precision and Recall |

<a id="image.classification.inference"></a>
### Inference { #image.classification.inference }

To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](#endpoint.create.inference.request).

<a id="image.classification.inference.response.format"></a>
#### Response Format

The score value for each image type (label) is answered.

[Example] Inference API response body of Cat-Dog classification

```json
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

<a id="semantic.segmentation"></a>
## Semantic Segmentation { #semantic.segmentation }

An algorithm (SegFormer-B3) that predicts the label of every pixel region within an image.

<a id="semantic.segmentation.hyperparameter"></a>
### Hyperparameter { #semantic.segmentation.hyperparameter }

| Hyperparameter Name | Required | Value Type | Default Value | Valid range | Description |
| --- | --- | --- |---------------| --- | --- |
| learning_rate | False | float | 2e-4          | [0.0~∞) | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 4             | [0~∞) |Training batch size per GPU/TPU core/CPU  |
| num_train_epochs | False | float | 3.0           | [0.0~∞) | The total number of times the entire training is performed  |
| logging_steps  | False | int | 500            | [500~∞)   | Step cycle to output logs |

<a id="semantic.segmentation.data.set"></a>
### Data Set { #semantic.segmentation.data.set }

Prepare train, validation, resources, and test data sets.

<a id="semantic.segmentation.data.set.train"></a>
#### Train (required)

A data set for training. Datasets should be prepared in a defined directory structure like this:

```text

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

- train.json
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

- image: Write the image file path.
- seg_map: Write the segmentation map file path.

<a id="semantic.segmentation.data.set.validation"></a>
#### Validation (required)

This is the data set for validation. Datasets should be prepared in a defined directory structure like this:

```text
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

- validation.json
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

- image: Write the image file path.
- seg_map: Write the segmentation map file path.

<a id="semantic.segmentation.data.set.resources"></a>
#### Resources (required)

Create a dictionary in key-value format to map label IDs to label classes required when setting up the model.

```text
folder/resources/id2lable.json
```

- id2lable.json

```json
{
    "0": "unlabeled",
    "1": "flat-road",
    "2": "flat-sidewalk",
    "3": "flat-crosswalk",
    "...": "..."
}
```

<a id="semantic.segmentation.data.set.test"></a>
#### Test (optional)

This is the data set for testing. Datasets should be prepared in a defined directory structure like this:

```text
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

- test.json
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

- image: Write the image file path.
- seg_map: Write the segmentation map file path.

<a id="semantic.segmentation.metric"></a>
### Indicators { #semantic.segmentation.metric }

The Semantic Segmentation algorithm generates the following metrics.
Indicators generated during training can be checked in **Training > Go to TensorBoard**.

| Indicator name | Description |
|--|--|
| mean_iou | The class average of the percentage of overlap between the area predicted by the model and the correct area |
| mean_accuracy| The class mean of the proportion of correct answers equal to the value predicted by the model |
| overall_accuracy | Average of all images with the same proportion of correct answers as the value predicted by the model |
| per_category_accuracy | Percentage of correct answers equal to the value predicted by the model for each class |
| per_category_iou | The overlapping ratio between the area predicted by the model for each class and the correct area |

<a id="semantic.segmentation.inference"></a>
### Inference { #semantic.segmentation.inference }

To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](#endpoint.create.inference.request).

<a id="semantic.segmentation.inference.response.format"></a>
#### Response Format

After resizing the requested image to 512 x 512, the label value for each pixel of each image is returned in the form of an array.

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

<a id="object.detection"></a>
## Object Detection { #object.detection }

An algorithm (detr-resnet-50) that predicts the position (bbox) and class (class) of all objects present in an image.

<a id="object.detection.hyperparameter"></a>
### Hyperparameter { #object.detection.hyperparameter }

| Hyperparameter Name | Required | Value Type | Default Value | Valid range     | Description |
| --- | --- | --- | -- |-----------| --- |
| learning_rate | False | float | 2e-4 | [0.0~∞)   | The initial learning rate value of the AdamW optimizer |
| per_device_train_batch_size | False | int | 4 | [1~∞)     | Training batch size per GPU/TPU core/CPU  |
| per_device_eval_batch_size | False | int | 4 | [1~∞)     | evaluation batch size per GPU/TPU core/CPU |
| num_train_epochs | False | float | 3.0 | [0.0~∞)   | The total number of times the entire training is performed |
| logging_steps  | False | int | 500 | [500~∞)   | Step cycle to output logs |

<a id="object.detection.data.set"></a>
### Data Set { #object.detection.data.set }

Prepare the train and test data sets.

<a id="object.detection.data.set.train"></a>
#### Train (required)

A data set for training. Datasets should be prepared in a defined directory structure like this:

```text
folder/train/_annotations.coco.json

folder/train/0001.png
folder/train/0002.png
folder/train/0003.png
...
```

- \_annotations.coco.json file
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).

[Example] Example of Balloon Object Detection

```json
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

<a id="object.detection.data.set.validation"></a>
#### Validation (required)

This is the data set for validation. Data sets should be prepared in a directory structure defined as follows.

```text
folder/validation/_annotations.coco.json

folder/validation/0001.png
folder/validation/0002.png
folder/validation/0003.png
...
```

- \_annotations.coco.json file
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).

<a id="object.detection.data.set.test"></a>
#### Test (required)

This is the data set for test. Datasets should be prepared in a defined directory structure like this:

```text
folder/test/_annotations.coco.json

folder/test/0001.png
folder/test/0002.png
folder/test/0003.png
...
```

- \_annotations.coco.json file
It is written in the format of COCO Dataset.
For detailed format, refer to Data format and Object Detection in the [format-data document of COCO Dataset](https://cocodataset.org/#format-data).

<a id="object.detection.inference"></a>
### Inference { #object.detection.inference }

To create an endpoint with a trained model and request inference, see [Create Endpoint and Request Inference](#endpoint.create.inference.request).

<a id="object.detection.inference.response.format"></a>
#### Response Format

Returns a list of bboxes (xmin, ymin, xmax, ymax) of detected objects.

```json
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

<a id="endpoint.create.inference.request"></a>
## Create Endpoint and Request Inference { #endpoint.create.inference.request }

Please refer to the following guide to create an endpoint and perform inference with a model that has been trained.

1. Select a completed training.
2. After clicking the **Create Model** button, write a model name and click the Create Model button to create a model.
3. (2) Click **Create Endpoint** with the model created in . After entering the endpoint setup information, create the endpoint.
4. Click the created endpoint name and select a stage.
5. You can request the real-time inference API through the stage endpoint URL.

<a id="endpoint.create.inference.request.format"></a>
### Request { #endpoint.create.inference.request.format }

- Request URI: POST <https://kr1-{apigwSeviceId}.api.nhncloudservice.com/inference>
- Request Body

```json
{
    "instances": [
        {
            "data": "image_to_bytes_array"
        }
    ]
}
```

- The image_to_bytes_array value is a value obtained by converting the image to a Base64 Byte Array. [Note] Please refer to the image byte array conversion python code.

<a id="endpoint.create.inference.request.note.image"></a>
### [Note] Image byte array conversion python code { #endpoint.create.inference.request.note.image }

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
