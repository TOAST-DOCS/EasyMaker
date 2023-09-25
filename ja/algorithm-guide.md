## Machine Learning > AI EasyMaker > NHN Cloud提供アルゴリズムガイド
NHN Cloud AI EasyMakerで提供するアルゴリズムを紹介します。
基本アルゴリズムを活用すれば、データセットを準備するだけで別途学習コードを作成しなくてもマシンラーニングモデルを生成できます。

## Image Classification 
画像の種類を分類するアルゴリズム(ResNet-50)です。

### ハイパーパラメータ 

| ハイパーパラメータ名 | 必須かどうか | Value Type | Default Value | 範囲     | 説明 |
| --- | --- | --- | -- |---------| --- |
| input_size | False | int | 28 | [4～∞)   | 出力画像の解像度 |
| learning_rate | False | float | 0.1 | [0.0～∞) | AdamWオプティマイザーの初期learning rate値 |
| per_device_train_batch_size | False | int | 16 | [2～∞)   | GPU/TPU core/CPUあたりtrainingバッチサイズ |
| per_device_eval_batch_size | False | int | 16 | [1～∞)   |GPU/TPU core/CPUあたりevaluationバッチサイズ |
| num_train_epochs | False | int | 3 | [1～∞)   | 全体trainingを実行する総回数 |
| save_steps  | False | int | 500 | [1～∞)   | チェックポイントを保存するstep周期 |
| logging_steps  | False | int | 10 | [1～∞)   | ログを出力するstep周期 |


### データセット
train、validation、testデータセットを準備します。

#### train(必須)
トレーニング用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 
```
folder/train/{lable}/image_file.png
```
画像種類のラベル({lable})ディレクトリを作成し、サブディレクトリに画像ファイルを保存します。

[例] Cat-Dog分類trainデータセット 
```
folder/train/cat/bengal.png
folder/train/cat/main_coon.png
folder/train/dog/chihuahua.png
folder/train/dog/golden_retriever.png
...
```

#### validation(必須)
検証用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

```
folder/validation/{lable}/image_file.png
```

画像種類のラベル({lable})ディレクトリを作成し、サブディレクトリに画像ファイルを保存します。

[例] Cat-Dog分類validationデータセット 
```
folder/validation/cat/abyssinian.png
folder/validation/cat/aegean.png
folder/validation/dog/billy.png
folder/validation/dog/calupoh.png
...
```


#### test(選択)
テスト用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

```
folder/test/{lable}/image_file.png
```

画像種類のラベル({lable})ディレクトリを作成し、サブディレクトリに画像ファイルを保存します。

[例] Cat-Dog分類testデータセット 
```
folder/test/cat/arabian_mau.png
folder/test/cat/american_curl.png
folder/test/dog/boerboel.png
folder/test/dog/cretan_hound.png
...
```

### 指標 

Image Classificationアルゴリズムは、次の指標を作成します。
学習中に作成された指標は**学習 > Tensorboardショートカット**で確認できます。


| 指標名 | 説明 |
| --- | --- |
| Accuracy | モデルが正しく予測したデータ数/実際のデータ数 |
| Precision | 各クラス別(モデルが正しく予測したデータ数/実際の該当クラスのデータ数)の平均 |
| Recall | 各クラス別(モデルが正しく予測したデータ数/モデルが該当クラスで予測したデータ数)の平均 |
| F1-Score | PrecisionとRecallの調和平均 |


### 推論 
学習されたモデルでエンドポイントを作成し、推論をリクエストするには、[エンドポイント作成と推論リクエスト](./algorithm-guide/#_15)文書を参照してください。

#### レスポンス形式 
画像種類(label)別のscore値がレスポンスされます。

[例] Cat-Dog分類の推論APIレスポンス本文 

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
画像内のすべてのピクセル領域のラベルを予測するアルゴリズム(SegFormer-B3)です。

### ハイパーパラメータ 

| ハイパーパラメータ名 | 必須かどうか | Value Type | Default Value | 有効範囲 | 説明 | 
| --- | --- | --- |---------------| --- | --- |
| learning_rate | False | float | 2e-4          | [0.0～∞) | AdamWオプティマイザーの初期learning rate値 |
| per_device_train_batch_size | False | int | 4             | [0～∞) |GPU/TPU core/CPUあたりのtrainingバッチサイズ |
| num_train_epochs | False | float | 3.0           | [0.0～∞) | 全体trainingを実行する総回数 |
| save_steps  | False | int | 500           | [1～∞) | チェックポイントを保存するstep周期 |
| logging_steps  | False | int | 10            | [1～∞)   | ログを出力するstep周期 |


### データセット
train、validation、resources、testデータセットを準備します。

#### train(必須)
トレーニング用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 
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
  imageとsegmentation mapのマッピングファイルを作成します。 

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

* image:画像ファイルのパスを作成します。
* seg_map: segmentation mapファイルパスを作成します。 

#### validation(必須)
検証用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

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
  imageとsegmentation mapのマッピングファイルを作成します。 

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

* image:画像ファイルのパスを作成します。
* seg_map: segmentation mapファイルパスを作成します。 


#### resources(必須)
モデル設定時に必要なラベルクラスにラベルIDをマッピングするためのKey-Value形式のDictionaryを作成します。

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

#### test(選択)
テスト用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

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
imageとsegmentation mapのマッピングファイルを作成します。 

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

* image:画像ファイルのパスを作成します。
* seg_map: segmentation mapファイルパスを作成します。 

### 指標 

Semantic Segmentationアルゴリズムは、次の指標を作成します。
学習中に作成された指標は**学習 > Tensorboardショートカット**で確認できます。

| 指標名 | 説明 |
|--|--|
| mean_iou | モデルが予測した領域と正解領域が重なる比率のクラス平均 |
| mean_accuracy| モデルが予測した値と正解が同じ比率のクラス平均 |
| overall_accuracy | モデルが予測した値と正解が同じ比率のすべての画像の平均 |
| per_category_accuracy | クラス別モデルが予測した値と正解が同じ比率 |
| per_category_iou | クラス別モデルが予測した領域と正解領域が重なる比率 |


### 推論 
学習されたモデルでエンドポイントを作成し、推論をリクエストするには、[エンドポイント作成と推論リクエスト](./algorithm-guide/#_15)文書を参照してください。

#### レスポンス形式
リクエスト画像を512 X 512サイズに調整後、各画像のピクセルごとにlabel値が配列形式でレスポンスされます。

```
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

画像内に存在するすべてのオブジェクトの位置(bbox)及び、種類(class)を予測するアルゴリズム(detr-resnet-50)です。

### ハイパーパラメータ 

| ハイパーパラメータ名 | 必須かどうか | Value Type | Default Value | 有効範囲    | 説明 | 
| --- | --- | --- | -- |-----------| --- |
| learning_rate | False | float | 2e-4 | [0.0～∞)   | AdamWオプティマイザーの初期learning rate値 |
| per_device_train_batch_size | False | int | 4 | [1～∞)     | GPU/TPU core/CPUあたりのtrainingバッチサイズ |
| per_device_eval_batch_size | False | int | 4 | [1～∞)     | GPU/TPU core/CPUあたりのevaluationバッチサイズ |
| num_train_epochs | False | float | 3.0 | [0.0～∞)   | 全体trainingを実行する総回数 |
| threshold | False | float | 0.5 | [0.0～1.0] | 推論Threshold | 
| save_steps  | False | int | 500 | [1～∞)     | チェックポイントを保存するstep周期 |
| logging_steps  | False | int | 10 | [1～∞)   | ログを出力するstep周期 |


### データセット
train、testデータセットを準備します。

#### train(必須)
トレーニング用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

```
folder/train/_annotations.coco.json

folder/train/0001.png
folder/train/0002.png
folder/train/0003.png
...
```

* _annotations.coco.jsonファイル 
COCO Datasetの形式で作成します。 
詳細な形式は、[COCO Datasetのformat-data](https://cocodataset.org/#format-data)文書のData formatとObject Detection内容を参照してください。

[例] Balloon Object Detection例
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

#### validation(必須)
検証用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。

```
folder/validation/_annotations.coco.json

folder/validation/0001.png
folder/validation/0002.png
folder/validation/0003.png
...
```

* _annotations.coco.jsonファイル
COCO Datasetの形式で作成します。
詳細な形式は、[COCO Datasetのformat-data](https://cocodataset.org/#format-data)文書のData formatとObject Detection内容を参照してください。


#### test(必須)
test用のデータセットです。データセットは次のように定義されたディレクトリ構造で準備する必要があります。 

```
folder/test/_annotations.coco.json

folder/test/0001.png
folder/test/0002.png
folder/test/0003.png
...
```

* _annotations.coco.jsonファイル 
COCO Datasetの形式で作成します。 
詳細な形式は、[COCO Datasetのformat-data](https://cocodataset.org/#format-data)文書のData formatとObject Detection内容を参照してください。


### 推論 
学習されたモデルでエンドポイントを作成し、推論をリクエストするには、[エンドポイント作成と推論リクエスト](./algorithm-guide/#_15)文書を参照してください。

#### レスポンス形式
detectionされたobjectのbbox(xmin、ymin、xmax、ymax)リストを返します。

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

## エンドポイント作成と推論リクエスト

学習が完了したモデルでエンドポイントを作成し、推論をするには次のガイドを参照してください。
1. 完了した学習を選択します。
2. **モデル作成**ボタンをクリックし、モデルの名前を作成し、モデル作成ボタンをクリックしてモデルを作成します。
3. (2)で作成したモデルで**エンドポイント作成**をクリックします。エンドポイント設定情報を入力後、エンドポイントを作成します。
4. 作成が完了したエンドポイントの名前をクリックし、ステージを選択します。
5. ステージエンドポイントURLを通じてリアルタイム推論APIをリクエストできます。

### リクエスト 

* Request URI: POST https://kr1-{apigwSeviceId}.api.nhncloudservice.com/inference
* Request Body

``` json
{
    "instances": [
        {
            "data": "image_to_bytes_array"
        }
    ]
}
```

* image_to_bytes_array値は画像をBase64 Byte Array変換した値です。[参照]画像のバイト配列変換Pythonコード内容を参照してください。


### [参照]画像のバイト配列変換Pythonコード

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
