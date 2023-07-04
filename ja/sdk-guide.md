## NHN Cloud > SDK使用ガイド > AI EasyMaker

## 開発ガイド

### AI EasyMaker Python SDKインストール

python -m pip install easymaker

* AI EasyMakerノートパソコンには基本的にインストールされています。


### AI EasyMaker SDK初期化
アプリケーションキー(Appkey)と秘密鍵(Secret key)はコンソール右上の**URL & Appkey**メニューで確認できます。
有効にしたAI EasyMaker商品のアプリケーションキー、秘密鍵、リージョン情報を入力します。
AI EasyMaker SDKを使用するには初期化コードが必要です。
```python
import easymaker

easymaker.init(
    appkey='EASYMAKER_APPKEY',
    region='kr1',
    secret_key='EASYMAKER_SECRET_KEY',
)
```

### 実験の作成
学習を作成する前に、学習を分類できる実験の作成が必要です。

[Parameter]

| 名前                 | タイプ  | 必須かどうか | デフォルト値 | 有効範囲   | 説明                                                     |
|------------------------|---------|-------|------|-------------|------------------------------------------------------------|
| experiment_name        | String  | 必須 | なし | 最大50文字  | 実験名                                                  |
| experiment_description | String  | 任意 | なし | 最大255文字 | 実験の説明                                              |
| wait                   | Boolean | 任意 | True | True, False | True：実験の作成が完了した後に実験IDを返す。False：作成リクエスト後、すぐに実験IDを返す |

```python
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### 学習作成

[Parameter]

| 名前                                     | タイプ  | 必須かどうか                | デフォルト値 | 有効範囲   | 説明                                                          |
|--------------------------------------------|---------|-----------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | 必須                | なし  | なし        | 実験ID                                                           |
| training_name                              | String  | 必須                | なし  | 最大50文字  | 学習名                                                       |
| training_description                       | String  | 選択                | なし  | 最大255文字 | 学習の説明                                                   |
| train_image_name                           | String  | 必須                | なし  | なし        | 学習に使用されるイメージ名(CLIで照会可能)                                      |
| train_instance_name                        | String  | 必須                | なし  | なし        | インスタンスタイプ名(CLIで照会可能)                                          |
| train_instance_count                       | Integer | 必須                | なし  | 1～10        | 学習に使用されるインスタンス数                                                 |
| data_storage_size                          | Integer | Obejct Storage使用時は必須 | なし  | 300～10000   | 学習に必要なデータをダウンロードする記憶領域サイズ(単位：GB)、NAS使用時は不要            |
| source_dir_uri                             | String  | 必須                | なし  | 最大255文字 | 学習に必要なファイルが入っているパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| model_upload_uri                           | String  | 必須                | なし  | 最大255文字 | 学習完了したモデルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)  |
| check_point_upload_uri                     | String  | 選択                | なし  | 最大255文字 | チェックポイントファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)   |
| entry_point                                | String  | 必須                | なし  | 最大255文字 | source_dir_uri内で初めて実行されるPythonファイル情報                         |
| timeout_hours                              | Integer | 任意                | 720   | 1～720       | 最大学習時間(単位:時間)                                                |
| hyperparameter_list                        | Array   | 選択                | なし  | 最大100個 | ハイパーパラメータ情報(hyperparameterKey/hyperparameterValueで構成)           |
| hyperparameter_list[0].hyperparameterKey   | String  | 選択                | なし  | 最大255文字 | ハイパーパラメータキー                                                      |
| hyperparameter_list[0].hyperparameterValue | String  | 選択                | なし  | 最大1000文字 | ハイパーパラメータ値                                                   |
| dataset_list                               | Array   | 選択                | なし  | 最大10個  | 学習に使用されるデータセット情報(datasetName/dataUriで構成)                      |
| dataset_list[0].datasetName                | String  | 選択                | なし  | 最大36文字  | データ名                                                      |
| dataset_list[0].datasetUri                 | String  | 選択                | なし  | 最大255文字 | データパス                                                      |
| tag_list                                   | Array   | 選択                | なし  | 最大10個  | タグ情報                                                       |
| tag_list[0].tagKey                         | String  | 選択                | なし  | 最大64文字  | タグキー                                                           |
| tag_list[0].tagValue                       | String  | 選択                | なし  | 最大255文字 | タグ値                                                        |
| use_log                                    | Boolean | 任意                | False | True, False | Log & Crash商品にログを残すかどうか                                     |
| wait                                       | Boolean | 選択                | True  | True、False | True：学習作成が完了した後に学習IDを返す。False：作成リクエスト後すぐに学習IDを返す     |

```python
training_id = easymaker.Training().run(
    experiment_id=experiment_id,
    training_name='training_name',
    training_description='training_description',
    train_image_name='Ubuntu 18.04 CPU TensorFlow Training',
    train_instance_name='m2.c4m8',
    train_instance_count=1,
    data_storage_size=300,  # minimum size ：300GB
    source_dir_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_list=[
        {
            "hyperparameterKey"："epochs",
            "hyperparameterValue"："10",
        },
        {
            "hyperparameterKey"："batch-size",
            "hyperparameterValue"："30",
        }
    ],
    timeout_hours=100,
    model_upload_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_upload_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        {
            "datasetName"："train",
            "dataUri"："obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_download_path}"
        },
        {
            "datasetName"："test",
            "dataUri"："obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_download_path}"
        }
    ],
    tag_list=[
        {
            "tagKey"："tag_num",
            "tagValue"："test_tag_1",
        },
        {
            "tagKey"："tag2",
            "tagValue"："test_tag_2",
        }
    ],
    use_log=True,
    # wait=False,
)
```

### モデル作成
学習ID値でモデルの作成をリクエストできます。
モデルはエンドポイント作成時に使用されます。

[Parameter]

| 名前               | タイプ | 必須かどうか | デフォルト値 | 有効範囲 | 説明        |
|----------------------|--------|-------|-----|---------|---------------|
| training_id          | String | 必須 | なし | なし    | モデルに作成する学習ID |
| model_name           | String | 必須 | なし | 最大50文字 | モデル名     |
| model_description    | String | 任意 | なし | 最大255文字 | モデルの説明 |
| tag_list             | Array  | 任意 | なし | 最大10個 | タグ情報     |
| tag_list[0].tagKey   | String | 任意 | なし | 最大64文字 | タグキー         |
| tag_list[0].tagValue | String | 任意 | なし | 最大255文字 | タグ値      |


```python
model_id = easymaker.Model().create(
    training_id=training_id,
    model_name='model_name',
    model_description='model_description',
)
```

学習IDがなくても、モデルが保存されたパス情報とフレームワークの種類を入力してモデルを作成できます。

[Parameter]

| 名前               | タイプ | 必須かどうか | デフォルト値 | 有効範囲                               | 説明                                              |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| framework_code       | Enum   | 必須 | なし | easymaker.TENSORFLOW、 easymaker.PYTORCH | 学習に使用されたフレームワーク情報                                |
| model_uri            | String | 必須 | なし | 最大255文字                             | モデルファイルパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| model_name           | String | 必須 | なし | 最大50文字                              | モデル名                                           |
| model_description    | String | 任意 | なし | 最大255文字                             | モデルの説明                                       |
| tag_list             | Array  | 任意 | なし | 最大10個                              | タグ情報                                           |
| tag_list[0].tagKey   | String | 任意 | なし | 最大64文字                              | タグキー                                               |
| tag_list[0].tagValue | String | 任意 | なし | 最大255文字                             | タグ値                                            |


```python
model_id = easymaker.Model().create_by_model_uri(
    framework_code=easymaker.TENSORFLOW,
    model_uri='obs://api-storage.cloud.toast.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### エンドポイントの作成

エンドポイント作成時に基本ステージが作成されます。

[Parameter]

| 名前                                | タイプ  | 必須かどうか | デフォルト値 | 有効範囲                  | 説明                                                                 |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| model_id                              | String  | 必須 | なし  | なし                       | エンドポイントで作成するモデルID                                                       |
| endpoint_name                         | String  | 必須 | なし  | 最大50文字                 | エンドポイント名                                                           |
| endpoint_description                  | String  | 任意 | なし  | 最大255文字                | エンドポイントの説明                                                       |
| endpoint_instance_name                | String  | 必須 | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                              |
| endpoint_instance_count               | Integer | 任意 | 1     | 1～10                       | エンドポイントに使用されるインスタンス数                                                     |
| apigw_resource_uri                    | String  | 必須 | なし  | 最大255文字                | /で始まるAPI Gatewayリソースパス                                         |
| tag_list                              | Array   | 任意 | なし  | 最大10個                 | タグ情報                                                              |
| tag_list[0].tagKey                    | String  | 任意 | なし  | 最大64文字                 | タグキー                                                                  |
| tag_list[0].tagValue                  | String  | 任意 | なし  | 最大255文字                | タグ値                                                               |
| use_log                               | Boolean | 任意 | False | True, False                | Log & Crash商品にログを残すかどうか                                            |        
| wait                                  | Boolean | 任意 | True  | True, False                | True：エンドポイントの作成が完了した後にエンドポイントIDを返す。False：エンドポイントリクエスト後、すぐにエンドポイントIDを返す |

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

作成しておいたエンドポイントの使用

```python
endpoint = easymaker.Endpoint()
```

### ステージの追加

既存エンドポイントに新規ステージを追加できます。

[Parameter]

| 名前                                | タイプ  | 必須かどうか | デフォルト値 | 有効範囲                  | 説明                                                             |
|---------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| model_id                              | String  | 必須 | なし  | なし                       | エンドポイントで作成するモデルID                                                   |
| stage_name                            | String  | 必須 | なし  | 最大50文字                 | ステージ名                                                        |
| stage_description                     | String  | 任意 | なし  | 最大255文字                | ステージの説明                                                    |
| endpoint_instance_name                | String  | 必須 | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                          |
| endpoint_instance_count               | Integer | 任意 | 1     | 1～10                       | エンドポイントに使用されるインスタンス数                                                 |
| tag_list                              | Array   | 任意 | なし  | 最大10個                 | タグ情報                                                          |
| tag_list[0].tagKey                    | String  | 任意 | なし  | 最大64文字                 | タグキー                                                              |
| tag_list[0].tagValue                  | String  | 任意 | なし  | 最大255文字                | タグ値                                                           |
| use_log                               | Boolean | 任意 | False | True, False                | Log & Crash商品にログを残すかどうか                                        |        
| wait                                  | Boolean | 任意 | True  | True, False                | True：ステージの作成が完了した後にステージIDを返す。False：ステージリクエスト後、すぐにステージIDを返す |
```python
stage_id = endpoint.create_stage(
    model_id=model_id,
    stage_name='stage01',  # 30文字以内小文字/数字
    stage_description='test endpoint',
    endpoint_instance_name='c2.c16m16',
    endpoint_instance_count=1,
    use_log=True,
    # wait=False,
)
```

### エンドポイントインファレンス

基本ステージにインファレンス

```python
input_data = [6.8, 2.8, 4.8, 1.4]
endpoint.predict(json={'instances'：[input_data]})
```

特定ステージを指定してインファレンス

```python
# ステージ情報照会
endpoint_stage_info_list = endpoint.get_endpoint_stage_info_list()
for endpoint_stage_info in endpoint_stage_info_list:
    print(f'endpoint_stage_info ：{endpoint_stage_info}')
    
# ステージを指定してインファレンスリクエスト
input_data = [6.0, 3.4, 4.5, 1.6]
for endpoint_stage_info in endpoint_stage_info_list:
    if endpoint_stage_info['stage_name'] == 'stage01':
        endpoint.predict(json={'instances'：[input_data]},
                         endpoint_stage_info=endpoint_stage_info)
```

### NHN Cloud - Log & Crashログ転送機能
```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default：INFO
                      project_version='2.0.0',  # default：1.0.0 
                      parameters={'serviceType'：'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storageファイル転送機能
Object Storage商品にファイルをアップロードし、ダウンロードする機能を提供します。
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
アプリケーションキー、秘密鍵、リージョン情報を知っている場合は、コンソールにアクセスせずにPython CLIを介してさまざまな情報を確認できます。

| 機能              | コマンド                                                                                    |
|---------------------|--------------------------------------------------------------------------------------------|
| instance typeリスト照会 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| imageリスト照会     | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| experimentリスト照会 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| trainingリスト照会  | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| modelリスト照会     | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| endpointリスト照会  | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |
