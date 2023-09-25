## NHN Cloud > SDK使用ガイド > AI EasyMaker

## 開発ガイド

### AI EasyMaker Python SDKインストール

python -m pip install easymaker

* AI EasyMakerノートパソコンには基本的にインストールされています。


### AI EasyMaker SDK初期化
アプリケーションキー(Appkey)と秘密鍵(Secret key)はコンソール右上の**URL & Appkey**メニューで確認できます。
有効にしたAI EasyMaker商品のアプリケーションキー、秘密鍵、リージョン情報を入力します。
AI EasyMaker SDKを使用するには初期化コードが必要です。
```
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

```
experiment_id = easymaker.Experiment().create(
    experiment_name='experiment_name',
    experiment_description='experiment_description',
    # wait=False,
)
```

### 実験の削除

[Parameter]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | 必須   | なし   | 最大36文字 | 実験ID |

```python
easymaker.Experiment().delete(experiment_id)
```

### 学習作成

[Parameter]

| 名前                                        | タイプ     | 必須かどうか                    | デフォルト値  | 有効範囲      | 説明                                                             |
|--------------------------------------------|---------|---------------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                              | String  | 必須                       | なし    | なし          | 実験ID                                                           |
| training_name                              | String  | 必須                       | なし    | 最大50文字     | 学習名                                                          |
| training_description                       | String  | 任意                       | なし    | 最大255文字    | 学習の説明                                                      |
| train_image_name                           | String  | 必須                       | なし    | なし          | 学習に使用されるイメージ名(CLIで照会可能)                                      |
| train_instance_name                        | String  | 必須                       | なし    | なし          | インスタンスタイプ名(CLIで照会可能)                                          |
| distributed_training_count                 | Integer | 必須                       | なし    | 1～10         | 学習に適用する分散学習数                                                 |
| data_storage_size                          | Integer | Obejct Storageを使用する場合は必須   | なし    | 300～10000   | 学習に必要なデータをダウンロードする記憶領域サイズ(単位: GB)、NAS使用時は不要              |
| algorithm_name                             | String  | NHN Cloud提供アルゴリズムを使用する場合は必須 | なし    | 最大64文字     | アルゴリズム名(CLIで照会可能)                                             |
| source_dir_uri                             | String  | 独自アルゴリズムを使用する場合は必須          | なし    | 最大255文字    | 学習に必要なファイルがあるパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| entry_point                                | String  | 独自アルゴリズムを使用する場合は必須          | なし    | 最大255文字    | source_dir_uri内で最初に実行されるPythonファイル情報                            |
| model_upload_uri                           | String  | 必須                       | なし    | 最大255文字    | 学習完了したモデルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)   |
| check_point_input_uri                      | String  | 任意                       | なし    | 最大255文字    | 入力チェックポイントファイルパス(NHN Cloud Object StorageまたはNHN Cloud NAS)                 |
| check_point_upload_uri                     | String  | 任意                       | なし    | 最大255文字    | チェックポイントファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)   |
| timeout_hours                              | Integer | 任意                       | 720   | 1～720       | 最大学習時間(単位:時間)                                                |
| hyperparameter_list                        | Array   | 任意                       | なし    | 最大100個    | ハイパーパラメータ情報(hyperparameterKey/hyperparameterValueで構成)           |
| hyperparameter_list[0].hyperparameterKey   | String  | 任意                       | なし    | 最大255文字    | ハイパーパラメータキー                                                       |
| hyperparameter_list[0].hyperparameterValue | String  | 任意                       | なし    | 最大1000文字   | ハイパーパラメータ値                                                      |
| dataset_list                               | Array   | 任意                       | なし    | 最大10個     | 学習に使用されるデータセット情報(datasetName/dataUriで構成)                      |
| dataset_list[0].datasetName                | String  | 任意                       | なし    | 最大36文字     | データ名                                                         |
| dataset_list[0].datasetUri                 | String  | 任意                       | なし    | 最大255文字    | データパス                                                         |
| tag_list                                   | Array   | 任意                       | なし    | 最大10個     | タグ情報                                                          |
| tag_list[0].tagKey                         | String  | 任意                       | なし    | 最大64文字     | タグキー                                                            |
| tag_list[0].tagValue                       | String  | 任意                       | なし    | 最大255文字    | タグ値                                                           |
| use_log                                    | Boolean | 任意                       | False | True、False | Log & Crash商品にログを残すかどうか                                      |
| wait                                       | Boolean | 任意                       | True  | True、False | True:学習作成が完了した後に学習IDを返す。False:作成リクエスト後すぐに学習IDを返す      |

```
training_id = easymaker.Training().run(
    experiment_id=experiment_id,
    training_name='training_name',
    training_description='training_description',
    train_image_name='Ubuntu 18.04 CPU TensorFlow Training',
    train_instance_name='m2.c4m8',
    distributed_training_count=1,
    data_storage_size=300,  # minimum size ：300GB
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
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
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    check_point_input_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_input_path}',
    check_point_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{checkpoint_upload_path}',
    dataset_list=[
        {
            "datasetName"："train",
            "dataUri"："obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{train_data_download_path}"
        },
        {
            "datasetName"："test",
            "dataUri"："obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{test_data_download_path}"
        }
    ],
    tag_list=[
        {
            "tagKey"："tag1",
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

### 学習の削除

[Parameter]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | 必須   | なし   | 最大36文字 | 学習ID |

```python
easymaker.Training().delete(training_id)
```

### ハイパーパラメータチューニング作成

[Parameter]

| 名前                                                            | タイプ            | 必須かどうか                                                | デフォルト値  | 有効範囲                                       | 説明                                                                        |
|----------------------------------------------------------------|----------------|-------------------------------------------------------|-------|----------------------------------------------|----------------------------------------------------------------------------|
| experiment_id                                                  | String         | 必須                                                   | なし    | なし                                           | 実験ID                                                                      |
| hyperparameter_tuning_name                                     | String         | 必須                                                   | なし    | 最大50文字                                      | ハイパーパラメータチューニング名                                                             |
| hyperparameter_tuning_description                              | String         | 任意                                                   | なし    | 最大255文字                                     | ハイパーパラメータチューニングについての説明                                                         |
| image_name                                                     | String         | 必須                                                   | なし    | なし                                           | ハイパーパラメータチューニングに使用されるイメージ名(CLIで照会可能)                                         |
| instance_name                                                  | String         | 必須                                                   | なし    | なし                                           | インスタンスタイプ名(CLIで照会可能)                                                     |
| distributed_training_count                                     | Integer        | 必須                                                   | 1      | distributed_training_countとparallel_trial_countの積が10以下 | ハイパーパラメータチューニングで各学習に適用する分散学習数                                                      |
| parallel_trial_count                                           | Integer        | 必須                                                   | 1      | distributed_training_countとparallel_trial_countの積が10以下 | ハイパーパラメータチューニングで並列実行する学習数                                                      |
| data_storage_size                                              | Integer        | Obejct Storageを使用する場合は必須                               | なし    | 300～10000                                    | ハイパーパラメータチューニングに必要なデータをダウンロードする記憶領域サイズ(単位：GB)、NAS使用時は不要                 |
| algorithm_name                                                 | String         | NHN Cloud提供アルゴリズムを使用する場合は必須                            | なし    | 最大64文字                                      | アルゴリズム名(CLIで照会可能)                                                        |
| source_dir_uri                                                 | String         | 独自アルゴリズムを使用する場合は必須                                      | なし    | 最大255文字                                     | ハイパーパラメータチューニングに必要なファイルがあるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)    |
| entry_point                                                    | String         | 独自アルゴリズムを使用する場合は必須                                      | なし    | 最大255文字                                     | source_dir_uri内で最初に実行されるPythonファイル情報                                       |
| model_upload_uri                                               | String         | 必須                                                   | なし    | 最大255文字                                     | ハイパーパラメータチューニングで学習完了したモデルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| check_point_input_uri                                          | String         | 任意                                                   | なし    | 最大255文字                                     | 入力チェックポイントファイルのパス(NHN Cloud Object StorageまたはNHN Cloud NAS)                 |
| check_point_upload_uri                                         | String         | 任意                                                   | なし    | 最大255文字                                     | チェックポイントファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)              |
| timeout_hours                                                  | Integer        | 任意                                                   | 720   | 1～720                                        | 最大ハイパーパラメータチューニング時間(単位：時間)                                                   |
| hyperparameter_spec_list                                       | Array          | 任意                                                   | なし    | 最大100個                                     | ハイパーパラメータのスペック情報                                                             |
| hyperparameter_spec_list[0].<br>hyperparameterName             | String         | 任意                                                   | なし    | 最大255文字                                     | ハイパーパラメータ名                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterTypeCode         | String         | 任意                                                   | なし    | INT、DOUBLE、DISCRETE、CATEGORICAL           | ハイパーパラメータタイプ                                                                |
| hyperparameter_spec_list[0].<br>hyperparameterMinValue         | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEの場合は必須           | なし    | なし                                           | ハイパーパラメータ最小値                                                               |
| hyperparameter_spec_list[0].<br>hyperparameterMaxValue         | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEの場合は必須           | なし    | なし                                           | ハイパーパラメータ最大値                                                               |
| hyperparameter_spec_list[0].<br>hyperparameterStep             | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEでGRID戦略の場合は必須 | なし    | なし                                           | "Grid"チューニング戦略を使用する際のハイパーパラメータ値の変化サイズ                                      |
| hyperparameter_spec_list[0].<br>hyperparameterSpecifiedValues  | String         | hyperparameterTypeCodeがDISCRETE、CATEGORICALの場合は必須  | なし    | 最大3000文字                                      | 決められたハイパーパラメータリスト(`,`で区切られた文字列や数字)                                          |
| dataset_list                                                   | Array          | 任意                                                   | なし    | 最大10個                                      | ハイパーパラメータチューニングに使用されるデータセット情報(datasetName/dataUriで構成)                         |
| dataset_list[0].datasetName                                    | String         | 任意                                                   | なし    | 最大36文字                                      | データ名                                                                    |
| dataset_list[0].datasetUri                                     | String         | 任意                                                   | なし    | 最大255文字                                     | データパス                                                                    |
| metric_list                                                    | Array          | 独自アルゴリズムを使用する場合は必須                                      | なし    | 最大10個(指標名で構成された文字列リスト)                    | 学習コードが出力するログの中からどの指標を収集するか定義します。                                       |
| metric_regex                                                   | String         | 独自アルゴリズムを使用する場合は任意                                      | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?) | 最大255文字                                                                   | 指標の収集に使用する正規表現を入力します。学習アルゴリズムが正規表現に合わせて指標を出力する必要があります。                                                          |
| objective_metric_name                                          | String         | 独自アルゴリズムを使用する場合は必須                                      | なし    | 最大36文字、metric_listの中で1つ                     | どの指標の最適化が目標なのか選択します。                                                 |
| objective_type_code                                            | String         | 独自アルゴリズムを使用する場合は必須                                      | なし    | MINIMIZE、MAXIMIZE                           | 目標指標最適化タイプを選択します。                                                       |
| objective_goal                                                 | Double         | 任意                                                   | なし    | なし                                           | 目標指標がこの値に達するとチューニング作業が終了します。                                             |
| max_failed_trial_count                                         | Integer        | 任意                                                   | なし    | なし                                           | 失敗した学習の最大数を定義します。失敗した学習の数がこの値に達するとチューニングが失敗となり終了します。                 |
| max_trial_count                                                | Integer        | 任意                                                   | なし    | なし                                           | 最大学習数を定義します。自動実行された学習の数がこの値に達するまでチューニングが実行されます。                     |
| tuning_strategy_name                                           | String         | 必須                                                   | なし    | なし                                           | どの戦略を使用して最適なハイパーパラメータを探すか選択します。                                        |
| tuning_strategy_random_state                                   | Integer        | 任意                                                   | なし    | なし                                           | 乱数作成を決定します。再現可能な結果のために固定された値で指定します。                                 |
| early_stopping_algorithm                                       | String         | 必須                                                   | なし    | EARLY_STOPPING_ALGORITHM.<br>MEDIAN          | 学習を継続してもモデルがそれ以上良くならない場合、早期に学習を終了します。                              |
| early_stopping_min_trial_count                                 | Integer        | 必須                                                   | 3     | なし                                           | 中間値を計算する際に、いくつの学習から目標指標値を取得するか定義します。                                |
| early_stopping_start_step                                      | Integer        | 必須                                                   | 4     | なし                                           | 何番目の学習段階から早期終了を適用するか設定します。                                            |
| tag_list                                                       | Array          | 任意                                                   | なし    | 最大10個                                      | タグ情報                                                                     |
| tag_list[0].tagKey                                             | String         | 任意                                                   | なし    | 最大64文字                                      | タグキー                                                                       |
| tag_list[0].tagValue                                           | String         | 任意                                                   | なし    | 最大255文字                                     | タグ値                                                                      |
| use_log                                                        | Boolean        | 任意                                                   | False | True、False                                  | Log & Crash製品にログを残すかどうか                                                 |
| wait                                                           | Boolean        | 任意                                                   | True  | True、False                                  | True:ハイパーパラメータチューニングの作成完了後、ハイパーパラメータチューニングIDを返却、False:作成リクエスト後すぐに学習IDを返却 |

```
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

### ハイパーパラメータチューニングの削除

[Parameter]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明          |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | 必須   | なし   | 最大36文字 | ハイパーパラメータチューニングID |

```python
easymaker.HyperparameterTuning().delete(hyperparameter_tuning_id)
```

### モデル作成
学習ID値でモデルの作成をリクエストできます。
モデルはエンドポイント作成時に使用されます。

[Parameter]

| 名前                      | タイプ    | 必須かどうか                             | デフォルト値 | 有効範囲  | 説明                                 |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | hyperparameter_tuning_idがない場合は必須 | なし  | なし      | モデルとして作成する学習ID                       |
| hyperparameter_tuning_id | String | training_idがない場合は必須             | なし  | なし      | モデルとして作成するハイパーパラメータチューニングID(最高学習で作成済み) |
| model_name               | String | 必須                                | なし  | 最大50文字 | モデル名                              |
| model_description        | String | 選択                                | なし  | 最大255文字 | モデルの説明                          |
| tag_list                 | Array  | 選択                                | なし  | 最大10個 | タグ情報                              |
| tag_list[0].tagKey       | String | 選択                                | なし  | 最大64文字 | タグキー                                |
| tag_list[0].tagValue     | String | 選択                                | なし  | 最大255文字 | タグ値                               |


```
model_id = easymaker.Model().create(
    training_id=training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning_id,
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


```
model_id = easymaker.Model().create_by_model_uri(
    framework_code=easymaker.TENSORFLOW,
    model_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    model_description='model_description',
)
```

### モデル削除

[Parameter]

| 名前                       | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | 必須   | なし   | 最大36文字 | モデルID |

```python
easymaker.Model().delete(model_id)
```

### エンドポイントの作成

エンドポイント作成時に基本ステージが作成されます。

[Parameter]

| 名前                                | タイプ  | 必須かどうか | デフォルト値 | 有効範囲                  | 説明                                                                 |
|---------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                         | String  | 必須 | なし  | 最大50文字                 | エンドポイント名                                                           |
| endpoint_description                  | String  | 任意 | なし  | 最大255文字                | エンドポイントの説明                                                       |
| endpoint_instance_name                | String  | 必須 | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                              |
| endpoint_instance_count               | Integer | 任意 | 1     | 1～10                       | エンドポイントに使用されるインスタンス数                                                     |
| endpoint_model_resource_list          | Array   | 必須   | なし   | 最大10個                     | ステージに使用されるリソース情報                                                |
| endpoint_model_resource_list[0].modelId           | String   | 必須   | なし   | なし                      | ステージリソースで作成するモデルID                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | 必須   | なし   | 最大255文字                 | /で始まるAPI Gatewayリソースパス                            |
| endpoint_model_resource_list[0].podCount          | Integer  | 必須   | なし   | 1～100                     | ステージリソースに使用されるPod数                                   |
| endpoint_model_resource_list[0].description       | String   | 任意   | なし   | 最大255文字                 | ステージリソースの説明                                      |
| tag_list                              | Array   | 任意 | なし  | 最大10個                 | タグ情報                                                              |
| tag_list[0].tagKey                    | String  | 任意 | なし  | 最大64文字                 | タグキー                                                                  |
| tag_list[0].tagValue                  | String  | 任意 | なし  | 最大255文字                | タグ値                                                               |
| use_log                               | Boolean | 任意 | False | True, False                | Log & Crash商品にログを残すかどうか                                            |
| wait                                  | Boolean | 任意 | True  | True, False                | True：エンドポイントの作成が完了した後にエンドポイントIDを返す。False：エンドポイントリクエスト後、すぐにエンドポイントIDを返す |

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

作成しておいたエンドポイントの使用

```python
endpoint = easymaker.Endpoint()
```

### ステージの追加

既存エンドポイントに新規ステージを追加できます。

[Parameter]

| 名前                                | タイプ  | 必須かどうか | デフォルト値 | 有効範囲                  | 説明                                                             |
|---------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| stage_name                            | String  | 必須 | なし  | 最大50文字                 | ステージ名                                                        |
| stage_description                     | String  | 任意 | なし  | 最大255文字                | ステージの説明                                                    |
| endpoint_instance_name                | String  | 必須 | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                          |
| endpoint_instance_count               | Integer | 任意 | 1     | 1～10                       | エンドポイントに使用されるインスタンス数                                                 |
| endpoint_model_resource_list          | Array   | 必須   | なし   | 最大10個                     | ステージに使用されるリソース情報                                                |
| endpoint_model_resource_list[0].modelId           | String   | 必須   | なし   | なし                      | ステージリソースで作成するモデルID                                   |
| endpoint_model_resource_list[0].apigwResourceUri  | String   | 必須   | なし   | 最大255文字                 | /で始まるAPI Gatewayリソースパス                            |
| endpoint_model_resource_list[0].podCount          | Integer  | 必須   | なし   | 1～100                     | ステージリソースに使用されるPod数                                   |
| endpoint_model_resource_list[0].description       | String   | 任意   | なし   | 最大255文字                 | ステージリソースの説明                                      |
| tag_list                              | Array   | 任意 | なし  | 最大10個                 | タグ情報                                                          |
| tag_list[0].tagKey                    | String  | 任意 | なし  | 最大64文字                 | タグキー                                                              |
| tag_list[0].tagValue                  | String  | 任意 | なし  | 最大255文字                | タグ値                                                           |
| use_log                               | Boolean | 任意 | False | True、False                | Log & Crash商品にログを残すかどうか                                       |
| wait                                  | Boolean | 任意 | True  | True, False                | True：ステージの作成が完了した後にステージIDを返す。False：ステージリクエスト後、すぐにステージIDを返す |
```python
stage_id = endpoint.create_stage(
    stage_name='stage01',  # 30文字以内小文字/数字
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

### エンドポイントインファレンス

基本ステージにインファレンス

```python
# 基本ステージ情報照会
endpoint_stage_info = endpoint.get_default_endpoint_stage()
print(f'endpoint_stage_info : {endpoint_stage_info}')

# ステージを指定してインファレンスリクエスト
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

特定ステージを指定してインファレンス

```python
# ステージ情報照会
endpoint_stage_info = endpoint.get_endpoint_stage_by_id(endpoint_stage_id=stage_id)
print(f'endpoint_stage_info : {endpoint_stage_info}')

# ステージを指定してインファレンスリクエスト
input_data = [6.0, 3.4, 4.5, 1.6]
endpoint.predict(endpoint_stage_info=endpoint_stage_info,
                 model_id=model_id,
                 json={'instances': [input_data]})
```

### エンドポイントの削除

[Parameter]

| 名前           | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明      |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | 必須   | なし   | 最大36文字 | エンドポイントID |

```python
endpoint.Endpoint().delete_endpoint(endpoint_id)
```

### エンドポイントステージの削除

[Parameter]

| 名前        | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明     |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | 必須   | なし   | 最大36文字 | ステージID |

```python
endpoint.Endpoint().delete_endpoint_stage(stage_id)
```

### NHN Cloud - Log & Crashログ転送機能
```
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default：INFO
                      project_version='2.0.0',  # default：1.0.0
                      parameters={'serviceType'：'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storageファイル転送機能
Object Storage商品にファイルをアップロードし、ダウンロードする機能を提供します。
```
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
アプリケーションキー、秘密鍵、リージョン情報を知っている場合は、コンソールにアクセスせずにPython CLIを介してさまざまな情報を確認できます。

| 機能                         | コマンド                                                                                       |
|-----------------------------|--------------------------------------------------------------------------------------------|
| Instance typeリスト照会        | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| Imageリスト照会                | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| Algorithmリスト照会            | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -algorithm  |
| Experimentリスト照会           | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| Trainingリスト照会             | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| Hyperparameter tuningリスト照会 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -tuning     |
| Modelリスト照会                | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| Endpointリスト照会             | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |
