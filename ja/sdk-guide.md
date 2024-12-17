## Machine Learning > AI EasyMaker > SDK使用ガイド

## SDK設定

### AI EasyMaker Python SDKインストール

python -m pip install easymaker

- AI EasyMakerノートパソコンには基本的にインストールされています。

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
    experiment_id="EXPERIMENT_ID", # Optional
)
```

## CLI Command

アプリキー、秘密鍵、リージョン情報を知っていれば、コンソールにアクセスしなくても、Python CLIを通じて様々な情報を確認できます。

| 機能                    | コマンド                                                                                      |
|-------------------------|--------------------------------------------------------------------------------------------|
| インスタンスタイプリスト照会         | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -instance   |
| 学習イメージリスト照会          | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -image      |
| NHN Cloud提供アルゴリズムリスト照会 | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -algorithm  |
| 実験リスト照会              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -experiment |
| 学習リスト照会              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -training   |
| ハイパーパラメータチューニングリスト照会      | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -tuning     |
| モデルリスト照会              | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -model      |
| エンドポイントリスト照会           | python -m easymaker --region kr1 --appkey EM_APPKEY --secret_key EM_SECRET_KEY -endpoint   |

## 実験

### 実験の作成

学習を作成する前に、学習を分類できる実験の作成が必要です。

[パラメータ]

| 名前                     | タイプ     | 必須かどうか | デフォルト値 | 有効範囲     | 説明                                                       |
|--------------------------|----------|-------|------|-------------|------------------------------------------------------------|
| experiment_name          | String   | 必須  | なし | 最大50文字    | 実験名                                                    |
| description   | String   | 選択  | なし | 最大255文字   | 実験の説明                                                |
| wait                     | Boolean  | 選択  | True | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
experiment  = easymaker.Experiment().create(
    experiment_name='experiment_name',
    description='experiment_description',
    # wait=False,
)
```

### 実験の削除

[パラメータ]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|------------------------|---------|-------|------|--------|-------|
| experiment_id          | String  | 必須   | なし   | 最大36文字 | 実験ID |

```python
easymaker.Experiment(experiment_id).delete()
```

## 学習

### 学習作成

[パラメータ]

| 名前                                  | タイプ    | 必須かどうか                   | デフォルト値 | 有効範囲     | 説明                                                            |
|---------------------------------------|---------|---------------------------|-------|-------------|-----------------------------------------------------------------|
| experiment_id                         | String  | easymaker.initで未入力の場合は必須 | なし  | 最大36文字        | 実験ID                                                           |
| training_name                         | String  | 必須                      | なし  | 最大50文字    | 学習名                                                         |
| description                  | String  | 選択                      | なし  | 最大255文字   | 学習の説明                                                     |
| image_name                      | String  | 必須                      | なし  | なし        | 学習に使用されるイメージ名(CLIで照会可能)                                      |
| instance_name                   | String  | 必須                      | なし  | なし        | インスタンスタイプ名(CLIで照会可能)                                          |
| distributed_node_count                | Integer | 必須                      | なし  | 1~10         | 分散学習を適用するノード数                                               |
| use_torchrun                          | Boolean | 選択                      | False  | True, False | torchrunの使用有無、Pytorchイメージでのみ使用可                          |
| nproc_per_node                        | Integer | use_torchrun Trueの場合は必須  | 1      | 1~(CPU数またはGPU数) | ノードあたりのプロセス数、 use_torchrunを使用する場合は必ず設定しなければならない値     |
| data_storage_size                     | Integer | Obejct Storageを使用する場合は必須  | なし  | 300~10000   | 学習に必要なデータをダウンロードする記憶領域サイズ(単位: GB)、NAS使用時は不要             |
| algorithm_name                        | String  | NHN Cloud提供アルゴリズムを使用する場合は必須 | なし  | 最大64文字    | アルゴリズム名(CLIで照会可能)                                             |
| source_dir_uri                        | String  | 独自アルゴリズムを使用する場合は必須         | なし  | 最大255文字   | 学習に必要なファイルがあるパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| entry_point                           | String  | 独自アルゴリズムを使用する場合は必須         | なし  | 最大255文字   | source_dir_uri内で最初に実行されるPythonファイル情報                           |
| model_upload_uri                      | String  | 必須                      | なし  | 最大255文字   | 学習完了したモデルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)   |
| check_point_input_uri                 | String  | 選択                      | なし  | 最大255文字   | 入力チェックポイントファイルパス(NHN Cloud Object StorageまたはNHN Cloud NAS)                 |
| check_point_upload_uri                | String  | 選択                      | なし  | 最大255文字   | チェックポイントファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)   |
| timeout_hours                         | Integer | 選択                      | 720   | 1~720       | 最大学習時間(単位:時間)                                                |
| hyperparameter_list                   | Array   | 選択                      | なし  | 最大100個   | ハイパーパラメータ情報(parameterName/parameterValueで構成)           |
| hyperparameter_list[0].parameterName  | String  | 選択                      | なし  | 最大255文字   | ハイパーパラメータキー                                                     |
| hyperparameter_list[0].parameterValue | String  | 任意                       | なし    | 最大1000文字   | ハイパーパラメータ値                                                      |
| dataset_list                          | Array   | 選択                      | なし  | 最大10個    | 学習に使用されるデータセット情報(datasetName/dataUriで構成)                      |
| dataset_list[0].datasetName           | String  | 選択                      | なし  | 最大36文字    | データ名                                                        |
| dataset_list[0].datasetUri            | String  | 選択                      | なし  | 最大255文字   | データパス                                                        |
| tag_list                              | Array   | 選択                      | なし  | 最大10個    | タグ情報                                                         |
| tag_list[0].tagKey                    | String  | 選択                      | なし  | 最大64文字    | タグキー                                                          |
| tag_list[0].tagValue                  | String  | 選択                      | なし  | 最大255文字   | タグ値                                                          |
| use_log                               | Boolean | 選択                      | False | True, False | Log & Crash Searchサービスにログを残すかどうか                                    |
| wait                                  | Boolean | 選択                      | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
training = easymaker.Training().run(
    experiment_id=experiment.experiment_id, # Optional if already set in init
    training_name='training_name',
    description='training_description',
    image_name='Ubuntu 18.04 CPU TensorFlow Training',
    instance_name='m2.c4m8',
    distributed_node_count=1,
    data_storage_size=300,  # minimum size ：300GB
    source_dir_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{soucre_download_path}',
    entry_point='training_start.py',
    hyperparameter_list=[
        {
            "parameterName"："epochs",
            "parameterValue"："10",
        },
        {
            "parameterName"："batch-size",
            "parameterValue"："30",
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

[パラメータ]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|------------------------|---------|-------|------|--------|-------|
| training_id          | String  | 必須   | なし   | 最大36文字 | 学習ID |

```python
easymaker.Training(training_id).delete()
```

## ハイパーパラメータチューニング

### ハイパーパラメータチューニング作成

[パラメータ]

| 名前                                                          | タイプ           | 必須かどうか                                               | デフォルト値 | 有効範囲                                      | 説明                                                                       |
|---------------------------------------------------------------|----------------|-------------------------------------------------------|-------|----------------------------------------------|----------------------------------------------------------------------------|
| experiment_id                                                 | String         | easymaker.initで未入力の場合は必須                           | なし  | 最大36文字                                         | 実験ID                                                                      |
| hyperparameter_tuning_name                                    | String         | 必須                                                  | なし  | 最大50文字                                     | ハイパーパラメータチューニング名                                                            |
| description                             | String         | 選択                                                  | なし  | 最大255文字                                    | ハイパーパラメータチューニングについての説明                                                        |
| image_name                                                    | String         | 必須                                                  | なし  | なし                                         | ハイパーパラメータチューニングに使用されるイメージ名(CLIで照会可能)                                         |
| instance_name                                                 | String         | 必須                                                  | なし  | なし                                         | インスタンスタイプ名(CLIで照会可能)                                                     |
| distributed_node_count                                        | Integer        | 必須                                                  | 1      | distributed_node_countとparallel_trial_countの積が10以下 | ハイパーパラメータチューニングで学習ごとに分散学習を適用するノード数                                                    |
| parallel_trial_count                                          | Integer        | 必須                                                  | 1      | distributed_node_countとparallel_trial_countの積が10以下 | ハイパーパラメータチューニングで並列実行する学習数                                                        |
| use_torchrun                                                  | Boolean        | 選択                                                  | False  | True, False | torchrunの使用有無、Pytorchイメージでのみ使用可                                               |
| nproc_per_node                                                | Integer        | use_torchrun Trueの場合は必須                              | 1      | 1~(CPU数またはGPU数) | ノードあたりのプロセス数、 use_torchrunを使用する場合は必ず設定しなければならない値                                       |
| data_storage_size                                             | Integer        | Obejct Storageを使用する場合は必須                              | なし  | 300~10000                                    | ハイパーパラメータチューニングに必要なデータをダウンロードする記憶領域サイズ(単位：GB)、NAS使用時は不要                |
| algorithm_name                                                | String         | NHN Cloud提供アルゴリズムを使用する場合は必須                           | なし  | 最大64文字                                     | アルゴリズム名(CLIで照会可能)                                                        |
| source_dir_uri                                                | String         | 独自アルゴリズムを使用する場合は必須                                     | なし  | 最大255文字                                    | ハイパーパラメータチューニングに必要なファイルがあるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)    |
| entry_point                                                   | String         | 独自アルゴリズムを使用する場合は必須                                     | なし  | 最大255文字                                    | source_dir_uri内で最初に実行されるPythonファイル情報                                      |
| model_upload_uri                                              | String         | 必須                                                  | なし  | 最大255文字                                    | ハイパーパラメータチューニングで学習完了したモデルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| check_point_input_uri                                         | String         | 選択                                                  | なし  | 最大255文字                                    | 入力チェックポイントファイルパス(NHN Cloud Object StorageまたはNHN Cloud NAS)                 |
| check_point_upload_uri                                        | String         | 選択                                                  | なし  | 最大255文字                                    | チェックポイントファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)              |
| timeout_hours                                                 | Integer        | 選択                                                  | 720   | 1~720                                        | 最大ハイパーパラメータチューニング時間(単位：時間)                                                   |
| hyperparameter_spec_list                                      | Array          | 選択                                                  | なし  | 最大100個                                    | ハイパーパラメータのスペック情報                                                            |
| hyperparameter_spec_list[0].<br>hyperparameterName            | String         | 選択                                                  | なし  | 最大255文字                                    | ハイパーパラメータ名                                                               |
| hyperparameter_spec_list[0].<br>hyperparameterTypeCode        | String         | 選択                                                  | なし  | INT, DOUBLE, DISCRETE, CATEGORICAL           | ハイパーパラメータタイプ                                                               |
| hyperparameter_spec_list[0].<br>hyperparameterMinValue        | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEの場合は必須          | なし  | なし                                         | ハイパーパラメータ最小値                                                              |
| hyperparameter_spec_list[0].<br>hyperparameterMaxValue        | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEの場合は必須          | なし  | なし                                         | ハイパーパラメータ最大値                                                              |
| hyperparameter_spec_list[0].<br>hyperparameterStep            | Integer/Double | hyperparameterTypeCodeがINT、DOUBLEでGRID戦略の場合は必須 | なし  | なし                                         | "Grid"チューニング戦略を使用する際のハイパーパラメータ値の変化サイズ                                     |
| hyperparameter_spec_list[0].<br>hyperparameterSpecifiedValues | String         | hyperparameterTypeCodeがDISCRETE、CATEGORICALの場合は必須 | なし  | 最大3000文字                                     | 決められたハイパーパラメータリスト(`,`で区切られた文字列や数字)                                          |
| dataset_list                                                  | Array          | 選択                                                  | なし  | 最大10個                                     | ハイパーパラメータチューニングに使用されるデータセット情報(datasetName/dataUriで構成)                         |
| dataset_list[0].datasetName                                   | String         | 選択                                                  | なし  | 最大36文字                                     | データ名                                                                   |
| dataset_list[0].datasetUri                                    | String         | 選択                                                  | なし  | 最大255文字                                    | データパス                                                                   |
| metric_list                                                   | Array          | 独自アルゴリズムを使用する場合は必須                                     | なし  | 最大10個(指標名で構成された文字列リスト)                    | 学習コードが出力するログの中からどの指標を収集するか定義                                     |
| metric_regex                                                  | String         | 独自アルゴリズムを使用する場合は任意                                     | ([\w\ | -]+)\s*=\s*([+-]?\d*(\.\d+)?([Ee][+-]?\d+)?) | 最大255文字                                                                  | 指標の収集に使用する正規表現を入力。学習アルゴリズムが正規表現に合わせて指標を出力する必要があります。                                                        |
| objective_metric_name                                         | String         | 独自アルゴリズムを使用する場合は必須                                     | なし  | 最大36文字、metric_listの中で1つ                   | どの指標の最適化が目標なのか選択                                               |
| objective_type_code                                           | String         | 独自アルゴリズムを使用する場合は必須                                     | なし  | MINIMIZE, MAXIMIZE                           | 目標指標最適化タイプ選択                                                     |
| objective_goal                                                | Double         | 選択                                                  | なし  | なし                                         | 目標指標がこの値に達するとチューニング作業が終了する                                            |
| max_failed_trial_count                                        | Integer        | 選択                                                  | なし  | なし                                         | 失敗した学習の最大数を定義。失敗した学習の数がこの値に達するとチューニングが失敗となり終了する。                 |
| max_trial_count                                               | Integer        | 選択                                                  | なし  | なし                                         | 最大学習数を定義。自動実行された学習の数がこの値に達するまでチューニングが実行される。                     |
| tuning_strategy_name                                          | String         | 必須                                                  | なし  | なし                                         | どの戦略を使用して最適なハイパーパラメータを探すか選択                                      |
| tuning_strategy_random_state                                  | Integer        | 選択                                                  | なし  | なし                                         | 乱数作成を決定。再現可能な結果のために固定された値で指定する。                                 |
| early_stopping_algorithm                                      | String         | 必須                                                  | なし  | EARLY_STOPPING_ALGORITHM.<br>MEDIAN          | 学習を継続してもモデルがそれ以上良くならない場合、早期に学習を終了                            |
| early_stopping_min_trial_count                                | Integer        | 必須                                                  | 3     | なし                                         | 中間値を計算する際に、いくつの学習から目標指標値を取得するか定義                              |
| early_stopping_start_step                                     | Integer        | 必須                                                  | 4     | なし                                         | 何番目の学習段階から早期終了を適用するか設定します。                                            |
| tag_list                                                      | Array          | 選択                                                  | なし  | 最大10個                                     | タグ情報                                                                    |
| tag_list[0].tagKey                                            | String         | 選択                                                  | なし  | 最大64文字                                     | タグキー                                                                     |
| tag_list[0].tagValue                                          | String         | 選択                                                  | なし  | 最大255文字                                    | タグ値                                                                     |
| use_log                                                       | Boolean        | 選択                                                  | False | True, False                                  | Log & Crash Searchサービスにログを残すかどうか                                               |
| wait                                                          | Boolean        | 選択                                                  | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
hyperparameter_tuning = easymaker.HyperparameterTuning().run(
    experiment_id=experiment.experiment_id, # Optional if already set in init
    hyperparameter_tuning_name='hyperparameter_tuning_name',
    description='hyperparameter_tuning_description',
    image_name='Ubuntu 18.04 CPU TensorFlow Training',
    instance_name='m2.c8m16',
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

### ハイパーパラメータチューニングの削除

[パラメータ]

| 名前                    | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明          |
|------------------------|---------|-------|------|--------|--------------|
| hyperparameter_tuning_id          | String  | 必須   | なし   | 最大36文字 | ハイパーパラメータチューニングID |

```python
easymaker.HyperparameterTuning(hyperparameter_tuning_id).delete()
```

## モデル

### モデル作成

学習ID値でモデルの作成をリクエストできます。
モデルはエンドポイント作成時に使用されます。

[パラメータ]

| 名前                      | タイプ    | 必須かどうか                             | デフォルト値 | 有効範囲  | 説明                                 |
|--------------------------|--------|------------------------------------|-----|---------|-------------------------------------|
| training_id              | String | hyperparameter_tuning_idがない場合は必須 | なし  | なし      | モデルとして作成する学習ID                       |
| hyperparameter_tuning_id | String | training_idがない場合は必須             | なし  | なし      | モデルとして作成するハイパーパラメータチューニングID(最高学習で作成済み) |
| model_name               | String | 必須                                | なし  | 最大50文字 | モデル名                              |
| description        | String | 選択                                | なし  | 最大255文字 | モデルの説明                          |
| tag_list                 | Array  | 選択                                | なし  | 最大10個 | タグ情報                              |
| tag_list[0].tagKey       | String | 選択                                | なし  | 最大64文字 | タグキー                                |
| tag_list[0].tagValue     | String | 選択                                | なし  | 最大255文字 | タグ値                               |

```python
model = easymaker.Model().create(
    training_id=training.training_id,  # or hyperparameter_tuning_id=hyperparameter_tuning.hyperparameter_tuning_id,
    model_name='model_name',
    description='model_description',
)
```

学習IDがなくても、モデルが保存されたパス情報とフレームワークの種類を入力してモデルを作成できます。

[パラメータ]

| 名前               | タイプ | 必須かどうか | デフォルト値 | 有効範囲                               | 説明                                              |
|----------------------|--------|-------|-----|-----------------------------------------|-----------------------------------------------------|
| model_type_code       | Enum   | 必須 | なし | easymaker.TENSORFLOW、 easymaker.PYTORCH | 学習に使用されたフレームワーク情報                                |
| model_upload_uri            | String | 必須 | なし | 最大255文字                             | モデルファイルパス(NHN Cloud Object StorageまたはNHN Cloud NAS) |
| model_name           | String | 必須 | なし | 最大50文字                              | モデル名                                           |
| description    | String | 任意 | なし | 最大255文字                             | モデルの説明                                       |
| parameter_list                   | Array  | 選択  | なし | 最大10個                                | パラメータ情報(parameterName/parameterValueで構成)         |
| parameter_list[0].parameterName  | String | 選択  | なし | 最大64文字                                | パラメータ名                                            |
| parameter_list[0].parameterValue | String | 選択  | なし | 最大255文字                               | パラメータ値                                              |
| tag_list             | Array  | 任意 | なし | 最大10個                              | タグ情報                                           |
| tag_list[0].tagKey   | String | 任意 | なし | 最大64文字                              | タグキー                                               |
| tag_list[0].tagValue | String | 任意 | なし | 最大255文字                             | タグ値                                            |

```python
# TensorFlowモデル
model = easymaker.Model().create_by_model_upload_uri(
    model_type_code=easymaker.TENSORFLOW,
    model_upload_uri='obs://kr1-api-object-storage.nhncloudservice.com/v1/AUTH_{tenant_id}/{container_name}/{model_upload_path}',
    model_name='model_name',
    description='model_description',
)
# HuggingFaceモデル
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

### モデル削除

[パラメータ]

| 名前                       | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明   |
|---------------------------|---------|-------|------|--------|-------|
| model_id | String  | 必須   | なし   | 最大36文字 | モデルID |

```python
easymaker.Model(model_id).delete()
```

## エンドポイント

### エンドポイントの作成

エンドポイント作成時に基本ステージが作成されます。

[パラメータ]

| 名前                                                        | タイプ    | 必須かどうか | デフォルト値 | 有効範囲                    | 説明                                                                   |
|-------------------------------------------------------------|---------|-------|-------|----------------------------|------------------------------------------------------------------------|
| endpoint_name                                               | String  | 必須  | なし  | 最大50文字                   | エンドポイント名                                                             |
| description                                        | String  | 選択  | なし  | 最大255文字                  | エンドポイントの説明                                                         |
| instance_name                                      | String  | 必須  | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                                |
| instance_count                                     | Integer | 選択  | 1     | 1~10                       | エンドポイントに使用されるインスタンス数                                                    |
| endpoint_model_resource_list                                | Array   | 必須  | なし  | 最大10個                   | ステージに使用されるリソース情報                                               |
| endpoint_model_resource_list[0].modelId                     | String   | 必須  | なし  | なし                     | ステージリソースで作成するモデルID                                   |
| endpoint_model_resource_list[0].resourceOptionDetail        | Object   | 必須  | なし  |                                  | ステージリソースの詳細情報               |
| endpoint_model_resource_list[0].resourceOptionDetail.cpu    | Double   | 必須  | なし  | 0.0~                             | ステージリソースに使用されるCPU                |
| endpoint_model_resource_list[0].resourceOptionDetail.memory | Object   | 必須  | なし  | 1Mi~                             | ステージリソースに使用されるメモリ           |
| endpoint_model_resource_list[0].podAutoScaleEnable          | Boolean  | 選択  | False   | True, False                      | ステージリソースに使用されるPodオートスケーラー |
| endpoint_model_resource_list[0].scaleMetricCode             | String   | 選択  | なし  | CONCURRENCY, REQUESTS_PER_SECOND | ステージリソースに使用される増設単位        |
| endpoint_model_resource_list[0].scaleMetricTarget           | Integer  | 選択  | なし  | 1~                               | ステージリソースに使用される増設しきい値   |
| endpoint_model_resource_list[0].description                 | String   | 選択  | なし  | 最大255文字                | ステージリソースの説明                                     |
| tag_list                                                    | Array   | 選択  | なし  | 最大10個                   | タグ情報                                                                |
| tag_list[0].tagKey                                          | String  | 選択  | なし  | 最大64文字                   | タグキー                                                                 |
| tag_list[0].tagValue                                        | String  | 選択  | なし  | 最大255文字                  | タグ値                                                                 |
| use_log                                                     | Boolean | 選択  | False | True, False                | Log & Crash Searchサービスにログを残すかどうか                                           |
| wait                                                        | Boolean | 選択  | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
endpoint = easymaker.Endpoint().create(
    endpoint_name='endpoint_name',
    description='endpoint_description',
    instance_name='c2.c16m16',
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

### ステージの追加

既存エンドポイントに新規ステージを追加できます。

[パラメータ]

| 名前                                                        | タイプ    | 必須かどうか | デフォルト値 | 有効範囲                    | 説明                                                               |
|-------------------------------------------------------------|---------|-------|-------|----------------------------|--------------------------------------------------------------------|
| endpoint_id                                                 | String  | 必須  | なし | 最大36文字                    | エンドポイントID                                                            |
| stage_name                                                  | String  | 必須  | なし  | 最大50文字                   | ステージ名                                                          |
| description                                           | String  | 選択  | なし  | 最大255文字                  | ステージの説明                                                      |
| instance_name                                      | String  | 必須  | なし  | なし                       | エンドポイントに使用されるインスタンスタイプ名                                            |
| instance_count                                     | Integer | 選択  | 1     | 1~10                       | エンドポイントに使用されるインスタンス数                                                |
| endpoint_model_resource_list                                | Array   | 必須  | なし  | 最大10個                   | ステージに使用されるリソース情報                                               |
| endpoint_model_resource_list[0].modelId                     | String   | 必須  | なし  | なし                     | ステージリソースで作成するモデルID                                   |
| endpoint_model_resource_list[0].resourceOptionDetail        | Object   | 必須  | なし  |                                  | ステージリソースの詳細情報               |
| endpoint_model_resource_list[0].resourceOptionDetail.cpu    | Double   | 必須  | なし  | 0.0~                             | ステージリソースに使用されるCPU                |
| endpoint_model_resource_list[0].resourceOptionDetail.memory | Object   | 必須  | なし  | 1Mi~                             | ステージリソースに使用されるメモリ           |
| endpoint_model_resource_list[0].podAutoScaleEnable          | Boolean  | 選択  | False   | True, False                      | ステージリソースに使用されるPodオートスケーラー |
| endpoint_model_resource_list[0].scaleMetricCode             | String   | 選択  | なし  | CONCURRENCY, REQUESTS_PER_SECOND | ステージリソースに使用される増設単位        |
| endpoint_model_resource_list[0].scaleMetricTarget           | Integer  | 選択  | なし  | 1~                               | ステージリソースに使用される増設しきい値   |
| endpoint_model_resource_list[0].description                 | String   | 選択  | なし  | 最大255文字                | ステージリソースの説明                                     |
| tag_list                                                    | Array   | 選択  | なし  | 最大10個                   | タグ情報                                                            |
| tag_list[0].tagKey                                          | String  | 選択  | なし  | 最大64文字                   | タグキー                                                             |
| tag_list[0].tagValue                                        | String  | 選択  | なし  | 最大255文字                  | タグ値                                                             |
| use_log                                                     | Boolean | 選択  | False | True, False                | Log & Crash Searchサービスにログを残すかどうか                                       |
| wait                                                        | Boolean | 選択  | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
endpoint_stage = easymaker.EndpointStage().create(
    endpoint_id=endpoint.endpoint_id,
    stage_name='stage01',  # 30文字以内小文字/数字
    description='test endpoint',
    instance_name='c2.c16m16',
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

### ステージリスト照会

エンドポイントステージリストを照会します。

```python
endpoint_stage_list = easymaker.Endpoint(endpoint_id).get_stage_list()
```

### エンドポイントインファレンス

基本ステージにインファレンス

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.Endpoint('endpoint_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

特定ステージを指定してインファレンス

```python
input_data = [6.0, 3.4, 4.5, 1.6]
easymaker.EndpointStage('endpoint_stage_id').predict(
    model_id=model_id,
    json={'instances': [input_data]},
)
```

### エンドポイントの削除

[パラメータ]

| 名前           | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明      |
|---------------|---------|-------|------|--------|----------|
| endpoint_id   | String  | 必須   | なし   | 最大36文字 | エンドポイントID |

```python
easymaker.Endpoint(endpoint_id).delete()
```

### エンドポイントステージの削除

[パラメータ]

| 名前        | タイプ     | 必須かどうか | デフォルト値 | 有効範囲 | 説明     |
|------------|---------|-------|------|--------|---------|
| stage_id   | String  | 必須   | なし   | 最大36文字 | ステージID |

```python
easymaker.EndpointStage(stage_id).delete()
```

## バッチ推論

### バッチ推論の作成

[パラメータ]

| 名前                     | タイプ   | 必須かどうか | デフォルト値 | 有効範囲  | 説明                                                                                 |
| ------------------------- | ------- | --------- | ------ | ----------- | ------------------------------------------------------------------------------------- |
| batch_inference_name      | String  | 必須     | なし   | 最大50文字  | バッチ推論名前                                                                       |
| instance_count            | Integer | 必須     | なし   | 1～10        | バッチ推論に使用するインスタンス数                                                       |
| timeout_hours             | Integer | 選択     | 720    | 1～720       | 最大バッチ推論時間(単位:時間)                                                       |
| instance_name             | String  | 必須     | なし   | なし        | インスタンスタイプ名(CLIで照会可能)                                                   |
| model_name                | String  | 必須     | なし   | なし        | モデル名(CLIで照会可能)                                                            |
| pod_count                 | Integer | 必須     | なし   | 1～100       | 分散学習を適用するノード数                                                           |
| batch_size                | Integer | 必須     | なし   | 1～1000      | 同時に処理されるデータサンプルの数                                                      |
| inference_timeout_seconds | Integer | 必須     | なし   | 1～1200      | 単一推論リクエストの最大許容時間                                                      |
| input_data_uri            | String  | 必須     | なし   | 最大255文字 | 入力データファイルのパス(NHN Cloud Object StorageまたはNHN Cloud NAS)                    |
| input_data_type           | String  | 必須     | なし   | JSON, JSONL | 入力データのタイプ                                                                   |
| include_glob_pattern      | String  | 選択     | なし   | 最大255文字 | ファイルセットを入力データに含めるGlobパターン                                         |
| exclude_glob_pattern      | String  | 選択     | なし   | 最大255文字 | ファイルセットを入力データから除外するGlobパターン                                         |
| output_upload_uri         | String  | 必須     | なし   | 最大255文字 | バッチ推論結果ファイルがアップロードされるパス(NHN Cloud Object StorageまたはNHN Cloud NAS)      |
| data_storage_size         | Integer | 必須     | なし   | 300～10000   | バッチ推論に必要なデータをダウンロードする記憶領域のサイズ(単位: GB)                       |
| description               | String  | 選択     | なし   | 最大255文字 | バッチ推論の説明                                                                |
| tag_list                  | Array   | 選択     | なし   | 最大10個  | タグ情報                                                                            |
| tag_list[0].tagKey        | String  | 選択     | なし   | 最大64文字  | タグキー                                                                               |
| tag_list[0].tagValue      | String  | 選択     | なし   | 最大255文字 | タグ値                                                                              |
| use_log                   | Boolean | 選択     | False  | True, False | Log & Crash Searchサービスにログを残すかどうか                                        |
| wait                      | Boolean | 選択    | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
batch_inference = easymaker.BatchInference().run(
    batch_inference_name='batch_inference_name',
    instance_count=1,
    timeout_hours=100,
    instance_name='m2.c4m8',
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

### バッチ推論削除

[パラメータ]

| 名前              | タイプ  | 必須かどうか | デフォルト値 | 有効範囲 | 説明        |
| ------------------ | ------ | --------- | ------ | --------- | ------------ |
| batch_inference_id | String | 必須     | なし   | 最大36文字 | バッチ推論ID |

```python
easymaker.BatchInference(batch_inference_id).delete()
```

## パイプライン

### パイプライン作成

[パラメータ]

| 名前                        | タイプ    | 必須かどうか | デフォルト値 | 有効範囲 | 説明                                      |
|-----------------------------|---------| --------- | ------ | --------- |-------------------------------------------|
| pipeline_name               | String  | 必須    | なし | 最大50文字 | パイプライン名                                |
| pipeline_spec_manifest_path | String  | 必須    | なし | 1~10      | アップロードするパイプラインファイルパス                        |
| description                 | String  | 選択    | なし | 最大255文字 | パイプラインの説明                            |
| tag_list                    | Array   | 選択    | なし | 最大10個 | タグ情報                                   |
| tag_list[0].tagKey          | String  | 選択    | なし | 最大64文字 | タグキー                                    |
| tag_list[0].tagValue        | String  | 選択    | なし | 最大255文字 | タグ値                                    |
| wait                        | Boolean | 選択    | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
pipeline = easymaker.Pipeline().upload(
    pipeline_name='pipeline_01',
    pipeline_spec_manifest_path='./sample-pipeline.yaml',
    description='test',
    tag_list=[],
    # wait=False,
)
```

### パイプライン削除

[パラメータ]

| 名前             | タイプ | 必須かどうか | デフォルト値 | 有効範囲 | 説明     |
| ------------------ | ------ | --------- | ------ | --------- |----------|
| pipeline_id | String | 必須    | なし | 最大36文字 | パイプラインID |

```python
easymaker.Pipeline(pipeline_id).delete()
```

### パイプライン実行作成

[パラメータ]

| 名前                             | タイプ    | 必須かどうか                   | デフォルト値 | 有効範囲     | 説明                                     |
|----------------------------------|---------|---------------------------| ------ |-------------|------------------------------------------|
| pipeline_run_name                | String  | 必須                      | なし | 最大50文字    | パイプライン実行名                            |
| pipeline_id                      | String  | 必須                      | なし | 最大36文字    | パイプラインスケジュール名                            |
| experiment_id                    | String  | easymaker.initで未入力の場合は必須 | なし  | 最大36文字    | 実験ID                                    |
| description                      | String  | 選択                      | なし | 最大255文字   | パイプライン実行の説明                        |
| instance_name                    | String  | 必須                      | なし | なし        | インスタンスタイプ名(CLIで照会可能)                   |
| instance_count                   | Integer | 必須                      | なし | 1~10        | 使用するインスタンス数                             |
| boot_storage_size                | Integer | 必須                      | なし | 50~         | パイプラインを実行するインスタンスのブートストレージサイズ(単位: GB)      |
| parameter_list                   | Array   | 選択                      | なし | なし        | パイプラインに伝達するパラメータ情報                     |
| parameter_list[0].parameterKey   | String  | 選択                      | なし | 最大255文字   | パラメータキー                                  |
| parameter_list[0].parameterValue | String  | 選択                      | なし | 最大1000文字  | パラメータ値                                 |
| nas_list                         | Array   | 選択                      | なし | 最大10個    | NAS情報                                 |
| nas_list[0].mountDirName         | String  | 選択                      | なし | 最大64文字    | インスタンスにマウントするディレクトリ名                     |
| nas_list[0].nasUri               | String  | 選択                      | なし | 最大255文字   | `nas://{NAS ID}:/{path}`形式のNASパス    |
| tag_list                         | Array   | 選択                      | なし | 最大10個    | タグ情報                                  |
| tag_list[0].tagKey               | String  | 選択                      | なし | 最大64文字    | タグキー                                   |
| tag_list[0].tagValue             | String  | 選択                      | なし | 最大255文字   | タグ値                                   |
| wait                             | Boolean | 選択                      | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す |

```python
pipeline_run = easymaker.PipelineRun().create(
    pipeline_run_name='pipeline_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_name='m2.c4m8',
    instance_count=1,
    boot_storage_size=50,
    # wait=False,
)
```

### パイプライン実行削除

[パラメータ]

| 名前             | タイプ | 必須かどうか | デフォルト値 | 有効範囲 | 説明        |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_run_id | String | 必須    | なし | 最大36文字 | パイプライン実行ID |

```python
easymaker.PipelineRun(pipeline_run_id).delete()
```

### パイプラインスケジュール作成

[パラメータ]

| 名前                             | タイプ    | 必須かどうか                            | デフォルト値 | 有効範囲     | 説明                                           |
|----------------------------------|---------|------------------------------------| ------ |-------------|------------------------------------------------|
| pipeline_recurring_run_name      | String  | 必須                               | なし | 最大50文字    | パイプラインスケジュール名                                  |
| pipeline_id                      | String  | 必須                               | なし | 最大36文字    | パイプラインスケジュール名                                  |
| experiment_id                    | String  | easymaker.initで未入力の場合は必須        | なし  | 最大36文字    | 実験ID                                          |
| description                      | String  | 選択                               | なし | 最大255文字   | パイプラインスケジュールの説明                              |
| instance_name                    | String  | 必須                               | なし | なし        | インスタンスタイプ名(CLIで照会可能)                         |
| instance_count                   | Integer | 必須                               | なし | 1~10        | 使用するインスタンス数                                   |
| boot_storage_size                | Integer | 必須                               | なし | 50~         | パイプラインを実行するインスタンスのブートストレージサイズ(単位: GB)            |
| schedule_periodic_minutes        | String  | schedule_cron_expression未入力の場合は必須 | なし | なし        | パイプラインを繰り返し実行する時間周期設定                       |
| schedule_cron_expression         | String  | schedule_periodic_minutes未入力の場合は必須 | なし | なし        | パイプラインを繰り返し実行するCron式設定                    |
| max_concurrency_count            | String  | 選択                               | なし | なし        | 同時実行最大数を指定して並列で実行される数を制限           |
| schedule_start_datetime          | String  | 選択                               | なし | なし        | パイプラインスケジュールの開始時間を設定、未入力の場合、設定した周期に合わせてパイプライン実行 |
| schedule_end_datetime            | String  | 選択                               | なし | なし        | パイプラインスケジュールの終了時間を設定、未入力時、停止するまでのパイプライン実行を作成 || use_catchup                      | Boolean | 選択                               | なし | なし        | 欠陥実行のキャッチアップ：パイプライン実行がスケジュールに遅れた場合、追いつくかどうかを選択 |
| parameter_list                   | Array   | 選択                               | なし | なし        | パイプラインに伝達するパラメータ情報                           |
| parameter_list[0].parameterKey   | String  | 選択                               | なし | 最大255文字   | パラメータキー                                        |
| parameter_list[0].parameterValue | String  | 選択                               | なし | 最大1000文字  | パラメータ値                                       |
| nas_list                         | Array   | 選択                               | なし | 最大10個    | NAS情報                                       |
| nas_list[0].mountDirName         | String  | 選択                               | なし | 最大64文字    | インスタンスにマウントするディレクトリ名                           |
| nas_list[0].nasUri               | String  | 選択                               | なし | 最大255文字   | `nas://{NAS ID}:/{path}`形式のNASパス          |
| tag_list                         | Array   | 選択                               | なし | 最大10個    | タグ情報                                        |
| tag_list[0].tagKey               | String  | 選択                               | なし | 最大64文字    | タグキー                                         |
| tag_list[0].tagValue             | String  | 選択                               | なし | 最大255文字   | タグ値                                         |
| wait                             | Boolean | 選択                               | True   | True, False | True：作成が完了した後に返す、False：作成リクエスト後、すぐに返す     |

```python
pipeline_recurring_run = easymaker.PipelineRecurringRun().create(
    pipeline_recurring_run_name='pipeline_recurring_run',
    description='test',
    pipeline_id=pipeline.pipeline_id,
    experiment_id=experiment.experiment_id, # Optional if already set in init
    instance_name='m2.c4m8',
    boot_storage_size=50,
    schedule_cron_expression='0 0 * * * ?',
    max_concurrency_count=1,
    schedule_start_datetime='2025-01-01T00:00:00+09:00'
    # wait=False,
)
```

### パイプラインスケジュールの停止/再起動

[パラメータ]

| 名前             | タイプ | 必須かどうか | デフォルト値 | 有効範囲 | 説明        |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | 必須    | なし | 最大36文字 | パイプラインスケジュールID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).stop()
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).start()
```

### パイプラインスケジュールの削除

[パラメータ]

| 名前             | タイプ | 必須かどうか | デフォルト値 | 有効範囲 | 説明        |
| ------------------ | ------ | --------- | ------ | --------- |-------------|
| pipeline_recurring_run_id | String | 必須    | なし | 最大36文字 | パイプラインスケジュールID |

```python
easymaker.PipelineRecurringRun(pipeline_recurring_run_id).delete()
```

## その他機能

### NHN Cloud - Log & Crash Searchログ転送

```python
easymaker_logger = easymaker.logger(logncrash_appkey='log&crash_product_app_key')
easymaker_logger.send('test log meassage')  # Output to stdout & send log to log&crash product
easymaker_logger.send(log_message='log meassage',
                      log_level='ERROR',  # default：INFO
                      project_version='2.0.0',  # default：1.0.0
                      parameters={'serviceType'：'EasyMakerSample'})  # Add custom parameters
```

### NHN Cloud - Object Storageファイル転送

Object Storage商品にファイルをアップロードし、ダウンロードする機能を提供します。

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
