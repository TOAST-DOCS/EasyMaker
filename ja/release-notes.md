<!-- pre-align:aligned sig=287622970352 -->

<a id="ai.easymaker"></a>
## Machine Learning > AI EasyMaker > リリースノート { #ai.easymaker }

<a id="ai.easymaker.release.notes.2026.07.28"></a>
### 2026. 07. 28. { #ai.easymaker.release.notes.2026.07.28 }

<a id="ai.easymaker.release.notes.2026.07.28.feature.change"></a>
#### 機能改善・変更

- ファインチューニング機能の追加
    - 事前学習された大規模言語モデルに、特定のドメインやタスクに合わせたデータセットで追加学習を実行し、モデルの性能を特化させることができます。
    - 詳細については、[ファインチューニングガイド](./console-guide/#fine.tuning)のドキュメントを参照してください。

{% if "gov" in build_flags -%}
<a id="ai.easymaker.release.notes.2025.11.25"></a>
### 2025. 11. 25. { #ai.easymaker.release.notes.2025.11.25 }

<a id="ai.easymaker.release.notes.2025.11.25.service"></a>
#### 新規サービスリリース

- AI EasyMakerは、機械学習の開発のための環境と学習および高度化、エンドポイントサービスのためのAIプラットフォームです。
{% endif %}
{% if "gov" not in build_flags -%}
<a id="ai.easymaker.release.notes.2025.10.28"></a>
### 2025. 10. 28. { #ai.easymaker.release.notes.2025.10.28 }

- RAG(Retrieval-Augmented Generation、検索拡張生成)機能の追加
    - LLMの応答精度を向上させるRAG機能が追加されました。

- NVIDIA Triton Inference Serverのサポート
    - Triton形式のモデルを作成し、デプロイできる機能が追加されました。

<a id="ai.easymaker.release.notes.2025.06.24"></a>
### 2025. 06. 24. { #ai.easymaker.release.notes.2025.06.24 }

<a id="ai.easymaker.release.notes.2025.06.24.feature.change"></a>
#### 機能改善・変更

- モデル評価機能追加
    - モデルの性能を測定し、比較できます。

<a id="ai.easymaker.release.notes.2024.10.29"></a>
### 2024. 10. 29. { #ai.easymaker.release.notes.2024.10.29 }

- エンドポイント機能改善
    - リソース割り当て値を直接設定できるようにサポートします。
- Hugging Faceモデルのサービングをサポート
    - Hugging FaceモデルをAI EasyMakerに登録してエンドポイント、バッチ推論でサービングできるようにサポートします。

<a id="ai.easymaker.release.notes.2024.07.23"></a>
### 2024. 07. 23. { #ai.easymaker.release.notes.2024.07.23 }

<a id="ai.easymaker.release.notes.2024.07.23.feature.change"></a>
#### 機能改善・変更

- MLパイプライン機能追加
    - MLパイプラインは、移植可能で拡張可能な機械学習ワークフローを管理・実行するための機能です。
    - 詳細は[MLパイプラインガイド](./console-guide/#pipeline)を参照してください。
- エンドポイント機能の改善
    - リソースの割り当てを最適化しました。
- PyTorch no-archiveモデルサービングをサポート
    - PyTorch no-archiveモデルをAI EasyMakerに登録し、エンドポイントとして提供できるようにサポートします。

<a id="ai.easymaker.release.notes.2024.05.10"></a>
### 2024. 05. 10. { #ai.easymaker.release.notes.2024.05.10 }

<a id="ai.easymaker.release.notes.2024.05.10.feature.change"></a>
#### 機能改善・変更

- ノートパソコン再起動機能を追加

<a id="ai.easymaker.release.notes.2024.04.23"></a>
### 2024. 04. 23. { #ai.easymaker.release.notes.2024.04.23 }

<a id="ai.easymaker.release.notes.2024.04.23.feature.change"></a>
#### 機能改善・変更

- バッチ推論機能を追加
    - AI EasyMakerのモデルでバッチ推論し、推論結果を統計で確認できる環境を提供します。
    - 詳細は、[バッチ推論ガイド](./console-guide/#batch.inference)文書を参照してください。
- リソース検索機能を追加
    - コンソール画面でリソースを検索し、リンクを介して他のリソース画面に移動できます。
- ノートブックNAS変更機能を追加
    - 実行中のノートブックのNHN Cloud NAS接続設定を変更できます。
- Scikit-learnサービングをサポート
    - Scikit-learnモデルをAI EasyMakerに登録し、エンドポイントとして提供できるようにサポートします。
- ノートブックのShared Memoryを有効化
    - 64MB以上のshared memoryを使用できるようにしました。
    - サイズはノートブック作成時に選択したインスタンスタイプによって異なります。
- NHN Cloud提供アルゴリズムからsave_stepsハイパーパラメータを削除
    - チェックポイント保存関連のハイパーパラメータsave_stepsを削除しました。
    - 適切なsave_stepsの数値をアルゴリズム内で自動的に計算し、最大3つまで保存します。

<a id="ai.easymaker.release.notes.2023.12.19"></a>
### 2023. 12. 19. { #ai.easymaker.release.notes.2023.12.19 }

<a id="ai.easymaker.release.notes.2023.12.19.feature.change"></a>
#### 機能改善・変更

- 個人イメージを利用したノートパソコン、学習
    - ユーザーがパーソナライズされたコンテナイメージを利用してノートパソコン、学習、ハイパーパラメータチューニングを駆動できます。
    - 個人イメージとレジストリアカウントを登録すると、簡単に個人イメージを選択してリソースを作成できます。

- ダッシュボード
    - リソース全体の利用状況、Top3エンドポイントサービスモニタリング、Top3 CPU/GPU使用率を一ページで確認できます。

- エンドポイント > オートスケーラー
    - エンドポイントノードの増設/縮小ポリシーを設定してノード数を動的に管理できます。

<a id="ai.easymaker.release.notes.2023.09.26"></a>
### 2023. 09. 26. { #ai.easymaker.release.notes.2023.09.26 }

<a id="ai.easymaker.release.notes.2023.09.26.feature.change"></a>
#### 機能改善・変更

- Ubuntu 22.04バージョン提供
    - 新規Ubuntu 22.04バージョンを提供します。Ubuntu 18.04バージョンはこれ以上提供されません。利用していた顧客は現在と同じようにサービスを利用できます。

- モニタリング機能提供
    - ノートパソコン、学習、エンドポイントのシステムモニタリング指標を確認できます。
    - エンドポイントで各APIリソースパスのAPI呼び出し指標を確認できます。

- 基本アルゴリズムハイパーパラメータチューニングをサポート
    - ハイパーパラメータチューニングによりAI EasyMakerが提供する基本アルゴリズムのハイパーパラメータを最適化できます。

- エンドポイント > 複数のモデルサービングをサポート
    - 1つのエンドポイントステージに複数の学習モデルをサービングできます。

- ハイパーパラメータチューニング並列学習をサポート
    - 並列学習数を調整してハイパーパラメータチューニング作業の性能を最適化できます。

<a id="ai.easymaker.release.notes.2023.06.27"></a>
### 2023. 06. 27. { #ai.easymaker.release.notes.2023.06.27 }

<a id="ai.easymaker.release.notes.2023.06.27.feature.change"></a>
#### 機能改善・変更

- ハイパーパラメータチューニング機能を追加
    - ハイパーパラメータチューニングは、マシンラーニングモデルの予測精度と性能を高めるために最適化されたハイパーパラメータを見つけられるように実験を自動化する機能です。
    - 詳細は[ハイパーパラメータチューニングガイド](./console-guide/#hyperparameter.tuning)文書を参照してください。
- NHN Cloud AI EasyMakerが提供する基本アルゴリズムを3種追加
    - 詳細については、各アルゴリズムのガイド文書を参照してください。
    - [Image Classificationガイド](./algorithm-guide/#image.classification)
    - [Object Detectionガイド](./algorithm-guide/#object.detection)
    - [Semantic Segmentationガイド](./algorithm-guide/#semantic.segmentation)

<a id="ai.easymaker.release.notes.2022.12.27"></a>
### 2022. 12. 27. { #ai.easymaker.release.notes.2022.12.27 }

<a id="ai.easymaker.release.notes.2022.12.27.service"></a>
#### 新規サービスリリース

- AI EasyMakerは、機械学習の開発のための環境と学習および高度化、エンドポイントサービスのためのAIプラットフォームです。
{% endif %}
