<!-- pre-align:aligned sig=fbf376f81e3f -->

<a id="ai.easymaker"></a>
## Machine Learning > AI EasyMaker > 릴리스 노트 { #ai.easymaker }

<a id="ai.easymaker.release.notes.2026.07.28"></a>
### 2026. 07. 28. { #ai.easymaker.release.notes.2026.07.28 }

<a id="ai.easymaker.release.notes.2026.07.28.feature.change"></a>
#### 기능 개선/변경

- 파인 튜닝 기능 추가
    - 사전 학습된 거대 언어 모델에 특정 도메인이나 작업에 맞춘 데이터 세트로 추가 학습을 수행하여 모델의 성능을 특화할 수 있습니다.
    - 자세한 내용은 [파인 튜닝 가이드](./console-guide/#fine.tuning) 문서를 참고하세요.

{% if "gov" in build_flags -%}
<a id="ai.easymaker.release.notes.2025.11.25"></a>
### 2025. 11. 25. { #ai.easymaker.release.notes.2025.11.25 }

<a id="ai.easymaker.release.notes.2025.11.25.service"></a>
#### 신규 서비스 출시

- AI EasyMaker는 머신 러닝 개발을 위한 환경과 학습 및 고도화, 엔드포인트 서비스를 위한 AI 플랫폼입니다.
{% endif %}
{% if "gov" not in build_flags -%}
<a id="ai.easymaker.release.notes.2025.10.28"></a>
### 2025. 10. 28. { #ai.easymaker.release.notes.2025.10.28 }

- RAG(Retrieval-Augmented Generation, 검색 증강 생성) 기능 추가
    - LLM의 응답 정확도를 향상시키는 RAG 기능이 추가되었습니다.

- NVIDIA Triton Inference Server 지원
    - Triton 형식의 모델을 생성하고 배포할 수 있는 기능이 추가되었습니다.

<a id="ai.easymaker.release.notes.2025.06.24"></a>
### 2025. 06. 24. { #ai.easymaker.release.notes.2025.06.24 }

<a id="ai.easymaker.release.notes.2025.06.24.feature.change"></a>
#### 기능 개선/변경

- 모델 평가 기능 추가
    - 모델의 성능을 측정하고 비교할 수 있습니다.

<a id="ai.easymaker.release.notes.2024.10.29"></a>
### 2024. 10. 29. { #ai.easymaker.release.notes.2024.10.29 }

- 엔드포인트 기능 개선
    - 리소스 할당 값을 직접 설정할 수 있도록 지원합니다.
- Hugging Face 모델 서빙 지원
    - Hugging Face 모델을 AI EasyMaker에 등록하여 엔드포인트, 배치 추론으로 서빙할 수 있도록 지원합니다.

<a id="ai.easymaker.release.notes.2024.07.23"></a>
### 2024. 07. 23. { #ai.easymaker.release.notes.2024.07.23 }

<a id="ai.easymaker.release.notes.2024.07.23.feature.change"></a>
#### 기능 개선/변경

- ML 파이프라인 기능 추가
    - ML 파이프라인은 이식 가능하고 확장 가능한 기계 학습 워크플로우를 관리하고 실행하기 위한 기능입니다.
    - 자세한 내용은 [ML 파이프라인 가이드](./console-guide/#pipeline) 문서를 참고하세요.
- 엔드포인트 기능 개선
    - 리소스 할당을 최적화했습니다.
- PyTorch no-archive 모델 서빙 지원
    - PyTorch no-archive 모델을 AI EasyMaker에 등록하여 엔드포인트로 서빙할 수 있도록 지원합니다.

<a id="ai.easymaker.release.notes.2024.05.10"></a>
### 2024. 05. 10. { #ai.easymaker.release.notes.2024.05.10 }

<a id="ai.easymaker.release.notes.2024.05.10.feature.change"></a>
#### 기능 개선/변경

- 노트북 재부팅 기능 추가

<a id="ai.easymaker.release.notes.2024.04.23"></a>
### 2024. 04. 23. { #ai.easymaker.release.notes.2024.04.23 }

<a id="ai.easymaker.release.notes.2024.04.23.feature.change"></a>
#### 기능 개선/변경

- 배치 추론 기능 추가
    - AI EasyMaker의 모델로 배치 추론하고 추론 결과를 통계로 확인할 수 있는 환경을 제공합니다.
    - 자세한 내용은 [배치 추론 가이드](./console-guide/#batch.inference) 문서를 참고하세요.
- 리소스 검색 기능 추가
    - 콘솔 화면에서 리소스를 검색하고, 링크를 통해서 다른 리소스 화면으로 이동할 수 있습니다.
- 노트북 NAS 변경 기능 추가
    - 실행 중인 노트북의 NHN Cloud NAS 연결 설정을 변경할 수 있습니다.
- Scikit-learn 서빙 지원
    - Scikit-learn 모델을 AI EasyMaker에 등록하여 엔드포인트로 서빙할 수 있도록 지원합니다.
- 노트북 Shared Memory 활성화
    - 64MB 이상의 shared memory를 사용할 수 있도록 활성화했습니다.
    - 크기는 노트북 생성 시 선택한 인스턴스 타입에 따라 다릅니다.
- NHN Cloud 제공 알고리즘에서 save_steps 하이퍼파라미터 제거
    - 체크포인트 저장 관련 하이퍼파라미터 save_steps를 제거했습니다.
    - 적절한 save_steps 수치를 알고리즘 내에서 자동으로 계산하고, 최대 3개 저장합니다.

<a id="ai.easymaker.release.notes.2023.12.19"></a>
### 2023. 12. 19. { #ai.easymaker.release.notes.2023.12.19 }

<a id="ai.easymaker.release.notes.2023.12.19.feature.change"></a>
#### 기능 개선/변경

- 개인 이미지를 이용한 노트북, 학습
    - 사용자가 개인화한 컨테이너 이미지를 이용하여 노트북, 학습, 하이퍼파라미터 튜닝을 구동할 수 있습니다.
    - 개인 이미지와 레지스트리 계정을 등록하면 손쉽게 개인 이미지를 선택하여 리소스를 생성할 수 있습니다.

- 대시보드
    - 전체 리소스의 이용 현황, Top3 엔드포인트 서비스 모니터링, Top3 CPU/GPU 사용률을 한 페이지에서 확인할 수 있습니다.

- 엔드포인트 > 오토 스케일러
    - 엔드포인트 노드의 증설/감축 정책을 설정하여 노드 수를 동적으로 관리할 수 있습니다.

<a id="ai.easymaker.release.notes.2023.09.26"></a>
### 2023. 09. 26. { #ai.easymaker.release.notes.2023.09.26 }

<a id="ai.easymaker.release.notes.2023.09.26.feature.change"></a>
#### 기능 개선/변경

- Ubuntu 22.04 버전 제공
    - 신규 Ubuntu 22.04 버전을 제공합니다. Ubuntu 18.04 버전은 더 이상 제공되지 않으며, 이용하던 고객은 현재와 동일하게 서비스를 이용할 수 있습니다.

- 모니터링 기능 제공
    - 노트북, 학습, 엔드포인트의 시스템 모니터링 지표를 확인할 수 있습니다.
    - 엔드포인트에서 각 API 리소스 경로에 대한 API 호출 지표를 확인할 수 있습니다.

- 기본 알고리즘 하이퍼파라미터 튜닝 지원
    - 하이퍼파라미터 튜닝을 통해 AI EasyMaker에서 제공하는 기본 알고리즘의 하이퍼파라미터를 최적화할 수 있습니다.

- 엔드포인트 > 복수 개의 모델 서빙 지원
    - 하나의 엔드포인트 스테이지에 여러 개의 학습 모델을 서빙할 수 있습니다.

- 하이퍼파라미터 튜닝 병렬 학습 지원
    - 병렬 학습 수를 조정하여 하이퍼파라미터 튜닝 작업의 성능을 최적화할 수 있습니다.

<a id="ai.easymaker.release.notes.2023.06.27"></a>
### 2023. 06. 27. { #ai.easymaker.release.notes.2023.06.27 }

<a id="ai.easymaker.release.notes.2023.06.27.feature.change"></a>
#### 기능 개선/변경

- 하이퍼파라미터 튜닝 기능 추가
    - 하이퍼파라미터 튜닝은 머신러닝 모델의 예측 정확도와 성능을 높이기 위해 최적화된 하이퍼파라미터를 찾아 낼 수 있도록 반복 실험을 자동화하는 기능입니다.
    - 자세한 내용은 [하이퍼파라미터 튜닝 가이드](./console-guide/#hyperparameter.tuning) 문서를 참고하세요.
- NHN Cloud AI EasyMaker에서 제공하는 기본 알고리즘 3종 추가
    - 자세한 내용은 각 알고리즘의 가이드 문서를 참고하세요.
    - [Image Classification 가이드](./algorithm-guide/#image.classification)
    - [Object Detection 가이드](./algorithm-guide/#object.detection)
    - [Semantic Segmentation 가이드](./algorithm-guide/#semantic.segmentation)

<a id="ai.easymaker.release.notes.2022.12.27"></a>
### 2022. 12. 27. { #ai.easymaker.release.notes.2022.12.27 }

<a id="ai.easymaker.release.notes.2022.12.27.service"></a>
#### 신규 서비스 출시

- AI EasyMaker는 머신 러닝 개발을 위한 환경과 학습 및 고도화, 엔드포인트 서비스를 위한 AI 플랫폼입니다.
{% endif %}
