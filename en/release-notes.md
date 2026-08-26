<!-- machine_translated: true -->

<!-- pre-align:aligned sig=287622970352 -->

<a id="ai.easymaker"></a>
## Machine Learning > AI EasyMaker > Release Notes { #ai.easymaker }

<a id="ai.easymaker.release.notes.2026.07.28"></a>
### July 28, 2026 { #ai.easymaker.release.notes.2026.07.28 }

<a id="ai.easymaker.release.notes.2026.07.28.feature.change"></a>
#### Feature Updates

- Added fine tuning feature
    - Specializes model performance by performing additional training on a pre-trained large language model using a dataset tailored to a specific domain or task.
    - For more information, see the [Fine Tuning Guide](./console-guide/#fine.tuning) document.

{% if "gov" in build_flags -%}

<a id="ai.easymaker.release.notes.2025.11.25"></a>
### November 25, 2025 { #ai.easymaker.release.notes.2025.11.25 }

<a id="ai.easymaker.release.notes.2025.11.25.service"></a>
#### New Service Launch

- AI EasyMaker is an AI platform that provides a machine learning development environment, training and advancement, and endpoint services.
{% endif %}
{% if "gov" not in build_flags -%}
<a id="ai.easymaker.release.notes.2025.10.28"></a>
### October 28, 2025 { #ai.easymaker.release.notes.2025.10.28 }

- Added the retrieval-augmented generation (RAG) feature
    - Added the RAG feature to improve LLM's response accuracy.

- Support for NVIDIA Triton Inference Server
    - Added the feature to allow users to generate and deploy Triton-format models.

<a id="ai.easymaker.release.notes.2025.06.24"></a>
### June 24, 2025 { #ai.easymaker.release.notes.2025.06.24 }

<a id="ai.easymaker.release.notes.2025.06.24.feature.change"></a>
#### Feature Updates

- Added Model Evaluation feature
    - You can measure and compare the performance of models.

<a id="ai.easymaker.release.notes.2024.10.29"></a>
### October 29, 2024 { #ai.easymaker.release.notes.2024.10.29 }

- Improved endpoint feature
    - Improved so that you can set resource allocation values directly.
- Added support for Hugging Face model
    - Added the Hugging Face model to AI EasyMaker to serve as an endpoint, batch inference.

<a id="ai.easymaker.release.notes.2024.07.23"></a>
### July 23, 2024 { #ai.easymaker.release.notes.2024.07.23 }

<a id="ai.easymaker.release.notes.2024.07.23.feature.change"></a>
#### Feature Updates

- Added ML pipeline feature
    - ML Pipeline is a feature for managing and executing portable and scalable machine learning workflows.
    - For more information, see the ML pipeline guide documentation.
- Improved endpoint functionality
    - Optimized resource allocation.
- Support for serving PyTorch no-archive model
    - Enabled PyTorch no-archive models to be registered in AI EasyMaker to serve as endpoints.

<a id="ai.easymaker.release.notes.2024.05.10"></a>
### May 10, 2024 { #ai.easymaker.release.notes.2024.05.10 }

<a id="ai.easymaker.release.notes.2024.05.10.feature.change"></a>
#### Feature Updates

- Added the feature to reboot notebooks

<a id="ai.easymaker.release.notes.2024.04.23"></a>
### April 23, 2024 { #ai.easymaker.release.notes.2024.04.23 }

<a id="ai.easymaker.release.notes.2024.04.23.feature.change"></a>
#### Feature Updates

- Added the batch inference feature
    - Provides an environment to make batch inferences from an AI EasyMaker model and view inference results in statistics.
    - For more information, see the [Batch Inference Guide](./console-guide/#batch.inference) article.
- Added the resource search feature
    - You can search for resources from the console screen, and navigate to other resource screens via links.
- Added the feature to change NAS of notebooks
    - You can change the NHN Cloud NAS connection settings for running notebooks.
- Scikit-learn serving support
    - Enabled Scikit-learn models to be registered in AI EasyMaker to serve as endpoints.
- Enable notebook shared memory
    - Enabled more than 64 MB of shared memory to be available.
    - The size depends on the instance type you selected when creating the notebook.
- Removed the save_steps hyperparameter from the NHN Cloud-provided algorithms
    - Removed the hyperparameter save_steps related to saving checkpoints.
    - The algorithm automatically calculates the appropriate number of save_steps and saves up to three.

<a id="ai.easymaker.release.notes.2023.12.19"></a>
### December 19, 2023 { #ai.easymaker.release.notes.2023.12.19 }

<a id="ai.easymaker.release.notes.2023.12.19.feature.change"></a>
#### Feature Updates

- Notebooks and training with private images
    - User-personalized container images can be used to power notebooks, training, and hyperparameter tuning.
    - By registering private image and registry account, you can easily select private images to create resources.

- Dashboard
    - See your overall resource utilization, top 3 endpoint service monitoring, and top 3 CPU/GPU utilization on one page.

- Endpoint > Autoscaler
    - You can dynamically manage the number of nodes by setting policies to scale up/scale down endpoint nodes.

<a id="ai.easymaker.release.notes.2023.09.26"></a>
### September 26, 2023 { #ai.easymaker.release.notes.2023.09.26 }

<a id="ai.easymaker.release.notes.2023.09.26.feature.change"></a>
#### Feature Updates

- Ubuntu 22.04 version provided
    - The new Ubuntu 22.04 version is provided. Ubuntu 18.04 version is no longer available, and existing customers can use the service as it is now.

- Monitoring feature provided
    - You can check system monitoring metrics for notebook, training, and endpoint.
    - You can view API call metrics for each API resource path in the endpoint.

- Basic algorithm for hyperparameter tuning
    - Through hyperparameter tuning, you can optimize the hyperparameters of the basic algorithm provided by AI EasyMaker.

- Endpoint > Serving multiple models
    - You can serve multiple training models on one endpoint stage.

- Parallel training for hyperparameter tuning
    - You can optimize the performance of hyperparameter tuning by adjusting the number of parallel trainings.

<a id="ai.easymaker.release.notes.2023.06.27"></a>
### June 27, 2023 { #ai.easymaker.release.notes.2023.06.27 }

<a id="ai.easymaker.release.notes.2023.06.27.feature.change"></a>
#### Feature Updates

- Added hyperparameter tuning feature
    - Hyperparameter tuning is the feature to automate repetitive experiments to find optimized hyperparameters to improve the predictive accuracy and performance of machine learning models.
    - For more information, please see the [Hyperparameter Tuning Guide](./console-guide/#hyperparameter.tuning).
- Added 3 basic algorithms provided by NHN Cloud AI EasyMaker
    - For more information, please see the guide document of each algorithm.
    - [Image Classification Guide](./algorithm-guide/#image.classification)
    - [Object Detection Guide](./algorithm-guide/#object.detection)
    - [Semantic Segmentation Guide](./algorithm-guide/#semantic.segmentation)

<a id="ai.easymaker.release.notes.2022.12.27"></a>
### December 27, 2022 { #ai.easymaker.release.notes.2022.12.27 }

<a id="ai.easymaker.release.notes.2022.12.27.service"></a>
#### Release of a New Service

- AI EasyMaker is an AI platform that provides environments for machine learning development, training and advancement, and endpoint services.
{% endif %}
