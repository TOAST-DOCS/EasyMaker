<!-- pre-align:aligned sig=098d9ec85da2 -->

<a id="ai.easymaker.overview"></a>
## Machine Learning > AI EasyMaker > Overview { #ai.easymaker.overview }

AI EasyMaker provides an AI platform to manage learning, models, and endpoints and offer development environment for AI-based learning and advancement. You can easily train and compare ML models and deploy them via endpoints.

<a id="main.feature"></a>
## Main Features { #main.feature }

1. Development Environment
    - Provides Jupyter Notebook with essential packages installed for machine learning development.
    - You can choose a TensorFlow or Pytorch framework notebook.
    - Notebooks are provided with an SDK for using the features of AI EasyMaker.

2. Training
    - Provides an environment to learn machine learning algorithms developed from the notebook.
    - You can create training by selecting an image of the TensorFlow or Pytorch framework environment suitable for the algorithm and specifying the desired GPU/CPU instance type and storage size.
    - Enables distributed training through up to 10 nodes.
    - You can analyze training result indicators in the TanserBoard.
    - You can train one algorithm several times to increase accuracy, grouping it into experimental units for comparative analysis.
    - Once you've prepared your dataset, you can create a machine learning model with the algorithms provided by AI EasyMaker without writing any training code.

3. Hyperparameter Tuning
    - Automates repetitive experiments to find the optimal hyperparameters to increase the predictive accuracy and performance of your machine learning model.

4. Fine Tuning
    - Specializes model performance by performing additional training on a pre-trained large language model using a dataset tailored to a specific domain or task.

5. Model Management
    - You can manage model artifacts that completed training.
    - You can measure and compare the performance of models.

6. Endpoint(Serving)
    - Provides an endpoint for model service.
    - Endpoints offer redundancy configuration.
    - You can use various features of API Gateway because endpoints are associated with the API Gateway service.
    - In order to deploy a new model with high accuracy to the endpoint in service, you can add the endpoint to the new test stage until the test ends and immediately apply it to the default domain through a feature to change the stage.

7. Pipeline
    - You can utilize the Kubeflow Pipelines (KFP) SDK to configure ML tasks such as data preprocessing, model training, evaluation, and deployment step by step and register them as reusable pipelines.
    - The registered pipeline can be executed immediately or scheduled to run automatically on a specific schedule.

8. Retrieval-Augmented Generation (RAG)
    - Provide RAG feature to improve response accuracy of Large Language Model (LLM).
    - Convert users' documents to vectors, save them, and provide the retrieved information to LLM to generate more accurate answers.
