## Machine Learning > AI EasyMaker > Overview

AI EasyMaker provides an AI platform to manage learning, models, and endpoints and offer development environment for AI-based learning and advancement. You can easily train and compare ML models and deploy them via endpoints.

## Main Features

1. Development Environment
    - Provides Jupyter Notebook with essential packages installed for machine learning development.
    - You can choose a TensorFlow or Pytorch framework notebook.
    - Notebooks are provided with an SDK for using the features of AI EasyMaker.

2. Training
    - Provides an environment to learn machine learning algorithms developed from the notebook.
    - You can create training by selecting an image of the TensorFlow or Pytorch framework environment suitable for the algorithm and specifying the desired GPU/CPU instance type and storage size.
    - Enables distributed training through up to 10 nodes.
    - You can analyze training result indicators using the TanserBoard.
    - You can train one algorithm several times to increase accuracy, grouping it into experimental units for comparative analysis.
    - Once you've prepared your dataset, you can create a machine learning model with the algorithms provided by AI EasyMaker without writing any training code.

3. Hyperparameter Tuning
    - Automate repetitive experiments to find the optimal hyperparameters to increase the predictive accuracy and performance of your machine learning model.

4. Model Management
    - You can manage model artifacts that completed training.

5. Endpoint(Serving)
    - Provides an endpoint for model service.
    - Endpoints offer redundancy configuration.
    - You can use various features of API Gateway because endpoints are associated with the API Gateway service.
    - In order to deploy a new model with high accuracy to the endpoint in service, you can add the endpoint to the new test stage until the test ends and immediately apply it to the default domain through a feature to change the stage.