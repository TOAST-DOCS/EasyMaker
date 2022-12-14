## Machine Learning > AI EasyMaker > Overview

## AI EasyMaker Overview
AI EasyMaker is an AI platform for environment, training and advancement, and endpoint services for machine learning development.

## Main Features

1. Development Environment
    - Provides Jupyter Notebook with essential packages installed for machine learning development.
    - You can choose a TensorFlow or Pytorch framework notebook.
    - Notebooks are provided with an SDK for using the features of AI EasyMaker.

2. AI-based Learning and Advancement
    - Provides an environment to learn machine learning algorithms developed from the notebook.
    - You can create training by selecting an image of the TensorFlow or Pytorch framework environment suitable for the algorithm and specifying the desired GPU/CPU instance type and storage size.
    - Enables distributed training through up to 10 nodes.
    - You can analyze training result indicators using the TanserBoard.
    - You can train one algorithm several times to increase accuracy, grouping it into experimental units for comparative analysis.

3. Model Management
    - You can manage model artifacts that completed training.

4. AI Service Endpoint
    - Provides an endpoint for model service.
    - You can configure endpoints for redundancy through multiple instances.
    - You can use various features of API Gateway because endpoints are associated with the API Gateway service.
    - In order to deploy a new model with high accuracy to the endpoint in service, you can add the endpoint to the new test stage until the test ends and immediately apply it to the default domain through a feature to change the stage.