## Machine Learning > AI EasyMaker > Console Guide

## Notebook
Create and manage Jupyter notebook with essential packages installed for machine learning development.

### Create Notebook   
Create a Jupyter notebook. 

- **Image**: Select OS image to be installed on the notebook instance.
    - **Core Type**: CPU, GPU core type of the image is displayed.
    - **Framework**: Installed framework is displayed on the image.
        - TENSORFLOW: Image with deep learning TensorFlow framework installed.
        - PYTORCH: Image with PyTorch deep learning framework installed.
        - PYTHON: Deep learning framework is not installed and Images with only Python languages installed.
    - **Framework Version**: Displays the version of the framework installed in the image.
    - **Python Version**: Displays the installed Python version in the image.

- **Notebook Information**
    - Enter name and description of notebook. 
    - Select instance type for notebook. The specifications of instance is selected based on Instance type selected.

- **Storage** 
    - Specifies size of notebook boot storage and data storage.
        - Boot storage is the storage on which Jupiter notebooks and underlying virtual environments are installed. This storage is initialized when the notebook is restarted.
        - Data storage is block storage mounted on the `/root/easymaker` directory path. Data on this storage is retained even when the notebook is restarted.
    - Storage size of created notebook cannot be changed, so please specify sufficient storage size at the time of creation.
    - The storage size can be entered in the unit of 10GB, maximum 2,040GB.
    - If necessary, you can add **NHN Cloud NAS** to which connect your notebook.
        - Mount Directory Name: Enter the name of the directory to mount on notebook.
        - NHN Cloud NAS Path: Enter directory path in the format `nas://{NAS ID}:/{path}`.

- **Additional Settings** 
    - Tags: Allows to specify tags in Key-Value format. You can enter maximum 10 tags.


> **[Caution] When using NHN Cloud NAS** 
> Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> **[Note] Time to create notebooks** 
> Notebooks can take several minutes to create. 
> Creation of the initial resources (notebooks, training, experiments, endpoint) takes additional few minutes to configure the service environment.

### Notebook List
A list of notebooks are displayed. Select a notebook in the list to check details and make changes to it.

- **Name**: Notebook name is displayed. You can change the name by clicking **Change** on the details screen.
- **Status**: Status of the notebook is displayed. Please refer to the table below for the main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED | Notebook creation is requested.  |
    | CREATE IN PROGRESS | Notebook instance is in the process of creation. |
    | ACTIVE (HEALTHY) | Notebook application is in normal operation.  |
    | ACTIVE (UNHEALTHY) | Notebook application is not operating properly. If this condition persists after restarting the notebook, please contact customer service center. |
    | STOP IN PROGRESS | Notebook stop in progress.  |
    | STOPPED | Notebook stopped.  |
    | START IN PROGRESS | Notebook start in progress |
    | DELETE IN PROGRESS | Notebook delete in progress.  |
    | CREATE FAILED | Failed to crate notebook. If keep fails to create, please contact Customer service center.   |
    | STOP FAILED | Failed to stop notebook. Please try to stop again.  |
    | START FAILED | Failed to start notebook. Please try to start again.  |
    | DELETE FAILED | Failed to delete notebook. Please try to delete again.  |

- **Action > Open Jupyter Notebook**: Click **Open Jupyter Notebook** button to open the notebook in a new browser window. The notebook is only accessible to users who are logged in to the console.
- **Tag**: Tag for notebook is displayed. You can change the tag by clicking **Change**.

### Configure User Virtual Execution Environment
AI EasyMaker notebook instance provides native Conda virtual environment with various libraries and kernels required for machine learning.
Default Conda virtual environment is initialized and driven when the laptop is stopped and started, but the virtual environment and external libraries that the user installs in any path are not automatically initialized and are not retained when the laptop is stopped and started.
To resolve this issue, you must create a virtual environment in directory path `/root/easymaker/custom-conda-envs` and install an external library in the created virtual environment.
AI EasyMaker notebook instance allows the virtual environment created in the `/root/easymaker/custom-conda-envs` directory path to initialize and drive when the notebook is stopped and started.

Please refer to the following guide to configure your virtual environment.

1. On the console menu, go to **Open Jupyter Notebook **>**Jupyter Notebook > Launcher>Terminal**.
2. Go to `/root/easymaker/custom-conda-envs` path.
    
        cd /root/easymaker/custom-conda-envs
    
3. To create virtual environment called `easymaker_env` in python 3.8 version, run the command `conda create` as follows
    
        conda create --prefix ./easymaker_env python=3.8
    
4. Created virtual environment can be checked with `conda env list` command. 

        (base) root@nb-xxxxxx-0:~# conda env list 
        # conda environments: 
        # 
                                /opt/intel/oneapi/intelpython/latest 
                                /opt/intel/oneapi/intelpython/latest/envs/2022.2.1 
        base                *   /opt/miniconda3 
        easymaker_env           /root/easymaker/custom-conda-envs/easymaker_env
           
### Stop Notebook
Stop the running notebook or start the stopped notebook.   

1. Select the notebook want to start or stop from Notebook List. 
2. Click **Start Notebook** or **Stop Notebook**.
3. Requested action cannot be cancelled. To proceed, please click **Confirm**

> **[Caution] How to retain your virtual environment and external libraries when starting the notebook after stopping it** 
> When stopping and starting the notebook, the virtual environment and external libraries that the user create can be initialized. 
> In order to retain, configure your virtual environment by referring to [User Virtual Execution Environment Configuration](./console-guide/#configure-user-virtual-execution-environment).


> **[Note] Time to start and stop notebooks** 
> It may take several minutes to start and stop notebooks.

### Change Notebook Instance Flavor 
Change the instance flavor of the created notebook.
Instance flavor you want to change can only be changed to the same core type instance flavor as the existing instance.

1. Select the notebook on which you want to change the instance flavor.
2. If the notebook is running (ACTIVE), click **Stop Notebook** to stop the notebook.
3. Click **Change Instance Flavor**.
4. Select the instance flavor you want to change and click Confirm.
    
> **[Note] Time to change instance flavors** 
> It may take several minutes to change the instance flavor.

### Delete Notebook   
Delete the created notebook. 

1. Select notebook you want to delete from the list. 
2. Click **Delete Notebook**
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

> **[Note] Storage** 
> When deleting a notebook, boot storage and data storage are to be deleted. 
> Connected NHN Cloud NAS is not deleted and must be deleted individually from **NHN Cloud NAS**.


## Training
Provides environment where you can train machine learning algorithms and check training results with statistics.

### Create Training  
Select the instance and OS image of training to be performed to set the environment, and proceed with the training by entering the algorithm information and input/output data path want to train.

- **Basic Information**: Select basic information about training and experiment to include training.
    - **Training Name**: Enter training name.
    - **Training Description**: Enter training description.
    - **Experiment**: Select the experiment to include the training. Experiments make group related training. If no experiments have been created, click **Add** to create the experiment.
- **Algorithm Information**: Enter information about the algorithm want to train.
    - **Algorithm Path**
        - **NHN Cloud Object Storage**: Enter the path to NHN Cloud Object Storage where the algorithm is stored.<br>
            - Enter the directory path in the format of obs://{Object Storage API endpoints}/{containerName}/{path}.
            - If using NHN Cloud Object Storage, please set permissions by referring to [Appendix>1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage](./console-guide/#1-add-ai-easymaker-system-account-permissions-to-nhn-cloud-object-storage). If you do not set the required permissions, model creation will fail.
        - **NHN Cloud NAS**: Enter the NHN Cloud NAS path where the algorithm is stored. <br>
            Enter the directory path in the format nas://{NASID}:/{path}.
            
    - **Entry Point**
        - Entry point is the entry point of algorithm execution at which the training begins. Creates entry point filename.
        - Entry point file must exist in the algorithm path.
        - Creating **requirements.txt** on the same path installs the Python package that script requires.
    - **Hyperparameter**
        - To add parameter for training, click the **the + button** to enter the parameter in Key-Value format. You can enter maximum 100 parameters.
        - Entered hyperparameters are entered as execution factors when entry point is executed. For more information on how to use it, refer to the [Appendix>3. Training Algorithm Creation Guide](./console-guide/#3-training-algorithm-creation-guide).

- **Image**: Select the image of instance for the environment in which you need to run the training.

- **Training Instance Information**
    - ** Training Instance Type**: Select type of instance on which you want to run the training.
    - **Number of Training Instances**: Enter the number of instances to run the training. If you enter more than one instance, the training run is parallel, allowing to complete training faster.

- **Data Information**
    - **Input Data**: Enter the data set on which you want to run the training. You can set maximum 10 data sets.
        - Dataset Name - Enter dataset name.
        - Data path: Enter the path to NHN Cloud Object Storage or NHN Cloud NAS.
    - **Output Data**: Enter the path to store the data to store the results of training run.
        - Enter the path to the NHN Cloud Object Storage or NHN Cloud NAS.

- **Additional Settings**
    - **Checkpoint**: If algorithm provides checkpoint, enter the storage path for checkpoint. 
        - Created checkpoints are available When to resume training from previous training.
        - Enter the path to the NHN Cloud Object Storage or NHN Cloud NAS.
    - **Data Storage Size**: Enter the data storage size of the instance on which you want to run the training.
        - Used only if you are using NHN Cloud Object Storage. Please specify sufficient size to ensure that all the data needed for training is stored.
    - **Maximum Training Time**: Specifies the maximum wait time for training to complete. Training that has exceeded the maximum wait time will be processed as an end.
    - **Log Management**: Logs that occur during training can be stored in the NHN Cloud Log & Crash service. 
        - For more details, refer to [Appendix>2. NHN Cloud Log & Crash Search Service User Guide and Check Logs](./console-guide/#2-nhn-cloud-log-crash-search-service-usage-guide-and-log-inquiry-guide).
    - **Tag**: To add tag, click the **the + button** to enter the tag in Key-Value format. You can enter maximum 10 tags.

> **[Caution] When using NHN Cloud NAS** 
> Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> **[Caution] Training failed when deleting training input data** 
> If not retained your input data until the training is completed, the training may fail.

### Training List
Training list is displayed. Select Train in the list to view details and change the information.

- **Description**: Training Description is displayed. You can change the description by clicking **Change** on the details screen.
- **Training Time**: Displays the time the training took place.
- **Status**: Training status is displayed. Please refer to the table below for main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED | Training creation is requested.  |
    | CREATE IN PROGRESS | Resources need for training is being created.  |
    | RUNNING | Training is running.  |
    | STOPPED | Training is stopped by users’ request.  |
    | COMPLETE | Training is properly completed.  |
    | STOP IN PROGRESS | Training stop in progress |
    | FAIL TRAIN | Failed state while training is in progress. Detailed failure information can be found in Log & Crash Search log when log management is enabled. |
    | CREATE FAILED | Training creation has failed. If creation keeps failing, please contact Customer service center. |
    | FAIL TRAIN IN PROGRESS, COMPLETE IN PROGRESS | Resources used for training resource clear is in progress.  |
    
- **Action**
    - **Go to TensorBoard **: TensorBoard opens in a new browser window where you can check training statistics.<br/>
    For information on how to leave TensorBoard logs, refer to [ Appendix>3. Training Algorithm Creation Guide](./console-guide/#3-training-algorithm-creation-guide). Only users who are logged into the console can access TensorBoard.
    - **Training Stop **: You can stop training in progress.

- **Hyperparameters**: You can check the hyper parameter values you set for training on the **Hyperparameters** tab of the detailed screen that displays when you select Training.

### Copy Training
Create new training with the same settings as existing training.

1. Select the training want to copy.
2. Click **Copy Training**.
3. Create Training screen is displayed with the same settings as the existing training.
4. If want to change information for settings, change it and click **Create Training** to create training.

### Create Model in Training
Create model with Training in completed state.

1. Select the Training you want to create as model.
2. Click **Create Model**. Only training in the COMPLETE state can be created as model.
3. Get moved to Model Creation page. After checking the contents, click **Create Model** to create a model. 
For more information on creating models, refer to [Model](./console-guide/#model) documentation.


### Delete Training  
Delete a training.

1. Select training you want to delete. 
2. Click **Delete Training**. Training in progress can be deleted after stopping.
3. Requested deletion action cannot be cancelled. To proceed, please click **Confirm**

> **[Note] Unable to delete training if associated models exists** 
> You cannot delete training if a model created by the training you want to delete is existed. Please delete the model first and then delete it.

## Experiment
Experiments manage associated training by grouping it into experiments.

### Create Experiment 

1. Click **Create Experiment**.
2. Enter name and description for experiment and click **Confirm**.

> **[Note] Time to create experiments** 
> Creating experiment can take several minutes. 
> Creation of initial resources (notebooks, training, experiments, endpoints) takes additional few minutes to configure the service environment.

### Experiments List
Experiments list is displayed. Selecting an experiment in the list allows to view detailed information and make changes to it.

- **Status**: Experiment status is displayed. Please refer to the table below for main status.

    | Status | Description |
    | --- | --- |
    | CREATE IN PROGRESS | Experiment create in progress. |
    | CREATE RESOURCE IN PROGRESS | Experiment create in progress. |
    | CREATE EXPERIMENT IN PROGRESS | Experiment create in progress. |
    | ACTIVE | Experiment is properly created. |
    | FAILED TO CREATE RESOURCE | Create experiment has failed.  |
    | FAILED TO CREATE EXPERIMENT | Create experiment has failed.  |

- **Action**
    - Click **Go to TensorBoard** to open a new browser window where you can view the training statistics included in the experiment. Only users who are logged into the console can access the TensorBoard.
    - **Retry**: If the experiment status is FAIL, you can recover the experiment by clicking **Retry**.
- **Training**: **Training** tab on the detailed screen that displays when select Training displays Training list included in the experiment.


### Delete Experiment  
Delete an experiment. 

1. Select the experiment to be deleted. 
2. Click ** Experiment Deletion **. You cannot delete an experiment if it is in creation progress.
3. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

> **[Note] Unable to delete training if associated experiment exists** 
> You cannot delete an experiment if training associated with the experiment exists. Please delete the associated training first and then delete it. 
> The associated training can be checked by clicking **Training** tab in details screen at the bottom when click the experiment you want to delete.


## Model 
Can manage models of AI EasyMaker's training outcomes or external models as artifacts.

### Create Model  

- **Basic Information**: Enter basic information of model.
    - **Name**: Enter model name.
        - If model's framework type is PyTorch, you must enter the same model name as PyTorch model name.
    - **Description**: Enter model description.
- **Framework Information**: Enter Framework Information
    - **Framework**: Select the model's framework from either TensorFlow or PyTorch.
    - **Framework Version**: Enter Model framework Version.
- **Model Information**: Enter the storage where model's artifacts are stored.
    - **NHN Cloud Object Storage**: Enter Object Storage path where model artifact was stored.
        - Enter a directory path in form of `obs://{Object Storage API endpoint}/{containerName}/{path}`. 
        - If using NHN Cloud Object Storage, please set permissions by referring to [Appendix>1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage](./console-guide/#1-add-ai-easymaker-system-account-permissions-to-nhn-cloud-object-storage). If do not set the required permissions, model creation will fail as unable to access to model artifact.
    - **NHN Cloud NAS**: Enter NHN Cloud NAS path where model artifact is stored. 
        - Enter directory path in form of `nas://{NAS ID}:/{path}`
- **Additional Settings**: Enter the additional information of model.
    - **Tag**: To add tag, click the **the + button** to enter the tag in Key-Value format. You can enter maximum 10 tags.

> **[Caution] When using NHN Cloud NAS** 
Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> **[Caution] Retain model artifacts in storage** 
> If not retained the model artifacts stored in storage, the creation of endpoints for that model fails.

### Model List 
Model list is displayed. Selecting a model in the list allows to check detailed information and make changes to it.

- **Name**: Model name and description are displayed. Model name and description can be changed by clicking **Change**.
- **Tag**: Model tag is displayed. Tag can be changed by clicking **Change**.
- **Model Artifact Path** displays the storage where the model's artifacts are stored.
- **Training Name**: For models created from training, training name that is based is displayed.
- **Training ID**: For models created from training, training ID that is based is displayed.
- **Framework**: Model's framework information is displayed.


### Create Endpoint from Model
Create an endpoint that can serve the selected model.

1. Select the model you want to create as an endpoint from the list.
2. Click **Create Endpoint**.
3. Get moved to **Create Endpoint** page. After checking the contents, click **Create Endpoint** to create a model. 
For more information on creating models, refer to **Endpoint** documents.


### Delete Model   
Delete a model.

1. Select the model want to delete from list. 
2. Click **Delete Model**.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

> **[Note] Unable to delete model if associated endpoint exists** 
> You cannot delete model if endpoint created by model want to delete is existed. 
> To delete, delete the endpoint created by the model first and then delete the model.


## Endpoint
Create and manage endpoints that can serve the model.

### Create Endpoint  

- **Enable API Gateway Service**
    - AI EasyMaker endpoints create API endpoints and manage APIs through NHN Cloud API Gateway service. API Gateway service must be enabled to take advantage of endpoint feature.
    - For more information on API Gateway services and fees, please refer to the following documents:
        - [API Gateway Service Guide](https://docs.toast.com/en/Application%20Service/API%20Gateway/en/overview/)
        - [API Gateway Usage Fee](https://www.toast.com/kr/pricing/by-service?c=Application%20Service&s=API%20Gateway)
- **Endpoint**: Select whether to add stage to new or existing endpoint.
    - **Create as New Endpoint**: Create new endpoint. Endpoint is created in API Gateway with new service and default stage.
    - **Add New Stage at Default Endpoint**: Endpoint is created as new stage in the service of API Gateway of existing endpoint. Select existing endpoint to add a stage.
- **Endpoint name**: Enter the endpoint name. Endpoint names cannot be duplicated.
- **Stage Name**: When adding new stage on existing endpoint, enter name for new stage. Stage names cannot be duplicated.
- **Description**: Enter the description of endpoint stage.
- **Model Information**: Enter the information for model artifacts to deploy to endpoint.
    - **Model**: Select the model you want to deploy to endpoint. If have not created model yet, please create model first.
    - **API Gateway Resource Path**: Enter API resource path for the model being deployed. For example, if set to `/inference`, you can request inference API at `POST https://{point-domain}/inference`.
- **Instance Information**: Enter instance information for the model to be served.
    - **Instance Flavor**: Select instance type.
    - **Number of Instances**: Enter the number of drives for instance.
- **Additional Settings > Tag**: To add a tag, click **the + button** to enter the tag in Key-Value format. You can enter maximum 10 tags.

> **[Note] Time to create endpoints** 
> Endpoint creation can take several minutes. 
> Creation of the initial resources (notebooks, training, experiments, endpoints) takes additional few minutes to configure the service environment.

> **[Note] Restrictions on API Gateway service resource provision when creating endpoints** 
> When you create a new endpoint, create a new API Gateway service. 
> Adding new stage on existing endpoint creates new stage in API Gateway service. 
> If you exceed the resource provision policy in [API Gateway Service Resource Provision Policy](https://docs.toast.com/en/TOAST/en/resource-policy/#resource-provision-policy-for-api-gateway-service), you might not be able to create endpoints in AI EasyMaker. In this case, adjust API Gateway service resource quota.

### Endpoint List
Endpoints list is displayed. Select an endpoint in the list to check details and make changes to the information.

- **Default Stage URL**: Displays URL of default stage among the stages on the endpoint.
- **Status**: Status of endpoint. Please refer to the table below for main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED |  Endpoint creation is requested.  |
    | CREATE IN PROGRESS | Endpoint creation is in progress.  |
    | UPDATE IN PROGRESS | Some of endpoint stages have tasks in progress.<br/>You can check the status of task for each stage in the endpoint stage list. |
    | DELETE IN PROGRESS | Endpoint deletion is in progress. |
    | ACTIVE | Endpoint is in normal operation. |
    | CREATE FAILED | Endpoint creation has failed. <br/>You must delete and recreate the endpoint. If the creation fails repeatedly, please contact the Customer Center. |
    | UPDATE FAILED | Some of endpoint stages are not serviced properly. You must delete and recreate the stages with issues. |

- **API Gateway Status**: Displays API Gateway status information for default stage of endpoint. Please refer to the table below for main status.
    
    | Status | Description |
    | --- | --- |
    | CREATE IN PROGRESS |  API Gateway Resource creation in progress.  |
    | STAGE DEPLOYING |  API Gateway default stage deploying in progress. |
    | ACTIVE |  API Gateway default stage is successfully deployed and activated. |
    | NOT FOUND: STAGE | Default stage for endpoint is not found.<br/>Please check if the stage exists in API Gateway console.<br/>If stage is deleted, the deleted API Gateway stage cannot be recovered, and the endpoint have to be deleted and recreated. |
    | NOT FOUND: STAGE DEPLOY RESULT | The deployment status of the endpoint default stage is not found.<br/>Please check if the default stage is deployed in API Gateway console. |
    | STAGE DEPLOY FAIL |  API Gateway default stage has failed to deploy. <br/>[Note] Please refer to **Recovery method when the stage's API Gateway in 'Deployment Failure' status** and recover from the deployment failed state. |
    

### Create Endpoint Stage  
Add new stage to existing endpoint. You can create and test the new stage without affecting default stage.

1. In Endpoint list, click **Endpoint Name**.
2. Click **+ Create Stage**.
3. Adding new stage from existing endpoint is automatically selected, and its setup method is the same as endpoint creation.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

### Endpoint Stage List
Stage list created under endpoint is displayed. Select stage in the list to check more information in the list.

- **Status**: Displays status of endpoint stage. Please refer to the table below for main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED |  Endpoint stage creation requested.  |
    | CREATE IN PROGRESS |  Endpoint stage creation is in progress.  |
    | DEPLOY IN PROGRESS |  Model deployment to the endpoint stage is in progress.|
    | DELETE IN PROGRESS |  Endpoint stage deletion is in progress.  |
    | ACTIVE |  Endpoint stage is normal operation. |
    | CREATE FAILED |  Endpoint stage creation has failed. Please try again.   |
    | DEPLOY FAILED |  Deployment to the endpoint stage has failed. Please try again.   |

- **API Gateway Status**: Displays stage status of API Gateway from where endpoint stage is deployed.
- **Default Stage Check**: Displays whether it is a default stage or not.
- **Stage URL**: Displays Stage URL of API Gateway where the model is served.
- **Stage Endpoint URL**: Endpoint URL of the model deployed on stage. API client can request API to displayed URL.
- **View API Gateway Settings**: Click **View Settings** to see settings that AI EasyMaker has deployed to API Gateway stage.
- **View API Gateway Statistics**: Click **View Statistics** to view API statistics of endpoints.
- **Model ID/Name/Artifact Path**: Displays information of models deployed on stage.
- **Instance Flavor**: Displays endpoint instance type the model is serving.
- **Number of Work Nodes/Pods In Progress**: Displays the number of nodes and pods being used by endpoint.

> **[Caution] Precautions when changing settings for API Gateway created by AI EasyMaker**
> When creating an endpoint or an endpoint stage, AI EasyMaker creates API Gateway services and stages for the endpoint.
> Please note the following precautions when changing API Gateway services and stages created by AI EasyMaker directly from API Gateway service console.
> 1. Avoid deleting API Gateway services and stages created by AI EasyMaker. Deletion may prevent the endpoint from displaying API Gateway information correctly, and changes made to endpoint may not be applied to API Gateway.
> 2. Avoid changing or deleting resources in API Gateway resource path that was entered when creating endpoints. Deletion may cause the endpoint's inference API call to fail
> 3. Avoid adding resources in API Gateway resource path that was entered when creating endpoints. The added resources may be deleted when adding or changing endpoint stages. 
> 4. In the stage settings of API Gateway, do not disable **Backend Endpoint Url Redifinition** or change the URL set in API Gateway resource path. If you change the url, endpoint's inference API call might fail.
> Other than above precautions, other settings are available with features provided by API Gateway as necessary. 
> For more information about how to use API Gateway, refer to [API Gateway Console Guide](https://docs.toast.com/en/Application%20Service/API%20Gateway/en/console-guide/).

> **[Note] Recovery method when the stage's API Gateway is in 'Deployment Failed' status**
> If stage settings of AI EasyMaker endpoint are not deployed to the API Gateway stage due to a temporary issue, deployment status is displayed as failed.
> In this case, you can deploy API Gateway stage manually by clicking Select Stage from the Stage list > View API Gateway Settings > 'Deploy Stage' in the bottom detail screen.
> If this guide couldn’t recover the deployment status, please contact the Customer Center.

### Call Endpoint Inference

1. When you click Stage in Endpoint > Endpoint Stage, Stage details screen is displayed at the bottom.
2. Check stage endpoint URL on detail screen.
3. When the stage endpoint URL is called the HTTP POST Method, inference API is called.
    - Request and response specifications of the inference API differ depending on the algorithm user created.

        // Inference API example: Request 
        curl --location --request POST '{Stage Endpoint URL}' \
                --header 'Content-Type: application/json' \
                --data-raw '{
            "instances": [
                [6.8,  2.8,  4.8,  1.4],
                [6.0,  3.4,  4.5,  1.6]
                ]
        }'
                
        // Inference API Example: Response 
        { 
        "predictions" : [ 
            [ 
                                0.337502569, 
                                0.332836747, 
                                0.329660654 
                            ], 
            [ 
                                0.337530434, 
                                0.332806051, 
                                0.329663515 
                            ] 
            ] 
        }


### Change Endpoint Default Stage

Change the default stage of the endpoint to another stage. 
To change the model of an endpoint without service stop, AI EasyMaker recommends deploying the model using stage capabilities.

1. Stages operating as actual services are operated by the default stage.
2. If replacing with new model, add new stage to the existing endpoint.
3. Verify that the endpoint service is not affected by the replaced model in the new stage.
4. Click **Change Default Stage**.
5. Select new stage that want to change as default stage from stage want to change.
6. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**
7. Stage that you want to change changes to the default stage, and resources of existing default stage are automatically deleted.

### Delete Endpoint Stage  

1. In Endpoint list, click **Endpoint Name** to go to Endpoint Stage list.
2. In Endpoint Stages list, select the endpoint stage want to delete. You cannot delete default stage.
3. Click **Delete Stage**.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

> **[Caution] Delete stage of API Gateway service when deleting the endpoint stage** 
> Deleting an endpoint stage in AI EasyMaker also deletes the stage in API Gateway service from which the endpoint's stage is deployed. 
> If there is an API running on the API Gateway stage to be deleted, please be noted that API calls cannot be made.

### Delete Endpoint  
Delete an endpoint.

1. Select the endpoint want to delete from endpoints list.
2. You cannot delete an endpoint if there is stage under endpoint other than the default stage. Please delete the other stages first.
3. Click **Delete Endpoint**.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

> **[Caution] Delete API Gateway service when deleting the endpoint stage** 
> Deleting an endpoint stage in AI EasyMaker also deletes API Gateway service from which the endpoint's stage was deployed. 
> If there is API running on the API Gateway service to be deleted, please be noted that API calls cannot be made.


## Appendix

### 1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage

Some features of AI EasyMaker use the user's NHN Cloud Object Storage as input/output storage 
You must allow read or write access to user’s AI EasyMaker system account in NHN Cloud Object Storage container for running normal features.

Allowing read/write permissions on the AI EasyMaker system account to the user's NHN Cloud Object Storage container is meaning that AI EasyMaker system account can read or write files in accordance with permissions granted to all files in the user's NHN Cloud Object Storage container.

You have to check this information to set up an access policy in User Object Storage only with the required accounts and permissions.

The 'User' take responsibility for all consequences of allowing the user to access Object Storage for an account other than the AI EasyMaker system account during the access policy setting process, and AI EasyMaker is not responsible for it.

[Note] According to features, AI EasyMaker accesses, reads or writes to Object Storage as follows.

| Feature | Access Right | Access target |
| --- | --- | --- |
| Training | Read | Algorithm path entered by user, training input data path |
| Training | Write | User-entered training output data, checkpoint path|
| Model | Read | Model artifact path entered by user |
| Endpoint | Read | Model artifact path entered by user |

To add read/write permissions to AI EasyMaker system account in Object Storage, refer to the following:

1. Click **[Training]** or **[Model]** Tab>**AI EasyMaker System Account Information**. 
2. Archive the AI EasyMaker system account information, **AI EasyMaker Tenant ID** and **AI EasyMaker API User ID**. 
3. Go to the NHN Cloud Object Storage console.
4. [Allow specific projects or users to read/write](https://docs.toast.com/en/Storage/Object%20Storage/en/acl-guide/#allow-readwrite-to-specific-projects-or-specific-users) Refer to documents to add required read and write permissions to AI EasyMaker system account in NHN Cloud Object Storage console.


### 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Inquiry Guide

#### NHN Cloud Log & Crash Search Service Usage Guide
Logs and events generated by the AI EasyMaker service can be stored in the NHN Cloud Log & Crash Search service. 
To store logs in the Log & Crash Search service, you have to enable Log & Crash service and separate usage fee will be charged.

- **Information on Log & Crash Search service use and fee** 
    - For more information and fees on the Log & Crash Search service, please refer to the following documents 
        - [Log & Crash Search Service Guide](https://docs.toast.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/Overview/)
        - [Log & Crash Search Fee](https://www.toast.com/kr/pricing/by-service?c=Data%20%26%20Analytics&s=Log%20%26%20Crash%20Search)

#### Log Query

1. Go to the Log & Crash Search service console page.
2. In Log & Crash Search service, enter search criteria to view the logs.
    * AI EasyMaker Training Log Query: Query logs with category field "easymaker.training"
        * Question: category:"easymaker.training"
    * AI EasyMaker endpoint logs query: Query logs with category field "easymaker.inference"
        * Question: category:"easymaker.inference"
    * AI EasyMaker Log Full Query: Query logs with logType field "NNHCloud-AIEasyMaker" 
        * Question: logType:"NHNCloud-AIEasyMaker"
3. For more information on how to use Log & Crash Search service, refer to [Log & Crash Search Service Console Guide](https://docs.toast.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/console-guide/).

AI EasyMaker service sends logs to Log & Crash Search service in the following defined fields:

- **Common Log Field** 

    | Name | Description | Valid range |
    | --- | --- | --- | 
    | easymakerAppKey | AI EasyMaker Appkey(AppKey) | - |
    | category | Log category | easymaker.training, easymaker.inference |
    | logLevel | Log level | INFO, WARNING, ERROR |
    | body | Log contents | - |
    | logType | Service name provided by log | NHNCloud-AIEasyMaker |
    | time | Log Occurrence Time (UTC Time) | - |

- **Training Log field**

    | Name | Description |
    | --- | --- |
    | trainingId | AI EasyMaker training ID |

- **Endpoint Log Field**

    | Name | Description |
    | --- | --- | 
    | endpointId | AI EasyMaker Endpoint ID |
    | endpointStageId | Endpoint stage ID | 
    | inferenceId | Inference request own ID | 
    | action | Action classification (Endpoint.Model) | 
    | modelName | Model name to be inferred | 

### 3. Hyperparameters

* Value in Key-Value format entered through the console.
* 엔트리 포인트 실행 시, 실행 인자(--{Key})로 전달됩니다.
* 환경 변수 값(EM_HP_{대문자로 변환된 Key})으로도 저장되어 활용할 수 있습니다.

As shown in the example below, you can use hyperparameter values entered during training creation.<br>
![HyperParameter Input Screen](http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_hyperparameter_en.png)

        import argparse
    
        model_version = os.environ.get("EM_HP_MODEL_VERSION")
    
        def parse_hyperparameters():
            parser = argparse.ArgumentParser()
    
            # 입력한 하이퍼파라미터 파싱
            parser.add_argument("--epochs", type=int, default=500)
            parser.add_argument("--batch_size", type=int, default=32)
            ...
    
            return parser.parse_known_args()

### 4. Environment Variables

* Information required for training is passed to training container with **Environment Variable** and the environment variables passed in **Training Script** can be utilized.
* Environment variable names created by user input are to be capitalized.
* 코드 상에서 학습이 완료된 모델은 반드시 EM_MODEL_DIR 경로에 저장해야 합니다.
* **Key Environment Variables**

    | Environment variable name              | Description |
----------------------------------------| --- | --- |
    | EM_SOURCE_DIR                          | Absolute path to the folder where the algorithm script entered at the time of training creation is downloaded |
    | EM_ENTRY_POINT                         | Algorithm entry point name entered at training creation |
    | EM_DATASET_${Data set name}            | Absolute path to the folder where each data set entered at the time of training creation is downloaded |
    | EM_DATASETS                            | Full data set list ( json format) |
    | EM_MODEL_DIR                           | Model storage path |
    | EM_CHECKPOINT_DIR                      | Checkpoint Storage Path |
    | EM_HP_${ 대문자로 변환된 Hyperparameter key } | Hyperparameter value corresponding to the hyperparameter key |
    | EM_HPS                                 | Full Hyperparameter List (in json format) |
    | EM_TENSORBOARD_LOG_DIR                 | TensorBoard log path for checking training results |
    | EM_REGION                              | Current Region Information |
    | EM_APPKEY                              | Appkey of AI EasyMaker service currently in use |

* **Example code for utilizing environment variables**

        import os
        import tensorflow

        dataset_dir = os.environ.get("EM_DATASET_TRAIN")
        train_data = read_data(dataset_dir, "train.csv")

        model = ... # 입력한 데이터를 이용해 모델 구현
        callbacks = [
            tensorflow.keras.callbacks.ModelCheckpoint(filepath=f'{os.environ.get("EM_CHECKPOINT_DIR")}/cp-{{epoch:04d}}.ckpt', save_freq='epoch', period=50),
            tensorflow.keras.callbacks.TensorBoard(log_dir=f'{os.environ.get("EM_TENSORBOARD_LOG_DIR")}'),
        ]
        model.fit(..., callbacks)

        model_dir = os.environ.get("EM_MODEL_DIR")
        model.save(model_dir)

### 5. Store Indicator Logs for TensorBoard Usage

* In order to check result indicators on the TensorBoard screen after training, the TensorBoard log storage space must be set to the specified location (`EM_TENSORBOARD_LOG_DIR`) when writing the training script.

* ** Example code for Tesnsorboard log storage (TensorFlow)**

        import tensorflow as tf 
 
        # Specify the TensorBoard log path 
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=os.environ.get("EM_TENSORBOARD_LOG_DIR")) 
        
        model = ... # model implementation
        
        model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                epochs=100, batch_size=20, callbacks=[tb_log])

![Check TensorBoard Log](http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_tensorboard.png)