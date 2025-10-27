## Machine Learning > AI EasyMaker > Console Guide

## Dashboard

You can view the usage status of all AI EasyMaker resources in the dashboard.

### Service Usage

Displays the number of resources in use per resource.

- Notebook: Number of notebooks in ACTIVE (HEALTHY) status that are in use.
- Training: Number of trainings that are COMPLETE
- Hyperparameter tuning: Number of hyperparameter tunings that are COMPLETE
- Endpoints: Number of endpoints in the ACTIVE state

### Monitoring Services

- Displays the top 3 endpoints with the most API calls.
- Select an endpoint to see the aggregate API success/failure metrics for the child endpoint stage.

### Resource Utilization

- You can see the most utilized resources by CPU and GPU core type.
- If you hover over a metric, it displays resource information.

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
    - If necessary, you can associate **NHN Cloud NAS** to which connect your notebook.
        - Mount Directory Name: Enter the name of the directory to mount on notebook.
        - NHN Cloud NAS Path: Enter directory path in the format `nas://{NAS ID}:/{path}`.

> [Caution] When using NHN Cloud NAS:
> Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> [Note] Time to create notebooks:
> Notebooks can take several minutes to create.
> Creation of the initial resources (notebooks, training, experiments, endpoint) takes additional few minutes to configure the service environment.

### Notebook List

A list of notebooks are displayed. Select a notebook in the list to check details and make changes to it.

- **Name**: Notebook name is displayed. You can change the name by clicking **Change** on the details screen.
- **Status**: Status of the notebook is displayed. Please refer to the table below for the main status.

    | Status                 | Description                                                                        |
    |--------------------|---------------------------------------------------------------------------|
    | CREATE REQUESTED   | Notebook creation is requested.                                                        |
    | CREATE IN PROGRESS | Notebook instance is in the process of creation.                                                    |
    | ACTIVE (HEALTHY)   | Notebook application is in normal operation.                                            |
    | ACTIVE (UNHEALTHY) | Notebook application is not operating properly. If this condition persists after restarting the notebook, please contact customer service center. |
    | STOP IN PROGRESS   | Notebook stop in progress.                                                         |
    | STOPPED            | Notebook stopped.                                                           |
    | START IN PROGRESS  | Notebook start in progress                                                         |
    | REBOOT IN PROGRESS | Notebook reboot in progress.                                                         |
    | DELETE IN PROGRESS | Notebook delete in progress.                                                         |
    | CREATE FAILED      | Failed to crate notebook. If keep fails to create, please contact Customer service center.                        |
    | STOP FAILED        | Failed to stop notebook. Please try to stop again.                                            |
    | START FAILED       | Failed to start notebook. Please try to start again.                                            |
    | REBOOT FAILED      | Failed to reboot notebook. Please try to start again.                                           |
    | DELETE FAILED      | Failed to delete notebook. Please try to delete again.                                            |

- **Action > Open Jupyter Notebook**: Click **Open Jupyter Notebook** button to open the notebook in a new browser window. The notebook is only accessible to users who are logged in to the console.

- **Monitoring**: On the **Monitoring** tab of the detail screen that appears when you select the notebook, you can see a list of monitored instances and a chart of basic metrics.
    - The **Monitoring** tab is disabled when the notebook is being created or when there is a task in progress.

### Configure User Virtual Execution Environment

AI EasyMaker notebook instance provides native Conda virtual environment with various libraries and kernels required for machine learning.
Default Conda virtual environment is initialized and driven when the laptop is stopped and started, but the virtual environment and external libraries that the user installs in any path are not automatically initialized and are not retained when the laptop is stopped and started.
To resolve this issue, you must create a virtual environment in directory path `/root/easymaker/custom-conda-envs` and install an external library in the created virtual environment.
AI EasyMaker notebook instance allows the virtual environment created in the `/root/easymaker/custom-conda-envs` directory path to initialize and drive when the notebook is stopped and started.

Please refer to the following guide to configure your virtual environment.

1. On the console menu, go to **Open Jupyter Notebook**>**Jupyter Notebook > Launcher>Terminal**.
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

### User Script

You can register scripts in the path `/root/easymaker/cont-init.d` that should run automatically when the notebook is stopped and started.
The scripts are executed in ascending alphanumeric order.

- Script location and permission
    - Only files located in the path `/root/easymaker/cont-init.d` are executed.
    - Only scripts for which you have permission to run are executed.
- Script content
    - The first line of scripts must start with `#!`.
    - Scripts are executed with the root permission.
- The script execution history is stored in the following locations.
    - Script exit code: `/root/easymaker/cont-init.d/{SCRIPT}.exitcode`
    - Script standard output and standard error streams: `/root/easymaker/cont-init.d/{SCRIPT}.output`
    - Full execution log: `/root/easymaker/cont-init.output`

### Stop Notebook

Stop the running notebook or start the stopped notebook.

1. Select the notebook want to start or stop from Notebook List.
2. Click **Start Notebook** or **Stop Notebook**.
3. Requested action cannot be cancelled. To proceed, please click **Confirm**

> [Caution] How to retain your virtual environment and external libraries when starting the notebook after stopping it:
> When stopping and starting the notebook, the virtual environment and external libraries that the user create can be initialized.
> In order to retain, configure your virtual environment by referring to [User Virtual Execution Environment Configuration](./console-guide/#configure-user-virtual-execution-environment).

> [Note] Time to start and stop notebooks:
> It may take several minutes to start and stop notebooks.

### Change Notebook Instance Type

Change the instance type of the created notebook.
Instance type you want to change can only be changed to the same core type instance type as the existing instance.

1. Select the notebook on which you want to change the instance type.
2. If the notebook is running (ACTIVE), click **Stop Notebook** to stop the notebook.
3. Click **Change Instance type**.
4. Select the instance type you want to change and click Confirm.

> [Note] Time to change instance types:
> It may take several minutes to change the instance type.

### Reboot Notebook

If a problem occurs while using the notebook, or if the status is ACTIVE but you can't access the notebook,
you can reboot the notebook.

1. Select notebook you want to reboot.
2. Click **Reboot Notebook**
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

> [Caution] How to retain your virtual environment and external libraries when rebooting the notebook:
> When rebooting the notebook, the virtual environment and external libraries that the user create can be initialized.
> In order to retain, configure your virtual environment by referring to [User Virtual Execution Environment Configuration](./console-guide/#_8).

### Delete Notebook

Delete the created notebook.

1. Select notebook you want to delete from the list.
2. Click **Delete Notebook**
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

> [Note] Storage:
> When deleting a notebook, boot storage and data storage are to be deleted.
> Connected NHN Cloud NAS is not deleted and must be deleted individually from **NHN Cloud NAS**.

## Experiment

Experiments are managed by grouping related trainings into experiments.

### Create Experiment

1. Click **Create Experiment**
2. Enter an experiment name and description and click **OK**.

> [Note] Experiment creation time:
Creating experiments can take several minutes.
When creating the initial resources (laptops, trainings, labs, endpoints), it takes an extra few minutes to configure the service environment.

### List of Experiments

Experiments appears. Select an experiment to view and modify detailed information.

- **Status**: Experiment status appears. Please refer to the table below for main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED | Creating an experiment is requested. |
    | CREATE IN PROGRESS | An experiment is being created. |
    | CREATE FAILED | Failed to create an experiment. Please try again. |
    | ACTIVE | The experiment is successfully created. |

- **Operation**
    - Click **Go to TensorBoard** to open the TensorBoard in a new browser window, where you can view statistical information about the training included in your experiment. The TensorBoard is only accessible to users who are logged into the console.
    - **Retry**: If the experiment status is FAIL, you can recover the experiment by clicking **Retry**.
- **Training**: The **Training** tab on the detailed screen that appears when selecting Training shows trainings included in the experiment.

### Delete Experiment

Delete an experiment.

1. Select an experiment to delete.
2. Click **Delete Experiment**.
3. Requested deletion cannot be undone. Click **OK** to proceed.

> [Note] Unable to delete experiment if an associated resource exists:
> You cannot delete an experiment if a pipeline schedule associated with the experiment exists, or if there are training, hyperparameter tuning, or pipeline execution in production. Delete the resources associated with the experiment first, then delete the experiment.
> For associated resources, you can check the list by clicking the **[Training]** tab in the detail screen at the bottom that is displayed when you click the experiment you want to delete.

## Training

Provides an training environment where you can learn and identify machine training algorithms based on training results.

### Create Training

Set the training environment by selecting the instance and OS image to be trained, and proceed with training by entering the algorithm information and input/output data path to learn.

- **Training template** : To set training information by loading a training template, select 'Use' and then select a training template to load.
- **Basic information** : Select basic information about the training and the experiment that the training will be included in.
    - **Training Name** : Enter a training name.
    - **Training Description** : Enter a description.
    - **Experiment** : Select an experiment to include training. Experiments group related trainings. If no experiments have been created, click **Add** to create one.
- **Algorithm information** : Enter information about the algorithm you want to learn.
    - **Algorithm Type** : Select the algorithm type.
        - **Algorithm provided by NHN Cloud** : Use the algorithm provided by AI EasyMaker. For detailed information on the provided algorithm, refer to [the Algorithm Guide document provided by NHN Cloud](./algorithm-guide/#).
            - **Algorithm** : Select an algorithm.
            - **Hyperparameter** : Enter the hyperparameter value required for training. For detailed information on hyperparameters for each algorithm, refer to [the Algorithm Guide document provided by NHN Cloud](./algorithm-guide/#).
            - **Algorithm Metrics** : Displays information about the metrics generated by the algorithm.
        - **Own Algorithm** : Uses an algorithm written by the user.
            - **algorithm path**
                - **NHN Cloud Object Storage** : Enter the path of NHN Cloud Object Storage where algorithms are stored.<br>
                    - obs://{Object Enter the directory path in the format Storage API endpoint}/{containerName}/{path}.
                    - When using NHN Cloud Object Storage, refer to [Appendix > 1. Adding AI EasyMaker System Account Permissions to NHN Cloud Object Storage](./console-guide/#1-add-ai-easymaker-system-account-permissions-to-nhn-cloud-object-storage) to set permissions. Model creation will fail if you do not set the necessary permissions.
                - **NHN Cloud NAS** : Enter the NHN Cloud NAS path where the algorithm is stored. <br>
                    nas://{NAS Enter the directory path in the format ID}:/{path}.

            - **entry point**
                - The entry point is the point of entry into the execution of the algorithm from which training begins. Creates the entry point file name.
                - The entry point file must exist in the algorithm path.
                - Creating **requirements.txt** in the same path will install the required python packages from the script.
            - **hyperparameter**
                - To add parameters for training, click **the + button** to enter parameters in Key-Value format. Up to 100 parameters can be entered.
                - The entered hyperparameters are entered as execution arguments when the entry point is executed. For detailed usage, please refer to [Appendix > 3. Hyperparameters](./console-guide/#3-hyperparameters).

- **Image** : Choose an image for your instance that matches the environment in which you need to run your training.

- **Training Resource Information**
    - **Training instance type** : Select an instance type to run training.
    - **Number of Distributed Nodes**: Enter a number of distributed nodes to be performed. Distributed training can be enabled through settings in the algorithm code. For more information, please refer to [Appendix > 6. Distributed Training Settings by Framework](./console-guide/#6).
    - **Enable torchrun**: Select whether to use torchrun, which is supported by the Pytorch framework. For more information, see [Appendix > 8. How to use torchrun](./console-guide/#8-how-to-use-torchrun).
    - **Number of processes per node**: If using torchrun, enter the number of processes per node. torchrun enables distributed training by running multiple processes on a single node. The number of processes affects memory usage.
- **Input Data**
    - **Data Set**: Enter the data set to run training on. You can set up to 10 data sets.
        - Dataset name: Enter a name for your data set.
        - Data Path: Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
    - **Checkpoint** : If you want to start training from a saved checkpoint, enter the save path of the checkpoint.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
- **Output Data**
    - **Output data** : Enter the data storage path to save the training execution results.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
    - **Checkpoint** : If the algorithm provides a checkpoint, enter the storage path of the checkpoint.
        - Created checkpoints can be used to resume training from previous training.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
- **Additional settings**
    - **Data storage size** : Enter the data storage size of the instance to run training.
        - Used only when using NHN Cloud Object Storage. Please specify a size large enough to store all the data required for training.
    - **Maximum training time** : Specifies the maximum waiting time until training is complete. training that exceeds the maximum waiting time will be terminated.
    - **Log Management** : Logs generated during training can be stored in the NHN Cloud Log & Crash service.
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](./console-guide/#2-nhn-cloud-log-crash-search-service-usage-guide-and-log-inquiry-guide).

> [Caution] When using NHN Cloud NAS:
> Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.

> [Caution] training failure when deleting training input data:
> Training may fail if the input data is deleted before training is completed.

### Training List

A list of studies is displayed. If you select a training from the list, you can check detailed information and change the information.

- **Training time** : Displays the training time.
- **Status** : Shows the status of training. Please refer to the table below for the main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED | You have requested to create a training. |
    | CREATE IN PROGRESS | This is a state in which resources necessary for training are being created. |
    | RUNNING | Training is in progress. |
    | STOPPED | Training is stopped at the user's request. |
    | COMPLETE | Training has been completed normally. |
    | STOP IN PROGRESS | Training is stopping. |
    | FAIL TRAIN | This is a failed state during training. Detailed failure information can be checked through the Log & Crash Search log when log management is enabled. |
    | CREATE FAILED | The training creation failed. If creation continues to fail, please contact customer service. |
    | FAIL TRAIN IN PROGRESS, COMPLETE IN PROGRESS | The resources used for training are being cleaned up. |

- **Operation**
    - **Go to TensorBoard** : TensorBoard, where you can check the statistical information of training, opens in a new browser window.<br/>
    For how to leave a TensorBoard log, refer to [Appendix > 5. Store Indicator Logs for TensorBoard Usage](./console-guide/#5-store-indicator-logs-for-tensorboard-usage). TensorBoard can only be accessed by users logged into the console.
    - **Stop training** : You can stop training in progress.

- **Hyperparameters** : You can check the hyperparameter values set for training on **the hyperparameter** tab of the detailed screen displayed when selecting training.

- **Monitoring**: When you select the endpoint stage, you can see a list of monitored instances and basic metrics charts in the **Monitoring** tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while an endpoint stage is being created.

### Copy Training

Create a new training with the same settings as an existing training.

1. Select the training you want to copy.
2. Click **Copy Training**.
3. The create training screen is displayed with the same settings as the existing training.
4. If there is any information you would like to change the settings for, make the changes and then click **Create Training** to create the training.

### Create a Model from Training

Create a model with training in the completed state.

1. Choose the training you want to create as a model.
2. Click **Generate Model** Only training in the COMPLETE state can be created as a model.
3. You will be taken to the model creation page. After checking the contents, click **Create Model** to create a model. For more information on model creation, see [the model](./console-guide/#model) documentation.

### Delete Training

Deletes a training.

1. Select the training you want to delete.
2. Click **Delete Training**. Training in progress can be deleted after stopping.
3. Requested deletion cannot be undone. Click **OK** to proceed.

> [Note] Training cannot be deleted if a related model exists:
Training cannot be deleted if a model created by the training to be deleted exists. Please delete the model first and then the training.

## Hyperparameter Tuning

Hyperparameter tuning is the process of optimizing hyperparameter values to maximize a model's predictive accuracy. If you don't use this feature, you'll have to manually tune the hyperparameters to find the optimal values while running many training jobs yourself.

### Create Hyperparameter Tuning

How to configure a hyperparameter tuning job.

- **Training Template**
    - **Use** : Select whether to use the training template. Using a training template, some configuration values for hyperparameter tuning are populated with pre-specified values.
    - **Training Template**: Select a training template to use to automatically populate some configuration values for hyperparameter tuning.
- **Basic Information**
    - **Hyperparameter Tuning Name**: Enter a name for the hyperparameter tuning job.
    - **Description**: Input when a description of the hyperparameter tuning task is required.
    - **Experiment**: Select an experiment to include hyperparameter tuning. Experiments group related hyperparameter tunings. If no experiments have been created, click **Add** to create one.-
- **Tuning Strategy**
    - **Strategy Name**: Choose which strategy to use to find the optimal hyperparameters.
    - **Random State**: Determines random number generation. Specify a fixed value for reproducible results.
- **Algorithm information** : Enter information about the algorithm you want to learn.
    - **Algorithm Type** : Select the algorithm type.
        - **Algorithm provided by NHN Cloud** : Use the algorithm provided by AI EasyMaker. For detailed information on the provided algorithm, refer to [the Algorithm Guide document provided by NHN Cloud](./algorithm-guide/#).
            - **Algorithm** : Select an algorithm.
            - **Hyperparameter Spec** : Enter the hyperparameter to use for hyperparameter tuning. For detailed information on hyperparameters for each algorithm, refer to [the Algorithm Guide document provided by NHN Cloud](./algorithm-guide/#).
            - **Name** : Defines which hyperparameters to tune It is determined by algorithm.
                - **Type** : Selects the data type of the hyperparameter. It is determined by algorithm.
                - **Value/Range**
                    - **Min**: Defines the minimum value.
                    - **Max**: Defines the maximum value.
                    - **Step**: Determines the size of the hyperparameter value change when using the "Grid" tuning strategy.
            - **Algorithm Metrics** : Displays information about the metrics generated by the algorithm.
        - **Own Algorithm**: Uses an algorithm written by the user.
            - **Algorithm Path**
                - **NHN Cloud Object Storage**: Enter the path of NHN Cloud Object Storage where algorithms are stored.<br>
                    - obs: Enter the directory path in the format of obs://{ObjectStorage API endpoint}/{containerName}/{path}.
                    - When using NHN Cloud Object Storage, please set permissions by referring to [Appendix > 1. Adding AI EasyMaker system account permissions to NHN Cloud Object Storage](./console-guide/#1-nhn-cloud-object-storage-ai-easymaker).If you do not set the required permissions, model creation will fail.
                - **NHN Cloud NAS**: Enter the NHN Cloud NAS path where the algorithm is stored.
                    - nas://{NAS Enter the directory path in the format ID}:/{path}.
            - **Entry Point**
                - The entry point is the point of entry into the execution of the algorithm from which training begins. Creates the entry point file name.
                - The entry point file must exist in the algorithm path.
                - Creating **requirements.txt** in the same path will install the required python packages from the script.
            - **Hyperparameter Specification**
                - **Name** : Defines which hyperparameters to tune.
                - **Type** : Select the data type of the hyperparameter.
                - **Value/Range**
                    - **Min**: Defines the minimum value.
                    - **Max**: Defines the maximum value.
                    - **Step**: Determines the size of the hyperparameter value change when using the "Grid" tuning strategy.
                    - **Comma-Separated Values**: Tune hyperparameters using static values (e.g. sgd, adam).
- **Image** : Choose an image for your instance that matches the environment in which you need to run your training.
- **Training Resource Information**
    - **Training instance type** : Select an instance type to run training.
    - **Number of Distributed Nodes**: Enter a number of distributed nodes to be performed. Distributed training can be enabled through settings in the algorithm code. For more information, please refer to [Appendix > 6. Distributed Training Settings by Framework](./console-guide/#6).
    - **Number of Parallel Trainings**: Enter a number of trainings to perform in parallel simultaneously.
    - **Enable torchrun**: Select whether to use torchrun, which is supported by the Pytorch framework. For more information, see [Appendix > 8. How to use torchrun](./console-guide/#8-how-to-use-torchrun).
    - **Number of processes per node**: If using torchrun, enter the number of processes per node. torchrun enables distributed training by running multiple processes on a single node. The number of processes affects memory usage.
- **Input Data**
    - **Data Set**: Enter the data set to run training on. You can set up to 10 data sets.
        - Dataset name: Enter a name for your data set.
        - Data Path: Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
    - **Checkpoint**: If you want to start training from a saved checkpoint, enter the save path of the checkpoint.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
- **Output Data**
    - **Output data** : Enter the data storage path to save the training execution results.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
    - **Checkpoint** : If the algorithm provides a checkpoint, enter the storage path of the checkpoint.
        - Created checkpoints can be used to resume training from previous training.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
- **Metrics**
    - **Metric Name**: Define which metric to collect from logs output by the training code.
    - **Metric Format**: Enter a regular expression to use to collect metrics. The training algorithm should output metrics to match the regular expression.
- **Target Indicator**
    - **Metric Name**: Choose which metric you want to optimize for.
    - **Goal Metric Type**: Choose an optimization type.
    - **Goal Metric Goal**: The tuning job will end when the goal metric reaches this value.
- **Tuning Resource Configuration**
    - **Maximum Number of Failed Trainings**: Define the maximum number of failed lessons. When the number of failed trainings reaches this value, tuning ends in failure.
    - **Maximum Number of Trainings**: Defines the maximum number of lessons. Tuning runs until the number of auto-run training reaches this value.
- **Early Stop Training**
    - **Name**: Stop training early if the model is no longer good even though training continues.
    - **Min Trainings Required**: Define how many trainings the target metric value will be taken from when calculating the median.
    - **Start Step**: Set the training step from which to apply early stop.
- **Additional settings**
    - **Data storage size** : Enter the data storage size of the instance to run training.
        - Used only when using NHN Cloud Object Storage. Please specify a size large enough to store all the data required for training.
    - **Maximum Progress Time**: Specifies the maximum progress time until training is completed. training that exceeds the maximum progress time will be terminated.
    - **Log Management** : Logs generated during training can be stored in the NHN Cloud Log & Crash service.
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](./console-guide/#2-nhn-cloud-log-crash-search-service-usage-guide-and-log-inquiry-guide).

> [Caution] When using NHN Cloud NAS:
> Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.

> [Caution] Training failure when deleting training input data:
> Training may fail if the input data is deleted before training is completed.

### Hyperparameter Tuning List

A list of hyperparameter tunings is displayed. Select a hyperparameter tuning from the list to view details and change information.

- **Time Spent** : Shows the time spent tuning hyperparameters.
- **Completed Training**: Indicates the number of completed trainings among the automatically generated trainings by hyperparameter tuning.
- **Training In Progress**: Indicates the number of trainings in progress.
- **Failed Training** : Indicates the number of failed lessons.
- **Best Training**: Indicates the target metric information of the training that recorded the highest target metric value among the training automatically generated by hyperparameter tuning.
- **Status** : Shows the status of hyperparameter tuning. Please refer to the table below for the main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED | Requested to create hyperparameter tuning. |
    | CREATE IN PROGRESS | Resources required for hyperparameter tuning are being created. |
    | RUNNING | Hyperparameter tuning is in progress. |
    | STOPPED | Hyperparameter tuning is stopped at the user's request. |
    | COMPLETE | Hyperparameter tuning has been successfully completed. |
    | STOP IN PROGRESS | Hyperparameter tuning is stopping. |
    | FAIL HYPERPARAMETER TUNING | A failed state during hyperparameter tuning in progress. Detailed failure information can be checked through the Log & Crash Search log when log management is enabled. |
    | CREATE FAILED | Hyperparameter tuning generation failed. If creation continues to fail, please contact customer service. |
    | FAIL HYPERPARAMETER TUNING IN PROGRESS, COMPLETE IN PROGRESS, STOP IN PROGRESS | Resources used for hyperparameter tuning are being cleaned up. |

- **Status Details**: The bracketed content in the `COMPLETE` status is the status details. See the table below for key details.

    | Details | Description |
    | --- | --- |
    | GoalReached | Details when training for hyperparameter tuning is complete by reaching the target value. |
    | MaxTrialsReached | Details when hyperparameter tuning has reached the maximum number of training runs and is complete. |
    | SuggestionEndReached | Details when the exploration algorithm in Hyperparameter Tuning has explored all hyperparameters. |

- **Operation**
    - **Go to TensorBoard** : TensorBoard, where you can check the statistical information of training, opens in a new browser window.<br/>
    For instructions on how to leave TensorBoard logs, please refer to [Appendix > 5. Store Indicator Logs for TensorBoard Usage](./console-guide/#5-store-indicator-logs-for-tensorboard-usage). TensorBoard can only be accessed by users logged into the console.
    - **Stop Hyperparameter Tuning** : You can stop hyperparameter tuning in progress.

- **Monitoring**: When you select hyperparameter tuning, you can check the list of monitored instances and basic indicator charts in the Monitoring tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while hyperparameter tuning is being created.

### List of Trainings for Hyperparameter Tuning

Displays a list of trainings auto-generated by hyperparameter tuning. Select a training from the list to check detailed information.

- **Target Metric Value**: Indicates the target metric value.
- **Status** : Shows the status of the training automatically generated by hyperparameter tuning. Please refer to the table below for the main status.

    | Status | Description |
    | --- | --- |
    | CREATED | Training has been created. |
    | RUNNING | Training is in progress. |
    | SUCCEEDED | Training has been completed normally. |
    | KILLED | Training is stopped by the system. |
    | FAILED| This is a failed state during training. Detailed failure information can be checked through the Log & Crash Search log when log management is enabled. |
    | METRICS_UNAVAILABLE | This is a state where target metrics cannot be collected. |
    | EARLY_STOPPED | Performance (goal metric) is not getting better while training is in progress, so it is in an early-stopped state. |

### Copy Hyperparameter Tuning

Create a new hyperparameter tuning with the same settings as the existing hyperparameter tuning.

1. Select the hyperparameter tuning you want to copy.
2. Click **Copy Hyperparameter Tuning**.
3. The Create Hyperparameter Tuning screen is displayed with the same settings as the existing hyperparameter tuning.
4. If there is any information you would like to change the settings for, make the changes and click **Create Hyperparameter Tuning** to create a hyperparameter tuning.

### Create a Model from Hyperparameter Tuning

Create a model with the best training of hyperparameter tuning in the completed state.

1. Choose the hyperparameter tuning you want to create as a model.
2. Click **Create Model**. Only hyperparameter tuning in the COMPLETE state can be created as a model.
3. You will be taken to the model creation page. After checking the contents, click **Create Model** to create a model.
For more information on model creation, see [the model](./console-guide/#model) documentation.

### Delete Hyperparameter Tuning

Delete a hyperparameter tuning.

1. Select the hyperparameter tuning you want to delete.
2. Click **Delete Hyperparameter Tuning**. Hyperparameter tuning in progress can be stopped and then deleted.
3. Requested deletion cannot be undone. Click **OK** to proceed.

> [Note] Hyperparameter tuning cannot be deleted if the associated model exists:
> Hyperparameter tuning cannot be deleted if the model created by the hyperparameter tuning you want to delete exists. Please delete the model first, then the hyperparameter tuning.

## Training Template

By creating a training template in advance, you can import the values entered into the template when creating training or hyperparameter tuning.

### Create Training Template

For information on what you can set in your training template, see [Creating a training](./console-guide/#create-training).

### List of Training Templates

Displays a list of training templates. Select a training template from the list to view details and change information.

- **Operation**
    - **Change** : You can change training template information.
- **Hyperparameters** : You can check the names of hyperparameters set in the training template on **the Hyperparameters** tab of the detailed screen displayed when you select a training template.

### Copy Training Template

Create a new training template with the same settings as an existing training template.

1. Select the training template you want to copy.
2. Click **Copy Training Template**.
3. The Create training Template screen appears with the same settings as the existing training template.
4. If there is any information you would like to change the settings for, change it and then click **Create Training Template** to create a training template.

### Delete Training Template

Delete the training template.

1. Select the training template you want to delete.
2. Click **Delete Training Template**
3. Requested deletion cannot be undone. Click **OK** to proceed.

## Model

Can manage models of AI EasyMaker's training outcomes or external models as artifacts.

### Create Model

- **Basic Information**: Enter basic information of model.
    - **Name**: Enter model name.
        - If model's framework type is PyTorch, you must enter the same model name as PyTorch model name.
    - **Description**: Enter model description.
- **Framework Information**: Enter Framework Information
    - **Framework**: Select the model's framework.
    - **Framework Version**: Enter Model framework Version.
- **Model Information**: Enter the storage where model's artifacts are stored.
    - **Model Artifact**: Select a repository where model artifacts are saved.
        - **NHN Cloud Object Storage**: Enter the path to Object Storage where the model artifacts are stored.
            - Enter the directory path in the format `obs://{Object Storage API endpoint}/{containerName}/{path}`.
            - If you are using NHN Cloud Object Storage, refer to [Appendix > 1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage](./console-guide/#1-nhn-cloud-object-storage-ai-easymaker) to set permissions. If you do not set permissions, you will not be able to access the model's artifacts and model creation will fail.
        - **NHN Cloud NAS**: Enter the path to the NHN Cloud NAS where the model artifact is stored.
            - Enter the directory path in the format `nas://{NAS ID}:/{path}`.
    - **Parameter**: Enter the model's parameter information.
        -**Parameter name**: Enter the name of the parameter in the model.
        -**Parameter value**: Enter the values of the parameters in the model.

> [Caution] When using NHN Cloud NAS:
Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> [Caution] Retain model artifacts in storage:
> If not retained the model artifacts stored in storage, the creation of endpoints for that model fails.

> [Note] Model Parameter:
> The values entered as model parameters are used when serving the model. Parameters can be used as arguments and environment variables:
> Arguments are used as the parameter name as entered, and environment variables are used with the parameter name converted to screaming snake notation.

> [Note] When creating HuggingFace model:
> When creating a HuggingFace model, you can create the model by entering the ID of the HuggingFace model as a parameter.
> The ID of the HuggingFace model can be found in the URL of the HuggingFace model page.
> For more information, see [Appendix > 11. Framework-specific serving notes](./console-guide/#11).

> [Caution] Supported Type for HuggingFace Model:
> The file type for the HuggingFace model are limited to safetensors.
> Safetensors is a safe and efficient machine learning model developed by HuggingFace.
> Other file types are not supported.

> [Caution] When creating TensorFlow (Triton), PyTorch (Triton), and ONNX (Triton) model:
> The model artifact path you enter must contain the model file and the `config.pbtxt` file in a structure that allows the model to be run with Triton.
> See the example below:

```
model_name/
├── config.pbtxt                              # Model selection file
└── 1/                                        # Version 1 directory
    └── model.savedmodel/                     # TensorFlow SavedModel directory
        ├── saved_model.pb                    # Metagraph and checkpoint data
        └── variables/                        # Model weight directory
            ├── variables.data-00000-of-00001
            └── variables.index
```

### Model List

Model list is displayed. Selecting a model in the list allows to check detailed information and make changes to it.

- **Name**: Model name and description are displayed. Model name and description can be changed by clicking **Change**.
- **Model Artifact Path** displays the storage where the model's artifacts are stored.
- **Status**: Model's status is displayed. For major statuses, see the following table.

    | Status               | Description                                                                              |
    | ------------------ | --------------------------------------------------------------------------------- |
    | CREATE REQUESTED   | Model creation is requested.                                                    |
    | CREATE IN PROGRESS | Resource required for the model is being created.                                        |
    | DELETE IN PROGRESS | Model is being deleted.                                                      |
    | ACTIVE             | Model is created successfully.                                              |
    | CREATE FAILED      | Failed to created a model. If creation fails repeatedly, contact the Customer Center. |
    | DELETE FAILED      | Failed to delete a model. Please try again.                                   |

- **Training Name**: For models created from training, training name that is based is displayed.
- **Training ID**: For models created from training, training ID that is based is displayed.
- **Framework**: Model's framework information is displayed.
- **Parameter**: Model's parameter is displayed. Parameters are used for inference.

### Create Endpoint from Model

Create an endpoint that can serve the selected model.

1. Select the model you want to create as an endpoint from the list.
2. Click **Create Endpoint**.
3. Go to **Create Endpoint** page. After checking the contents, click **Create Endpoint** to create a model.
For more information on creating models, refer to **Endpoint** documents.

### Create Batch Inference in a Model

Create batch inferences with the selected model and view the inference results as statistics.

1. Select the model you want to create with batch inference from the list.
2. Click **Create Batch Inference**.
3. You will be taken to the **Create Batch Inference** page. Check the contents and click Create Batch Inference.
For more information about creating batch inferences, see [Batch Inference](./console-guide/#_54).

### Delete Model

Delete a model.

1. Select the model want to delete from list.
2. Click **Delete Model**.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

> [Note] Unable to delete model if associated endpoint exists:
> You cannot delete model if endpoint created by model want to delete is existed.
> To delete, delete the endpoint created by the model first and then delete the model.

## Evaluate models

Measure the performance of models, and compare performance across different models.

### Create a model evaluation

Batch inferences are automatically created during the model evaluation process.

- **Basic Information**: Enter basic information about the model evaluation.
    - **Name**: Enter a name for the model evaluation.
    - **Description**: Enter a description of the model evaluation.
- **Model Information**: Enter information about the model to evaluate.
    - **Model**: Select the model to evaluate. Only classification models and regression models are supported.
    - **Class name**: Enter a class name for the model.
- **Model Evaluation Instance Information**
    - **Instance type**: Select the instance type to run model evaluation on. This is used for data preprocessing and evaluation calculations.
- **Input Data**: The input data is used to compare the prediction generated by batch inference with the ground truth. It requires the fields used in the inference and the answer fields.
    - **Data path**: Enter the path where the input data is located.
        - **Input Data Type**: Select the format of the input data. CSV and JSONL formats are supported, and the file extension must be .csv and .jsonl, respectively.
        - **Evaluation Target Field**: Enter the field name for the ground truth label.
- **Batch Inference Output Data**
    - **Data path**: Enter the path where the results of batch inference will be stored.
- **Batch Inference Information**
    - **Instance Type**: Select the instance type to run batch inference on.
    - **Number of Instances**: Enter the number of instances to perform batch inference on.
    - **Number of Pods**: Enter the number of pods for batch inference.
    - **Batch Size**: Enter the number of data samples that are processed simultaneously in one inference job.
    - **Inference Time Limit (Seconds)**: Enter the time limit for batch inference. Sets the maximum allowable time for a single inference request to be processed and return results.
- **Additional Settings**
    - **Maximum Duration**: Specify the maximum progress time until the model evaluation is complete. Model evaluations that exceed the maximum progress time are terminated.
    - **Log management**: Logs generated during model evaluation can be stored in the NHN Cloud Log & Crash Search service.
        - For more information, see [Appendix > 2. How to use NHN Cloud Log & Crash Search service and check logs](./console-guide/#2-nhn-cloud-log-crash-search).

> [Caution] When using NHN Cloud NAS
Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> [Caution] Input data size:
The size of the input data used to evaluate the model must be 20GB or less.

> [Caution] Number of classes in a classification model evaluation:
The number of classes in a classification model evaluation must be 50 or fewer.

### Model Evaluation List

A list of model evaluations is displayed. Select a model evaluation in the list to view details and make changes to the information.

- **Name**: Displays the name of the model evaluation.
- **Model**: Displays the name of the model used to evaluate the model.
- **Status**: Displays the status of the model evaluation. See the table below for the main statuses.

    | Status                                                      | Description                                                                                 |
    |----------------------------------------------------------|------------------------------------------------------------------------------------|
    | CREATE REQUESTED                                         | Model evaluation creation is requested.                                                               |
    | CREATE IN PROGRESS                                       | Model evaluation is being created.                                                                |
    | CREATE FAILED                                            | Model evaluation creation has failed. Please try again.                                                     |
    | RUNNING                                                  | Model evaluation is in progress.                                                                |
    | COMPLETE IN PROGRESS, FAIL MODEL EVALUATION IN PROGRESS  | Resources used for model evaluation are being cleaned up.                                                       |
    | COMPLETE                                                 | Model evaluation completed successfully.                                                            |
    | STOP IN PROGRESS                                         | Model evaluation is stopping.                                                                |
    | STOPPED                                                  | Model evaluation has been stopped at the user's request.                                                        |
    | FAIL MODEL EVALUATION                                    | Model evaluation has failed. If log management is enabled, you can check the detailed failure information in the Log & Crash Search logs. |
    | DELETE IN PROGRESS                                       | Model evaluation is being deleted.                                                                |

- **Task**
    - **Stop**: You can stop an ongoing model evaluation.

### Classification Model Evaluation Metrics

- **PR AUC**: The area under Precision-Recall (PR) curve. It is effective for measuring a model's classification performance on unbalanced datasets.
- **ROC AUC**: The area under Recall-False Positive Rate (ROC) curve indicates model performance. The closer the value is to 1, the better the performance.
- **Log Loss**: The loss value calculated using a logarithmic function based on the difference between predicted probabilities and actual labels. A lower value indicates more reliable model predictions.
- **F1 Score**: The harmonic mean of precision and recall. It is useful when there is class imbalance, and values closer to 1 indicate better performance.
- **Precision**: The proportion of positive predictions that are actually positive. It focuses on reducing false positives.
- **Recall**: The proportion of actual positives that are correctly predicted as positive by the model. It is important for reducing false negatives.
- **Precision-recall curve**: A curve visualizing the relationship between precision and recall at various thresholds. It is useful for adjusting the model's decision threshold.
- **ROC curve**: A curve showing the relationship between recall and false positive rate at different thresholds. It is used for setting classification thresholds and comparing models.
- **Precision-recall curve by threshold**: A graph illustrating how precision and recall change at a specific threshold. It serves as a reference when defining operational criteria.
- **Confusion matrix**: A matrix that categorizes prediction results into true positives, false positives, false negatives, and true negatives. It allows easy identification of error types for each class.

### Regression Model Evaluation Metrics

- **MAE(mean absolute error)**: The mean absolute error between actual and predicted values. It intuitively shows the magnitude of prediction errors.
- **MAPE(mean absolute percentage error)**: The mean of prediction errors divided by actual values. Since it is ratio-based, it may be unsuitable for data with values close to zero.
- **R-squared(coefficient of determination)**: Indicates how well the model explains the actual data, with values closer to 1 representing higher explanatory power.
- **RMSE(root mean squared error)**: The square root of the mean squared error. It is more sensitive to large errors and interprets results on the same scale as the original units.
- **RMSLE(root mean squared logarithmic error)**: Calculated from the difference between log-transformed actual and predicted values. It is less sensitive to differences in magnitude and useful for evaluating exponentially growing data.

### Compare Model Evaluations

Compare evaluation metrics across models.

1. In the list, select the model evaluations to compare.
2. Click **Compare**.

### Delete Model Evaluation

Delete a model evaluation.

1. Select the model evaluation to delete.
2. Click **Delete**. An ongoing model evaluation can be stopped and then deleted.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

## Endpoint

Create and manage endpoints that can serve the model.

### Create Endpoint

- **Enable API Gateway Service**
    - AI EasyMaker endpoints create API endpoints and manage APIs through NHN Cloud API Gateway service. API Gateway service must be enabled to take advantage of endpoint feature.
    - For more information on API Gateway services and fees, please refer to the following documents:
        - [API Gateway Service Guide](https://docs.nhncloud.com/en/Application%20Service/API%20Gateway/en/overview/)
        - [API Gateway Usage Fee](https://www.nhncloud.com/kr/pricing/by-service?c=Application%20Service&s=API%20Gateway)
- **Endpoint**: Select whether to add stage to new or existing endpoint.
    - **Create as New Endpoint**: Create new endpoint. Endpoint is created in API Gateway with new service and default stage.
    - **Add New Stage at Default Endpoint**: Endpoint is created as new stage in the service of API Gateway of existing endpoint. Select existing endpoint to add a stage.
- **Endpoint name**: Enter the endpoint name. Endpoint names cannot be duplicated.
- **Stage Name**: When adding new stage on existing endpoint, enter name for new stage. Stage names cannot be duplicated.
- **Description**: Enter the description of endpoint stage.
- **Instance Information**: Enter instance information for the model to be served.
    - **Instance Type**: Select instance type.
    - **Number of Instances**: Enter the number of drives for instance.
    - **Autoscaler**: The autoscaler is a feature that automatically adjusts the number of nodes based on resource usage policies. The autoscaler is set on a per-stage basis.
        - **Enable/Disable**: Select whether to enable the autoscaler. If enabled, the number of instances will scale in or out based on the instance load.
        - **Minimum number of nodes**: The minimum number of nodes that can be scaled down
        - **Maximum number of nodes**: The maximum number of nodes that can be scaled up
        - **Scale-down**: Set whether to enable node scale-down
        - **Resource Usage Threshold**: The default for resource usage threshold, which is the reference point for a scale down
        - **Threshold Duration (min)**: The resource usage duration at or below the threshold for the nodes to be scaled down
        - **Scale-up to scale-down latency (min)**: Delay before starting to monitor for scale-down targets after scaling up
- **Stage Information**: Enter the information for model artifacts to deploy to endpoint. When you deploy the same model to multiple stage resources, requests are distributed and processed.
    - **Model**: Select a model you want to deploy to the endpoint. If you haven't created a model, create one first. For information on model framework-specific serving, please see [Appendix > 11. Serving by Framework](./console-guide/#11).
    - **API Gateway Resource Path**: Enter the path to the API resource to which the model is deployed. For example, if you set it to `/inference`, you can request the inference API with `POST https://{enpdoint-domain}/inference`.
    - **Resource Allocation (%)**: Enter the resource you want to allocate to the model. Allocate a fixed percentage of the actual resource usage by instance.
        - **cpu**: Enter the CPU quota. Enter if you are allocating directly without using an allocation percentage (%).
        - **memory**: Enter the Memory quota. Enter if you are allocating directly without using an allocation percentage (%).
        - **gpu**: Enter the GPU quota. Enter if you are allocating directly without using an allocation percentage (%).
    - **Description**: Enter a stage resource description.
    - **Pod Autoscaler**: The feature to automatically adjust the number of pods based on the request volume of your model. The autoscaler is set on a per-model basis.
        - **Enable/Disable**: Select whether to use the auto scaler. When enabled, the number of Pods scales in or out based on the model load.
        - **Scaling Unit**: Enter the Pod Scaling Unit.
            - **CPU**: Adjust the pod count depending on CPU usage.
            - **Memory**: Adust the memory count depending on CPU usage.
        - **Threshold (%)**: The threshold value per increment at which the Pod will be scaled up.
    - **Resource Information:**: You can see the resources you're actually using. Allocates resource room usage to each model based on the quota for the model you entered. For more information, please see [Appendix > 9. Resource Information](./console-guide/#9).

> [Note] API Specification for Inference Request:
> The AI EasyMaker service provides endpoints based on the open inference protocol (OIP) specification. For the endpoint API specification, see [Appendix > 10. Endpoint API specification](./console-guide/#10-api).
> To use a separate endpoint, refer to the resources created in the API Gateway service and create a new resource to use it.
> For more information about the OIP specification, see [OIP specification](https://github.com/kserve/open-inference-protocol).

> [Note] Time to create endpoints:
> Endpoint creation can take several minutes.
> Creation of the initial resources (notebooks, training, experiments, endpoints) takes additional few minutes to configure the service environment.

> [Note] Restrictions on API Gateway service resource provision when creating endpoints:
> When you create a new endpoint, create a new API Gateway service.
> Adding new stage on existing endpoint creates new stage in API Gateway service.
> If you exceed the default provision in [API Gateway Service Resource Provision Policy](https://docs.nhncloud.com/en/TOAST/en/resource-policy/#resource-provision-policy-for-api-gateway-service), you might not be able to create endpoints in AI EasyMaker. In this case, adjust API Gateway service resource quota.

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
- **View API Gateway Settings**: Click **View Settings** to see settings that AI EasyMaker has deployed to API Gateway stage.
- **View API Gateway Statistics**: Click **View Statistics** to view API statistics of endpoints.
- **Instance Type**: Displays endpoint instance type the model is serving.
- **Number of Work Nodes/Pods In Progress**: Displays the number of nodes and pods being used by endpoint.
- **Stage Resource**: Displays information about model artifacts deployed to the stage.
- **Monitoring**: When you select the endpoint stage, you can see a list of monitored instances and basic metrics charts in the **Monitoring** tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while an endpoint stage is being created.
- **API Statistics**: You can check the API statistics information of the endpoint stage in the **API Statistics** tab of the details screen that appears when you select the endpoint stage.
    - The **API Statistics** tab is disabled while the endpoint stage is being created.

> [Caution] Precautions when changing settings for API Gateway created by AI EasyMaker:
> When creating an endpoint or an endpoint stage, AI EasyMaker creates API Gateway services and stages for the endpoint.
> Please note the following precautions when changing API Gateway services and stages created by AI EasyMaker directly from API Gateway service console.
>
> 1. Avoid deleting API Gateway services and stages created by AI EasyMaker. Deletion may prevent the endpoint from displaying API Gateway information correctly, and changes made to endpoint may not be applied to API Gateway.
> 2. Avoid changing or deleting resources in API Gateway resource path that was entered when creating endpoints. Deletion may cause the endpoint's inference API call to fail
> 3. Avoid adding resources in API Gateway resource path that was entered when creating endpoints. The added resources may be deleted when adding or changing endpoint stages.
> 4. In the stage settings of API Gateway, do not disable **Backend Endpoint Url Redifinition** or change the URL set in API Gateway resource path. If you change the url, endpoint's inference API call might fail.
> Other than above precautions, other settings are available with features provided by API Gateway as necessary.
> For more information about how to use API Gateway, refer to [API Gateway Console Guide](https://docs.nhncloud.com/en/Application%20Service/API%20Gateway/en/console-guide/).

> [Note] Recovery method when the stage's API Gateway is in 'Deployment Failed' status:
> If stage settings of AI EasyMaker endpoint are not deployed to the API Gateway stage due to a temporary issue, deployment status is displayed as failed.
> In this case, you can deploy API Gateway stage manually by clicking Select Stage from the Stage list > View API Gateway Settings > 'Deploy Stage' in the bottom detail screen.
> If this guide couldn’t recover the deployment status, please contact the Customer Center.

### Create Stage Resource

Add a new resource to an existing endpoint stage.

- **Model**: Select the model you want to deploy to your endpoints. If you have not created a model, please create one first.

- **Resource quota(%)**: Enter the resources you want to allocate to the model. Allocate a fixed percentage of the instance's resource room usage.
    - **CPU**: Enter the CPU quota. Enter if you are allocating directly without using an allocation percentage (%).
    - **Memory**: Enter the memory quota. Enter if you are allocating directly without using an allocation percentage (%).

- **Number of Pods**: Enter a number of pods in the stage resource.

- **Description**: Enter a description for the stage resource.

- **Pod Auto Scaler**: The feature to automatically adjust the number of Pods based on the request volume of your model. The autoscaler is set on a per-model basis.
    - **Enable/Disable**: Select whether to use the auto scaler. If enabled, the number of Pods will scale in or out based on the model load.
    - **Scale Unit**: Enter the pod scale unit.
        - **CPU**: Adjust the pod count depending on CPU usage.
        - **Memory**: Adjust the pod count depending on memory usage.
        - **Threshold value**: The threshold value per increment that the Pod will be scaled to.

### Stage Resource List

A list of resources created under the endpoint stage is displayed.

- **Status** : Shows the status of stage resource. Please refer to the table below for the main status.

    | Status | Description |
    | --- | --- |
    | CREATE REQUESTED |  Creating stage resource requested. |
    | CREATE IN PROGRESS |  Stage resource is being created. |
    | Training is properly completed. |  Stage resource is being deleted. |
    | ACTIVE |  Stage resource is deployed normally. |
    | CREATE FAILED |  Creating stage resource failed. Please try again. |

- **Model Name**: The name of the model deployed to the stage.
- **API Gateway Resource Path**: The inference URL of the model deployed to the stage. API clients can request inference at the displayed URL. For more information, see [Appendix > 10. Endpoint API Specfication](./console-guide/#10-api).
- **Number of Pods**: Shows the number of healthy pods and total pods in use on the resource.

### Call Endpoint Inference

1. When you click Stage in Endpoint > Endpoint Stage, Stage details screen is displayed at the bottom.
2. Check the API Gateway resource path from the details screen on the Stage Resource tab.
3. When the API Gateway resource path is called the HTTP POST Method, inference API is called.
    - Request and response specifications of the inference API differ depending on the algorithm user created.

            // Inference API example: Request
            curl --location --request POST '{API Gateway Resource Path}' \
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

### Delete Stage Resource

1. In the endpoint list, click the **endpoint name** to move it to the Endpoint Stage list.
2. In the endpoint stage list, click the endpoint stage on which the stage resource you want to delete is deployed. When you click, the stage details screen will be displayed at the bottom.
3. On the **Stage Resource** tab of the details screen, select the stage resource you want to delete.
3. Click **Delete Stage Resource**.
4. Requested deletion cannot be undone. Click **OK** to proceed.

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

> [Caution] Delete stage of API Gateway service when deleting the endpoint stage:
> Deleting an endpoint stage in AI EasyMaker also deletes the stage in API Gateway service from which the endpoint's stage is deployed.
> If there is an API running on the API Gateway stage to be deleted, please be noted that API calls cannot be made.

### Delete Endpoint

Delete an endpoint.

1. Select the endpoint want to delete from endpoints list.
2. You cannot delete an endpoint if there is stage under endpoint other than the default stage. Please delete the other stages first.
3. Click **Delete Endpoint**.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

> [Caution] Delete API Gateway service when deleting the endpoint stage:
> Deleting an endpoint stage in AI EasyMaker also deletes API Gateway service from which the endpoint's stage was deployed.
> If there is API running on the API Gateway service to be deleted, please be noted that API calls cannot be made.

## Batch Inference

Provides an environment to make batch inferences from an AI EasyMaker model and view inference results in statistics.

### Create Batch Inference

Set up the environment in which batch inference will be performed by selecting an instance and OS image, and enter the paths to the input/output data to be inferred to proceed with batch inference.

- **Basic Information**: Enter basic information about a batch inference.
    - **Batch Inference Name**: Enter a name for the batch inference.
    - **Batch Inference Description**: Enter a description.
- **Instance information**
    - **Instance Type**: Select the instance type to run batch inference on.
    - **Number of Instances**: The number of instances to perform batch inference on.
- **Model Information**
    - **Model**: Select the model from which you want to make a batch inference. If you did not create a model, create one first.
    - **Number of Pods**: Enter the number of pods in the model.
    - **Resource Information**: You can see the actual resources used by the model. The actual usage is split and allocated to each pod based on the number of pods you entered. For more information, see [Appendix > 9. Resource Information](./console-guide/#9).
- **Input Data**
    - **Data Path**: Enter the path to the data that you want to run batch inference on.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
    - **Input Data Type**: Select the type of data you want to run batch inference on.
        - **JSON**: Use valid JSON data from a file as input.
        - **JSONL**: Use JSON lines files where each line is valid JSON as input.
            - Note: [https://jsonlines.org/](https://jsonlines.org/)
    - **Glob Pattern**
        - **Specify File to Include**: Enter a set of files to include in the input data in a Glob pattern.
        - **Specify File to Exclude**: Enter a set of files to exclude from the input data in a Glob pattern.
- **Output Data**
    - **Output Data**: Enter the data storage path to save the batch inference results.
        - Enter the NHN Cloud Object Storage or NHN Cloud NAS path.
- **Additional Settings**
    - **Batch Options**
        - **Batch Size**: Enter the number of data samples that are processed simultaneously in one inference job.
        - **Inference Timeout (in seconds)**: Enter the timeout period for batch inference. You can set the maximum allowable time before a single inference request is processed and results are returned.
    - **Data Storage Size** : Enter the data storage size of the instance to run batch inference.
        - Used only when using NHN Cloud Object Storage. Please specify a size large enough to store all the data required for batch inference.
    - **Maximum Batch Inference Time** : Specify the maximum waiting time until batch inference is complete. Batch inference that exceeds the maximum waiting time will be terminated.
    - **Log Management** : Logs generated during batch inference can be stored in the NHN Cloud Log & Crash Search service.
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](./console-guide/#2-nhn-cloud-log-crash-search-service-usage-guide-and-log-inquiry-guide).

> [Caution] When using NHN Cloud NAS:
> Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

> [Caution] Batch inference fails when batch inference input data is deleted:
> Batch inference can fail if you delete input data before batch inference is complete.

> [Caution] When setting input data detailed options:
> If the Glob pattern is not entered properly, batch inference may not work properly because the input data cannot be found.
> When used together with the **Include Glob pattern**, the **Exclude Glob pattern** takes precedence.

> [Caution] When setting batch options:
> You must set the **batch size** and **inference timeout** appropriately based on the performance of the model you are batch inferring.
> If the settings you enter are incorrect, batch inference might not perform well enough.

> [Caution] When using GPU instances:
> Batch inference using GPU instances allocates GPU instances based on the number of Pods
> If `Number of Pods / Number of GPUs` is not divisible by an integer, you may encounter unallocated GPUs
> Unallocated GPUs are not used by batch inference, so set the number of Pods appropriately to use GPU instances efficiently.

### Batch Inference List

Displays a list of batch inferences. Select a batch inference from the list to check the details and change the information.

- **Inference Time**: Displays how long the batch inference has been running.
- **Status** : Displays the status of batch inference. Please refer to the table below for the main status.

    | **Failed Training** : Indicates the number of failed lessons. | **Best Training**: Indicates the target metric information of the training that recorded the highest target metric value among the training automatically generated by hyperparameter tuning. |
    | --- | --- |
    | **Status** : Shows the status of hyperparameter tuning. Please refer to the table below for the main status. | You have requested to create a batch inference. |
    | **API Gateway Status**: Displays API Gateway status information for default stage of endpoint. Please refer to the table below for main status. | This is a state in which resources necessary for batch inference are being created. |
    | Description | Batch inference is in progress. |
    | Resources required for hyperparameter tuning are being created. | Batch inference is stopped at the user's request. |
    | COMPLETE | Batch inference has been completed successfully. |
    | STOP IN PROGRESS | Batch inference is stopping. |
    | FAIL BATCH INFERENCE | This is a failed state during batch inference. Detailed failure information can be checked through the Log & Crash Search log when log management is enabled. |
    | Stage resource is being deleted. | The batch inference creation failed. If creation continues to fail, please contact customer service. |
    | FAIL BATCH INFERENCE IN PROGRESS, COMPLETE IN PROGRESS | The resources used for batch inference are being cleaned up. |

- **Operation**
    - **Stop**: You can stop batch inference in progress.
- **Monitoring**: When you select a batch inference, you can check the list of monitored instances and basic indicator charts in the **Monitoring** tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while batch inference is being created.

### Copy Batch Inference

Create a new batch inference with the same settings as an existing batch inference.

1. Select the batch inference you want to copy.
2. Click **Copy Batch Inference**.
3. The Create batch inference screen appears with the same settings as an existing batch inference.
4. If there is any information you would like to change the settings for, make the changes and then click **Create Batch Inference** to create the batch inference.

### Delete Batch Inference

Delete a batch inference.

1. Select the batch inference you want to delete.
2. Click **Delete Batch Inference**. Batch inference in progress can be deleted after stopping.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

## Private Image

User-personalized container images can be used to drive notebooks, training, and hyperparameter tuning.
Only private images derived from the notebook/deep learning images provided by AI EasyMaker can be used when creating resources in AI EasyMaker.
See the table below for the base images in AI EasyMaker.

#### Notebook Image

 Image Name | CoreType | Framework | Framework version | Python version | Image address |
| --- | --- | --- | --- | --- | --- |
| Ubuntu 22.04 CPU Python Notebook     | CPU  | Python     | 3.10.12  | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/python-notebook:3.10.12-cpu-py310-ubuntu2204   |
| Ubuntu 22.04 GPU Python Notebook     | GPU  | Python     | 3.10.12  | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/python-notebook:3.10.12-gpu-py310-ubuntu2204   |
| Ubuntu 22.04 CPU PyTorch Notebook    | CPU  | PyTorch    | 2.0.1    | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/pytorch-notebook:2.0.1-cpu-py310-ubuntu2204    |
| Ubuntu 22.04 GPU PyTorch Notebook    | GPU  | PyTorch    | 2.0.1    | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/pytorch-notebook:2.0.1-gpu-py310-ubuntu2204    |
| Ubuntu 22.04 CPU TensorFlow Notebook | CPU  | TensorFlow | 2.12.0   | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/tensorflow-notebook:2.12.0-cpu-py310-ubuntu2204|
| Ubuntu 22.04 GPU TensorFlow Notebook | GPU  | TensorFlow | 2.12.0   | 3.10   | fb34a0a4-en1-registry.container.nhncloud.com/easymaker/tensorflow-notebook:2.12.0-gpu-py310-ubuntu2204|

#### Deep Learning Images

| Image Name | CoreType | Framework | Framework version | Python version | Image address |
| --- | --- | --- | --- | --- | --- |
| Ubuntu 22.04 CPU PyTorch Training    | CPU  | PyTorch    | 2.0.1    | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/pytorch-train:2.0.1-cpu-py310-ubuntu2204        |
| Ubuntu 22.04 GPU PyTorch Training    | GPU  | PyTorch    | 2.0.1    | 3.10   | fb34a0a4-en1-registry.container.nhncloud.com/easymaker/pytorch-train:2.0.1-gpu-py310-ubuntu2204        |
| Ubuntu 22.04 CPU TensorFlow Training | CPU  | TensorFlow | 2.12.0   | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/tensorflow-train:2.12.0-cpu-py310-ubuntu2204 |
| Ubuntu 22.04 GPU TensorFlow Training | GPU  | TensorFlow | 2.12.0   | 3.10   | fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/tensorflow-train:2.12.0-gpu-py310-ubuntu2204 |

> [Note] Limitations on using private images:
>
> Only private images derived from base images provided by AI EasyMaker can be used.
> Only NHN Container Registry (NCR) can be integrated as a container registry service where private images are stored. (As of December 2023)

### Create Private Image

The following document explains how to create a container image with an AI EasyMaker-based image using Docker, and using a private image for notebooks in AI EasyMaker.

1. Create a DockerFile of private image.

            FROM fb34a0a4-kr1-registry.container.nhncloud.com/easymaker/python-notebook:3.10.12-cpu-py310-ubuntu2204 as easymaker-notebook
            RUN conda create -n example python=3.10
            RUN conda activate example
            RUN pip install torch torchvision

2. Build a private image and push to the container registry
Build an image with a Dockerfile and save (push) the image to the NCR registry.

            docker build -t {image name}:{tags} . .
            docker tag {image name}:{tag} docker push {NCR registry address}/{image name}:{tag}
            docker push {NCR registry address}/{image name}:{tag} .


            (Example)
            docker build -t custom-training:v1 .
            docker tag custom-training:v1 example-kr1-registry.container.nhncloud.com/registry/custom-training:v1
            docker push example-kr1-registry.container.nhncloud.com/registry/custom-training:v1

3. Create a private image in AI EasyMaker of the image you saved (pushed) to the NCR.

    1. Go to the **Image** menu in the AI EasyMaker console.
    2. Click the **Create image** button to enter the information for the image you created.
        - Name, description: Enter a name and description for the image.
        - Address: Enter the address of the registry image.
        - Type: Enter the type of container image. Select **Notebook** or **Training**.
        - Account: Select the account for the AI EasyMaker notebook/learning node to access your registry storage.
            - New: Register a new registry account.
                - Name, Description: Enter a name and description for the registry account.
                - Category: Select a container registry service.
                - ID: Enter the ID of the registry storage.
                - Password: Enter the password for the registry storage.
            - Use an existing account: Select a registry account that is already registered.

4. Create a notebook with the private image you created.
    1. Go to the **Notebook** menu. Click the **Create notebook** button to go to the Create notebook page.
    2. Under Image information, click the **Private Image** tab.
    3. Select a private image to use as the notebook container image.
    4. After filling out and creating the other notebook information, the notebook will be running with your private image.

> [Note] Where to use private images:
> Private images can be used for notebooks, training, and hyperparameter tuning to create resources.

> [Note] Container registry service: NHN Container Registry (NCR)
> Only NCR service can be used as a container registry service. (As of December 2023)
> Enter the following values for the account ID and password for the NCR service.
> ID: User Access Key of NHN Cloud user account
> Password: User Secret Key of NHN Cloud user account

## Registry Account

In order for AI EasyMaker to pull an image from a user's registry where private images are stored to power the container, they need to be logged into the user's registry.
If you save your login information with a registry account, you can reuse it in images linked to that registry account.
To manage your registry accounts, go to the **Image** menu in the AI EasyMaker console, then select the **Registry Account** tab.

### Create Registry Account

Create a new registry account.

- Name: Enter the name of registry account.
- Description: Enter a description of the registry account.
- Category: Select a container registry service.
- ID: Enter the ID of the registry account.
- Password: Enter the password for the registry account.

### Modify Registry Account

#### Modify registry ID and password

- Click **Change Registry Account**.
- Enter an ID and password, then click **Confirm**.

> [Note]
> When you change your registry account, you sign in to the registry service with the changed username and password when using images associated with that account.
> If you enter an incorrect registry username and password, the login during a private image pull fails and the resource creation fails.
> If there are resources being created with a private image that has a registry account associated with it, or if there are studies and hyperparameters in progress, you cannot modify them.

#### Registry Account > Change Name, Description

1. In the Registry Accounts list, select the account you want to change.
2. Click **Change** on the bottom screen.
3. After changing the name and description, click the **Confirm** button.

### Delete Registry Account

Select the registry account you want to delete from the list, and click **Delete Registry Account**.

> [Note]
> You cannot delete a registry account associated with an image. To delete, delete the associated image first and then delete the registry account.

## Pipeline

ML Pipeline is a feature for managing and executing portable and scalable machine learning workflows.
You can use the Kubeflow Pipelines (KFP) Python SDK to write components and pipelines, compile pipelines into intermediate representation YAML, and run them in AI EasyMaker.

> [Note] What is a pipeline?
> A pipeline is a definition of a workflow that combines one or more components to form a directed acyclic graph (DAG).
> Each component runs a single container during execution, which can generate ML artifacts.

> [Note] What are ML artifacts?
> Components can take inputs and produce outputs. There are two types of I/O types. Parameters and artifacts:
>
> 1. Parameters are useful for passing small amounts of data between components.
> 2. Artifact types are for ML artifact outputs, such as datasets, models, metrics, etc. Provides a convenient mechanism for saving to object storage.

> [Note] View Pipeline Execution logs
> The feature to view console output generated while executing a pipeline is not provided.
> To check the logs of pipeline code, use the [SDK's Log Send feature] (./sdk-guide/#nhn-cloud-log-crash-search) to send the logs to Log & Crash Search.

Most pipelines aim to produce one or more ML artifacts, such as datasets, models, evaluation metrics, etc.

> [Reference] Kubeflow Pipelines (KFP) official documentation
>
> - [KFP User Guide](https://www.kubeflow.org/docs/components/pipelines/user-guides/)
> - [KFP SDK Reference](https://kubeflow-pipelines.readthedocs.io/en/stable/)

### Upload a Pipeline

Upload a pipeline.

- **Name**: Enter a pipeline name.
- **Description**: Enter description.
- **File registration**: Select the YAML file to upload.

> [Note] Pipeline upload time:
> Uploading a pipeline can take a few minutes.
> The initial resource creation requires an additional few minutes of time to configure the service environment.

### Pipeline List

A list of pipelines is displayed. Select a pipeline in the list to view details and make changes to the information.

- **Status**: The status of the pipeline is displayed. See the table below for key statuses.

    | Status                | Description                             |
    |--------------------|--------------------------------|
    | CREATE REQUESTED   | Pipeline creation has been requested.           |
    | CREATE IN PROGRESS | Pipeline creation is in progress.         |
    | CREATE FAILED      | Pipeline creation failed. Try again. |
    | ACTIVE             | The pipeline was created successfully.        |

### Pipeline Graph

A pipeline graph is displayed. Select a node in the graph to see more information.

A graph is a pictorial representation of a pipeline. Each node in the graph represents a step in the pipeline, with arrows indicating the parent/child relationship between the pipeline components represented by each step.

### Delete a Pipeline

Delete the pipeline.

1. Select the pipeline you want to delete.
2. Click **Delete Pipeline**. You can't delete a pipeline while it's being created.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

> [Note] Cannot delete a pipeline if an associated pipeline schedule exists:
> You cannot delete a pipeline if a schedule created with the pipeline you want to delete exists. Delete the pipeline schedule first, then delete the pipeline.

## Run a Pipeline

You can run and manage your uploaded pipelines in AI EasyMaker.

### Create a Pipeline Run

Run the pipeline.

- **Basic Information**
    - **Name**: Enter a name for the pipeline run.
    - **Description**: Enter description.
    - **Pipeline**: Select the pipeline you want to run.
    - **Experiment**: Select an experiment that will include pipeline execution. Experiments group related pipeline runs. If no experiments have been created, click **Add** to create an experiment.
- **Execution Information**
    - **Execution Parameters**: Enter a value if the pipeline has defined input parameters.
    - **Execution Type**: Select the type of pipeline execution. If you select **One-time**, the pipeline runs only once. To run the pipeline repeatedly at regular intervals, select **Enable Recurring Run** and then see [Create Recurring Run](./console-guide/#_82) to configure recurring runs.
- **Instance Information**
    - **Instance Type**: Select the instance type to run the pipeline on.
    - **Number of Instances**: Enter the number of instances to use to run the pipeline.
- **Additional Settings**
    - **Boot Storage Size**: Enter the boot storage size of the instance on which you want to run the pipeline.
    - **NHN Cloud NAS**: You can connect an **NHN Cloud NAS** to the instance where you want to run the pipeline.
        - **The name of the mount directory**: Enter the name of the directory to mount on the instance.
        - **NAS Path**: Enter the path in the following format: `nas://{NAS ID}:/{path}`.
    - **Manage Logs**: Logs that occur during pipeline execution can be stored in the NHN Cloud Log & Crash Search service.
        - For more information, refer to [Appendix > 2. NHN Cloud Log & Crash Search service usage guide and checking logs](./console-guide/#2-nhn-cloud-log-crash-search).

> [Caution] If you are using NHN Cloud NAS:
> Only NHN Cloud NAS created in the same project as AI EasyMaker is available.

> [Note] Pipeline run generation time:
> Creating a pipeline run can take a few minutes.
> The initial resource creation requires an additional few minutes of time to configure the service environment.

### Pipeline Run List

A list of pipeline runs is displayed. Select a pipeline run in the list to view details and make changes to the information.

- **Status**: The status of the pipeline execution is displayed. See the table below for key statuses.

    | Status                           | Description                                                                                    |
    |-------------------------------|---------------------------------------------------------------------------------------|
    | CREATE REQUESTED              | Pipeline execution creation is requested.                                                               |
    | CREATE IN PROGRESS            | Pipeline run creation is in progress.                                                             |
    | CREATE FAILED                 | Pipeline execution creation failed. Try again.                                                     |
    | RUNNING                       | Pipeline execution is in progress.                                                                |
    | COMPLETE IN PROGRESS          | The resources used to run the pipeline are being cleaned up.                                                       |
    | COMPLETE                      | The pipeline execution has completed successfully.                                                            |
    | Hyperparameter tuning is stopped at the user's request.              | The pipeline is stopping running.                                                                |
    | STOPPED                       | The pipeline execution has been stopped at the user's request.                                                        |
    | FAIL PIPELINE RUN IN PROGRESS | The resources used to run the pipeline are being cleaned up.                                                       |
    | FAIL PIPELINE RUN             | The pipeline execution has failed. Detailed failure information can be found in the Log & Crash Search log when log management is enabled. |

- **Operation**
    - **Stop**: You can stop running a pipeline in progress.
- **Monitoring**: When you select Run a pipeline from the list, you can see a list of monitored instances and a basic metrics chart on the **Monitoring** tab of the detail screen that appears.
    - The **Monitoring** tab is disabled while a pipeline run is being created.

### Pipeline Run Graph

A graph of the pipeline run is displayed. Select a node in the graph to see more information.

The graph is a pictorial representation of the pipeline execution. This graph shows the steps that have already been executed and the steps that are currently executing during pipeline execution, with arrows indicating the parent/child relationship between the pipeline components represented by each step. Each node in the graph represents a step in the pipeline.

With node-specific details, you can download the generated artifacts.

> [Caution] Pipeline artifact storage cycle:
> Artifacts older than 120 days are automatically deleted.

### Stop Pipeline Run

Stop running pipelines in progress.

1. Select the pipeline execution you want to stop from the list.
2. Click **Stop running**.
3. The requested action can't be canceled. Click **Confirm** to continue.

> [Note] How long it takes to stop running a pipeline:
> Stopping pipeline execution can take a few minutes.

### Copy Pipeline Run

Create a new pipeline run with the same settings as an existing pipeline run.

1. Select the pipeline run you want to copy.
2. Click **Copy Pipeline Run**.
3. The Create pipeline run screen displays with the same settings as an existing pipeline run.
4. If you want to change any settings, make any changes, and then click **Create Pipeline Run**.

### Delete a Pipeline Run

Delete a pipeline run.

1. Select the pipeline run you want to delete.
2. Click **Delete Pipeline Run**. You cannot delete a pipeline run that is in progress.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

## Pipeline Recurring Run

You can create and manage a recurring run to periodically run the uploaded pipeline repeatedly in AI EasyMaker.

### Create a Recurring Run

Create a recurring run to run the pipeline in periodic iterations.

For information beyond the items below that you can set in creating a pipeline schedule, see [Create Recurring Run](./console-guide/#_75).

- **Execution Information**
    - **Execution Type**: Select the type of pipeline execution. If you select **Enable Recurring Run**, the pipeline will repeat periodically. Select **One-time** to run the pipeline only once.
    - **Trigger Type**: Select the type of pipeline execution trigger. You can choose **Time Interval** or **Cron Expression**.
        - To run a pipeline repeatedly with a time interval setting, select a **Time Interval** and enter a number and time units.
        - To run the pipeline repeatedly through a Cron expression setup, select **Cron Expression**, and then enter a Cron expression.
    - **Setting up Concurrency**: Depending on the trigger cycle (**time interval** or **Cron expression**), a new pipeline run may be created before the previously created pipeline run ends. You can specify a maximum number of concurrent runs to limit the number of runs in parallel.
    - **Start Time**: You can set the start time of a pipeline recurring run. Generates pipeline executions at the interval you set when not entered.
    - **End Time**: You can set the end time of a pipeline recurring run. On no input, generate pipeline execution until stopped.
    - **Catching up on missed runs**: If a pipeline run falls behind recurring run, determine if it needs to be caught up.
        - For example, if a pipeline recurring run is briefly stopped and later restarted, Setting **Use** will catch up on missed pipeline runs.
        - If the pipeline handles backfill internally, it should be **disabled** to prevent duplicate backfill operations.

> [Note] How long it takes to create a pipeline recurring run:
> Creating a recurring run can take a few minutes.
> The initial resource creation requires an additional few minutes of time to configure the service environment.

> [Note] Cron expression format:
> The Cron expression uses six space-separated fields to represent the time.
> For more information, see the [Cron Expression Format](https://pkg.go.dev/github.com/robfig/cron#hdr-CRON_Expression_Format) documentation.

### Pipeline Recurring Runs

A list of pipeline schedules is displayed. Select a pipeline recurring run in the list to view details and make changes to the information.

- **Status**: The status of the pipeline recurring run is displayed. See the table below for key statuses.

    | Status                           | Description                                          |
    |-------------------------------|---------------------------------------------|
    | CREATE REQUESTED              | Pipeline recurring run creation has been requested.                     |
    | CREATE FAILED                 | Pipeline recurring run creation failed. Try again.           |
    | ENABLED                       | The pipeline recurring run has started normally.                  |
    | ENABLED(EXPIRED)              | The pipeline recurring run started successfully but has passed the end time you set. |
    | DISABLED                      | The pipeline recurring run has been stopped at the user's request.              |

- **Manage Execution**: When you select a pipeline recurring run in the list, you can view the list of runs generated by the pipeline recurring run on the **Manage Run** tab of the detail screen that appears.

### Start and Stop Recurring Run

Stop a started pipeline recurring run or start a stopped pipeline recurring run.

1. Select the pipeline recurring run you want to start or stop from the list.
2. Click **Start Recurring Run** or **Stop Recurring Run**.

### Copy a Pipeline Recurring Run

Create a new pipeline recurring run with the same settings as an existing pipeline recurring run.

1. Select the pipeline recurring run you want to copy.
2. Click **Copy Pipeline Recurring Run**.
3. The Create pipeline schedule screen displays with the same settings as an existing pipeline schedule.
4. Make any changes to the settings you want to make, and then click **Create Pipeline Recurring Run**.

### Delete a pipeline recurring run

Delete a pipeline recurring run.

1. Select the pipeline recurring run you want to delete.
2. Click **Delete Pipeline Recurring Run**.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

> [Note] You cannot delete a pipeline schedule if an associated pipeline run is in progress:
> You cannot delete a run generated by the pipeline schedule you want to delete if it is in progress. Delete the pipeline schedule after the pipeline run is complete.

<a id="rag"></a>

## RAG

Retrieval-Augmented Generation (RAG) is a technology that vectorizes and stores users' documents, retrieves content related to the question, and improves the accuracy of Large Language Model (LLM) responses. AI EasyMaker allows you to integrate vector store, embedding model, and LLM to create and manage RAG systems.

<a id="rag_create"></a>

### Create a RAG

Create a new RAG.

- **Enalbe API Gateway service**
    - AI EasyMaker RAG creates and manages API enpoints by using the NHN Cloud API Gateway service. To use RAG feature, you must enable the API Gateway service.
    - For more information and fee about the API Gateway service, see below:
        - [About API Gateway Service](https://docs.nhncloud.com/ko/Application%20Service/API%20Gateway/ko/overview/)
        - [API Gateway Fee](https://www.nhncloud.com/kr/pricing/by-service?c=Application%20Service&s=API%20Gateway)
- **Default Setting**
    - **Name**: enter a RAG name. It cannot be duplicated.
    - **Description**: Enter a description for the RAG.
    - **Instance flavor**: Select the instance flavor to execute RAG endpoint.
    - **No. of instances**: Enter the number of the instances to execute RAG endpoint.
    - **Prompt**: a prompt to be used for RAG endpoint. You can check the prompt's entire content by clicking **View Content**.
- **Vector Store Setting**
    - **Vector store type**: Select a vector store type.
        - **RDS for PostgreSQL**
            - **Enable RDS for PostgreSQL**
                - AI EasyMaker RAG creates and manages by using NHN Cloud RDS for PostgreSQL. If selecting this option, you must enable RDS for PostgreSQL.
                - For more information and fee about the RDS for PostgreSQL, see below:
                    - [RDS for PostgreSQL Guide](https://docs.nhncloud.com/ko/Database/RDS%20for%20PostgreSQL/ko/overview/)
                    - [RDS for PostgreSQL Usage Fee](https://www.nhncloud.com/kr/pricing/by-service?c=Database&s=RDS%20for%20PostgreSQL)
            - **Instance flavor**: select the instance flavor to be used for RDS for PostgreSQL.
            - **Storage type**: select the storage flavor to be used for RDS for PostgreSQL.
            - **Storage size**: a storage size for RDS for PostgreSQL.
            - **User ID**: enter the user ID to be used for connecting to PostgreSQL.
            - **Password**: enter the password to be used for connecting to PostgreSQL.
            - **Confirm password**: re-enter the password to confirm.
            - **VPC ID**: enter the VPC ID to be used for RDS for PostgreSQL.
            - **Subnet ID**: enter the subnet ID to be used for RDS for PostgreSQL.
        - **PostgreSQL Instance**: use user-created NHN Cloud PostgreSQL instance as a vector store.
            - **User ID**: enter the user ID set when creating PostgreSQL Instance.
            - **Password**: enter the password set when creating PostgreSQL Instance.
            - **VPC ID**: enter the VPC ID used for PostgreSQL Instance.
            - **Subnet ID**: enter the subnet ID used for PostgreSQL Instance.
            - **PostgreSQL instance IP**: enter the IP address of the created PostgreSQL Instance.
    - **Ingestion setting**
        - **Data path**: enter the data path where the documents to be collected are stored in the vector store.
    - **Embedding model**
        - **Model**: select an embedding model to use when vectorizing documents and queries.
        - **Instance flavor**: an instance flavor to execute embedding model.
        - **No. of instances**: the number of the instances to execute embedding model.
- **LLM Setting**
    - **Model**: select the LLM to be used when creating a response.
    - **Instance flavor**: an instance flavor to execute LLM.
    - **No. of instances**: the number of the instances to execute LLM.
- **Addtional Settings**
    - **Log Management**: You can save logs generated during RAG execution to the NHN Cloud Log & Crash Search service.
        - For more information, refer to [Addpendix > 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Check](./console-guide/#2-nhn-cloud-log-crash-search).

> [Caution] When using a PostgreSQL Instance, the port must be set to `15432`.
> For how to create an instance, refer to [PostgreSQL Instance User Guide](https://docs.nhncloud.com/ko/Compute/Instance/ko/component-guide/#postgresql-instance).

> [Caution] When using NHN Cloud NAS
> Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.

> [Note] There may be limitations on the format, size, and number of files available for ingestion. For more information, see [Collect Sync](#rag_ingestion_sync).

<a id="rag_list"></a>

### RAG List

View and manage the list of generated RAGs. Select a RAG from the list to view detailed information.

- **Status**: a RAG status. Please refer to the table below for the main statuses:

| Status | Description |
| --- | --- |
| CREATE REQUESTED | RAG creation has been requested. |
| CREATE IN PROGRESS | RAG creation is in progress. |
| ACTIVE | RAG is operating normally. |
| UPDATE IN PROGRESS | RAG ingestion is in progress. |
| DELETE IN PROGRESS | RAG deletion is in progress. |
| CREATE FAILED | RAG creation has failed.<br/>Delete the RAG and create it again. If creation fails repeatedly, contact Customer Service. |
| UPDATE FAILED | RAG ingestion has failed.<br/>Try **Synchronize ingestions** again. If update fails repeatedly, contact Customer Service. |
| DELETE FAILED | RAG deletion has failed.<br/>Try deletion again. If deletion fails repeatedly, contact Customer Service. |

- **API Gateway Status**: the deployment status information for API Gateway basic stage.

| Status | Description |
| --- | --- |
| DEPLOYING | API Gateway Basic Stage is deploying. |
| COMPLETE | API Gateway Basic Stage has been successfully deployed and is enabled. |
| FAILURE | API Gateway Basic Stage deployment has failed. |

- **Ingestion History**: You can check the execution history of the document ingestion task in the **Ingestion History** tab of the details screen displayed when you select a RAG.
- **API Statistics**: You can check API statistics in the **API Statistics** tab of the detail screen displayed when you select a RAG.
- **Monitoring**: You can check the list of monitored instances and basic metric charts in the **Monitoring** tab of the details screen displayed when you select a RAG.

<a id="rag_ingestion_sync"></a>

### Synchronize Ingestions

- The Synchronize Ingestions feature is available in the **Vector Store** tab of the details screen displayed when you select RAG.
- If documents are added, deleted, or modified in the ingestion data path, you can run **Synchronize Ingestions** to reflect the changes.
- The format, size, and number of files available for ingestion may be limited. See the table below for details:

| Item | Limitation |
|-----|------|
| Total file size | 100GB |
| Maximum no. of files | 1,000,000 |

| Category | Supported format | Maximum file size |
|--------|---------|------------|
| Text document | `.txt`, `.text`, `.md` | 3MB |
| Document | `.doc`, `.docx`, `.pdf` | 50MB |
| Spreadsheet | `.csv`, `.xls`, `.xlsx` | 3MB |
| Presentation | `.ppt`, `.pptx` | 50MB |

<a id="rag_delete"></a>

### Delete RAG

- You cannot delete the RAG that is on creation or deletion.
- The requested deletion task cannot be canceled.

<a id="rag_query_guide"></a>

### Guide to Asking RAG Questions

- When requesting a question, include `model` and `messages` in the request body, similar to the OpenAI Chat Completion API. For `model`, include the RAG name.
- For detailed request examples, please refer to the information below:

```bash
curl -X POST https://{API endpoint address}/rag/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{RAG name}",
    "messages": [
      {
        "role": "user",
        "content": "{query_text}"
      }
    ]
  }'
```

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
4. [Allow specific projects or users to read/write](https://docs.nhncloud.com/en/Storage/Object%20Storage/en/acl-guide/#allow-readwrite-to-specific-projects-or-specific-users) Refer to documents to add required read and write permissions to AI EasyMaker system account in NHN Cloud Object Storage console.

### 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Inquiry Guide

#### NHN Cloud Log & Crash Search Service Usage Guide

Logs and events generated by the AI EasyMaker service can be stored in the NHN Cloud Log & Crash Search service.
To store logs in the Log & Crash Search service, you have to enable Log & Crash service and separate usage fee will be charged.

- **Information on Log & Crash Search service use and fee**
    - For more information and fees on the Log & Crash Search service, please refer to the following documents
        - [Log & Crash Search Service Guide](https://docs.nhncloud.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/Overview/)
        - [Log & Crash Search Fee](https://www.nhncloud.com/kr/pricing/by-service?c=Data%20%26%20Analytics&s=Log%20%26%20Crash%20Search)

#### Log Query

1. Go to the Log & Crash Search service console page.
2. In Log & Crash Search service, enter search criteria to view the logs.
    - AI EasyMaker Training Log Query: Query logs with category field "easymaker.training"
        - Question: category:"easymaker.training"
    - AI EasyMaker endpoint logs query: Query logs with category field "easymaker.inference"
        - Question: category:"easymaker.inference"
    - AI EasyMaker Log Full Query: Query logs with logType field "NNHCloud-AIEasyMaker"
        - Question: logType:"NHNCloud-AIEasyMaker"
3. For more information on how to use Log & Crash Search service, refer to [Log & Crash Search Service Console Guide](https://docs.nhncloud.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/console-guide/).

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

- **Training Log Field**

    | Name | Description |
    |---------------------| --- |
    | trainingId | AI EasyMaker training ID  |

- **Hyperparameter Tuning Log Field**

    | Name | Description |
    | --- | --- |
    | hyperparameterTuningId | AI EasyMaker hyperparameter tuning ID |

- **Endpoint Log Field**

    | Name | Description |
    | --- | --- |
    | endpointId | AI EasyMaker Endpoint ID |
    | endpointStageId | Endpoint stage ID |
    | inferenceId | Inference request own ID |
    | action | Action classification (Endpoint.Model) |
    | modelName | Model name to be inferred |

- **Batch Inference Log Field**

    | Name | Description |
    | --- | --- |
    | batchInferenceId | AI EasyMaker batch inference ID |

### 3. Hyperparameters

- Value in Key-Value format entered through the console.
- When entry point is executed, it is passed to the execution factor (---{Key}).
- It can be stored and used as an environment variable (EM_HP_{Key converted to uppercase letter}).

As shown in the example below, you can use hyperparameter values entered during training creation.<br>
![HyperParameter Input Screen](http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_hyperparameter_en.png)

        import argparse

        model_version = os.environ.get("EM_HP_MODEL_VERSION")

        def parse_hyperparameters():
            parser = argparse.ArgumentParser()

            # Parsing the entered hyper parameter
            parser.add_argument("--epochs", type=int, default=500)
            parser.add_argument("--batch_size", type=int, default=32)
            ...

            return parser.parse_known_args()

### 4. Environment Variables

- Information required for training is passed to training container with **Environment Variable** and the environment variables passed in **Training Script** can be utilized.
- Environment variable names created by user input are to be capitalized.
- Models that have been trained in the code must be saved in the EM_MODEL_DIR path.
- **Key Environment Variables**

    | Environment Variable Name                      | Description                                                                        |
    |-----------------------------| --------------------------------------------------------------------------- |
    | EM_SOURCE_DIR               | Absolute path to the folder where the algorithm script entered at the time of training creation is downloaded  |
    | EM_ENTRY_POINT              | Algorithm entry point name entered at training creation                             |
    | EM_DATASET_${Data set name}     | Absolute path to the folder where each data set entered at the time of training creation is downloaded |
    | EM_DATASETS                 | Full data set list ( json format)                                            |
    | EM_MODEL_DIR                | Model storage path                                                              |
    | EM_CHECKPOINT_INPUT_DIR     | Input checkpoint storage path                                                  |
    | EM_CHECKPOINT_DIR           | Output checkpoint storage path                                                  |
    | EM_HP_${ Upper case converted Hyperparameter key } | Hyperparameter value corresponding to the hyperparameter key                              |
    | EM_HPS                      | Full Hyperparameter List (in json format)                                         |
    | EM_TENSORBOARD_LOG_DIR      | TensorBoard log path for checking training results                                    |
    | EM_REGION                   | Current Region Information                                                              |
    | EM_APPKEY                   | Appkey of AI EasyMaker service currently in use                                   |


- **Example code for utilizing environment variables**

        import os
        import tensorflow

        dataset_dir = os.environ.get("EM_DATASET_TRAIN")
        train_data = read_data(dataset_dir, "train.csv")

        model = ... # Implement the model using input data
        model.load_weights(os.environ.get('EM_CHECKPOINT_INPUT_DIR', None))
        callbacks = [
            tensorflow.keras.callbacks.ModelCheckpoint(filepath=f'{os.environ.get("EM_CHECKPOINT_DIR")}/cp-{{epoch:04d}}.ckpt', save_freq='epoch', period=50),
            tensorflow.keras.callbacks.TensorBoard(log_dir=f'{os.environ.get("EM_TENSORBOARD_LOG_DIR")}'),
        ]
        model.fit(..., callbacks)

        model_dir = os.environ.get("EM_MODEL_DIR")
        model.save(model_dir)

### 5. Store Indicator Logs for TensorBoard Usage

- In order to check result indicators on the TensorBoard screen after training, the TensorBoard log storage space must be set to the specified location (`EM_TENSORBOARD_LOG_DIR`) when writing the training script.

> [Caution] TensorBoard metrics logs storage cycle:
> Metrics older than 120 days will be deleted automatically.

- **Example code for Tesnsorboard log storage (TensorFlow)**

        import tensorflow as tf

        # Specify the TensorBoard log path
        tb_log = tf.keras.callbacks.TensorBoard(log_dir=os.environ.get("EM_TENSORBOARD_LOG_DIR"))

        model = ... # model implementation

        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                epochs=100, batch_size=20, callbacks=[tb_log])

![Check TensorBoard Log](http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_tensorboard.png)

### 6. Distributed Training Settings by Framework

- **Tensorflow**
    - The environment variable `TF_CONFIG` required for distributed training is automatically set. For more information, please refer to the [Tensorflow guide document](https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy).
- **Pytorch**
    - `Backends` settings are required for distributed training. If distributed training is performed on CPU, set it to gloo, and if distributed training is performed on GPU, set it to nccl. For more information, please refer to the [Pytorch guide document](https://pytorch.org/docs/stable/distributed.html).

### 7. Upgrade the cluster version

The AI EasyMaker service periodically upgrades the cluster version to provide stable service and new features.
When a new cluster version is deployed, you need to move the notebooks and endpoints that are running on the old version of the cluster to the new cluster.
Explains how to move new clusters by resource.

#### Upgrade Notebook Cluster Version

On the **Notebook** list screen, notebooks that need to be moved to the new cluster display a **Restart** button to the left of their name.
Hovering the mouse pointer over the**Restart** button displays restart instructions and an expiration date.

- Before expiration, be sure to read the following caveats before clicking the **Restart** button.
    - Upon restart, data stored in the data storage (/root/easymaker directory path) will remain intact.
    - When you run a restart, data stored in boot storage is initialized and may be lost. Move your data to data storage before restarting.

Restarts take about 25 minutes for the first run, and about 10 minutes for subsequent runs.
Failed restarts are automatically reported to the administrator.

#### Upgrade the endpoint cluster version

On the **endpoints list** screen, endpoints that need to be moved to the new cluster will have a **! Notice** to the left of the name.
If you hover over the **! Notice**, it displays a version upgrade announcement and an expiration date.
Before the expiration, you must follow these instructions to move stages running on the old version cluster to the new version cluster.

##### Upgrade the cluster version of a general stage

1. Delete a general stage that is not the default stage. Make sure the stage is in service before deleting.
2. Recreate the stage.
3. When a new stage becomes ACTIVE, check whether API calls and inference responses come normally to the stage endpoint.

> [Caution]
> Deleting a stage will shut down the endpoint, preventing API calls. Ensure that the stage is not in service before deleting it.

##### Upgrade the cluster version of the default stage

The default stage is the stage on which the actual service operates.
To move the cluster version of the default stage without disrupting the service, use the following guide to move it.

1. Create a new stage to replace the default stage in an older version of the cluster.
2. Verify that API calls and inference responses are coming from the new stage endpoint as normal.
3. Click **Change Default Stage**. Select a new stage to change it to the default stage.
4. When the change is complete, the new stage is set as the default stage, and the existing default stage is deleted.

### 8. How to Use Torchrun

- The code has been written to enable distributed learning in Pytorch, and if you enter the number of distributed nodes and the number of processes per node, distributed learning using torchrun and distributed learning using multi-processes will be performed.
- Training and hyperparameter tuning can fail due to insufficient memory, depending on factors such as the total number of processes, model size, input data size, batch size, etc. If it fails due to insufficient memory, it may leave the following error messages. However, not all of the messages below are due to insufficient memory. Please set the appropriate instance type according to your memory usage.

```plaintext
exit code : -9 (pid: {pid})
```

- For more information about torchrun, see the [Pytorch Guide](https://pytorch.org/docs/stable/elastic/run.html).

### 9. Resource Information

When you create batch inferences and endpoints in AI EasyMaker, it allocates resources on the selected instance type, less the default usage.
The amount of resources you need depends on the demand and complexity of your model, so carefully set the number of pods and resource quota along with the appropriate instance type.

Batch inference allocates resources to each pod by dividing the actual usage by the number of pods. Endpoint cannot allow the quota you enter to exceed the actual usage of your instance, so check your resource usage beforehand.
Note that both batch inference and endpoints can fail to create if the allocated resources are less than the minimum usage required by the inference.

### 10. Endpoint API Specification

The AI EasyMaker service provides endpoints based on the open inference protocol (OIP) specification.
For more information about the OIP specification, see [OIP Specification](https://github.com/kserve/open-inference-protocol).

| Name                              | Method | API path                                                                |
| --------------------------------- | ------ | ----------------------------------------------------------------------- |
| Model List                        | GET    | /{model_name}/v1/models                                                 |
| Model Ready                       | GET    | /{model_name}/v1/models/{model_name}                                    |
| Inference                         | POST   | /{model_name}/v1/models/{model_name}/predict                            |
| Description                       | POST   | /{model_name}/v1/models/{model_name}/explain                            |
| Server Information                | GET    | /{model_name}/v2                                                        |
| Server Live                       | GET    | /{model_name}/v2/health/live                                            |
| Server Ready                      | GET    | /{model_name}/v2/health/ready                                           |
| Model Information                 | GET    | /{model_name}/v2/models/{model_name}\[/versions/{model_version}\]       |
| Model Ready                       | GET    | /{model_name}/v2/models/{model_name}\[/versions/{model_version}\]/ready |
| Inference                         | POST   | /{model_name}/v2/models/{model_name}\[/versions/{model_version}\]/infer |
| OpenAI generative model inference | POST   | /{model_name}/openai/v1/completions                                     |
| OpenAI generative model inference | POST   | /{model_name}/openai/v1/chat/completions                                |

> [Note] OpenAI generative model inference
> OpenAI generative model inference is used when using a generative model, such as OpenAI's GPT-4o.
> The inputs required for inference must be entered according to OpenAI's API specification. For more information, see the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat).
> For models that support the Completion and Chat Completion APIs provided by AI EasyMaker, see [Model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility).

### 11. Considerations for framework-specific serving

#### TensorFlow Framework

The TensorFlow model serving provided by AI EasyMaker uses the SavedModel (.pb) recommended by TensorFlow.
To use checkpoints, save the checkpoint variables directory saved as a SavedModel along with the model directory, which will be used to serve the model.
Reference: [https://www.tensorflow.org/guide/saved_model](https://www.tensorflow.org/guide/saved_model)

#### PyTorch Framework

AI EasyMaker serves PyTorch models (.mar) with TorchServe.
We recommend using MAR files created using model-archiver, weight files can also be served, but there are files that are required along with the weight files.
See the table below and the [model-archiver documentation](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) for the required files and detailed descriptions.

| File name                    | Necessity | Description                                                              |
| ---------------------------- | --------- | ----------------------------------------------------------------- |
| model.py                     | Required      | The model structure file passed in the model-file parameter.              |
| handler.py                   | Required      | The file passed to the handler parameter to handle the inference logic. |
| weight files (.pt, .pth, .bin) | Required      | The file that stores the weights and structure of the model.                         |
| requirements.txt             | Optional      | Files for installing Python packages needed when serving.        |
| extra/                       | Optional      | The files in the directory are passed in the extra-files parameter.         |

>[Note] There are differences in the request format between using TorchServe directly and using AI EasyMaker serving, so take care when writing the handler.py.
Refer to the example below to see what values are passed, and implement the handler accordingly.

```bash
# Example request
curl --location --request POST '{API Gateway resource path}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "instances": [].
        [1.0, 2.0],
        [3.0, 4.0]
    ]
}'
```

```python
class TestHandler(BaseHandler):
    # ...
    def preprocess(self, data): # Example: data = [[1.0, 2.0], [3.0, 4.0]]
        features = []
        for row in data:
            # Example: row = [1.0, 2.0] content = row
            features.append(content)
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        return tensor
    # ...
```

#### Scikit-learn Framework

AI EasyMaker uses mlserver to serve Scikit-learn models (.joblib).
The `model-settings.json`, which is required when using mlserver directly, is not required when using AI EasyMaker serving.

#### Hugging Face Framework

The Hugging Face model can be served using the Runtime provided by AI EasyMaker, TensorFlow Serving, or TorchServe.

##### Hugging Face Runtime

This is a simple way to serve Hugging Face models.
Hugging Face Runtime Serving does not support fine-tuning. To serve fine-tuned models, use the TensorFlow/Pytorch Serving method.

1. In Hugging Face, identify the model you want to serve.
2. Copy the Hugging Face model ID.
3. On the Create AI EasyMaker Model page, select the Hugging Face framework, and enter the Hugging Face model ID.
4. Create a model by entering the required inputs based on the model.
5. Verify the created model, and create an endpoint.

> [Note] Supported Hugging Face Tasks:
> Currently, the Hugging Face Runtime does not support the full range of Tasks in Hugging Face.
> The following tasks are supported: `sequence_classification`, `token_classification`, `fill_mask`, `text_generation`, and `text2text_generation`.
> To use unsupported Tasks, use the TensorFlow/Pytorch Serving method.

> [Note] Gated Model:
> To serve a gated model, you must enter the token of an account that is allowed access as a model parameter.
> If you do not enter a token, or if you enter a token from an account that is not allowed, the model deployment fails.

##### TensorFlow/PyTorch Serving

How to serve a Hugging Face model trained with TensorFlow and PyTorch.

1. Download the Hugging Face model.
    - You can download it using the AutoTokenizer, AutoConfig, and AutoModel from the transformers library, as shown in the example code below.

            from transformers import AutoTokenizer, AutoConfig, AutoModel

            model_id = "<model_id>"
            revision = "main"

            model_dir = f"./models/{model_id}/{revision}"

            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model_config = AutoConfig.from_pretrained(model_id, revision=revision)
            model = AutoModel.from_config(model_config)

            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

    - If the model fails to download, try importing the correct class for your non-AutoModel model and downloading it.
    - If you need to fine-tune, you can follow the [Hugging Face fine-tuning guide](https://huggingface.co/docs/transformers/main/en/training) to learn how to write your own code.
        - For more information about AI EasyMaker training, see [Training](./console-guide/#_18).

2. View the Hugging Face model information and generate the files needed to serve it.
    - Save the model in the form required for framework-specific serving.
    - For more information, see the TensorFlow, PyTorch framework notes.
3. Upload the model file to OBS or NAS.
4. For the rest of the process, check out our guides to [creating models and](./console-guide/#_37) [creating endpoints](./console-guide/#_43).
