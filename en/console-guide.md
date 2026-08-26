<!-- machine_translated: true -->

{%- set em_registry = "0516e3a7-kr-registry.container.gov-nhncloud.com" if "gov" in build_flags else "fb34a0a4-kr1-registry.container.nhncloud.com" -%}
<!-- pre-align:aligned sig=13fa2e880aa4 -->

<a id="ai.easymaker.console.guide"></a>
## Machine Learning > AI EasyMaker > Console Guide { #ai.easymaker.console.guide }

<a id="dashboard"></a>
## Dashboard { #dashboard }

You can view the usage status of all AI EasyMaker resources in the dashboard.

<a id="dashboard.service.usage.status"></a>
### Service Usage { #dashboard.service.usage.status }

Displays the number of resources in use per resource.

- Notebook: Number of notebooks in ACTIVE (HEALTHY) status that are in use.
- Training: Number of trainings that are COMPLETE
- Hyperparameter tuning: Number of hyperparameter tunings that are COMPLETE
- Endpoints: Number of endpoints in the ACTIVE state

<a id="dashboard.service.monitoring"></a>
### Monitoring Services { #dashboard.service.monitoring }

- Displays the top 3 endpoints with the most API calls.
- Select an endpoint to see the aggregate API success/failure metrics for the child endpoint stage.

<a id="dashboard.resource.usage"></a>
### Resource Utilization { #dashboard.resource.usage }

- You can see the most utilized resources by CPU and GPU core type.
- If you hover over a metric, it displays resource information.

<a id="notebook"></a>
## Notebook { #notebook }

Create and manage Jupyter notebook with essential packages installed for machine learning development.

<a id="notebook.create"></a>
### Create Notebook { #notebook.create }

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

!!! tip "Note"
    Notebooks can take several minutes to create.
    Creation of the initial resources (notebooks, training, experiments, endpoint) takes additional few minutes to configure the service environment.

!!! danger "Caution"
    Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

<a id="notebook.list"></a>
### Notebook List { #notebook.list }

A list of notebooks are displayed. Select a notebook in the list to check details and make changes to it.

- **Name**: Notebook name is displayed. You can change the name by clicking **Change** on the details screen.
- **Status**: Status of the notebook is displayed. Please refer to the table below for the main status.

    | Status                 | Description                                                                        |
    |--------------------|---------------------------------------------------------------------------|
    | CREATE REQUESTED   | Notebook creation is requested.                                                        |
    | CREATE IN PROGRESS | Notebook instance is in the process of creation.                                                    |
    | ACTIVE (HEALTHY)   | Notebook application is in normal operation.                                            |
    | ACTIVE (UNHEALTHY) | Notebook application is not operating properly. If this condition persists after restarting the notebook, please contact Customer Support. |
    | STOP IN PROGRESS   | Notebook stop in progress.                                                         |
    | STOPPED            | Notebook stopped.                                                           |
    | START IN PROGRESS  | Notebook start in progress                                                         |
    | REBOOT IN PROGRESS | Notebook reboot in progress.                                                         |
    | DELETE IN PROGRESS | Notebook delete in progress.                                                         |
    | CREATE FAILED      | Failed to crate notebook. If keep fails to create, please contact Customer Support.                        |
    | STOP FAILED        | Failed to stop notebook. Please try to stop again.                                            |
    | START FAILED       | Failed to start notebook. Please try to start again.                                            |
    | REBOOT FAILED      | Failed to reboot notebook. Please try to start again.                                           |
    | DELETE FAILED      | Failed to delete notebook. Please try to delete again.                                            |

- **Action > Open Jupyter Notebook**: Click **Open Jupyter Notebook** button to open the notebook in a new browser window. The notebook is only accessible to users who are logged in to the console.

- **Monitoring**: On the **Monitoring** tab of the detail screen that appears when you select the notebook, you can see a list of monitored instances and a chart of basic metrics.
    - The **Monitoring** tab is disabled when the notebook is being created or when there is a task in progress.

<a id="notebook.user.virtual.run.environment.configuration"></a>
### Configure User Virtual Execution Environment { #notebook.user.virtual.run.environment.configuration }

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

<a id="notebook.user.script"></a>
### User Script { #notebook.user.script }

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

<a id="notebook.stop"></a>
### Stop Notebook { #notebook.stop }

Stop the running notebook or start the stopped notebook.

1. Select the notebook want to start or stop from Notebook List.
2. Click **Start Notebook** or **Stop Notebook**.
3. Requested action cannot be cancelled. To proceed, please click **Confirm**

!!! tip "Note"
    It may take several minutes to start and stop notebooks.

!!! danger "Caution"
    When stopping and starting the notebook, the virtual environment and external libraries that the user create can be initialized.
    In order to retain, configure your virtual environment by referring to [User Virtual Execution Environment Configuration](#notebook.user.virtual.run.environment.configuration).

<a id="notebook.instance.type.change"></a>
### Change Notebook Instance Type { #notebook.instance.type.change }

Change the instance type of the created notebook.
Instance type you want to change can only be changed to the same core type instance type as the existing instance.

1. Select the notebook on which you want to change the instance type.
2. If the notebook is running (ACTIVE), click **Stop Notebook** to stop the notebook.
3. Click **Change Instance type**.
4. Select the instance type you want to change and click Confirm.

!!! tip "Note"
    It may take several minutes to change the instance type.

<a id="notebook.reboot"></a>
### Reboot Notebook { #notebook.reboot }

If a problem occurs while using the notebook, or if the status is ACTIVE but you can't access the notebook,
you can reboot the notebook.

1. Select notebook you want to reboot.
2. Click **Reboot Notebook**
3. The requested task cannot be cancelled. To proceed, please click **Confirm**

!!! danger "Caution"
    When rebooting the notebook, the virtual environment and external libraries that the user create can be initialized.
    In order to retain, configure your virtual environment by referring to [User Virtual Execution Environment Configuration](#notebook.user.virtual.run.environment.configuration).

<a id="notebook.delete"></a>
### Delete Notebook { #notebook.delete }

Delete the created notebook.

1. Select notebook you want to delete from the list.
2. Click **Delete Notebook**
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

!!! tip "Note"
    When deleting a notebook, boot storage and data storage are to be deleted.
    Connected NHN Cloud NAS is not deleted and must be deleted individually from **NHN Cloud NAS**.

<a id="experiment"></a>
## Experiment { #experiment }

Experiments are managed by grouping related trainings into experiments.

<a id="experiment.create"></a>
### Create Experiment { #experiment.create }

1. Click **Create Experiment**
2. Enter an experiment name and description and click **OK**.

!!! tip "Note"
Creating experiments can take several minutes.
When creating the initial resources (laptops, trainings, labs, endpoints), it takes an extra few minutes to configure the service environment.

<a id="experiment.list"></a>
### List of Experiments { #experiment.list }

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

<a id="experiment.delete"></a>
### Delete Experiment { #experiment.delete }

Delete an experiment.

1. Select an experiment to delete.
2. Click **Delete Experiment**.
3. Requested deletion cannot be undone. Click **OK** to proceed.

!!! tip "Note"
    You cannot delete an experiment if a pipeline schedule associated with the experiment exists, or if there are training, hyperparameter tuning, or pipeline execution in production. Delete the resources associated with the experiment first, then delete the experiment.
    For associated resources, you can check the list by clicking the **[Training]** tab in the detail screen at the bottom that is displayed when you click the experiment you want to delete.

<a id="training"></a>
## Training { #training }

Provides an training environment where you can learn and identify machine training algorithms based on training results.

<a id="training.create"></a>
### Create Training { #training.create }

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
                    - When using NHN Cloud Object Storage, refer to [Appendix > 1. Adding AI EasyMaker System Account Permissions to NHN Cloud Object Storage](#appendix.1.object.storage.account.permission) to set permissions. Model creation will fail if you do not set the necessary permissions.
                - **NHN Cloud NAS** : Enter the NHN Cloud NAS path where the algorithm is stored. <br>
                    nas://{NAS Enter the directory path in the format ID}:/{path}.

            - **entry point**
                - The entry point is the point of entry into the execution of the algorithm from which training begins. Creates the entry point file name.
                - The entry point file must exist in the algorithm path.
                - Creating **requirements.txt** in the same path will install the required python packages from the script.
            - **hyperparameter**
                - To add parameters for training, click **the + button** to enter parameters in Key-Value format. Up to 100 parameters can be entered.
                - The entered hyperparameters are entered as execution arguments when the entry point is executed. For detailed usage, please refer to [Appendix > 3. Hyperparameters](#appendix.3.hyperparameter).

- **Image** : Choose an image for your instance that matches the environment in which you need to run your training.

- **Training Resource Information**
    - **Training instance type** : Select an instance type to run training.
    - **Number of Distributed Nodes**: Enter a number of distributed nodes to be performed. Distributed training can be enabled through settings in the algorithm code. For more information, please refer to [Appendix > 6. Distributed Training Settings by Framework](#appendix.6.framework.training.settings).
    - **Enable torchrun**: Select whether to use torchrun, which is supported by the Pytorch framework. For more information, see [Appendix > 8. How to use torchrun](#appendix.8.torchrun.usage).
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
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! danger "Caution"
    - Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.
    - Training may fail if the input data is deleted before training is completed.

<a id="training.list"></a>
### Training List { #training.list }

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
    | CREATE FAILED | The training creation failed. If creation continues to fail, please contact Customer Support. |
    | FAIL TRAIN IN PROGRESS, COMPLETE IN PROGRESS | The resources used for training are being cleaned up. |

- **Operation**
    - **Go to TensorBoard** : TensorBoard, where you can check the statistical information of training, opens in a new browser window.<br/>
    For how to leave a TensorBoard log, refer to [Appendix > 5. Store Indicator Logs for TensorBoard Usage](#appendix.5.tensorboard.store.metric.log). TensorBoard can only be accessed by users logged into the console.
    - **Stop training** : You can stop training in progress.

- **Hyperparameters** : You can check the hyperparameter values set for training on **the hyperparameter** tab of the detailed screen displayed when selecting training.

- **Monitoring**: When you select the endpoint stage, you can see a list of monitored instances and basic metrics charts in the **Monitoring** tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while an endpoint stage is being created.

<a id="training.copy"></a>
### Copy Training { #training.copy }

Create a new training with the same settings as an existing training.

1. Select the training you want to copy.
2. Click **Copy Training**.
3. The create training screen is displayed with the same settings as the existing training.
4. If there is any information you would like to change the settings for, make the changes and then click **Create Training** to create the training.

<a id="training.model.create"></a>
### Create a Model from Training { #training.model.create }

Create a model with training in the completed state.

1. Choose the training you want to create as a model.
2. Click **Generate Model** Only training in the COMPLETE state can be created as a model.
3. You will be taken to the model creation page. After checking the contents, click **Create Model** to create a model. For more information on model creation, see [the model](#model) documentation.

<a id="training.delete"></a>
### Delete Training { #training.delete }

Deletes a training.

1. Select the training you want to delete.
2. Click **Delete Training**. Training in progress can be deleted after stopping.
3. Requested deletion cannot be undone. Click **OK** to proceed.

!!! tip "Note"
    Training cannot be deleted if a model created by the training to be deleted exists. Please delete the model first and then the training.

<a id="hyperparameter.tuning"></a>
## Hyperparameter Tuning { #hyperparameter.tuning }

Hyperparameter tuning is the process of optimizing hyperparameter values to maximize a model's predictive accuracy. If you don't use this feature, you'll have to manually tune the hyperparameters to find the optimal values while running many training jobs yourself.

<a id="hyperparameter.tuning.create"></a>
### Create Hyperparameter Tuning { #hyperparameter.tuning.create }

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
                    - When using NHN Cloud Object Storage, please set permissions by referring to [Appendix > 1. Adding AI EasyMaker system account permissions to NHN Cloud Object Storage](#appendix.1.object.storage.account.permission).If you do not set the required permissions, model creation will fail.
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
    - **Number of Distributed Nodes**: Enter a number of distributed nodes to be performed. Distributed training can be enabled through settings in the algorithm code. For more information, please refer to [Appendix > 6. Distributed Training Settings by Framework](#appendix.6.framework.training.settings).
    - **Number of Parallel Trainings**: Enter a number of trainings to perform in parallel simultaneously.
    - **Enable torchrun**: Select whether to use torchrun, which is supported by the Pytorch framework. For more information, see [Appendix > 8. How to use torchrun](#appendix.8.torchrun.usage).
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
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! danger "Caution"
    - Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.
    - Training may fail if the input data is deleted before training is completed.

<a id="hyperparameter.tuning.list"></a>
### Hyperparameter Tuning List { #hyperparameter.tuning.list }

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
    | CREATE FAILED | Hyperparameter tuning generation failed. If creation continues to fail, please contact Customer Support. |
    | FAIL HYPERPARAMETER TUNING IN PROGRESS, COMPLETE IN PROGRESS, STOP IN PROGRESS | Resources used for hyperparameter tuning are being cleaned up. |

- **Status Details**: The bracketed content in the `COMPLETE` status is the status details. See the table below for key details.

    | Details | Description |
    | --- | --- |
    | GoalReached | Details when training for hyperparameter tuning is complete by reaching the target value. |
    | MaxTrialsReached | Details when hyperparameter tuning has reached the maximum number of training runs and is complete. |
    | SuggestionEndReached | Details when the exploration algorithm in Hyperparameter Tuning has explored all hyperparameters. |

- **Operation**
    - **Go to TensorBoard** : TensorBoard, where you can check the statistical information of training, opens in a new browser window.<br/>
    For instructions on how to leave TensorBoard logs, please refer to [Appendix > 5. Store Indicator Logs for TensorBoard Usage](#appendix.5.tensorboard.store.metric.log). TensorBoard can only be accessed by users logged into the console.
    - **Stop Hyperparameter Tuning** : You can stop hyperparameter tuning in progress.

- **Monitoring**: When you select hyperparameter tuning, you can check the list of monitored instances and basic indicator charts in the Monitoring tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while hyperparameter tuning is being created.

<a id="hyperparameter.tuning.training.list"></a>
### List of Trainings for Hyperparameter Tuning { #hyperparameter.tuning.training.list }

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

<a id="hyperparameter.tuning.copy"></a>
### Copy Hyperparameter Tuning { #hyperparameter.tuning.copy }

Create a new hyperparameter tuning with the same settings as the existing hyperparameter tuning.

1. Select the hyperparameter tuning you want to copy.
2. Click **Copy Hyperparameter Tuning**.
3. The Create Hyperparameter Tuning screen is displayed with the same settings as the existing hyperparameter tuning.
4. If there is any information you would like to change the settings for, make the changes and click **Create Hyperparameter Tuning** to create a hyperparameter tuning.

<a id="hyperparameter.tuning.model.create"></a>
### Create a Model from Hyperparameter Tuning { #hyperparameter.tuning.model.create }

Create a model with the best training of hyperparameter tuning in the completed state.

1. Choose the hyperparameter tuning you want to create as a model.
2. Click **Create Model**. Only hyperparameter tuning in the COMPLETE state can be created as a model.
3. You will be taken to the model creation page. After checking the contents, click **Create Model** to create a model.
For more information on model creation, see [the model](#model) documentation.

<a id="hyperparameter.tuning.delete"></a>
### Delete Hyperparameter Tuning { #hyperparameter.tuning.delete }

Delete a hyperparameter tuning.

1. Select the hyperparameter tuning you want to delete.
2. Click **Delete Hyperparameter Tuning**. Hyperparameter tuning in progress can be stopped and then deleted.
3. Requested deletion cannot be undone. Click **OK** to proceed.

!!! tip "Note"
    Hyperparameter tuning cannot be deleted if the model created by the hyperparameter tuning you want to delete exists. Please delete the model first, then the hyperparameter tuning.

<a id="training.template"></a>
## Training Template { #training.template }

By creating a training template in advance, you can import the values entered into the template when creating training or hyperparameter tuning.

<a id="training.template.create"></a>
### Create Training Template { #training.template.create }

For information on what you can set in your training template, see [Creating a training](#training.create).

<a id="training.template.list"></a>
### List of Training Templates { #training.template.list }

Displays a list of training templates. Select a training template from the list to view details and change information.

- **Operation**
    - **Change** : You can change training template information.
- **Hyperparameters** : You can check the names of hyperparameters set in the training template on **the Hyperparameters** tab of the detailed screen displayed when you select a training template.

<a id="training.template.copy"></a>
### Copy Training Template { #training.template.copy }

Create a new training template with the same settings as an existing training template.

1. Select the training template you want to copy.
2. Click **Copy Training Template**.
3. The Create training Template screen appears with the same settings as the existing training template.
4. If there is any information you would like to change the settings for, change it and then click **Create Training Template** to create a training template.

<a id="training.template.delete"></a>
### Delete Training Template { #training.template.delete }

Delete the training template.

1. Select the training template you want to delete.
2. Click **Delete Training Template**
3. Requested deletion cannot be undone. Click **OK** to proceed.

<a id="fine.tuning"></a>
## Fine Tuning { #fine.tuning }

A feature that specializes model performance by performing additional training on a pre-trained large language model using a dataset tailored to a specific domain or task. AI EasyMaker provides multiple models as base models for additional training. Select one of the base models, then enter the training data and hyperparameters to perform fine tuning.

<a id="fine.tuning.create"></a>
### Create Fine Tuning { #fine.tuning.create }

Configure the base model, training data, and hyperparameters to perform fine tuning.

- **Basic Information**
    - **Name**: Enter a name for the fine tuning.
    - **Description**: Enter a description if needed.
    - **Experiment**: Select an experiment to include the fine tuning in. Experiments group related tasks. If no experiment has been created, click **Add** to create one.
- **Base Model Information**
    - **Base Model**: Select the base model to use for fine tuning. Click **Base Model** to open the model selection window, then select one of the base models available for fine tuning.
- **Parameter Information**
    - **Hyperparameter**: Displays a list of hyperparameters supported by the selected base model. Default values are pre-filled for each parameter, and values can be changed for editable parameters.
- **Instance Information**
    - **Instance Type**: Select the instance type to run fine tuning on. Only instance types available for the selected base model are displayed.
    - **Instance Count**: The number of instances to use for fine tuning.
- **Input Data**
    - **Training Data**: Enter the training dataset to use for fine tuning. A minimum of 1 and a maximum of 10 training datasets can be configured.
        - **Dataset Name**: Enter a name for the dataset.
        - **Data Path**: Enter the NHN Cloud Object Storage or NHN Cloud NAS path where the data is stored.
            - If using NHN Cloud Object Storage, configure the required permissions by referring to [Appendix > 1. Add AI EasyMaker System Account Permissions to NHN Cloud Object Storage](#appendix.1.object.storage.account.permission). Fine tuning will fail if the required permissions are not configured.
        - **Data Format**: The format of the training data. (Fixed as `chat_template`) For more information on dataset formats, see [Appendix > 12. Fine Tuning Dataset Format](#appendix.12.fine.tuning.dataset.format).
    - **Validation Data**: Enter the validation dataset to use for fine tuning. Up to 10 validation datasets can be configured, and the input fields are the same as those for training data.
    - **Validation Data Percentage**: If you want to use a portion of the training data for validation without a separate validation dataset, enter the percentage (%) to use for validation.
        - If a validation dataset is entered, this is fixed at 0 and cannot be modified.
- **Output Data**
    - **Model Upload Path**: Enter the path where the completed fine tuning model will be saved.
        - Enter an NHN Cloud Object Storage or NHN Cloud NAS path.
- **Additional Settings**
    - **Data Storage Size**: Enter the data storage size for the instance running fine tuning.
        - Used only when NHN Cloud Object Storage is used. Specify a size large enough to store all data required for fine tuning.
    - **Maximum Duration**: Specify the maximum wait time for fine tuning to complete. Fine tuning that exceeds the maximum wait time will be terminated.
    - **Log Management**: Logs generated during fine tuning can be saved to the NHN Cloud Log & Crash service.
        - For more information, see [Appendix > 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Inquiry](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! danger "Caution"
    - Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.
    - Deleting input data before fine tuning is complete may cause fine tuning to fail.

<a id="fine.tuning.list"></a>
### Fine Tuning List { #fine.tuning.list }

Displays the list of fine tunings. Select a fine tuning from the list to view its details.

- **Base Model**: Displays the name of the base model used for fine tuning.
- **Duration**: Displays the time taken for fine tuning.
- **Status**: Displays the status of the fine tuning. Refer to the table below for major statuses.

    | Status | Description |
    | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
    | CREATE IN PROGRESS | Resources required for fine tuning are being created. |
    | RUNNING | Fine tuning is in progress. |
    | STOPPED | Fine tuning has been stopped at the user's request. |
    | COMPLETE | Fine tuning has completed successfully. |
    | STOP IN PROGRESS | Fine tuning is being stopped. |
    | FAIL FINE TUNING | Fine tuning failed during the process. If log management is enabled, detailed failure information can be checked through Log & Crash Search logs. |
    | CREATE FAILED | Fine tuning creation failed. If creation continues to fail, contact customer support. |
    | STOP FAILED | Fine tuning stop failed. |
    | FAIL FINE TUNING IN PROGRESS, COMPLETE IN PROGRESS, STOP IN PROGRESS | Resources used for fine tuning are being cleaned up. |

- **Actions**
    - **Go to TensorBoard**: Opens TensorBoard, where statistical information about the fine tuning can be viewed, in a new browser window. TensorBoard can only be accessed by users logged in to the console.
    - **Stop**: Stops the fine tuning that is in progress.

- **Hyperparameter**: Select a fine tuning to open the details screen, then go to the **Hyperparameter** tab to view the hyperparameter values configured for the fine tuning.

- **Monitoring**: Select a fine tuning to open the details screen, then go to the **Monitoring** tab to view the list of monitored instances and basic metric charts.
    - The **Monitoring** tab is disabled while fine tuning is being created.

<a id="fine.tuning.copy"></a>
### Copy Fine Tuning { #fine.tuning.copy }

Creates a new fine tuning with the same settings as an existing fine tuning.

1. Select the fine tuning to copy. This feature is available only when a single fine tuning is selected.
2. Click **Copy**.
3. The fine tuning creation screen is displayed with the same settings as the existing fine tuning.
4. If there are any settings to change, make the changes, then click **Create Fine Tuning** to create the fine tuning.

<a id="fine.tuning.model.create"></a>
### Create a Model from Fine Tuning { #fine.tuning.model.create }

Creates a model from a fine tuning in the completed state.

1. Select the fine tuning to create as a model.
2. Click **Create Model**. Only fine tunings in the COMPLETE status can be created as a model.
3. You will be redirected to the model creation page. Review the information and click **Create Model** to create the model. For more information on model creation, see the [Model](#model) document.

<a id="fine.tuning.delete"></a>
### Delete Fine Tuning { #fine.tuning.delete }

Deletes a fine tuning.

1. Select the fine tuning to delete. Only fine tunings in the CREATE FAILED, FAIL FINE TUNING, COMPLETE, or STOPPED status can be deleted.
2. Click **Delete**.
3. The requested deletion cannot be canceled. Click **Confirm** to proceed.

!!! tip "Note"
    If a model created from the fine tuning to be deleted exists, the fine tuning cannot be deleted. Delete the model first, then delete the fine tuning.

<a id="model"></a>
## Model { #model }

Can manage models of AI EasyMaker's training outcomes or external models as artifacts.

<a id="model.create"></a>
### Create Model { #model.create }

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
            - If you are using NHN Cloud Object Storage, refer to [Appendix > 1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage](#appendix.1.object.storage.account.permission) to set permissions. If you do not set permissions, you will not be able to access the model's artifacts and model creation will fail.
        - **NHN Cloud NAS**: Enter the path to the NHN Cloud NAS where the model artifact is stored.
            - Enter the directory path in the format `nas://{NAS ID}:/{path}`.
    - **Parameter**: Enter the model's parameter information.
        -**Parameter name**: Enter the name of the parameter in the model.
        -**Parameter value**: Enter the values of the parameters in the model.

!!! tip "Note"
    The values entered as model parameters are used when serving the model. Parameters can be used as arguments and environment variables:
    Arguments are used as the parameter name as entered, and environment variables are used with the parameter name converted to screaming snake notation.

!!! tip "Note"
    When creating a HuggingFace model, you can create the model by entering the ID of the HuggingFace model as a parameter.
    The ID of the HuggingFace model can be found in the URL of the HuggingFace model page.
    For more information, see [Appendix > 11. Framework-specific serving notes](#appendix.11.framework.note).

!!! danger "Caution"
    Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.

!!! danger "Caution"
    If not retained the model artifacts stored in storage, the creation of endpoints for that model fails.

!!! danger "Caution"
    The file type for the HuggingFace model are limited to safetensors.
    Safetensors is a safe and efficient machine learning model developed by HuggingFace.
    Other file types are not supported.

!!! danger "Caution"
    Triton models support only TensorFlow, PyTorch, and ONNX backends.
    When creating a Triton model, the model artifact path you enter must contain the model files and a `config.pbtxt` file structured to run the model on Triton.
    See the example below:
    <details>
    <summary><strong>Example</strong></summary>

        model_name/
        ├── config.pbtxt                              # Model selection file
        └── 1/                                        # Version 1 directory
            └── model.savedmodel/                     # TensorFlow SavedModel directory
                ├── saved_model.pb                    # Metagraph and checkpoint data
                └── variables/                        # Model weight directory
                    ├── variables.data-00000-of-00001
                    └── variables.index

    </details>

<a id="model.list"></a>
### Model List { #model.list }

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
    | CREATE FAILED      | Failed to created a model. If creation fails repeatedly, contact Customer Support. |
    | DELETE FAILED      | Failed to delete a model. Please try again.                                   |

- **Training Name**: For models created from training, training name that is based is displayed.
- **Training ID**: For models created from training, training ID that is based is displayed.
- **Framework**: Model's framework information is displayed.
- **Parameter**: Model's parameter is displayed. Parameters are used for inference.

<a id="model.endpoint.create"></a>
### Create Endpoint from Model { #model.endpoint.create }

Create an endpoint that can serve the selected model.

1. Select the model you want to create as an endpoint from the list.
2. Click **Create Endpoint**.
3. Go to **Create Endpoint** page. After checking the contents, click **Create Endpoint** to create a model.
For more information on creating models, refer to **Endpoint** documents.

<a id="model.batch.inference.create"></a>
### Create Batch Inference in a Model { #model.batch.inference.create }

Create batch inferences with the selected model and view the inference results as statistics.

1. Select the model you want to create with batch inference from the list.
2. Click **Create Batch Inference**.
3. You will be taken to the **Create Batch Inference** page. Check the contents and click Create Batch Inference.
For more information about creating batch inferences, see [Batch Inference](#batch.inference).

<a id="model.delete"></a>
### Delete Model { #model.delete }

Delete a model.

1. Select the model want to delete from list.
2. Click **Delete Model**.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

!!! tip "Note"
    You cannot delete model if endpoint created by model want to delete is existed.
    To delete, delete the endpoint created by the model first and then delete the model.

<a id="model.evaluation"></a>
## Evaluate models { #model.evaluation }

Measure the performance of models, and compare performance across different models.

<a id="model.evaluation.create"></a>
### Create a model evaluation { #model.evaluation.create }

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
        - For more information, see [Appendix > 2. How to use NHN Cloud Log & Crash Search service and check logs](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! danger "Caution"
    - Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.
    - The size of the input data used to evaluate the model must be 20GB or less.
    - The number of classes in a classification model evaluation must be 50 or fewer.

<a id="model.evaluation.list"></a>
### Model Evaluation List { #model.evaluation.list }

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

<a id="model.evaluation.classification.metric"></a>
### Classification Model Evaluation Metrics { #model.evaluation.classification.metric }

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

<a id="model.evaluation.regression.metric"></a>
### Regression Model Evaluation Metrics { #model.evaluation.regression.metric }

- **MAE(mean absolute error)**: The mean absolute error between actual and predicted values. It intuitively shows the magnitude of prediction errors.
- **MAPE(mean absolute percentage error)**: The mean of prediction errors divided by actual values. Since it is ratio-based, it may be unsuitable for data with values close to zero.
- **R-squared(coefficient of determination)**: Indicates how well the model explains the actual data, with values closer to 1 representing higher explanatory power.
- **RMSE(root mean squared error)**: The square root of the mean squared error. It is more sensitive to large errors and interprets results on the same scale as the original units.
- **RMSLE(root mean squared logarithmic error)**: Calculated from the difference between log-transformed actual and predicted values. It is less sensitive to differences in magnitude and useful for evaluating exponentially growing data.

<a id="model.evaluation.compare"></a>
### Compare Model Evaluations { #model.evaluation.compare }

Compare evaluation metrics across models.

1. In the list, select the model evaluations to compare.
2. Click **Compare**.

<a id="model.evaluation.delete"></a>
### Delete Model Evaluation { #model.evaluation.delete }

Delete a model evaluation.

1. Select the model evaluation to delete.
2. Click **Delete**. An ongoing model evaluation can be stopped and then deleted.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

<a id="endpoint"></a>
## Endpoint { #endpoint }

Create and manage endpoints that can serve the model.

<a id="endpoint.create"></a>
### Create Endpoint { #endpoint.create }

- **Enable API Gateway Service**
    - AI EasyMaker endpoints create API endpoints and manage APIs through the NHN Cloud API Gateway service. To use the endpoint feature, you must enable the API Gateway service.
    - For more information and pricing on the API Gateway service, see the following:
        - [API Gateway Service Overview](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Application%20Service/API%20Gateway/ko/overview/)
        - [API Gateway Pricing](https://www.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/kr/pricing/by-service?c=Application%20Service&s=API%20Gateway)
- **Endpoint**: Select whether to add a stage to a new or existing endpoint.
    - **Create as a new endpoint**: Creates a new endpoint. The endpoint is created in API Gateway with a new service and a default stage.
    - **Add a new stage to an existing endpoint**: The endpoint is created as a new stage in the API Gateway service of the existing endpoint. Select the existing endpoint to which you want to add a stage.
- **Endpoint name**: Enter the endpoint name. Endpoint names must be unique.
- **Stage name**: If you are adding a new stage to an existing endpoint, enter the name of the new stage. Stage names must be unique.
- **Description**: Enter a description for the endpoint stage.
- **Instance information**: Enter the information for the instance on which the model will be served.
    - **Instance type**: Select the instance type.
    - **Number of instances**: Enter the number of instances to run.
    - **Autoscaler**: The autoscaler automatically adjusts the number of nodes according to resource usage policies. The autoscaler is configured at the stage level.
        - **Enable/Disable**: Select whether to use the autoscaler. When enabled, the number of instances scales in or out according to instance load.
        - **Minimum number of nodes**: Minimum number of nodes that can be scaled down to
        - **Maximum number of nodes**: Maximum number of nodes that can be scaled up to
        - **Scale down**: Configure whether to enable node scale-down
        - **Resource usage threshold**: The threshold value for the resource usage threshold range that serves as the criterion for scale-down
        - **Threshold duration (minutes)**: The duration for retaining resource usage of target nodes to scale down below the threshold
        - **Scale-down delay after scale-up (minutes)**: Delay before starting to monitor for scale-down targets after scaling up
- **Stage information**: Enter the information for the Model Artifact to deploy to the endpoint. If you deploy the same model across multiple stage resources, requests are distributed and processed.
    - **Model**: Select the model to deploy to the endpoint. If you have not created a model, create one first. For serving notes by model framework, see [Appendix > 11. Serving Notes by Framework](#appendix.11.framework.note).
    - **Resource allocation (%)**: Enter the resources to allocate to the model. The actual resource usage of the instance is allocated at a fixed ratio.
        - **cpu**: Enter the CPU allocation. Enter this value if you want to allocate directly without using the allocation ratio (%).
        - **memory**: Enter the memory allocation. Enter this value if you want to allocate directly without using the allocation ratio (%).
        - **gpu**: Enter the GPU allocation. Enter this value if you want to allocate directly without using the allocation ratio (%).
    - **Description**: Enter a description for the stage resource.
    - **Pod Autoscaler**: A feature that automatically adjusts the number of pods according to the request volume of the model. The autoscaler is configured at the model level.
        - **Enable/Disable**: Select whether to use the autoscaler. When enabled, the number of pods scales in or out according to model load.
        - **Scale-up unit**: Enter the pod scale-up unit.
            - **CPU**: The number of pods is adjusted according to CPU usage.
            - **Memory**: The number of pods is adjusted according to memory usage.
        - **Threshold value**: The threshold value for each scale-up unit at which pods are scaled up.
    - **Resource information**: You can check the actual resources in use. The actual resource usage is allocated to each model according to the allocation quota of the entered model. For more information, see [Appendix > 9. Resource Information](#appendix.9.resource.info).

!!! tip "Note"
    The AI EasyMaker service provides endpoints based on the open inference protocol (OIP) specification. For the endpoint API specification, see [Appendix > 10. Endpoint API specification](#appendix.10.endpoint.api.specification).
    To use a separate endpoint, refer to the resources created in the API Gateway service and create a new resource to use it.
    For more information about the OIP specification, see [OIP specification](https://github.com/kserve/open-inference-protocol).

!!! tip "Note"
    Endpoint creation can take several minutes.
    Creation of the initial resources takes additional few minutes to configure the service environment.

!!! tip "Note"
    When you create a new endpoint, a new API Gateway service is created.
    When you add a new stage to an existing endpoint, a new stage is created in the API Gateway service.
    If the default quota in the [API Gateway Service Resource Provision Policy](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/nhncloud/ko/resource-policy{% if "gov" in build_flags %}-gov/#api-gateway{% else %}/#resource-provision-policy-for-api-gateway-service{% endif %}) is exceeded, endpoint creation in AI EasyMaker may not be possible. In this case, you can resolve this issue by adjusting the API Gateway service resource quota.

<a id="endpoint.list"></a>
### Endpoint List { #endpoint.list }

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
    | CREATE FAILED | Endpoint creation has failed. <br/>You must delete and recreate the endpoint. If the creation fails repeatedly, please contact Customer Support. |
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

<a id="endpoint.stage.create"></a>
### Create Endpoint Stage { #endpoint.stage.create }

Add new stage to existing endpoint. You can create and test the new stage without affecting default stage.

1. In Endpoint list, click **Endpoint Name**.
2. Click **+ Create Stage**.
3. Adding new stage from existing endpoint is automatically selected, and its setup method is the same as endpoint creation.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

<a id="endpoint.stage.list"></a>
### Endpoint Stage List { #endpoint.stage.list }

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

!!! danger "Caution"
    When you create an endpoint or an endpoint stage, AI EasyMaker creates API Gateway services and stages for the endpoint.
    If you change the API Gateway services and stages created by AI EasyMaker directly from the API Gateway service console, note the following precautions.

    1. Don't delete the API Gateway services and stages created by AI EasyMaker. If you delete them, API Gateway information may not display correctly on the endpoint, and changes to the endpoint may not be applied to API Gateway.
    2. Don't change or delete the resources at the API Gateway resource path that you specified when creating the endpoint. If you delete them, inference API calls from the endpoint may fail.
    3. Don't add resources under the API Gateway resource path that you specified when creating the endpoint. Resources that you add may be deleted when you add or modify an endpoint stage.
    4. In the stage settings in API Gateway, don't disable **Backend Endpoint URL Override** configured for the API Gateway resource path or change its URL. If you change it, inference API calls from the endpoint may fail.
       For settings other than the precautions listed above, you can use features provided by API Gateway as needed.
       For more information about using API Gateway, see the [API Gateway Console Guide](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Application%20Service/API%20Gateway/ko/console-guide{% if "gov" in build_flags %}-gov{% endif %}/).

!!! tip "Note"
    If stage settings of AI EasyMaker endpoint are not deployed to the API Gateway stage due to a temporary issue, deployment status is displayed as failed.
    In this case, you can deploy API Gateway stage manually by clicking Select Stage from the Stage list > View API Gateway Settings > 'Deploy Stage' in the bottom detail screen.
    If this guide couldn’t recover the deployment status, please contact the Customer Center.

<a id="endpoint.stage.resource.create"></a>
### Create Stage Resource { #endpoint.stage.resource.create }

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

<a id="endpoint.stage.resource.list"></a>
### Stage Resource List { #endpoint.stage.resource.list }

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
- **API Gateway Resource Path**: The inference URL of the model deployed to the stage. API clients can request inference at the displayed URL. For more information, see [Appendix > 10. Endpoint API Specfication](#appendix.10.endpoint.api.specification).
- **Number of Pods**: Shows the number of healthy pods and total pods in use on the resource.

<a id="endpoint.inference.call"></a>
### Call Endpoint Inference { #endpoint.inference.call }

1. When you click Stage in **Endpoint** > **Endpoint Stage**, Stage details screen is displayed at the bottom.
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

<a id="endpoint.stage.resource.delete"></a>
### Delete Stage Resource { #endpoint.stage.resource.delete }

1. In the endpoint list, click the **endpoint name** to move it to the Endpoint Stage list.
2. In the endpoint stage list, click the endpoint stage on which the stage resource you want to delete is deployed. When you click, the stage details screen will be displayed at the bottom.
3. On the **Stage Resource** tab of the details screen, select the stage resource you want to delete.
4. Click **Delete Stage Resource**.
5. Requested deletion cannot be undone. Click **OK** to proceed.

<a id="endpoint.default.stage.change"></a>
### Change Endpoint Default Stage { #endpoint.default.stage.change }

Change the default stage of the endpoint to another stage.
To change the model of an endpoint without service stop, AI EasyMaker recommends deploying the model using stage capabilities.

1. Stages operating as actual services are operated by the default stage.
2. If replacing with new model, add new stage to the existing endpoint.
3. Verify that the endpoint service is not affected by the replaced model in the new stage.
4. Click **Change Default Stage**.
5. Select new stage that want to change as default stage from stage want to change.
6. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**
7. Stage that you want to change changes to the default stage, and resources of existing default stage are automatically deleted.

<a id="endpoint.stage.delete"></a>
### Delete Endpoint Stage { #endpoint.stage.delete }

1. In Endpoint list, click **Endpoint Name** to go to Endpoint Stage list.
2. In Endpoint Stages list, select the endpoint stage want to delete. You cannot delete default stage.
3. Click **Delete Stage**.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

!!! danger "Caution"
    Deleting an endpoint stage in AI EasyMaker also deletes the stage in API Gateway service from which the endpoint's stage is deployed.
    If there is an API running on the API Gateway stage to be deleted, please be noted that API calls cannot be made.

<a id="endpoint.delete"></a>
### Delete Endpoint { #endpoint.delete }

Delete an endpoint.

1. Select the endpoint want to delete from endpoints list.
2. You cannot delete an endpoint if there is stage under endpoint other than the default stage. Please delete the other stages first.
3. Click **Delete Endpoint**.
4. Requested deletion task cannot be cancelled. If want to proceed, please click **Confirm**

!!! danger "Caution"
    Deleting an endpoint stage in AI EasyMaker also deletes API Gateway service from which the endpoint's stage was deployed.
    If there is API running on the API Gateway service to be deleted, please be noted that API calls cannot be made.

<a id="batch.inference"></a>
## Batch Inference { #batch.inference }

Provides an environment to make batch inferences from an AI EasyMaker model and view inference results in statistics.

<a id="batch.inference.create"></a>
### Create Batch Inference { #batch.inference.create }

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
    - **Resource Information**: You can see the actual resources used by the model. The actual usage is split and allocated to each pod based on the number of pods you entered. For more information, see [Appendix > 9. Resource Information](#appendix.9.resource.info).
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
        - For more information, please refer to [Appendix > 2. NHN Cloud Log & Crash Search Service User Guide and Log Check](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! tip "Note"
    - If the Glob pattern is not entered properly, batch inference may not work properly because the input data cannot be found.
    When used together with the **Include Glob pattern**, the **Exclude Glob pattern** takes precedence.
    - You must set the **batch size** and **inference timeout** appropriately based on the performance of the model you are batch inferring.
    If the settings you enter are incorrect, batch inference might not perform well enough.

!!! danger "Caution"
    - Only NHN Cloud NAS created on the same project as AI EasyMaker is available to use.
    - Batch inference can fail if you delete input data before batch inference is complete.

!!! danger "Caution"
    Batch inference using GPU instances allocates GPU instances based on the number of Pods
    If `Number of Pods / Number of GPUs` is not divisible by an integer, you may encounter unallocated GPUs
    Unallocated GPUs are not used by batch inference, so set the number of Pods appropriately to use GPU instances efficiently.

<a id="batch.inference.list"></a>
### Batch Inference List { #batch.inference.list }

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
    | CREATE FAILED | The batch inference creation failed. If creation continues to fail, please contact Customer Support. |
    | FAIL BATCH INFERENCE IN PROGRESS, COMPLETE IN PROGRESS | The resources used for batch inference are being cleaned up. |

- **Operation**
    - **Stop**: You can stop batch inference in progress.
- **Monitoring**: When you select a batch inference, you can check the list of monitored instances and basic indicator charts in the **Monitoring** tab of the detailed screen that appears.
    - The **Monitoring** tab is disabled while batch inference is being created.

<a id="batch.inference.copy"></a>
### Copy Batch Inference { #batch.inference.copy }

Create a new batch inference with the same settings as an existing batch inference.

1. Select the batch inference you want to copy.
2. Click **Copy Batch Inference**.
3. The Create batch inference screen appears with the same settings as an existing batch inference.
4. If there is any information you would like to change the settings for, make the changes and then click **Create Batch Inference** to create the batch inference.

<a id="batch.inference.delete"></a>
### Delete Batch Inference { #batch.inference.delete }

Delete a batch inference.

1. Select the batch inference you want to delete.
2. Click **Delete Batch Inference**. Batch inference in progress can be deleted after stopping.
3. Requested deletion task cannot be cancelled. To proceed, please click **Confirm**

<a id="personal.image"></a>
## Private Image { #personal.image }

User-personalized container images can be used to drive notebooks, training, and hyperparameter tuning.
Only private images derived from the notebook/deep learning images provided by AI EasyMaker can be used when creating resources in AI EasyMaker.
See the table below for the base images in AI EasyMaker.

<a id="personal.image.notebook.image"></a>
#### Notebook Image

| Image Name                           | Core Type | Framework  | Framework Version | Python Version | Image Address                                                                                          |
| ------------------------------------ | --------- | ---------- | ----------------- | -------------- | ------------------------------------------------------------------------------------------------------ |
| Ubuntu 22.04 CPU Python Notebook     | CPU       | Python     | 3.10.12           | 3.10           | $[ em_registry ]$/easymaker/python-notebook:3.10.12-cpu-py310-ubuntu2204    |
| Ubuntu 22.04 GPU Python Notebook     | GPU       | Python     | 3.10.12           | 3.10           | $[ em_registry ]$/easymaker/python-notebook:3.10.12-gpu-py310-ubuntu2204    |
| Ubuntu 22.04 CPU PyTorch Notebook    | CPU       | PyTorch    | 2.0.1             | 3.10           | $[ em_registry ]$/easymaker/pytorch-notebook:2.0.1-cpu-py310-ubuntu2204     |
| Ubuntu 22.04 GPU PyTorch Notebook    | GPU       | PyTorch    | 2.0.1             | 3.10           | $[ em_registry ]$/easymaker/pytorch-notebook:2.0.1-gpu-py310-ubuntu2204     |
| Ubuntu 22.04 CPU TensorFlow Notebook | CPU       | TensorFlow | 2.12.0            | 3.10           | $[ em_registry ]$/easymaker/tensorflow-notebook:2.12.0-cpu-py310-ubuntu2204 |
| Ubuntu 22.04 GPU TensorFlow Notebook | GPU       | TensorFlow | 2.12.0            | 3.10           | $[ em_registry ]$/easymaker/tensorflow-notebook:2.12.0-gpu-py310-ubuntu2204 |

<a id="personal.image.deep.learning.image"></a>
#### Deep Learning Images

| Image Name | CoreType | Framework | Framework version | Python version | Image address |
| --- | --- | --- | --- | --- | --- |
| Ubuntu 22.04 CPU PyTorch Training    | CPU      | PyTorch    | 2.0.1           | 3.10        | $[ em_registry ]$/easymaker/pytorch-train:2.0.1-cpu-py310-ubuntu2204     |
| Ubuntu 22.04 GPU PyTorch Training    | GPU      | PyTorch    | 2.0.1           | 3.10        | $[ em_registry ]$/easymaker/pytorch-train:2.0.1-gpu-py310-ubuntu2204     |
| Ubuntu 22.04 CPU TensorFlow Training | CPU      | TensorFlow | 2.12.0          | 3.10        | $[ em_registry ]$/easymaker/tensorflow-train:2.12.0-cpu-py310-ubuntu2204 |
| Ubuntu 22.04 GPU TensorFlow Training | GPU      | TensorFlow | 2.12.0          | 3.10        | $[ em_registry ]$/easymaker/tensorflow-train:2.12.0-gpu-py310-ubuntu2204 |

!!! tip "Note"
    Only NHN Container Registry (NCR) can be integrated as a container registry service where private images are stored. (As of December 2023)

!!! danger "Caution"
    Only private images derived from base images provided by AI EasyMaker can be used.

<a id="personal.image.create"></a>
### Create Private Image { #personal.image.create }

The following document explains how to create a container image with an AI EasyMaker-based image using Docker, and using a private image for notebooks in AI EasyMaker.

1. Create a DockerFile of private image.

FROM $[ em_registry ]$/easymaker/python-notebook:3.10.12-cpu-py310-ubuntu2204 as easymaker-notebook
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
        docker tag custom-training:v1 example-kr1-registry.container.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/registry/custom-training:v1
        docker push example-kr1-registry.container.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/registry/custom-training:v1

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

!!! tip "Note"
    Private images can be used for notebooks, training, and hyperparameter tuning to create resources.

!!! tip "Note"
    Only NCR service can be used as a container registry service. (As of December 2023)
    Enter the following values for the account ID and password for the NCR service.
    ID: User Access Key of NHN Cloud user account
    Password: User Secret Key of NHN Cloud user account

<a id="registry.account"></a>
## Registry Account { #registry.account }

In order for AI EasyMaker to pull an image from a user's registry where private images are stored to power the container, they need to be logged into the user's registry.
If you save your login information with a registry account, you can reuse it in images linked to that registry account.
To manage your registry accounts, go to the **Image** menu in the AI EasyMaker console, then select the **Registry Account** tab.

<a id="registry.account.create"></a>
### Create Registry Account { #registry.account.create }

Create a new registry account.

- Name: Enter the name of registry account.
- Description: Enter a description of the registry account.
- Category: Select a container registry service.
- ID: Enter the ID of the registry account.
- Password: Enter the password for the registry account.

<a id="registry.account.modify"></a>
### Modify Registry Account { #registry.account.modify }

<a id="registry.account.modify.account.modify"></a>
#### Modify registry ID and password

- Click **Change Registry Account**.
- Enter an ID and password, then click **Confirm**.

!!! tip "Note"
    When you change your registry account, you sign in to the registry service with the changed username and password when using images associated with that account.
    If you enter an incorrect registry username and password, the login during a private image pull fails and the resource creation fails.

!!! danger "Caution"
    If there are resources being created with a private image that has a registry account associated with it, or if there are studies and hyperparameters in progress, you cannot modify them.

<a id="registry.account.modify.account.info.modify"></a>
#### Registry Account > Change Name, Description

1. In the Registry Accounts list, select the account you want to change.
2. Click **Change** on the bottom screen.
3. After changing the name and description, click the **Confirm** button.

<a id="registry.account.delete"></a>
### Delete Registry Account { #registry.account.delete }

Select the registry account you want to delete from the list, and click **Delete Registry Account**.

!!! tip "Note"
    You cannot delete a registry account associated with an image. To delete, delete the associated image first and then delete the registry account.

<a id="pipeline"></a>
## Pipeline { #pipeline }

ML Pipeline is a feature for managing and executing portable and scalable machine learning workflows.
You can use the Kubeflow Pipelines (KFP) Python SDK to write components and pipelines, compile pipelines into intermediate representation YAML, and run them in AI EasyMaker.
Most pipelines are designed to produce one or more ML artifacts, such as datasets, models, and evaluation metrics.

!!! tip "Note"
    A **pipeline** is a definition of a workflow that combines one or more components to form a directed acyclic graph (DAG).
    - Each component runs a single container during execution, which can generate ML artifacts.
    - Components can take inputs and produce outputs. There are two types of I/O types. Parameters and artifacts:
    - Parameters are useful for passing small amounts of data between components.
    - Artifact types are for ML artifact outputs, such as datasets, models, metrics, etc. Provides a convenient mechanism for saving to object storage.

!!! tip "Note"
    The feature to view console output generated while executing a pipeline is not provided.
    To check the logs of pipeline code, use the [SDK's Log Send feature] (./sdk-guide/#feature.lncs.log.send) to send the logs to Log & Crash Search.

!!! tip "Note"
    Kubeflow Pipelines (KFP) official documentation
    - [KFP User Guide](https://www.kubeflow.org/docs/components/pipelines/user-guides/)
    - [KFP SDK Reference](https://kubeflow-pipelines.readthedocs.io/en/stable/)

<a id="pipeline.upload"></a>
### Upload a Pipeline { #pipeline.upload }

Upload a pipeline.

- **Name**: Enter a pipeline name.
- **Description**: Enter description.
- **File registration**: Select the YAML file to upload.

!!! tip "Note"
    Uploading a pipeline can take a few minutes.
    The initial resource creation requires an additional few minutes of time to configure the service environment.

<a id="pipeline.list"></a>
### Pipeline List { #pipeline.list }

A list of pipelines is displayed. Select a pipeline in the list to view details and make changes to the information.

- **Status**: The status of the pipeline is displayed. See the table below for key statuses.

    | Status                | Description                             |
    |--------------------|--------------------------------|
    | CREATE REQUESTED   | Pipeline creation has been requested.           |
    | CREATE IN PROGRESS | Pipeline creation is in progress.         |
    | CREATE FAILED      | Pipeline creation failed. Try again. |
    | ACTIVE             | The pipeline was created successfully.        |

<a id="pipeline.graph"></a>
### Pipeline Graph { #pipeline.graph }

A pipeline graph is displayed. Select a node in the graph to see more information.

A graph is a pictorial representation of a pipeline. Each node in the graph represents a step in the pipeline, with arrows indicating the parent/child relationship between the pipeline components represented by each step.

<a id="pipeline.delete"></a>
### Delete a Pipeline { #pipeline.delete }

Delete the pipeline.

1. Select the pipeline you want to delete.
2. Click **Delete Pipeline**. You can't delete a pipeline while it's being created.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

!!! tip "Note"
    You cannot delete a pipeline if a schedule created with the pipeline you want to delete exists. Delete the pipeline schedule first, then delete the pipeline.

<a id="pipeline.run"></a>
## Run a Pipeline { #pipeline.run }

You can run and manage your uploaded pipelines in AI EasyMaker.

<a id="pipeline.run.create"></a>
### Create a Pipeline Run { #pipeline.run.create }

Run the pipeline.

- **Basic Information**
    - **Name**: Enter a name for the pipeline run.
    - **Description**: Enter description.
    - **Pipeline**: Select the pipeline you want to run.
    - **Experiment**: Select an experiment that will include pipeline execution. Experiments group related pipeline runs. If no experiments have been created, click **Add** to create an experiment.
- **Execution Information**
    - **Execution Parameters**: Enter a value if the pipeline has defined input parameters.
    - **Execution Type**: Select the type of pipeline execution. If you select **One-time**, the pipeline runs only once. To run the pipeline repeatedly at regular intervals, select **Enable Recurring Run** and then see [Create Recurring Run](#pipeline.recurring.run.create) to configure recurring runs.
- **Instance Information**
    - **Instance Type**: Select the instance type to run the pipeline on.
    - **Number of Instances**: Enter the number of instances to use to run the pipeline.
- **Additional Settings**
    - **Boot Storage Size**: Enter the boot storage size of the instance on which you want to run the pipeline.
    - **NHN Cloud NAS**: You can connect an **NHN Cloud NAS** to the instance where you want to run the pipeline.
        - **The name of the mount directory**: Enter the name of the directory to mount on the instance.
        - **NAS Path**: Enter the path in the following format: `nas://{NAS ID}:/{path}`.
    - **Manage Logs**: Logs that occur during pipeline execution can be stored in the NHN Cloud Log & Crash Search service.
        - For more information, refer to [Appendix > 2. NHN Cloud Log & Crash Search service usage guide and checking logs](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! tip "Note"
    Creating a pipeline run can take a few minutes.
    The initial resource creation requires an additional few minutes of time to configure the service environment.

!!! danger "Caution"
    Only NHN Cloud NAS created in the same project as AI EasyMaker is available.

<a id="pipeline.run.list"></a>
### Pipeline Run List { #pipeline.run.list }

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

<a id="pipeline.run.graph"></a>
### Pipeline Run Graph { #pipeline.run.graph }

A graph of the pipeline run is displayed. Select a node in the graph to see more information.

The graph is a pictorial representation of the pipeline execution. This graph shows the steps that have already been executed and the steps that are currently executing during pipeline execution, with arrows indicating the parent/child relationship between the pipeline components represented by each step. Each node in the graph represents a step in the pipeline.

With node-specific details, you can download the generated artifacts.

!!! danger "Caution"
    Artifacts older than 120 days are automatically deleted.

<a id="pipeline.run.stop"></a>
### Stop Pipeline Run { #pipeline.run.stop }

Stop running pipelines in progress.

1. Select the pipeline execution you want to stop from the list.
2. Click **Stop running**.
3. The requested action can't be canceled. Click **Confirm** to continue.

!!! tip "Note"
    Stopping pipeline execution can take a few minutes.

<a id="pipeline.run.copy"></a>
### Copy Pipeline Run { #pipeline.run.copy }

Create a new pipeline run with the same settings as an existing pipeline run.

1. Select the pipeline run you want to copy.
2. Click **Copy Pipeline Run**.
3. The Create pipeline run screen displays with the same settings as an existing pipeline run.
4. If you want to change any settings, make any changes, and then click **Create Pipeline Run**.

<a id="pipeline.run.delete"></a>
### Delete a Pipeline Run { #pipeline.run.delete }

Delete a pipeline run.

1. Select the pipeline run you want to delete.
2. Click **Delete Pipeline Run**. You cannot delete a pipeline run that is in progress.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

<a id="pipeline.schedule"></a>
## Pipeline Recurring Run { #pipeline.schedule }

You can create and manage a recurring run to periodically run the uploaded pipeline repeatedly in AI EasyMaker.

<a id="pipeline.recurring.run.create"></a>
### Create a Recurring Run { #pipeline.recurring.run.create }

Create a recurring run to run the pipeline in periodic iterations.

For information beyond the items below that you can set in creating a pipeline schedule, see [Create Recurring Run](#pipeline.run.create).

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

!!! tip "Note"
    Creating a recurring run can take a few minutes.
    The initial resource creation requires an additional few minutes of time to configure the service environment.

!!! tip "Note"
    The Cron expression uses six space-separated fields to represent the time.
    For more information, see the [Cron Expression Format](https://pkg.go.dev/github.com/robfig/cron#hdr-CRON_Expression_Format) documentation.

<a id="pipeline.recurring.run.list"></a>
### Pipeline Recurring Runs { #pipeline.recurring.run.list }

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

<a id="pipeline.recurring.run.start.stop"></a>
### Start and Stop Recurring Run { #pipeline.recurring.run.start.stop }

Stop a started pipeline recurring run or start a stopped pipeline recurring run.

1. Select the pipeline recurring run you want to start or stop from the list.
2. Click **Start Recurring Run** or **Stop Recurring Run**.

<a id="pipeline.recurring.run.copy"></a>
### Copy a Pipeline Recurring Run { #pipeline.recurring.run.copy }

Create a new pipeline recurring run with the same settings as an existing pipeline recurring run.

1. Select the pipeline recurring run you want to copy.
2. Click **Copy Pipeline Recurring Run**.
3. The Create pipeline schedule screen displays with the same settings as an existing pipeline schedule.
4. Make any changes to the settings you want to make, and then click **Create Pipeline Recurring Run**.

<a id="pipeline.recurring.run.delete"></a>
### Delete a pipeline recurring run { #pipeline.recurring.run.delete }

Delete a pipeline recurring run.

1. Select the pipeline recurring run you want to delete.
2. Click **Delete Pipeline Recurring Run**.
3. The requested delete task cannot be canceled. Click **Delete** to proceed.

!!! tip "Note"
    You cannot delete a run generated by the pipeline schedule you want to delete if it is in progress. Delete the pipeline schedule after the pipeline run is complete.

<a id="rag"></a>
## RAG { #rag }

Retrieval-Augmented Generation (RAG) is a technology that vectorizes and stores users' documents, retrieves content related to the question, and improves the accuracy of Large Language Model (LLM) responses. AI EasyMaker allows you to integrate vector store, embedding model, and LLM to create and manage RAG systems.

<a id="rag.create"></a>
### Create a RAG { #rag.create }

Create a new RAG.

- **Enable the API Gateway Service**
    - AI EasyMaker RAG uses the NHN Cloud API Gateway service to create and manage API endpoints. To use the RAG feature, you must enable the API Gateway service.
    - For more information on the API Gateway service and pricing, see the following:
        - [API Gateway Service Guide](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Application%20Service/API%20Gateway/ko/overview/)
        - [API Gateway Pricing](https://www.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/kr/pricing/by-service?c=Application%20Service&s=API%20Gateway)
- **Basic Settings**
    - **Name**: Enter a name for the RAG. RAG names must be unique.
    - **Description**: Enter a description for the RAG.
    - **Instance Type**: Select the instance type to run the RAG endpoint.
    - **Instance Count**: Enter the number of instances to run the RAG endpoint.
    - **Prompt**: The prompt to use in the RAG endpoint. Click **View Content** to see the full content of the prompt.
- **Vector Store Settings**
    - **Vector Store Type**: Select the vector store type.
{%- if "gov" not in build_flags %}
        - **RDS for PostgreSQL**
            - **Enable RDS for PostgreSQL**
                - AI EasyMaker RAG uses NHN Cloud RDS for PostgreSQL to create and manage the vector store. If you select this option, you must enable RDS for PostgreSQL.
                - For more information on RDS for PostgreSQL and pricing, see the following:
                    - [RDS for PostgreSQL Guide](/Database/RDS%20for%20PostgreSQL/en/overview/)
                    - [RDS for PostgreSQL Pricing](https://www.nhncloud.com/kr/pricing/by-service?c=Database&s=RDS%20for%20PostgreSQL)
            - **Instance Type**: Select the instance type for RDS for PostgreSQL.
            - **Storage Type**: Select the storage type for RDS for PostgreSQL.
            - **Storage Size**: The storage size for RDS for PostgreSQL.
            - **User ID**: Enter the user ID for connecting to PostgreSQL.
            - **Password**: Enter the password for connecting to PostgreSQL.
            - **Confirm Password**: Re-enter the password to confirm.
            - **VPC ID**: Enter the VPC ID for RDS for PostgreSQL.
            - **Subnet ID**: Enter the subnet ID for RDS for PostgreSQL.
{%- endif %}
        - **PostgreSQL Instance**: Use a user-created NHN Cloud PostgreSQL Instance as the vector store.
            - **User ID**: Enter the user ID for accessing the PostgreSQL Instance.
            - **Password**: Enter the password for accessing the PostgreSQL Instance.
            - **VPC ID**: Enter the VPC ID of the PostgreSQL Instance.
            - **Subnet ID**: Enter the subnet ID of the PostgreSQL Instance.
            - **PostgreSQL Instance IP**: Enter the IP address of the PostgreSQL Instance.
    - **Ingestion Settings**
        - **Data Path**: Enter the data path where the documents to be ingested into the vector store are stored.
    - **Embedding Model**
        - **Model**: Select the embedding model to use for vectorizing documents and queries.
        - **Instance Type**: The instance type to run the embedding model.
        - **Instance Count**: Enter the number of instances to run the embedding model.
- **LLM Settings**
    - **Model**: Select the LLM to use for generating responses.
    - **Instance Type**: The instance type to run the LLM.
    - **Instance Count**: The number of instances to run the LLM.
- **Additional Settings**
    - **Log Management**: You can save logs generated during RAG execution to the NHN Cloud Log & Crash Search service.
        - For more information, see [Appendix > 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Inquiry](#appendix.2.lncs.service.usage.guide.and.log.inquiry.guide).

!!! tip "Note"
    There may be limitations on the format, size, and number of files available for ingestion. For more information, see [Collect Sync](#rag.ingestion.sync).

!!! danger "Caution"
    Set the port to `15432` when using a PostgreSQL Instance.
    For instructions on how to create an instance, refer to [PostgreSQL Instance](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Compute/Instance/ko/component-guide{% if "gov" in build_flags %}-gov{% endif %}/#postgresql-instance).
    Configure the security group to allow access to port `15432` from the instance's subnet range.

!!! danger "Caution"
    Only NHN Cloud NAS created in the same project as AI EasyMaker can be used.

<a id="rag.list"></a>
### RAG List { #rag.list }

View and manage the list of generated RAGs. Select a RAG from the list to view detailed information.

- **Status**: a RAG status. Please refer to the table below for the main statuses:

| Status | Description |
| --- | --- |
| CREATE REQUESTED | RAG creation has been requested. |
| CREATE IN PROGRESS | RAG creation is in progress. |
| ACTIVE | RAG is operating normally. |
| UPDATE IN PROGRESS | RAG ingestion is in progress. |
| DELETE IN PROGRESS | RAG deletion is in progress. |
| CREATE FAILED | RAG creation has failed.<br/>Delete the RAG and create it again. If creation fails repeatedly, contact Customer Support. |
| UPDATE FAILED | RAG ingestion has failed.<br/>Try **Synchronize ingestions** again. If update fails repeatedly, contact Customer Support. |
| DELETE FAILED | RAG deletion has failed.<br/>Try deletion again. If deletion fails repeatedly, contact Customer Support. |

- **API Gateway Status**: the deployment status information for API Gateway basic stage.

| Status | Description |
| --- | --- |
| DEPLOYING | API Gateway Basic Stage is deploying. |
| COMPLETE | API Gateway Basic Stage has been successfully deployed and is enabled. |
| FAILURE | API Gateway Basic Stage deployment has failed. |

- **Ingestion History**: You can check the execution history of the document ingestion task in the **Ingestion History** tab of the details screen displayed when you select a RAG.
- **API Statistics**: You can check API statistics in the **API Statistics** tab of the detail screen displayed when you select a RAG.
- **Monitoring**: You can check the list of monitored instances and basic metric charts in the **Monitoring** tab of the details screen displayed when you select a RAG.

<a id="rag.ingestion.sync"></a>
### Synchronize Ingestions { #rag.ingestion.sync }

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

<a id="rag.delete"></a>
### Delete RAG { #rag.delete }

- You cannot delete the RAG that is on creation or deletion.
- The requested deletion task cannot be canceled.

<a id="rag.query.request.guide"></a>
### Guide to Asking RAG Questions { #rag.query.request.guide }

- When requesting a question, include `model` and `messages` in the request body, similar to the OpenAI Chat Completion API. For `model`, include the RAG name.
- For detailed request examples, please refer to the information below:

<details>
<summary><strong>Reqeust Example(cURL)</strong></summary>

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

</details>

<details>
<summary><strong>Stream request example (cURL)</strong></summary>

```bash
#!/bin/bash
set -euo pipefail

DEFAULT_URL="https://{API endpoint address}/rag/v1/query"
DEFAULT_MODEL="{RAG name}"
DEFAULT_PROMPT="Describe AI EasyMaker service."

usage() {
  cat <<'EOF'
How to use:
  <File name> -k <API_KEY> [-u URL] [-m MODEL] [-p PROMPT]

Option:
  -k   API key (x-nhn-apikey: send to <API_KEY> header)
  -u   Call URL
  -m   Model name
  -p   User prompt
  -h   Help

Description:
  - Call stream=true with an OpenAI-compatible specification,
    and sequentially write only the choices[].delta.content of
    each chunk delivered via streaming to standard output.

Required tool:
  - curl, jq
EOF
}

API_KEY=""
URL="$DEFAULT_URL"
MODEL="$DEFAULT_MODEL"
PROMPT="$DEFAULT_PROMPT"

while getopts ":k:u:m:p:h" opt; do
  case "$opt" in
    k) API_KEY="$OPTARG" ;;
    u) URL="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    p) PROMPT="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "The option -$OPTARG needs the value." >&2; usage; exit 2 ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl required." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq required." >&2
  exit 1
fi

# Create JSON Payload (OpenAI Chat Completions compatible)
payload="$(jq -n \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  '{
    model: $model,
    messages: [ { role: "user", content: $prompt } ],
    stream: true
  }'
)"

headers=( -H "Content-Type: application/json" )
if [[ -n "$API_KEY" ]]; then
  headers+=( -H "x-nhn-apikey: $API_KEY" )
fi

echo "request URL: $URL" >&2
echo "model: $MODEL" >&2
echo "---------------- Start stream ----------------" >&2

# Streaming processing: Extract only delta.content from the data: {json} line
curl -sS -N -X POST "$URL" "${headers[@]}" --data-raw "$payload" \
| while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == "data: [DONE]"* ]]; then
      break
    fi
    if [[ "$line" == data:* ]]; then
      json="${line#data: }"
      # There may be multiple choices, so print them all.
      # Delta.content may not be present, so it is treated as empty.
      while IFS= read -r piece; do
        printf "%s" "$piece"
      done < <(printf '%s\n' "$json" | jq -r '.choices[]?.delta?.content // empty')
    fi
  done

echo
echo "---------------- End stream ----------------" >&2
```

</details>

<a id="appendix"></a>
## Appendix { #appendix }

<a id="appendix.1.object.storage.account.permission"></a>
### 1. Add AI EasyMaker system account permissions to NHN Cloud Object Storage { #appendix.1.object.storage.account.permission }

Some features of AI EasyMaker use the user's NHN Cloud Object Storage as input/output storage
You must allow read or write access to user’s AI EasyMaker system account in NHN Cloud Object Storage container for running normal features.

Allowing read/write permissions on the AI EasyMaker system account to the user's NHN Cloud Object Storage container is meaning that AI EasyMaker system account can read or write files in accordance with permissions granted to all files in the user's NHN Cloud Object Storage container.

You have to check this information to set up an access policy in User Object Storage only with the required accounts and permissions.

The 'User' take responsibility for all consequences of allowing the user to access Object Storage for an account other than the AI EasyMaker system account during the access policy setting process, and AI EasyMaker is not responsible for it.

!!! tip "Note"
    According to features, AI EasyMaker accesses, reads or writes to Object Storage as follows.

| Feature | Access Right | Access target |
| --- | --- | --- |
| Training, hyperparameter tuning | Read | Algorithm path entered by user, training input data path |
| Training, hyperparameter tuning | Write | User-entered training output data, checkpoint path|
| Model | Read | Model artifact path entered by user |
| Model evaluation | Read | User-supplied input data path |
| Model evaluation | Write | User-supplied output data path |
| Batch inference | Read | User-supplied input data path |
| Batch inference | Write | User-supplied output data path |
| RAG | read | User-supplied ingestion data path |

To add read/write permissions to AI EasyMaker system account in Object Storage, refer to the following:

1. Click the **[Training]** or **[Model]** tab > **AI EasyMaker System Account Info**.
2. Save the **AI EasyMaker Tenant ID** and **AI EasyMaker API User ID** as the AI EasyMaker system account information.
3. Go to the NHN Cloud Object Storage console.
4. Refer to the [Allow read/write to specific projects or specific users](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Storage/Object%20Storage/ko/acl-guide{% if "gov" in build_flags %}-gov{% endif %}/#role-based-access-allow-rw-project-or-user) documentation to add the necessary read and write permissions for the AI EasyMaker system account in the NHN Cloud Object Storage console.

<a id="appendix.2.lncs.service.usage.guide.and.log.inquiry.guide"></a>
### 2. NHN Cloud Log & Crash Search Service Usage Guide and Log Inquiry Guide { #appendix.2.lncs.service.usage.guide.and.log.inquiry.guide }

<a id="appendix.2.lncs.service.usage.guide"></a>
#### NHN Cloud Log & Crash Search Service Usage Guide

Logs and events generated by the AI EasyMaker service can be stored in the NHN Cloud Log & Crash Search service.
To store logs in the Log & Crash Search service, you have to enable Log & Crash service and separate usage fee will be charged.

- **Log & Crash Search Service Usage and Pricing Information**
    - For more information on the Log & Crash Search service and pricing, see the following:
        - [Log & Crash Search Service Guide](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/{% if "gov" in build_flags %}gov-overview{% else %}Overview{% endif %}/)
        - [Log & Crash Search Pricing](https://www.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/kr/pricing/by-service?c=Data%20%26%20Analytics&s=Log%20%26%20Crash%20Search)

<a id="appendix.2.lncs.service.log.inquiry.guide"></a>
#### Log Query

1. Go to the Log & Crash Search service console page.
2. In the Log & Crash Search service, enter search criteria to view the logs.
    - AI EasyMaker Training log query: Retrieve logs where the category field is "easymaker.training".
        - Query: category:"easymaker.training"
    - AI EasyMaker Endpoint log query: Retrieve logs where the category field is "easymaker.inference".
        - Query: category:"easymaker.inference"
    - AI EasyMaker full log query: Retrieve logs where the logType field is "NNHCloud-AIEasyMaker".
        - Query: logType:"NHNCloud\-AIEasyMaker"
3. For details on how to use the Log & Crash Search service, refer to the [Log & Crash Search service console guide](https://docs.{% if "gov" in build_flags %}gov-{% endif %}nhncloud.com/en/Data%20&%20Analytics/Log%20&%20Crash%20Search/en/{% if "gov" in build_flags %}gov-console-guide{% else %}console-guide{% endif %}/).

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

<a id="appendix.3.hyperparameter"></a>
### 3. Hyperparameters { #appendix.3.hyperparameter }

- Value in Key-Value format entered through the console.
- When entry point is executed, it is passed to the execution factor (---{Key}).
- It can be stored and used as an environment variable (EM_HP_{Key converted to uppercase letter}).

As shown in the example below, you can use hyperparameter values entered during training creation.<br>
![HyperParameter Input Screen](http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_hyperparameter_en.png)

```python
import argparse

model_version = os.environ.get("EM_HP_MODEL_VERSION")

def parse_hyperparameters():
    parser = argparse.ArgumentParser()

    # Parsing the entered hyper parameter
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    ...

    return parser.parse_known_args()
```

<a id="appendix.4.environment"></a>
### 4. Environment Variables { #appendix.4.environment }

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

```python
import os
import tensorflow

dataset_dir = os.environ.get("EM_DATASET_TRAIN")
train_data = read_data(dataset_dir, "train.csv")

model = ... # Implement the model using the input data
model.load_weights(os.environ.get('EM_CHECKPOINT_INPUT_DIR', None))
callbacks = [
    tensorflow.keras.callbacks.ModelCheckpoint(filepath=f'{os.environ.get("EM_CHECKPOINT_DIR")}/cp-{{epoch:04d}}.ckpt', save_freq='epoch', period=50),
    tensorflow.keras.callbacks.TensorBoard(log_dir=f'{os.environ.get("EM_TENSORBOARD_LOG_DIR")}'),
]
model.fit(..., callbacks)

model_dir = os.environ.get("EM_MODEL_DIR")
model.save(model_dir)
```

<a id="appendix.5.tensorboard.store.metric.log"></a>
### 5. Store Indicator Logs for TensorBoard Usage { #appendix.5.tensorboard.store.metric.log }

- In order to check result indicators on the TensorBoard screen after training, the TensorBoard log storage space must be set to the specified location (`EM_TENSORBOARD_LOG_DIR`) when writing the training script.

<details>
<summary><strong>Example</strong></summary>

```python
import tensorflow as tf

# Specify the TensorBoard log path
tb_log = tf.keras.callbacks.TensorBoard(log_dir=os.environ.get("EM_TENSORBOARD_LOG_DIR"))

model = ... # model implementation

model.fit(x_train, y_train, validation_data=(x_test, y_test),
        epochs=100, batch_size=20, callbacks=[tb_log])
```

</details>

<details>
<summary><strong>Check TensorBoard Log</strong></summary>

<img src="http://static.toastoven.net/prod_ai_easymaker/console-guide_appendix_tensorboard.png" alt="Check TensorBoard Log">

</details>

!!! danger "Caution"
    Metrics older than 120 days will be deleted automatically.

<a id="appendix.6.framework.training.settings"></a>
### 6. Distributed Training Settings by Framework { #appendix.6.framework.training.settings }

- **Tensorflow**
    - The environment variable `TF_CONFIG` required for distributed training is automatically set. For more information, please refer to the [Tensorflow guide document](https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy).
- **Pytorch**
    - `Backends` settings are required for distributed training. If distributed training is performed on CPU, set it to gloo, and if distributed training is performed on GPU, set it to nccl. For more information, please refer to the [Pytorch guide document](https://pytorch.org/docs/stable/distributed.html).

<a id="appendix.7.cluster.upgrade"></a>
### 7. Upgrade the cluster version { #appendix.7.cluster.upgrade }

The AI EasyMaker service periodically upgrades the cluster version to provide stable service and new features.
When a new cluster version is deployed, you need to move the notebooks and endpoints that are running on the old version of the cluster to the new cluster.
Explains how to move new clusters by resource.

<a id="appendix.7.cluster.upgrade.notebook"></a>
#### Upgrade Notebook Cluster Version

On the **Notebook** list screen, notebooks that need to be moved to the new cluster display a **Restart** button to the left of their name.
Hovering the mouse pointer over the**Restart** button displays restart instructions and an expiration date.

- Before expiration, be sure to read the following caveats before clicking the **Restart** button.
    - Upon restart, data stored in the data storage (/root/easymaker directory path) will remain intact.
    - When you run a restart, data stored in boot storage is initialized and may be lost. Move your data to data storage before restarting.

Restarts take about 25 minutes for the first run, and about 10 minutes for subsequent runs.
Failed restarts are automatically reported to the administrator.

<a id="appendix.7.cluster.upgrade.endpoint"></a>
#### Upgrade the endpoint cluster version

On the **endpoints list** screen, endpoints that need to be moved to the new cluster will have a **! Notice** to the left of the name.
If you hover over the **! Notice**, it displays a version upgrade announcement and an expiration date.
Before the expiration, you must follow these instructions to move stages running on the old version cluster to the new version cluster.

<a id="appendix.7.cluster.upgrade.endpoint.stage"></a>
##### Upgrade the cluster version of a general stage

1. Delete a general stage that is not the default stage. Make sure the stage is in service before deleting.
2. Recreate the stage.
3. When a new stage becomes ACTIVE, check whether API calls and inference responses come normally to the stage endpoint.

!!! danger "Caution"
    Deleting a stage will shut down the endpoint, preventing API calls. Ensure that the stage is not in service before deleting it.

<a id="appendix.7.cluster.upgrade.endpoint.default.stage"></a>
##### Upgrade the cluster version of the default stage

The default stage is the stage on which the actual service operates.
To move the cluster version of the default stage without disrupting the service, use the following guide to move it.

1. Create a new stage to replace the default stage in an older version of the cluster.
2. Verify that API calls and inference responses are coming from the new stage endpoint as normal.
3. Click **Change Default Stage**. Select a new stage to change it to the default stage.
4. When the change is complete, the new stage is set as the default stage, and the existing default stage is deleted.

<a id="appendix.8.torchrun.usage"></a>
### 8. How to Use Torchrun { #appendix.8.torchrun.usage }

- The code has been written to enable distributed learning in Pytorch, and if you enter the number of distributed nodes and the number of processes per node, distributed learning using torchrun and distributed learning using multi-processes will be performed.
- Training and hyperparameter tuning can fail due to insufficient memory, depending on factors such as the total number of processes, model size, input data size, batch size, etc. If it fails due to insufficient memory, it may leave the following error messages. However, not all of the messages below are due to insufficient memory. Please set the appropriate instance type according to your memory usage.

```plaintext
exit code : -9 (pid: {pid})
```

- For more information about torchrun, see the [Pytorch Guide](https://pytorch.org/docs/stable/elastic/run.html).

<a id="appendix.9.resource.info"></a>
### 9. Resource Information { #appendix.9.resource.info }

When you create batch inferences and endpoints in AI EasyMaker, it allocates resources on the selected instance type, less the default usage.
The amount of resources you need depends on the demand and complexity of your model, so carefully set the number of pods and resource quota along with the appropriate instance type.

Batch inference allocates resources to each pod by dividing the actual usage by the number of pods. Endpoint cannot allow the quota you enter to exceed the actual usage of your instance, so check your resource usage beforehand.
Note that both batch inference and endpoints can fail to create if the allocated resources are less than the minimum usage required by the inference.

<a id="appendix.10.endpoint.api.specification"></a>
### 10. Endpoint API Specification { #appendix.10.endpoint.api.specification }

The AI EasyMaker service provides endpoints based on the open inference protocol (OIP) specification.
For more information about the OIP specification, see [OIP Specification](https://github.com/kserve/open-inference-protocol).

| Name                              | Method | API path                                                   |
|-----------------------------------|--------|------------------------------------------------------------|
| Model List                        | GET    | /v1/models                                                 |
| Model Ready                       | GET    | /v1/models/{model_name}                                    |
| Inference                         | POST   | /v1/models/{model_name}/predict                            |
| Description                       | POST   | /v1/models/{model_name}/explain                            |
| Server Information                | GET    | /v2                                                        |
| Server Live                       | GET    | /v2/health/live                                            |
| Server Ready                      | GET    | /v2/health/ready                                           |
| Model Information                 | GET    | /v2/models/{model_name}\[/versions/{model_version}\]       |
| Model Ready                       | GET    | /v2/models/{model_name}\[/versions/{model_version}\]/ready |
| Inference                         | POST   | /v2/models/{model_name}\[/versions/{model_version}\]/infer |
| OpenAI generative model inference | POST   | /v1/completions                                            |
| OpenAI generative model inference | POST   | /v1/chat/completions                                       |

!!! tip "Note"
    OpenAI generative model inference is used when using a generative model such as OpenAI's GPT-4o.
    The input values required for inference must be entered according to OpenAI's API specification. For more information, see the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat).
    For models that support the Completion and Chat Completion APIs provided by AI EasyMaker, see [Models](https://platform.openai.com/docs/models).

<a id="appendix.11.framework.note"></a>
### 11. Considerations for framework-specific serving { #appendix.11.framework.note }

<a id="appendix.11.framework.note.tensorflow.framework"></a>
#### TensorFlow Framework

The TensorFlow model serving provided by AI EasyMaker uses the SavedModel (.pb) recommended by TensorFlow.
To use checkpoints, save the checkpoint variables directory saved as a SavedModel along with the model directory, which will be used to serve the model.
Reference: [https://www.tensorflow.org/guide/saved_model](https://www.tensorflow.org/guide/saved_model)

<a id="appendix.11.framework.note.pytorch.framework"></a>
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

!!! tip "Note"
    There are differences in the request format between using TorchServe directly and using AI EasyMaker serving, so take care when writing the handler.py.
    Refer to the example below to see what values are passed, and implement the handler accordingly.

<details>
<summary><strong>Example(cURL)</strong></summary>

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

</details>

<details>
<summary><strong>Example(handler.py)</strong></summary>

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

</details>

<a id="appendix.11.framework.note.scikitlearn.framework"></a>
#### Scikit-learn Framework

AI EasyMaker uses mlserver to serve Scikit-learn models (.joblib).
The `model-settings.json`, which is required when using mlserver directly, is not required when using AI EasyMaker serving.

<a id="appendix.11.framework.note.hugging.face.framework"></a>
#### Hugging Face Framework

The Hugging Face model can be served using the Runtime provided by AI EasyMaker, TensorFlow Serving, or TorchServe.

<a id="appendix.11.framework.note.hugging.face.framework.runtime"></a>
##### Hugging Face Runtime

This is a simple way to serve Hugging Face models.
Hugging Face Runtime Serving does not support fine-tuning. To serve fine-tuned models, use the TensorFlow/Pytorch Serving method.

1. In Hugging Face, identify the model you want to serve.
2. Copy the Hugging Face model ID.
3. On the Create AI EasyMaker Model page, select the Hugging Face framework, and enter the Hugging Face model ID.
4. Create a model by entering the required inputs based on the model.
5. Verify the created model, and create an endpoint.

!!! tip "Note"
    Currently, the Hugging Face Runtime does not support the full range of Tasks in Hugging Face.
    The following tasks are supported: `sequence_classification`, `token_classification`, `fill_mask`, `text_generation`, and `text2text_generation`.
    To use unsupported Tasks, use the TensorFlow/Pytorch Serving method.

!!! tip "Note"
    To serve a gated model, you must enter the token of an account that is allowed access as a model parameter.
    If you do not enter a token, or if you enter a token from an account that is not allowed, the model deployment fails.

<a id="appendix.11.framework.note.hugging.face.framework.tensorflow.pytorch.serving"></a>
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
        - For more information about AI EasyMaker training, see [Training](#training).

2. View the Hugging Face model information and generate the files needed to serve it.
    - Save the model in the form required for framework-specific serving.
    - For more information, see the TensorFlow, PyTorch framework notes.
3. Upload the model file to OBS or NAS.
4. For the rest of the process, check out our guides to [creating models and](#model.create) [creating endpoints](#endpoint.create).


<a id="appendix.12.fine.tuning.dataset.format"></a>
### 12. Fine Tuning Dataset Format { #appendix.12.fine.tuning.dataset.format }

Prepare the training data and validation data for fine tuning as JSONL files in the `chat_template` format.

- Files are in JSONL (JSON Lines) format, with one conversation sample (JSON object) per line.
- Each sample consists of a `messages` array, where each item in the array contains a `role` and `content`.
    - `role`: The entity delivering the message. Use `system`, `user`, or `assistant`.
        - `system`: Defines the model's role or instructions (optional).
        - `user`: User input.
        - `assistant`: The correct response that the model should generate.
    - `content`: The text content of the corresponding role.

**Example (.jsonl)**

```json
{"messages": [{"role": "system", "content": "당신은 친절한 AI 비서입니다."}, {"role": "user", "content": "안녕하세요?"}, {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]}
{"messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}, {"role": "assistant", "content": "대한민국의 수도는 서울입니다."}]}
```

!!! tip "Note"
    - Use the `.jsonl` file extension, with only one JSON object per line.
    - The `assistant` message is the correct answer that the model learns from.
    - Validation data must be written in the same format as training data.
    - The maximum length of a single sample (one line) is 5,120 tokens. Samples that exceed the maximum length are excluded from training.