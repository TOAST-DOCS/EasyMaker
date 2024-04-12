## Machine Learning > AI EasyMaker > Release Notes

### April 23, 2024

#### Feature Updates

* Added the batch inference feature
    * Provides an environment to make batch inferences from an AI EasyMaker model and view inference results in statistics.
    * For more information, see the [Batch Inference Guide](./console-guide/#_51) article.
* Added the resource search feature
    * You can search for resources from the console screen, and navigate to other resource screens via links.
* Added the feature to change NAS of notebooks
    * You can change the NHN Cloud NAS connection settings for running notebooks.
* Scikit-learn serving support
    * Enabled Scikit-learn models to be registered in AI EasyMaker to serve as endpoints.
* Enable notebook shared memory
    * Enabled more than 64 MB of shared memory to be available.
    * The size depends on the instance type you selected when creating the notebook.
* Removed the save_steps hyperparameter from the NHN Cloud-provided algorithms
    * Removed the hyperparameter save_steps related to saving checkpoints.
    * The algorithm automatically calculates the appropriate number of save_steps and saves up to three.

### December 19, 2023

#### Feature Updates 

* Notebooks and training with private images  
    * User-personalized container images can be used to power notebooks, training, and hyperparameter tuning.
    * By registering private image and registry account, you can easily select private images to create resources.

* Dashboard
    * See your overall resource utilization, top 3 endpoint service monitoring, and top 3 CPU/GPU utilization on one page.

* Endpoint > Autoscaler
    * You can dynamically manage the number of nodes by setting policies to scale up/scale down endpoint nodes. 
    
### September 26, 2023

#### Feature Updates 

* Ubuntu 22.04 version provided 
    * The new Ubuntu 22.04 version is provided. Ubuntu 18.04 version is no longer available, and existing customers can use the service as it is now.

* Monitoring feature provided 
    * You can check system monitoring metrics for notebook, training, and endpoint.
    * You can view API call metrics for each API resource path in the endpoint.

* Basic algorithm for hyperparameter tuning 
    * Through hyperparameter tuning, you can optimize the hyperparameters of the basic algorithm provided by AI EasyMaker. 

* Endpoint > Serving multiple models
    * You can serve multiple training models on one endpoint stage.

* Parallel training for hyperparameter tuning
    * You can optimize the performance of hyperparameter tuning by adjusting the number of parallel trainings.


### June 27, 2023

#### Feature Updates

* Added hyperparameter tuning feature
    * Hyperparameter tuning is the feature to automate repetitive experiments to find optimized hyperparameters to improve the predictive accuracy and performance of machine learning models.
    * For more information, please see the [Hyperparameter Tuning Guide](./console-guide/#hyperparameter-tuning).
* Added 3 basic algorithms provided by NHN Cloud AI EasyMaker
    * For more information, please see the guide document of each algorithm.
    * [Image Classification Guide](./algorithm-guide/#image-classification)
    * [Object Detection Guide](./algorithm-guide/#object-detection)
    * [Semantic Segmentation Guide](./algorithm-guide/#semantic-segmentation)


### December 27, 2022
#### Release of a New Service 
* AI EasyMaker is an AI platform for environment, training and advancement, and endpoint services for machine learning development.