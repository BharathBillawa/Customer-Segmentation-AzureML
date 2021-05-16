
# Customer-Segmentation-AzureML
Illustrates customer segmentation in Azure ML.
## Dataset
The dataset in the project is taken from: kaggle - customer segmentation [dataset](https://www.kaggle.com/abisheksudarshan/customer-segmentation).  It is made up of data from an automobile company's market research, as well as a history of customer behavior. 

## Files
 1. `upload-data.py` handles creation of datastore used in model creation.
 2. `run-cust-seg.py` runs the model trainng on Azure compute cluster.
 3. `register-model.py` registers the saved model in AzureML for deployment.

## Models and performance
Listed below are models and their performance (interms of accuracy) for an example run:
| Model | Accuracy | 
| ------------- |:-------------:|
| Linear SVM |  0.4962825278810409 | 
| RBF SVM |   0.4547707558859975 | 
| Decision Tree |   0.5346964064436184 | 
| Random Forest |   0.5297397769516728 | 
| Neural Net |   0.509913258983891 | 
| AdaBoost |   0.5396530359355638 | 
| Naive Bayes |   0.5018587360594795 | 
| QDA  |   0.5210656753407683 | 


## References

 - https://docs.microsoft.com/en-us/azure/machine-learning/
 -  https://towardsdatascience.com/deploying-ml-models-on-azure-a948c106f7b5
 - https://towardsdatascience.com/how-to-deploy-a-local-ml-model-as-a-web-service-on-azure-machine-learning-studio-5eb788a2884c
 - https://stackoverflow.com/questions/41236871/how-to-download-the-trained-models-from-azure-machine-studio
 - https://stackoverflow.com/questions/55277334/how-can-i-register-in-azure-ml-service-a-machine-learning-model-trained-locally

