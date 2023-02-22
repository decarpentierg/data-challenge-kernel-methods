The evaluation page describes how submissions will be scored and how students should format their submissions. 

The metric is the AUC (area under curve). The data contains 2 classes. 

## Submission Format

 Submission files should contain two columns: Id and Prediction. The file should contain a header and have the format described below. Id represents the identifier of the test example, ranging from 1 to 2000. The prediction is the corresponding logit which is a real number.
 
 Ex: 
 ```
 Id, Prediction 
 1, -1.1 
 2, 3.2 
 3, -2.4 
 4, -0.5 
 5, 2.1 
 6, 0.1 
 7, -0.9
 ```
 
 Below, you will also find a piece of code for reading/writing the data. 
 
 ```python 
 import pickle as pkl 
 import pandas as pd 
 
 with open('training_data.pkl', 'rb') as file: 
    train_graphs = pkl.load(file) 

with open('test_data.pkl', 'rb') as file: 
    test_graphs = pkl.load(file) 

with open('training_labels.pkl', 'rb') as file: 
    train_labels = pkl.load(file) 

# define your learning algorithm here 
# for instance, define an object called ``classifier''
classifier.train(train_labels,train_graphs) 

# predict on the test data 
# for instance, 
test_preds = classifier.predict(test_graphs) 
Yte = {'Prediction' : test_preds} 
dataframe = pd.DataFrame(Yte) 
dataframe.index += 1 
dataframe.to_csv('test_pred.csv',index_label='Id') 
```