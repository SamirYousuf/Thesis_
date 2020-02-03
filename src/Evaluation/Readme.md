## Evaluation 
Evaluation is done on the test dataset

#### 1. Evaluation of a model 
Evaluation of any model for example (*q+c+h*) can be do using the *eval_model.py* 

Shell command:
```
python3 eval_model.py q+c+h --bilstm --gpu 0 --batch_size 128 --min 3 --max 5
```

Command take parse inputs
  * File name - eval_model.py
  * Model name - q+c+h (or any combination works from the given list [q,c,s,i,o,h]
  * Bilstm - Default is *False* 
  * GPU - Default is '0' Options [0,1,2,3]
  * Batch size - Default is 64
  * Minimum number of objects - Default is 0
  * Maximum number of objects - Default is 20
  
Outputs:
- Metrics of the model
  - Loss
  - Accuracy 
  

#### 2. Evaluation of a game 
Evaluation of any particular game in the dataset using *evaluate_game.ipynb*

The code in the notebook outputs
  - List of questions in the game
  - Category of the object
  - Status of the game
  - List of answers in the game
  - Predicted answers using the model
  - Image in the game


#### Results
Error rate for different models

| Model Name | Without History | With History | BiLSTM |
| :---: | :---: | :---: | :---: |
| q+c | 26.4 | 25.8 | 25.8 |
| q+c+s | 21.9 | 21.3 | 21.4 |

| Model Name | Without History | With History | 
| :---: | :---: | :---: | 
| q | 41.4 | 40.3 | 
| q+i | 41.2 | 38.7 |
| q+o | 39.0 | 
| q+s | 31.0 | 
| q+c+i |   | 27.2 |
| q+s+i |   | 31.07 | 
|q+c+s+i |   | 23.1 |
