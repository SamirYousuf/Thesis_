<h1 align="center">
  Master Thesis (Master in Language Technology)
</h1>

## GuessWhat? - From what we discussed before
*Improving the VQA task in goal-oriented conversational games using the context of the preceding dialogue*


#### Requirements

* Python 3
* Tensorflow
* Keras
* NLTK
* Pandas
* Numpy

#### Files and Folder
```
--- images
--- src
  -- Evaluation           # evaluate model (.py & .ipynb)
  -- log_files
  -- models               # models (example, functions, all models)
  -- plot_graphs          # loss and accuracy plots
  -- preprocessed_data    # category_list, vocab_list, saved_data
  -- saved_models
    - model_loss_acc      # Loss and accuracy for each trained models
    - model_summary       # Summary for each model
  -- sh_files             # command to train models from command_line
  -- config.json
  -- preprocessor.py
  -- train.py
  -- visualise.py
--- thesis_report
  -- report               # Master thesis report
--- README.md
  ```
  
#### Parameters and settings
* Vocabulary size for question: 5875, answers: 3
* Maximum length of the question is 45 and the longest dialogue consists of 54 question/answer pairs
* Pre-trained VGG16  is used for image and object features
* Batchsize
  * Training: 64
  * Validation and Testing: 128
* Learning rate: *1e<sup>-4</sup>*, optimiser: *ADAM*, loss function: *sparse_categorical_crossentropy*
* Epochs: 15, EarlyStopping: 7 epochs on *validation_loss*
* ModelCheckpoint saves the best model based on *validation\_accuracy*

#### Model Architecture

Proposed model with Baseline module, history module and caption module.

###### Inputs

* Baseline Module
  * Question Q
  * Image I 
  * Target object O
  * Spatial location of the target object S
  * Category of the target object C
* History Module
  * History of previous question/answer pairs H
* Caption Module
  * Caption *cap<sub>i</sub>*
  
###### Model (Oracle)
<p align="center">
  <img width="460" height="300" src="https://github.com/SamirYousuf/Thesis_/blob/master/images/history_caption.pdf">
</p>

#### Results

Error-rate for baseline models, encoded history models and GuessWhat(de Vries)

| Model Name | GuessWhat(de Vries) | Baseline Model | History Model |
| :--- | :---: | :---: | :---: |
| Question | 41.2% | 40.5% | **37.2%** |
| Question + Image | 39.8% | 41.2% | 38.7% |
| Question + Category | 25.7% | 26.4% | 25.8% |
| Question + Spatial | 31.3% | 31.1% | |
| Question + Category + Spatial | 21.5% | 21.9% | 21.3% |
| Question + Category + Image | 27.4% | | 27.2% |
| Question + Category + Spatial + Image | 23.5% | | 23.1% |

###### Qualitative evaluation and comparision

<p align="center">
  <img width="460" height="300" src="https://github.com/SamirYousuf/Thesis_/blob/master/images/thesis_images_3.pdf">
</p>

