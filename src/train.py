# ----
# parse the commands from shell
import argparse

parser = argparse.ArgumentParser(description="model type based on the inputs")
parser.add_argument("inputs", type=str)
parser.add_argument("--pilot", action="store_true", help="If true then model is trained on first 10000 samples")
parser.add_argument("--bilstm", action="store_true", help="If true then history input 'h' use BiLSTM otherwise LSTM")
parser.add_argument("--imgcompress", action="store_true", help="If true then image and object are compress to (512,) from (7,7,512) to minimise the memory usage")
parser.add_argument("--encode_qa", action="store_true", help="If true then QA in history are encoded separately otherwise concatenation is done before Timedistributed")
parser.add_argument("--onlyYES", action="store_true", help="If true then QA in history are encoded only with yes answers")
parser.add_argument("--gpu", nargs="?", const=1, type=int)
parser.add_argument("--dropout", type=float, default=0.0, help="any value in the range of 0.0-1.0")

args = parser.parse_args()
model_name = args.inputs
pilot_run = args.pilot
bilstm_mode = args.bilstm
img_compress = args.imgcompress
encode_qa = args.encode_qa
onlyYES = args.onlyYES
gpu_number = str(args.gpu) if args.gpu is not None else "0"

print("run pilot:", pilot_run)
print("model name:", model_name)
print("BiLSTM mode:", bilstm_mode)
print("Dropout:", args.dropout)
print("Encode QA in history separately:", encode_qa)
print("History has only YES answers:", onlyYES)

# models based on input:
# quesiton
# quesiton+category
# quesiton+image
# quesiton+spatial_features
# quesiton+history
# ...
# feature list => question, category, image, spatial, object, history
feature_list = {
    "q":"question",
    "c":"category", 
    "i":"whole image",
    "s":"spatial feature",
    "o":"cropped object",
    "h":"dialog history"
}
model_inputs = model_name.split('+')
model_input_fn = lambda all_inputs: tuple(all_inputs[feature_list[_t]] for _t in model_inputs)

# -----
import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_number


from models import all_models
#model_function = eval("all_models.{}_model".format(''.join(model_inputs)))
if bilstm_mode:
    model_name = model_name + "_bilstm"

if img_compress:
    model_name = model_name + "_compress"

if encode_qa:
    model_name = model_name + "_hisQ_enc_sep_testing"
    
if onlyYES:
    model_name = model_name + "_hisOnlyYes"

import numpy as np
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Concatenate, Dropout, TimeDistributed, Masking, Bidirectional
from keras.layers import Input, merge, InputLayer, Lambda, Multiply, Subtract
from keras.layers import AveragePooling2D, Reshape, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import keras.backend as K
from keras.optimizers import Adam
import json, pickle

# Preprocessing the data for the model
# "vect_data" - vectorized data file contains image, object, location, category_name, question-answer pairs and status of game
# "hist_data" - history of question-answer from each game
# "cat_data" - category_id 
def preparation(vect_data, hist_data, cat_data):
    def spatial_features(location):
       x_width = location[2]
       y_height = location[3]
       x_left = location[0]
       x_right = location[0] + location[2]
       y_upper = 224 - location[1]
       y_lower = y_upper - y_height
       x_center = x_left + 0.5*x_width
       y_center = y_lower + 0.5*y_height
       x_left = round((1.*x_left / 224) * 2 - 1,4)
       x_right = round((1.*x_right / 224) * 2 - 1,4)
       x_center = round((1.*x_center / 224) * 2 - 1,4)

       y_lower = round((1.*y_lower / 224) * 2 - 1,4)
       y_upper = round((1.*y_upper / 224) * 2 - 1,4)
       y_center = round((1.*y_center / 224) * 2 - 1,4)

       x_width = round((1.*x_width / 224) * 2,4)
       y_height = round((1.*y_height / 224) * 2,4)
       
       return [x_left,x_right,y_upper,y_lower,x_width,y_height,x_center,y_center]
    
    # load them based on games
    image_list = [i[0] for i in vect_data]
    object_list = [i[1] for i in vect_data]
    spatial_list = [spatial_features(i[2]) for i in vect_data]
    category_list = list(cat_data) # [i[3] for i in vect_data]
    qa = [i[4] for i in vect_data]
    status = [i[5] for i in vect_data]

    # append them based on question-answers 
    list_image = []
    list_object = []
    list_spatial = []
    list_category = []
    list_status = []
    for item in range(len(qa)):
        for i in qa[item]:
            list_image.append(image_list[item])
            list_object.append(object_list[item])
            list_spatial.append(spatial_list[item])
            list_category.append(category_list[item])
            list_status.append(status[item])

    list_data = []
    for i in range(len(hist_data)):
        list_data.append((hist_data[i][:-1], hist_data[i][-1][:-1], hist_data[i][-1][-1], list_image[i], list_object[i], list_spatial[i], list_category[i], list_status[i]))
        
    history, question, answer, image, crop, location, category, status  = zip(*list_data)
    
    return list(history), list(question), list(answer), list(image), list(crop), list(location), list(category)
 

def sort_yes(history1):
    zero_seq = history1[0][0]
    for i in history1:
        for x,y in enumerate(i):
            if y[-1] != 1:
                i[x] = zero_seq
    new_hist = []
    for i in history1:
        xx,yy = [],[]
        for j in i:
            if any(v!=0 for v in j):
                xx.append(j)
            else:
                yy.append(j)
        yy.extend(xx)
        new_hist.append(yy)
    print("History has only yes answers")
    return new_hist


# Compress function to reduce the memory usage while using image and object fc8 input
def compress(data):
    comp_data = []
    for i in data:
        img = np.sum(i,0)
        img = np.sum(img,0)
        img = np.divide(img,49)
        comp_data.append(img)
    return comp_data

# Generator for training data
# Data is shuffle on each epochs
# This is a sample generator, it need to be modified according to the model 
def train_gen(batch_size, x_data, y_data):
    index = np.arange(len(y_data))
    while True:
        np.random.shuffle(index)
        for i in range(0, len(y_data), batch_size):
            yield ([feature[index[i:i+batch_size]] for feature in x_data], y_data[index[i:i+batch_size]])
           
            
# Generator for validation data
# This is a sample generator, it need to be modified according to the model 
def val_gen(batch_size, x_data, y_data):
    while True:
        for i in range(0, len(y_data), batch_size):
            yield ([feature[i:i+batch_size] for feature in x_data], np.array(y_data[i:i+batch_size]))

# Training the model 
# Returns metrics and summary of the model
def train_model(train_data, valid_data):
    model = all_models.model_builder(model_inputs,args.dropout,args.bilstm,args.imgcompress,args.encode_qa)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    mc = ModelCheckpoint('saved_models/{0}.h5'.format(model_name), monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    history = model.fit_generator(train_gen(64, train_data[0], train_data[1]), steps_per_epoch=int(len(train_data[1])/64), validation_data=val_gen(128, valid_data[0], valid_data[1]), validation_steps=int(len(valid_data[1])/128), epochs=15, callbacks=[es,mc])
    return history, model


if __name__ == "__main__":
    print("load training data ...")
    vectorized_train = np.load('preprocessed_data/scratch/gw_train_vec.npy')
    history_train = np.load('preprocessed_data/scratch/hist_train.npy')
    cat_id_train = np.load('preprocessed_data/scratch/gw_train_cat_id.npy')
    print("prepare training data")
    history, question, answer, image, crop, location, category = preparation(vectorized_train,history_train,cat_id_train)

    print("load validation data ...")
    vectorized_valid = np.load('preprocessed_data/scratch/gw_valid_vec.npy')
    history_valid = np.load('preprocessed_data/scratch/hist_valid.npy')
    cat_id_valid = np.load('preprocessed_data/scratch/gw_valid_cat_id.npy')
    
    print("prepare validation data")
    v_history, v_question, v_answer, v_image, v_crop, v_location, v_category = preparation(vectorized_valid,history_valid,cat_id_valid)
    
    if onlyYES:
        history = sort_yes(history)
        v_history = sort_yes(v_history)
       
    if img_compress:
        image = compress(image)
        crop = compress(crop)
        v_image = compress(v_image)
        v_crop = compress(v_crop)
                          
    if pilot_run:
        # only on 10000 items
        history, question, answer, image, crop, location, category = (history[:10000], question[:10000], answer[:10000], image[:10000], crop[:10000], location[:10000], category[:10000])
        v_history, v_question, v_answer, v_image, v_crop, v_location, v_category = (v_history[:1000], v_question[:1000], v_answer[:1000], v_image[:1000], v_crop[:1000], v_location[:1000], v_category[:1000])

    print("convert all inputs to numpy format")
    x_train_data = model_input_fn({
        "question": np.array(question),
        "category": np.array(category), 
        "whole image": np.array(image),
        "spatial feature": np.array(location),
        "cropped object": np.array(crop),
        "dialog history": np.array(history)
    })
    y_train_data = np.array(answer)
    
    x_valid_data = model_input_fn({
        "question": np.array(v_question),
        "category": np.array(v_category), 
        "whole image": np.array(v_image),
        "spatial feature": np.array(v_location),
        "cropped object": np.array(v_crop),
        "dialog history": np.array(v_history)
    })
    y_valid_data = np.array(v_answer)
    
    print("start training")
    history, model = train_model((x_train_data,y_train_data),(x_valid_data,y_valid_data))

    with open('saved_models/model_loss_acc/loss_acc_{0}'.format(model_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Loss and Accuracy are saved in saved_models/model_loss_acc/ folder")

    with open('saved_models/model_summary/{0}_model_summary.txt'.format(model_name), 'w') as f:
        with redirect_stdout(f):
            model.summary()
        print("Summary of the trained model is saved in saved_models/model_summary/ folder") 

    print("Model is trained")