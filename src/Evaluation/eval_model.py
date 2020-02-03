####
# Parse input from the shell
####

import argparse

parser = argparse.ArgumentParser(description="evaluation of model based on the inputs")
parser.add_argument("inputs", type=str)
parser.add_argument("--bilstm", action="store_true", help="to evaluate a BiLSTM model")
parser.add_argument("--gpu", nargs="?", const=1, type=int, help="[0,1,2,3], default is 0")
parser.add_argument("--min", type=int, default=1, choices=range(1,21),
                    help="specify minimum number of objects in the range of 1-20, default is 1")
parser.add_argument("--max", type=int, default=20, choices=range(1,21),
                    help="specify maximum number of objects in the range of 1-20, default is 20")
parser.add_argument("--batch_size", type=int, default=64,
                    help="specify batch size, default is 64")

args = parser.parse_args()
model_name = args.inputs
bilstm_mode = args.bilstm
min_obj = args.min
max_obj = args.max
batch_size = args.batch_size
gpu_number = str(args.gpu) if args.gpu is not None else "0"

print("model name: ", model_name)
print("BiLSTM mode: ", bilstm_mode)
print("Min: ", min_obj)
print("Max: ", max_obj)
print("GPU: ", gpu_number)
print("Batch size: ", batch_size)

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

if bilstm_mode:
    model_name = model_name + "_bilstm"
    
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.models import Sequential, Model

def process_test_data(data_list, obj_min, obj_max):
    test_data = []
    for i in data_list:
        if obj_min <= i[6] <= obj_max:
            test_data.append(i)
    return test_data

def preparation(vect_test,hist_test, min_obj, max_obj):
    # load them based on games
    image_list = [i[0] for i in vect_test]
    object_list = [i[1] for i in vect_test]
    spatial_list = [i[2] for i in vect_test]
    category_list = [i[3] for i in vect_test]
    qa = [i[4] for i in vect_test]
    status = [i[5] for i in vect_test]
    num_obj = [i[6] for i in vect_test]

    # append them based on question-answers 
    list_image = []
    list_object = []
    list_spatial = []
    list_category = []
    list_status = []
    list_num_obj = []
    for item in range(len(qa)):
        for i in qa[item]:
            list_image.append(image_list[item])
            list_object.append(object_list[item])
            list_spatial.append(spatial_list[item])
            list_category.append(category_list[item])
            list_status.append(status[item])
            list_num_obj.append(num_obj[item])

    list_data = []
    for i in range(len(hist_test)):
        list_data.append((hist_test[i][:-1], hist_test[i][-1][:-1], hist_test[i][-1][-1], list_image[i], list_object[i], list_spatial[i], list_category[i], list_status[i], list_num_obj[i]))
        
    test_data = process_test_data(list_data, min_obj, max_obj)
    
    history, question, answer, image, crop, spatial, category, status, num_obj  = zip(*test_data)
    return list(history), list(question), list(answer), list(image), list(crop), list(spatial), list(category)
        
def eval_gen(batch_size, x_data, y_data):
    while True:
        for i in range(0, len(y_data), batch_size):
            yield ([feature[i:i+batch_size] for feature in x_data], np.array(y_data[i:i+batch_size]))

    
if __name__ == "__main__":
    print("load test data ...")
    vectorized_test = np.load("../preprocessed_data/scratch/gw_test_data.npy")
    history_test = np.load("../preprocessed_data/scratch/hist_test.npy")
    print("prepare test data for evaluation")
    history, question, answer, image, crop, spatial, category = preparation(vectorized_test,history_test, min_obj, max_obj)
    
    print("convert all inputs to numpy format")
    x_test_data = model_input_fn({
        "question": np.array(question),
        "category": np.array(category), 
        "whole image": np.array(image),
        "spatial feature": np.array(spatial),
        "cropped object": np.array(crop),
        "dialog history": np.array(history)
    })
    y_test_data = np.array(answer)
    
    print("Evaluating model {0}".format(model_name))
    model = load_model("../saved_models/{0}.h5".format(model_name))
    loss, acc = model.evaluate_generator(eval_gen(batch_size, x_test_data, y_test_data), steps=int(len(y_test_data)/batch_size))
    print("Loss : ", loss)
    print("Accuarcy : ", acc)
    print("Error Summary: Model: %s, Accuracy: %f, Loss: %f" % (model_name, (1-acc)*100, (1-loss)*100))
    print("Model successfully evaluated")


