import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as kimage
from matplotlib import pyplot as plt
import json, time
import string
from collections import Counter, defaultdict 

# Pretrained VGG model
pretrained_cnn_model = VGG16(weights='imagenet', include_top=False)

def load_data(data):
    if data == "train":
        gw_train = [
            json.loads(line)
            for line in open('gw/guesswhat.train.jsonl')
        ]
        return gw_train
    elif data == "valid":
        gw_valid = [
            json.loads(line)
            for line in open('gw/guesswhat.valid.jsonl')
        ]
        return gw_valid
    elif data == "test":
        gw_test = [
          json.loads(line)
          for line in open('gw/guesswhat.test.jsonl')
        ]
        return gw_test
    else:
        print("Input is invalid")
    
def list_que_ans(data):
    list_questions = [] 
    list_answers = []
    for i in data:
        for j in i['qas']:
            list_questions.append(j['question'])
            list_answers.append(j['answer'])    
    list_qa = []
    for i in range(len(list_questions)):
        list_qa.append(list_questions[i])
        list_qa.append(list_answers[i])

    return list_qa 
 
def preprocess_qa(list_qa):
    # useful metadata:
    # vocabulary, longest_sentence_length

    # vocabulary for each file
    metadata = {
        'questions': {'vocab': Counter(), 'max_len': 0},
        'answers': {'vocab': Counter(), 'max_len': 0},
    }
    
    def process(s, mask_numbers=True, metadata=None):
        word_sequence = [w.strip().rstrip(string.punctuation) for w in s.lower().split()]
        metadata['vocab'].update(word_sequence)
        metadata['max_len'] = max(len(word_sequence), metadata['max_len'])
        
        # stateful reading of the file, each line changes the states as following:
        # question => answer => question
    state = 'questions'
    for line in list_qa:
        process(line, metadata=metadata[state])
        if state == 'questions':
            state = 'answers'
        elif state == 'answers':
            state = 'questions'
    return metadata

def indexing_metadata(metadata):
    question_len = metadata['questions']['max_len']
    answers = metadata['answers']['vocab']
    vocab = metadata['questions']['vocab']
    # dump the frequencies:
    vocab = ['<pad>', '<yes>', '<no>', '<n/a>', '<unk>', '?'] + [w for w,f in vocab.items() if f >= 3] # don't keep the rare words.
    word2index = defaultdict(lambda: vocab.index('<unk>'), zip(vocab, range(len(vocab)))) # this will set <unk> for unknwon words
    answer2index = {
      'yes': 0,
      'no': 1,
      'n/a': 2
    }
    answer2word_index ={
        i: word2index['<'+w.lower()+'>']
        for w,i in answer2index.items()
    }
  
    with open("preprocessed_data/vocabulary", "w") as f:
        json.dump({"answer_index" : list(answer2index.keys()), "vocabulary": vocab}, f)
    
# This function is used to create the vectorized data for "context image","crop image","question" and "answer" set
def load_processing(item):
    if item['image']['file_name'].find('train2014') > -1:
        img_pil = kimage.load_img("coco/train2014/{0}".format(item['image']['file_name']))
    elif item['image']['file_name'].find('val2014') > -1:
        img_pil = kimage.load_img("coco/val2014/{0}".format(item['image']['file_name']))
    elif item['image']['file_name'].find('test2014') > -1:
        img_pil = kimage.load_img("coco/test2014/{0}".format(item['image']['file_name']))
    
    ####
    X_questions = []
    Y_answers = []
    list_que = []
    list_ans = []
    for s in item['qas']:
        list_que.append([w.strip().rstrip(string.punctuation) for w in s['question'].lower().split()])
        list_ans.append(s['answer'])
    X_questions = [
    [word2index['<pad>']]*(question_len-len(line))+[word2index[w] for w in line]+[word2index['?']]
    for line in list_que
    ]
    Y_answers = [
    [answer2index[w.strip()] for w in line.split()]
    for line in list_ans
    ]
    QA = []
    for i in range(len(X_questions)):
        QA.append(X_questions[i]+[answer2word_index[ix] for ix in Y_answers[i]])
    ####
    width = item['image']['width']
    height = item['image']['height']
    img_pil1 = img_pil.resize((224,224))
    
    for obj in item['objects']:
        if obj['id'] == item['object_id']:
            x,y,w,h = obj['bbox']
            x = x*(224/width)
            y = y*(224/height)
            w = ((w+1)*(224/width))-1
            h = ((h+1)*(224/height))-1
            location = [round(x), round(y), round(w), round(h)]
            obj_img_pil = img_pil1.crop([int(x), int(y), int(x+w), int(y+h)])
            target_obj = preprocess_input(kimage.img_to_array(obj_img_pil.resize((224,224))))
            category_name = obj['category']
            category = obj['category_id']
       
    img = preprocess_input(kimage.img_to_array(img_pil1))
    
    # process all images (including objects) with CNN 
    img_features = pretrained_cnn_model.predict(np.array([img]+[target_obj])) 
        
    # return the feature vector of the image and the list of object ids with feature representation
    return (img_features[0], img_features[1], location, category, QA, item['status'])

# Create_history function takes vectorized qa data and total data as argument
# Length of max number of question in a game in obtain from total dataset
# Zero sequences of the max length are added before the question to have same shape
def create_history(data,total):
    # List = [Q1,Q2,Q3,Q4] - list of qa in a game
    # List_of_list = [[Q1],
                      #[Q1,Q2],
                      #[Q1,Q2,Q3],
                      #[Q1,Q2,Q3,Q4]]
    list_of_list = []
    for i in data:
        list_temp = []
        for j in i:
            list_temp.append(np.array(j))
            list_of_list.append(list(list_temp))
    
    # History format
    # Z = zero_sequence
    # history_data = [[Z,Z,Z,Z,Q1],
                      #[Z,Z,Z,Q1,Q2],
                      #[Z,Z,Q1,Q2,Q3],
                      #[Z,Q1,Q2,Q3,Q4]]
    history_data = []
    for i in list_of_list:
        temp_list = []
        for j in range(max([len(i['qas']) for i in total])+1 - len(i)):
            temp_list.append([0]*len(i[0]))
        history_data.append(temp_list+i)
        
    return np.array(history_data)
    
if __name__ == '__main__':
    train = load_data("train") # loading training data
    print("Training dataset is loaded")
    valid = load_data("valid") # loading validation data
    print("Validation dataset is loaded")
    test = load_data("test") # load testing data
    print("Testing dataset is loaded")
    total = train+valid+test
    list_qa = list_que_ans(total) # list of question answer
    metadata = preprocess_qa(list_qa) # metadata for question and answer pair
    indexing_metadata(metadata)
    print("Vocabulary of questions and answers are saved in preprocessed_data/vocabulary")

    # This code below takes longer time ....
    # Vectorized data for training, validation and testing dataset
    # Vectorized data format is a tuple of (image, object, location, category, question/answer, game_status)
    
    # Vectorizing train data and saving in numpy format
    vectorized_train = np.array([
        load_processing(item)
        for item in train
    ])
    np.save('preprocessed_data/scratch/gw_train_vec.npy', vectorized_train)

    # Vectorizing validation data and saving in numpy format
    vectorized_valid = np.array([
        load_processing(item)
        for item in valid
    ])
    np.save('preprocessed_data/scratch/gw_valid_vec.npy', vectorized_test)

    # Vectorizing test data and saving in numpy format
    vectorized_test = np.array([
        load_processing(item)
        for item in test
    ])
    np.save('preprocessed_data/scratch/gw_test_vec.npy', vectorized_test)

    # Creating history data from the question/answer pairs in the Vectorized datasets
    # Creating history for qa pairs in training dataset and saving as numpy
    data = [i[4] for i in vectorized_train]
    train_hist = create_history(data,total)
    np.save('preprocessed_data/scratch/hist_train.npy', train_hist)
    
    # Creating history for qa pairs in training dataset and saving as numpy
    data = [i[4] for i in vectorized_valid]
    train_valid = create_history(data,total)
    np.save('preprocessed_data/scratch/hist_valid.npy', valid_hist)
    
    # Creating history for qa pairs in training dataset and saving as numpy
    data = [i[4] for i in vectorized_test]
    train_test = create_history(data,total)
    np.save('preprocessed_data/scratch/hist_test.npy', test_hist)
    
    print("Pre-processing is done")
