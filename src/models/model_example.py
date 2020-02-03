from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Concatenate, Dropout, TimeDistributed, Masking, Bidirectional
from keras.layers import Input, merge, InputLayer
from keras.layers import AveragePooling2D, Reshape, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json

max_hist_len = 54
question_len=44
answer_len = 3 # len(answers)
vocab_len= 4000 # len(vocab)
category_len = 80
learning_rate = 1e-4
adam1 = Adam(lr = learning_rate)
scc = 'sparse_categorical_crossentropy'
acc = ['accuracy']

def qcto_model():
    # History part
    qa_lstm = LSTM(512)
    hist_lstm = LSTM(512)
    # Input history question
    input_his_q = Input([max_hist_len, question_len+1,])
    q_embs = TimeDistributed(Embedding(vocab_len, 300))(input_his_q)
    q_embs = TimeDistributed(qa_lstm)(q_embs)
    # Input history answer
    input_his_a = Input([max_hist_len, 1])
    a_embs = TimeDistributed(Embedding(answer_len, 300))(input_his_a)
    a_embs = TimeDistributed(qa_lstm)(a_embs)
    # Concat both
    con_con = Concatenate()([q_embs, a_embs])
    hist_emb = hist_lstm(con_con)
     
    # Question
    input_que = Input([question_len+1,])
    que_embedding = Embedding(vocab_len, 300)(input_que)
    que_embedding = LSTM(512)(que_embedding)
    # Category
    input_cat = Input([1,])
    category = Flatten()(Embedding(category_len, 512)(input_cat))
    # Spatial
    spatial = Input([8,])
    
    # Concat
    con = Concatenate()([que_embedding, category, spatial, hist_emb])
    con = Dense(512, activation='relu')(con)
    answer = Dense(answer_len, activation='softmax')(con)

    model = Model([input_que, input_cat, spatial, input_hist, input_ans], answer)
    model.compile(optimizer=adam1, loss=scc, metrics=acc)
    return model
