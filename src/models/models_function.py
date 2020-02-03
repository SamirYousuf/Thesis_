from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Concatenate, Dropout, TimeDistributed, Masking, Bidirectional
from keras.layers import Input, merge, InputLayer, concatenate, Add
from keras.layers import AveragePooling2D, Reshape, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
from keras.regularizers import l2


adam1 = Adam(lr = learning_rate)
scc = 'sparse_categorical_crossentropy'
acc = ['accuracy']

# Question -> Answer model
def que_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
  answer = Dense(config['answer_len'], activation='softmax')(con)
  model = Model(input_que, answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model
                   
# Question + Image -> Answer model
def que_img_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                    
  input_img = Input([[7,7,512])
  img_embedding = GlobalAveragePooling2D()(input_img)
  img_embedding = Dense(1024, activation='relu')(img_embedding)
                     
  con = Concatenate()([que_embedding, img_embedding])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_img], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model      
                    
# Question + Object -> Answer model
def que_obj_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                    
  input_obj = Input([[7,7,512])
  obj_embedding = GlobalAveragePooling2D()(input_obj)
  obj_embedding = Dense(1024, activation='relu')(obj_embedding)
                     
  con = Concatenate()([que_embedding, obj_embedding])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_obj], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model  
                     
# Question + Category -> Answer model
def que_cat_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                    
  input_cat = Input([config['category_len'],])
  category = Flatten()(Embedding(config['no_of_categories'], config['emb_units'])(input_cat))
                     
  con = Concatenate()([que_embedding, category])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_img], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model
 
# Question + Category + Image -> Answer model
def que_cat_img_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                    
  input_cat = Input([config['category_len'],])
  category = Flatten()(Embedding(config['no_of_categories'], config['emb_units'])(input_cat))
                              
  input_img = Input([[7,7,512])
  img_embedding = GlobalAveragePooling2D()(input_img)
  img_embedding = Dense(1024, activation='relu')(img_embedding)
  
  con = Concatenate()([que_embedding, category, img_embedding])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_cat, input_img], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model

# Question + Image + Object -> Answer model
def que_img_obj_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                                                
  input_img = Input([[7,7,512])
  img_embedding = GlobalAveragePooling2D()(input_img)
  img_embedding = Dense(1024, activation='relu')(img_embedding)
  
  input_obj = Input([[7,7,512])
  obj_embedding = GlobalAveragePooling2D()(input_obj)
  obj_embedding = Dense(1024, activation='relu')(obj_embedding)
  
  con = Concatenate()([que_embedding, img_embedding, obj_embedding])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_img, input_obj], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model 

# Question + Category + Object -> Answer model
def que_cat_obj_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                    
  input_cat = Input([config['category_len'],])
  category = Flatten()(Embedding(config['no_of_categories'], config['emb_units'])(input_cat))
                              
  input_obj = Input([[7,7,512])
  obj_embedding = GlobalAveragePooling2D()(input_obj)
  obj_embedding = Dense(1024, activation='relu')(obj_embedding)
  
  con = Concatenate()([que_embedding, category, obj_embedding])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_cat, input_obj], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model
 
# Question + Spatial -> Answer model
def que_spa_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
                              
  spatial = Input([config['spatial_coor_len'],])
                             
  con = Concatenate()([que_embedding, spatial])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, spatial], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model
   
# Question + Category + Spatial -> Answer model
def que_cat_spa_model():
  input_que = Input([config['question_len']+1,])
  que_embedding = Embedding(config['vocab_len'], config['emb_units'])(input_que)
  que_embedding = LSTM(config['lstm_units')(que_embedding)
   
  input_cat = Input([config['category_len'],])
  category = Flatten()(Embedding(config['no_of_categories'], config['emb_units'])(input_cat))
  
  spatial = Input([config['spatial_coor_len'],])
                             
  con = Concatenate()([que_embedding, category, spatial])
  con = Dense(config['hidden_units'], activation='relu')(con)
  answer = Dense(config['answer_len'], activation='softmax')(con)
    
  model = Model([input_que, input_cat, spatial], answer)
  model.compile(optimizer=adam1, loss=scc, metrics=acc)
  return model                           
