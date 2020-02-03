from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Concatenate, Dropout, TimeDistributed, Masking, Bidirectional
from keras.layers import Input, merge, InputLayer, concatenate, Add, Lambda, Multiply, Subtract
from keras.layers import AveragePooling2D, Reshape, Flatten, GlobalAveragePooling2D
import keras.backend as K
from keras.optimizers import Adam
import json
from keras.regularizers import l2


learning_rate = 1e-4
opt = Adam(lr = learning_rate)
#opt = 'rmsprop'
scc = 'sparse_categorical_crossentropy'
acc = ['accuracy']
config = json.load(open('config.json'))

# This can build any model but it is only used in some models.
def model_builder(inputs, dropout, blstm=False, img_compress=False, encode_qa=False):
    # if any of these wasn't used it will be ignored :)
    embedding_layer = Embedding(config['vocab_len'], config['no_of_dim'])
    if blstm:
        qa_lstm = Bidirectional(LSTM(config['lstm_units']))
        hist_lstm = Bidirectional(LSTM(config['lstm_units']))
    else:
        qa_lstm = LSTM(config['lstm_units'])
        hist_lstm = LSTM(config['lstm_units'])
        
    embs = []
    inps = []
    if 'q' in inputs:
        input_que = Input([config['question_len']+1,])
        que_emb = embedding_layer(input_que)
        que_emb = qa_lstm(que_emb)
        embs.append(que_emb)
        inps.append(input_que)

    if 'c' in inputs:
        input_cat = Input([config['category_len'],])
        cat_emb = Flatten()(Embedding(config['no_of_categories'], config['no_of_cat_dim'])(input_cat))
        embs.append(cat_emb)
        inps.append(input_cat)

    if 's' in inputs:
        spatial = Input([config['spatial_coor_len'],])
        embs.append(spatial)
        inps.append(spatial)

    if 'o' in inputs:
        if img_compress:
            input_obj = Input([512,])
            obj_emb = Dense(512, activation='relu')(input_obj)
            embs.append(obj_emb)
            inps.append(input_obj)
        else:
            input_obj = Input([7,7,512])
            obj_emb = GlobalAveragePooling2D()(input_obj)
            obj_emb = Dense(512, activation='relu')(obj_emb)
            embs.append(obj_emb)
            inps.append(input_obj)

    if 'i' in inputs:
        if img_compress:
            input_img = Input([512,])
            img_emb = Dense(512, activation='relu')(input_img)
            embs.append(img_emb)
            inps.append(input_img)
        else:
            input_img = Input([7,7,512])
            img_emb = GlobalAveragePooling2D()(input_img)
            img_emb = Dense(512, activation='relu')(img_emb)
            embs.append(img_emb)
            inps.append(input_img)

    if 'h' in inputs:
        if encode_qa:
            input_hist = Input([config['max_hist_len'], config['question_len']+2,])
            hist_questions, hist_answers = Lambda(lambda x: [x[:,:,:-1], x[:,:,-1]])(input_hist)
            q_embs = TimeDistributed(embedding_layer)(hist_questions)
            q_embs = TimeDistributed(qa_lstm)(q_embs)
            #q_embs = Dense(300, activation='relu')(q_embs)
            #a_embs = embedding_layer(hist_answers)
            #qa_embs = Concatenate()([
            #    q_embs, 
            #    a_embs, 
            #    Multiply()([q_embs, a_embs]), 
            #    Subtract()([q_embs, a_embs]),
            #])
            #qa_embs = Concatenate()([q_embs, a_embs, ])
            hist_emb = hist_lstm(q_embs)
            embs.append(hist_emb)
            inps.append(input_hist)
        else:
            input_hist = Input([config['max_hist_len'], config['question_len']+2,])
            qa_embs = TimeDistributed(embedding_layer)(input_hist)
            qa_embs = TimeDistributed(qa_lstm)(qa_embs)
            hist_emb = hist_lstm(qa_embs)
            embs.append(hist_emb)
            inps.append(input_hist)

    con = Concatenate()(embs)
    con = Dense(512, activation='relu')(con)
    
    if dropout != 0.0 and float(dropout) < 1.0:
        dropout_layer = Dropout(dropout)(con)
    else:
        dropout_layer = con
    # Dropout here? We can see the results and use dropout layer later on the best models
    answer = Dense(config['answer_len'], activation='softmax')(dropout_layer)

    model = Model(inps, answer)
    model.compile(optimizer=opt, loss=scc, metrics=acc)
    return model