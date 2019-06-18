import numpy as np
from keras import Sequential
import os
from keras.models import model_from_json
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Helpers.read_deid_surrogate import readSurrogate, tokenize_fa
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Dropout,LSTM,Bidirectional
from keras.optimizers import SGD,Nadam
import pickle

class CNN_BLSTM(object):
    def __init__(self, EPOCHS,DROPOUT,DROPOUT_RECURRENT,LSTM_STATE_SIZE,CONV_SIZE,LEARNING_RATE,OPTIMIZER):
        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER
        self.MAX_SEQUENCE_LENGTH = 200
        self.EMBEDDING_DIM = 300
        self.MAX_NB_WORDS = 2200000

    def loadData(self,path):
        documents = readSurrogate(path)
        train_docs = documents#[:600]
        print("Tokenizing")
        self.trainSequences = tokenize_fa(train_docs)
        print("Tokenized")

    def train(self):
        self.model.fit(self.X_train,self.Y_train,epochs=45,validation_split=0.1,batch_size=64)

    def save_model(self,model_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.model.save_weights(model_path+".h5")
        print("Saved model to disk")

    def load_model(self,model_path):
        json_file = open(model_path+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(model_path+".h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print("Loaded model from disk")

    def test_model(self):
        Y_pred = self.model.predict(self.X_test)
        #print(Y_pred)
        from sklearn import metrics
        # Y_testing = []
        labels = [1,2,3,4,5,6,7,8,9]
        #labels = ["DATE", "LOCATION", "NAME", "ID", "AGE", "CONTACT", "PROFESSION", "PHI"]

        Y_pred_F = []

        for i in range(0,len(Y_pred)):
            for j in range(0,len(Y_pred[i])):
                max_k = 0
                max_k_val =0
                for k in range(0,len(Y_pred[i][j])):
                    if Y_pred[i][j][k]>max_k_val:
                        max_k_val = Y_pred[i][j][k]
                        max_k = k
                Y_pred_F.append(max_k)

        Y_test_F = []
        for i in range(0,len(self.Y_test)):
            for j in range(0,len(self.Y_test[i])):
                max_k = 0
                max_k_val =0
                for k in range(0,len(self.Y_test[i][j])):
                    if self.Y_test[i][j][k]>max_k_val:
                        max_k_val = self.Y_test[i][j][k]
                        max_k = k
                Y_test_F.append(max_k)

        print(metrics.classification_report(Y_test_F, Y_pred_F,labels))

    def word2labels(self,token):
        return token[1]


    def word2features(self,token):
        return token[0]

    def createModel(self, text):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'),encoding='utf')
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            #word = values[0]
            coefs = np.asarray(values[-300:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(text)

        self.word_index = tokenizer.word_index
        pickle.dump(self.word_index,open("../Models/DeId/word_index.pkl",'wb'))

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        print(self.embedding_matrix.shape)
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(self.word_index) + 1,
                                         self.EMBEDDING_DIM,
                                         weights=[self.embedding_matrix],
                                         input_length=70,
                                         trainable=True)
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.6, return_sequences=True)))#{'sum', 'mul', 'concat', 'ave', None}
        self.model.add(Bidirectional(LSTM(60, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(9, activation='softmax')))  # a dense layer as suggested by neuralNer
        self.model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        self.model.summary()
        pass


    def build_tensor(self,sequences,numrecs,word2index,maxlen,makecategorical=False,num_classes=0,is_label=False):
        data = np.empty((numrecs,),dtype=list)
        label_index = {'O': 0}
        label_set = ["DATE", "LOCATION", "NAME", "ID", "AGE", "CONTACT", "PROFESSION", "PHI"]
        for lbl in label_set:
            label_index[lbl] = len(label_index)

        lb = LabelBinarizer()
        lb.fit(list(label_index.values()))
        i = 0
        plabels = []
        for sent in tqdm(sequences, desc='Building tensor'):
            wids = []
            pl = []
            for word, label in sent:
                if is_label == False:
                    if word in word2index:
                        wids.append(word2index[word])
                    else:
                        wids.append(word2index['the'])
                else:
                    pl.append(label_index[label])
            plabels.append(pl)
            if not is_label:
                data[i] = wids
            i +=1
        if is_label:
            plabels = sequence.pad_sequences(plabels, maxlen=maxlen)
            print(plabels.shape)
            pdata = np.array([lb.transform(l) for l in plabels])
        else:
            pdata = sequence.pad_sequences(data, maxlen=maxlen)
        return pdata

    def make_sequnces_labels(self):
        print("Training set creation")
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        for seq in self.trainSequences:
            features_seq = []
            labels_seq = []
            for i in range(0, len(seq)):
                features_seq.append(self.word2features(seq[i]))
                labels_seq.append(self.word2labels(seq[i]))
            self.X_train.append(features_seq)
            self.y_train.append(labels_seq)
        print("Training set created")

GLOVE_DIR = "../Resources/"
EPOCHS = 30               # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 275     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0055    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended
cnblstm = CNN_BLSTM(EPOCHS,DROPOUT,DROPOUT_RECURRENT,LSTM_STATE_SIZE,CONV_SIZE,LEARNING_RATE,OPTIMIZER)
path = "../Datasets/i2b2_data/training-PHI-Gold-Set1"
cnblstm.loadData(path)
cnblstm.make_sequnces_labels()
cnblstm.createModel(cnblstm.X_train)
cnblstm.word_index = pickle.load(open("../Models/DeId/word_index.pkl",'rb'))
X = cnblstm.build_tensor(cnblstm.trainSequences,len(cnblstm.trainSequences),cnblstm.word_index,70)
Y = cnblstm.build_tensor(cnblstm.trainSequences,len(cnblstm.trainSequences),cnblstm.word_index,70,True,9,True)
cnblstm.X_train,cnblstm.X_test,cnblstm.Y_train,cnblstm.Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

cnblstm.train()
cnblstm.save_model("../Models/DeId/BiLSTM_Glove_de_identification_model")
cnblstm.load_model("../Models/DeId/BiLSTM_Glove_de_identification_model")
cnblstm.test_model()