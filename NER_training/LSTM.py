from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from DataProcessors.CoNLL2003Processor import CoNLL2003Processor


class LSTM_NER():
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 2000
        self.EMBEDDING_DIM = 300
        self.MAX_NB_WORDS = 20000
        pass


    def build_tensor(self,sequences,numrecs,word2index,maxlen,makecategorical=False,num_classes=0,is_label=False):
        data = np.empty((numrecs,),dtype=list)
        encoder = LabelBinarizer()
        transfomed_label = encoder.fit_transform(["B-geo", "B-gpe", "B-per", "I-geo", "B-org", "I-org", "B-tim", "B-art", "I-art", "I-per", "I-gpe",
                                                  "I-tim", "B-nat", "B-eve", "I-eve", "I-nat", "O"])
        print(transfomed_label)
        i = 0
        for sent in sequences:
            wids = []
            for word, lablel in sent:
                if is_label == False:
                    if word in word2index:
                        wids.append(word2index[word])
                       # print(word2index[word])
                    else:
                        wids.append(word2index['nothing'])
            if makecategorical and is_label:
                widsl = []
                for word, label in sent:
                    widsl.append(encoder.transform([label]))
                data[i]=widsl
            else:
                data[i] = wids
                #print(data[i])
            i +=1
        pdata = sequence.pad_sequences(data,maxlen=maxlen)
        return pdata

    def createModel(self):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)

        word_index = tokenizer.word_index

        self.embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        print(self.embedding_matrix.shape)
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
        #print(word_index)

        self.embedding_layer = Embedding(len(word_index) + 1,
                                         self.EMBEDDING_DIM,
                                         weights=[self.embedding_matrix],
                                         input_length=1,
                                         trainable=False)
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(17, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        pass

    def load_GLoVe_embeddings(self,GLOVE_DIR,text):
        pass

    def train(self):
        self.model.fit(self.X_train,self.Y_train,32,epochs=200,validation_split=0.1)
        pass

    def save_mode(self):
        pass

GLOVE_DIR = "../Resources/"
conll = CoNLL2003Processor("../Datasets/CoNLL2003/ner_dataset.csv")
sentences = conll.full_texts()
labels = conll.get_sentences_lablels_only()
text = conll.full_texts()

lstm = LSTM_NER()
lstm.load_GLoVe_embeddings(GLOVE_DIR,text)
lstm.createModel()
X = lstm.build_tensor(conll.sentences,len(conll.sentences),lstm.embeddings_index,100)
Y = lstm.build_tensor(conll.sentences,len(conll.sentences),lstm.embeddings_index,100,True,17,True)
print(X)
print(Y)
lstm.X_train,lstm.Y_train,lstm.X_test,lstm.Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# lstm.X_train = np.array(sentences[0:40000])
# lstm.X_train.reshape(lstm.X_train.shape[0],1)
# lstm.X_test = np.array(sentences[40000:])
# lstm.X_test.reshape(lstm.X_test.shape[0],1)
# lstm.Y_train = np.array(labels[0:40000])
# lstm.Y_train.reshape(lstm.Y_train.shape[0],1)
# lstm.Y_test = np.array(labels[40000:])
# lstm.Y_test.reshape(lstm.Y_test.shape[0],1)

lstm.train()