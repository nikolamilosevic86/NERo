from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten,TimeDistributed
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import tqdm
from DataProcessors.CoNLL2003Processor import CoNLL2003Processor


class LSTM_NER():
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 2000
        self.EMBEDDING_DIM = 300
        self.MAX_NB_WORDS = 20000
        pass


    def build_tensor(self,sequences,numrecs,word2index,maxlen,makecategorical=False,num_classes=0,is_label=False):
        data = np.empty((numrecs,),dtype=list)
        label_index = {'O': 0}
        label_set = ["B-geo", "B-gpe", "B-per", "I-geo", "B-org", "I-org", "B-tim", "B-art", "I-art", "I-per", "I-gpe",
                                                  "I-tim", "B-nat", "B-eve", "I-eve", "I-nat"]
        for lbl in label_set:
            label_index[lbl] = len(label_index)

        lb = LabelBinarizer()
        lb.fit(list(label_index.values()))
        i = 0
        plabels = []
        for sent in tqdm.tqdm(sequences, desc='Building tensor'):
            wids = []
            pl = []
            for word, label in sent:
                if is_label == False:
                    # wids.append(word2index[word])
                    if word in word2index:
                        wids.append(word2index[word])
                       # print(word2index[word])
                    else:
                        wids.append(word2index['the'])
                else:
                    pl.append(label_index[label])
            plabels.append(pl)
            if not is_label:
                data[i] = wids
                #print(data[i])
            i +=1
        # if makecategorical and is_label:
        #     pdata = sequence.pa100d_sequences(data,maxlen=maxlen)
        #     return pdata
        if is_label:
            plabels = sequence.pad_sequences(plabels, maxlen=maxlen)
            print(plabels.shape)
            pdata = np.array([lb.transform(l) for l in plabels])
        else:
            pdata = sequence.pad_sequences(data, maxlen=maxlen)
        return pdata
        #return data

    def createModel(self, text):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)

        self.word_index = tokenizer.word_index

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        print(self.embedding_matrix.shape)
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
        #print(word_index)

        self.embedding_layer = Embedding(len(self.word_index) + 1,
                                         self.EMBEDDING_DIM,
                                         weights=[self.embedding_matrix],
                                         input_length=70,
                                         trainable=False)
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        self.model.add(TimeDistributed(Dense(17, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        pass

    def load_GLoVe_embeddings(self,GLOVE_DIR,text):
        pass

    def train(self):
        self.model.fit(self.X_train,self.Y_train,epochs=200,validation_split=0.1,batch_size=128)
        pass

    def save_mode(self):
        pass

GLOVE_DIR = "../Resources/"
conll = CoNLL2003Processor("../Datasets/CoNLL2003/ner_dataset.csv")
sentences = conll.full_texts()
labels = conll.get_sentences_lablels_only()
text = conll.full_texts()
text = text[:20000]
sents = conll.sentences[:20000]
ml = max([len(s) for s in sents])
print("Maxlen observed: %d" % ml)
#
# text = ['test1 test2', 'test21 test22']
# sents = [
#     [('test1', 'O'), ('test2', 'B-geo')], [('test21', 'O'), ('test22', 'O')]
# ]
lstm = LSTM_NER()
# lstm.load_GLoVe_embeddings(GLOVE_DIR,text)
lstm.createModel(text)

X = lstm.build_tensor(sents,len(sents),lstm.word_index,70)
Y = lstm.build_tensor(sents,len(sents),lstm.word_index,70,True,17,True)

lstm.X_train,lstm.X_test,lstm.Y_train,lstm.Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(lstm.X_train.shape)
print(lstm.Y_train.shape)
print(lstm.X_train[0].shape)
print(lstm.Y_train[0].shape)
lstm.train()