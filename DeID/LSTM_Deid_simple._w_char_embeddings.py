import numpy as np
from keras import Sequential
import os

from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Helpers.read_deid_surrogate import readSurrogate, tokenize_fa
from Helpers.validation import compute_f1
from keras.models import Model,load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate, Reshape
from Helpers.prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformation,padding
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.optimizers import SGD,Nadam

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
        self.whole_text = ""
        for t in documents:
            self.whole_text = self.whole_text + " "+ t['text']
        train_docs = documents#[:600]
        #test_docs = documents[600:]
        print("Tokenizing")
        self.trainSequences = tokenize_fa(train_docs)
        #self.testSequences = tokenize_fa(test_docs)
        print("Tokenized")

    def train(self):
        self.model.fit([self.X_train,self.X_char_train],self.Y_train,epochs=5,validation_split=0.1,batch_size=64)

    def test_model(self):
        Y_pred = self.model.predict([self.X_test,self.X_char_test])
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
        self.chars = sorted(list(set(self.whole_text)))
        char_indices = dict((c, i) for i, c in enumerate(self.chars))
        indices_char = dict((i, c) for i, c in enumerate(self.chars))
        maxlen_char = 70
        maxlen = 70
        n_chars = len(self.chars)
        step = 3
        use_pca = False
        embedding_dim_char = 300
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen_char, step):
            sentences.append(text[i: i + maxlen_char])
            next_chars.append(text[i + maxlen_char])
        print('nb sequences:', len(sentences))

        print('Processing pretrained character embeds...')
        embedding_vectors = {}
        with open(os.path.join(GLOVE_DIR, 'glove.840B.300d-char.txt'), 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                char = line_split[0]
                embedding_vectors[char] = vec

        embedding_matrix = np.zeros((len(self.chars), 300))
        # embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
        for char, i in char_indices.items():
            # print ("{}, {}".format(char, i))
            embedding_vector = embedding_vectors.get(char)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Use PCA from sklearn to reduce 300D -> 50D
        if use_pca:
            pca = PCA(n_components=embedding_dim_char)
            pca.fit(embedding_matrix)
            embedding_matrix_pca = np.array(pca.transform(embedding_matrix))
            print(embedding_matrix_pca)
            print(embedding_matrix_pca.shape)

        print('Build model...')
        char_input = Input(shape=(maxlen, maxlen_char,))
        embedding_layer_char = TimeDistributed(Embedding(
            len(self.chars)+2, embedding_dim_char, input_length=maxlen_char,
            weights=[embedding_matrix_pca] if use_pca else [embedding_matrix]))
        # embedding_layer = Embedding(
        #     len(chars), embedding_dim, input_length=maxlen)
        char_embedded = embedding_layer_char(char_input)

        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(char_embedded)


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

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        print(self.embedding_matrix.shape)
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector



        token_input = Input(shape=(70,))

        self.embedding_layer = Embedding(len(self.word_index) + 1,
                                         self.EMBEDDING_DIM,
                                         weights=[self.embedding_matrix],
                                         input_length=70,
                                         trainable=False)
        token_embedded = self.embedding_layer(token_input)
        comb_emb = concatenate([token_embedded, char_enc])
        # self.model = Sequential()
        # embedding1_model = Sequential()
        # embedding1_model.add(self.embedding_layer)
        # embedding2_model = Sequential()
        # embedding2_model.add(embedding_layer_char)
        #self.model.add(self.embedding_layer)
        #self.model.add(concatenate([embedding1_model,embedding2_model]))
        #self.model.add(Reshape((600,)))
        #self.model.add(self.embedding_lay)
        BiLSTM_Layer = Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.6, return_sequences=True))#{'sum', 'mul', 'concat', 'ave', None}
        BiLSTM_out = BiLSTM_Layer(comb_emb)
        out = TimeDistributed(Dense(9, activation='softmax'))(BiLSTM_out)
        #self.model.add(Bidirectional(LSTM(60, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))
        #self.model.add(TimeDistributed(Dense(50, activation='relu')))
        #self.model.add(TimeDistributed(Dense(9, activation='softmax')))  # a dense layer as suggested by neuralNer
        #crf = CRF(17, sparse_target=True)
        #self.model.add(crf)
        #self.model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_viterbi_accuracy])
        self.model = Model(inputs=[token_input, char_input],
                      outputs=[out])
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


        self.chars = sorted(list(set(self.whole_text)))
        maxlen_char = 70
        n_chars = len(self.chars)
        char2idx = {c: i + 2 for i, c in enumerate(self.chars)}
        char2idx["UNK"] = 1
        char2idx["PAD"] = 0
        self.X_char = []
        for sentence in self.trainSequences:
            sent_seq = []
            for i in range(70):
                word_seq = []
                for j in range(70):
                    try:
                        word_seq.append(char2idx.get(sentence[i][0][j]))
                    except:
                        word_seq.append(char2idx.get("PAD"))
                sent_seq.append(word_seq)
            self.X_char.append(np.array(sent_seq))
        for seq in self.trainSequences:
            features_seq = []
            labels_seq = []
            for i in range(0, len(seq)):
                features_seq.append(self.word2features(seq[i]))
                labels_seq.append(self.word2labels(seq[i]))
            self.X_train.append(features_seq)
            self.y_train.append(labels_seq)
        print("Training set created")
        # print("Testing set creation")
        # for seq in self.testSequences:
        #     features_seq = []
        #     labels_seq = []
        #     for i in range(0, len(seq)):
        #         features_seq.append(self.word2features(seq[i]))
        #         labels_seq.append(self.word2labels(seq[i]))
        #     self.X_test.append(features_seq)
        #     self.y_test.append(labels_seq)
        # print("Testing set created")

GLOVE_DIR = "../Resources/"
EPOCHS = 30               # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 275     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0055    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended
cnblstm = CNN_BLSTM(EPOCHS,DROPOUT,DROPOUT_RECURRENT,LSTM_STATE_SIZE,CONV_SIZE,LEARNING_RATE,OPTIMIZER)
path = "../Datasets/i2b2_data/training-PHI-Gold-Set1-small"
cnblstm.loadData(path)
cnblstm.make_sequnces_labels()
cnblstm.createModel(cnblstm.X_train)
X = cnblstm.build_tensor(cnblstm.trainSequences,len(cnblstm.trainSequences),cnblstm.word_index,70)
Y = cnblstm.build_tensor(cnblstm.trainSequences,len(cnblstm.trainSequences),cnblstm.word_index,70,True,9,True)
#X_test = cnblstm.build_tensor(cnblstm.trainSequences,len(cnblstm.trainSequences),cnblstm.word_index,70)
#Y_test = cnblstm.build_tensor(cnblstm.testSequences,len(cnblstm.testSequences),cnblstm.word_index,70,True,9,True)
cnblstm.X_train,cnblstm.X_test,cnblstm.Y_train,cnblstm.Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
cnblstm.X_char_train,cnblstm.X_char_test,_,_ = train_test_split(X,Y,test_size=0.2,random_state=42)
#cnblstm.X_train = X
#cnblstm.Y_train = Y
#cnblstm.X_test = X_test
#cnblstm.Y_test = Y_test

cnblstm.train()
#lstm.save_mode()
cnblstm.test_model()