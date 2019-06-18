import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import sys
sys.path.append('/home/mbaxknm4/NERo')
from Helpers.read_deid_surrogate import readSurrogate, tokenize_fa


class CRF_DeId_NER():
    def __init__(self):
        pass

    def shape(self,word):
        shape = ""
        for letter in word:
            if letter.isdigit():
                shape = shape + "d"
            elif letter.isalpha():
                if letter.isupper():
                    shape = shape + "W"
                else:
                    shape = shape + "w"
            else:
                shape = shape + letter
        return shape

    def word2features(self,sent, i):
        word = sent[i][0]
        #postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.shape()':self.shape(word),
            'word.isalnum()':word.isalnum(),
            'word.isalpha()':word.isalpha(),
            # 'postag': postag,
            # 'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            #postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.isalnum()':word1.isalnum(),
                '-1:word.isalpha()':word1.isalpha(),
                # '-1:postag': postag1,
                # '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i > 1:
            word2 = sent[i - 2][0]
            #postag2 = sent[i - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.isalnum()': word2.isalnum(),
                '-2:word.isalpha()': word2.isalpha(),
                # '-2:postag': postag2,
                # '-2:postag[:2]': postag2[:2],
            })
        else:
            features['BOS1'] = True
        if i > 2:
            word3 = sent[i - 3][0]
            #postag3 = sent[i - 3][1]
            features.update({
                '-3:word.lower()': word3.lower(),
                '-3:word.istitle()': word3.istitle(),
                '-3:word.isupper()': word3.isupper(),
                '-3:word.isdigit()': word3.isdigit(),
                '-3:word.isalnum()': word3.isalnum(),
                '-3:word.isalpha()': word3.isalpha(),
                # '-3:postag': postag3,
                # '-3:postag[:2]': postag3[:2],
            })
        else:
            features['BOS2'] = True

        if i > 3:
            word4 = sent[i - 4][0]
            #postag4 = sent[i - 4][1]
            features.update({
                '-4:word.lower()': word4.lower(),
                '-4:word.istitle()': word4.istitle(),
                '-4:word.isupper()': word4.isupper(),
                '-4:word.isdigit()': word4.isdigit(),
                '-4:word.isalnum()': word4.isalnum(),
                '-4:word.isalpha()': word4.isalpha(),
                # '-4:postag': postag4,
                # '-4:postag[:2]': postag4[:2],
            })
        else:
            features['BOS2'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.isalnum()': word1.isalnum(),
                '+1:word.isalpha()': word1.isalpha(),
                # '+1:postag': postag1,
                # '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        if i < len(sent) - 2:
            word12 = sent[i + 2][0]
            #postag12 = sent[i + 2][1]
            features.update({
                '+2:word.lower()': word12.lower(),
                '+2:word.istitle()': word12.istitle(),
                '+2:word.isupper()': word12.isupper(),
                '+2:word.isdigit()': word12.isdigit(),
                '+2:word.isalnum()': word12.isalnum(),
                '+2:word.isalpha()': word12.isalpha(),
                # '+2:postag': postag12,
                # '+2:postag[:2]': postag12[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 3:
            word13 = sent[i + 3][0]
            #postag13 = sent[i + 3][1]
            features.update({
                '+3:word.lower()': word13.lower(),
                '+3:word.istitle()': word13.istitle(),
                '+3:word.isupper()': word13.isupper(),
                '+3:word.isdigit()': word13.isdigit(),
                '+3:word.isalnum()': word13.isalnum(),
                '+3:word.isalpha()': word13.isalpha(),
                # '+3:postag': postag13,
                # '+3:postag[:2]': postag13[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 4:
            word14 = sent[i + 4][0]
            #postag14 = sent[i + 4][1]
            features.update({
                '+4:word.lower()': word14.lower(),
                '+4:word.istitle()': word14.istitle(),
                '+4:word.isupper()': word14.isupper(),
                '+4:word.isdigit()': word14.isdigit(),
                '+4:word.isalnum()': word14.isalnum(),
                '+4:word.isalpha()': word14.isalpha(),
                # '+4:postag': postag14,
                # '+4:postag[:2]': postag14[:2],
            })
        else:
            features['EOS2'] = True
        return features

    def doc2features(self,sent):
        return [self.word2features(sent['tokens'], i) for i in range(len(sent['tokens']))]

    def word2labels(self, sent):
        return sent[1]

    def sent2tokens(self,sent):
        return [token for token, postag,capitalized, label in sent]
    def prepare_features(self):
        pass

    def train(self):
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.05,
            max_iterations=200,
            all_possible_transitions=True
        )
        self.crf_model.fit(self.X_train, self.y_train)
    def save_model(self,path):
        pickle.dump(crf.crf_model, open(path, 'wb'))
    def predict(self,text):
        pass
print("Dataset reading")
documents = readSurrogate("../Datasets/i2b2_data/training-PHI-Gold-Set1")
print("Dataset read")
train_docs = documents[:400]
test_docs = documents[400:]
print("Tokenizing")
train_sequences = tokenize_fa(train_docs)
test_sequences = tokenize_fa(test_docs)
print("Tokenized")
train_tokens = []
test_tokens = []

crf = CRF_DeId_NER()
crf.X_train = []
crf.y_train = []
crf.X_test = []
crf.y_test = []
print("Training set creation")
for seq in train_sequences:
    features_seq = []
    labels_seq = []
    for i in range(0,len(seq)):
        features_seq.append(crf.word2features(seq, i))
        labels_seq.append(crf.word2labels(seq[i]))
    crf.X_train.append(features_seq)
    crf.y_train.append(labels_seq)
print("Training set created")
print("Testing set creation")
for seq in test_sequences:
    features_seq = []
    labels_seq = []
    for i in range(0,len(seq)):
        features_seq.append(crf.word2features(seq, i))
        labels_seq.append(crf.word2labels(seq[i]))
    crf.X_test.append(features_seq)
    crf.y_test.append(labels_seq)
print("Testing set created")
print("Training")
crf.train()
print("Train end")
labels = list(crf.crf_model.classes_)
labels.remove('O')
print(labels)
# #
y_pred = crf.crf_model.predict(crf.X_test)
f1_score = metrics.flat_f1_score(crf.y_test, y_pred,
                       average='weighted', labels=labels)
#
precision_score = metrics.flat_precision_score(crf.y_test, y_pred,
                       average='weighted', labels=labels)
#
recall_score = metrics.flat_recall_score(crf.y_test, y_pred,
                       average='weighted', labels=labels)
stats = metrics.flat_classification_report(crf.y_test, y_pred,
                        labels=labels)
print("Precision: "+str(precision_score))
print("Recall: "+str(recall_score))
print("F1-score: "+str(recall_score))
print(stats)
filename = '../Models/crf_baseline_model.sav'
pickle.dump(crf.crf_model, open(filename, 'wb'))
print("Done with all")