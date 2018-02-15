import csv
import nltk

class CoNLL2003Processor:
    def __init__(self,datapath):
        self.data_path = datapath
        self.dataset = []
        self.sentences = []
        with open(datapath, 'r') as csvfile:
            conllreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            i = 0
            tokens = []
            for line in conllreader:
                if i == 0:
                    i = i+1
                    continue
                i = i+1
                tokens.append((line[1],line[3]))
                if line[0]!="" and i>2:
                    self.sentences.append(tokens[:-1])
                    lt = tokens[-1]
                    tokens = []
                    tokens.append(lt)
                self.dataset.append((line[1],line[3]))

    def isCapitalized(self,word):
        if word[0].isupper():
            return True
        return False
    def addPoS_words(self):
        words = [x[0] for x in self.dataset]
        tags = nltk.pos_tag(words)
        ds = []
        for i in range(0, len(tags)):
            capitalized = "normal"
            if self.isCapitalized(self.dataset[i][0]):
                capitalized = "capitalized"
            ds.append((self.dataset[i][0], tags[i][1], capitalized, self.dataset[i][1]))
        self.dataset = ds
    def addPoS_sentences(self):
        sent = []
        for s in self.sentences:
            words2 = [x[0] for x in s]
            tags2 = nltk.pos_tag(words2)
            ds = []
            for i in range(0, len(tags2)):
                capitalized = "normal"
                if self.isCapitalized(self.dataset[i][0]):
                    capitalized = "capitalized"
                ds.append((s[i][0], tags2[i][1], capitalized, s[i][1]))
            sent.append(ds)
        self.sentences = sent

    def get_sentences_words_only(self):
        word_sent = []
        for sent in self.sentences:
            word_list = []
            for word,label in sent:
                word_list.append(word)
            word_sent.append(word_list)
        return word_sent

    def get_sentences_lablels_only(self):
        label_sent = []
        for sent in self.sentences:
            lable_list = []
            for word, label in sent:
                lable_list.append(label)
                label_sent.append(lable_list)
        return label_sent
    def full_text(self):
        full_text = ""
        for sent in self.sentences:
            for word,label in sent:
                full_text = full_text + " "+word
        return full_text

    def full_texts(self):
        full_texts = []
        for sent in self.sentences:
            text = ""
            for word, label in sent:
                text = text + " " + word
            full_texts.append(text)
        return full_texts
        #print(self.sentences[0])




#conll = CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
#print(conll.isCapitalized("Hello"))
#print(conll.isCapitalized("nothing"))
#conll.addPoS()
#for d in conll.dataset:
#    print(d)


