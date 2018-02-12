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
    def addPoS(self):
        words = [x[0] for x in self.dataset]
        tags = nltk.pos_tag(words)
        ds = []
        for i in range(0,len(tags)):
            capitalized = "normal"
            if self.isCapitalized(self.dataset[i][0]):
                capitalized = "capitalized"
            ds.append((self.dataset[i][0],tags[i][1],capitalized,self.dataset[i][1]))
        self.dataset = ds
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
        print(self.sentences)




#conll = CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
#print(conll.isCapitalized("Hello"))
#print(conll.isCapitalized("nothing"))
#conll.addPoS()
#for d in conll.dataset:
#    print(d)


