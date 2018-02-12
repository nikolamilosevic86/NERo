import csv
import nltk

class CoNLL2003Processor:
    def __init__(self,datapath):
        self.data_path = datapath
        self.dataset = []
        self.labels = []
        with open(datapath, 'r') as csvfile:
            conllreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            i = 0
            for line in conllreader:
                if i == 0:
                    i = i+1
                    continue
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



#conll = CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
#print(conll.isCapitalized("Hello"))
#print(conll.isCapitalized("nothing"))
#conll.addPoS()
#for d in conll.dataset:
#    print(d)


