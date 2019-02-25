from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
from nltk.tokenize.treebank import TreebankWordTokenizer

def readSurrogate(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    documents = []
    for file in onlyfiles:
        tree = ET.parse(path+"/"+file)
        root = tree.getroot()
        document_tags = []
        for child in root:
            if child.tag == "TEXT":
                text = child.text
            if child.tag == "TAGS":
                for chch in child:
                    tag = chch.tag
                    attributes = chch.attrib
                    start = attributes["start"]
                    end = attributes["end"]
                    content = attributes["text"]
                    type= attributes["TYPE"]
                    document_tags.append({"tag":tag,"start":start,"end":end,"text":content,"type":type})
        documents.append({"id":file,"text":text,"tags":document_tags})
    return documents

def tokenize(documents):
    tbw = TreebankWordTokenizer()
    real_tokens = []
    documents2 = []
    for doc in documents:
        text = doc["text"]
        file = doc["id"]
        text = text.replace("\"","'")
        #text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace(".", " ")
        tokens = tbw.span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            for tag in doc["tags"]:
                if int(tag["start"])<=token[0] and int(tag["end"])>=token[1]:
                    token_tag = tag["tag"]
                    token_tag_type = tag["type"]
                    found = True
            if found==False:
                token_tag = "O"
                token_tag_type = "O"

            real_tokens.append({"token":token_txt,"start":token[0],"end":token[1],"tag":token_tag,"tag_type":token_tag_type})
        documents2.append({"id": file, "text": text, "tags": doc["tags"],"tokens":real_tokens})
    return documents2


documents = readSurrogate("../Datasets/i2b2_data/training-PHI-Gold-Set1")
documents = tokenize(documents)
print("Hi")