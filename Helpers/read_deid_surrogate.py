from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import re
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens

_treebank_word_tokenizer = TreebankWordTokenizer()
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
    real_tokens = []
    documents2 = []
    tbw = TreebankWordTokenizer()
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

def tokenize_fa(documents):
    sequences = []
    sequence = []
    for doc in documents:
        if len(sequence)>0:
            sequences.append(sequence)
        sequence = []
        text = doc["text"]
        file = doc["id"]
        text = text.replace("\"", "'")
        text = text.replace("`", "'")
        text = text.replace("``", "")
        text = text.replace("''", "")
        tokens = custom_span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            for tag in doc["tags"]:
                if int(tag["start"])<=token[0] and int(tag["end"])>=token[1]:
                    token_tag = tag["tag"]
                    #token_tag_type = tag["type"]
                    found = True
            if found==False:
                token_tag = "O"
                #token_tag_type = "O"
            sequence.append((token_txt,token_tag))
            if token_txt == ".":
                sequences.append(sequence)
                sequence = []
        sequences.append(sequence)
    return sequences

def custom_word_tokenize(text, language='english', preserve_line=True):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :param text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserver_line: bool
    """
    tokens = []
    sentences = [text] if preserve_line else nltk.sent_tokenize(text, language)
    for sent in sentences:
        for token in _treebank_word_tokenizer.tokenize(sent):
            if "-" in token:
                m = re.compile("(\d+)(-)([a-zA-z-]+)")
                g = m.match(token)
                if g:
                    for group in g.groups():
                        tokens.append(group)
                else:
                    tokens.append(token)
            else:
                tokens.append(token)
    return tokens

def custom_span_tokenize(text, language='english', preserve_line=False):
    tokens = custom_word_tokenize(text)
    tokens = ['"' if tok in ['``',"''"] else tok for tok in tokens]
    return align_tokens(tokens, text)

#print(custom_span_tokenize("He was a 47-year-old man born on 10/12/1975. His phone number is 170-574-2276"))

# documents = readSurrogate("../Datasets/i2b2_data/training-PHI-Gold-Set1")
# documents = tokenize(documents)
# print("Hi")