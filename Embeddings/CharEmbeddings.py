from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load
#in_filename = '../Datasets/NewsCorpora/char_sequences.txt'
raw_text = """Tolstoy began writing War and Peace in 1862, the year that he finally married and settled down at his country estate. The first half of the book was written under the name "1805". During the writing of the second half, he read widely and acknowledged Schopenhauer as one of his main inspirations. Tolstoy wrote in a letter to Afanasy Fet that what he has written in War and Peace is also said by Schopenhauer in The World as Will and Representation. However, Tolstoy approaches "it from the other side."

The first draft of the novel was completed in 1863. In 1865, the periodical Russkiy Vestnik (The Russian Messenger) published the first part of this draft under the title 1805 and published more the following year. Tolstoy was dissatisfied with this version, although he allowed several parts of it to be published with a different ending in 1867. He heavily rewrote the entire novel between 1866 and 1869.[5][9] Tolstoy's wife, Sophia Tolstaya, copied as many as seven separate complete manuscripts before Tolstoy considered it again ready for publication. The version that was published in Russkiy Vestnik had a very different ending from the version eventually published under the title War and Peace in 1869. Russians who had read the serialized version were anxious to buy the complete novel, and it sold out almost immediately. The novel was translated almost immediately after publication into many other languages.[citation needed]

It is unknown why Tolstoy changed the name to War and Peace. He may have borrowed the title from the 1861 work of Pierre-Joseph Proudhon: La Guerre et la Paix ("The War and the Peace" in French).[4] The title may also be another reference to Titus, described as being a master of "war and peace" in The Twelve Caesars, written by Suetonius in 119 CE. The completed novel was then called Voyna i mir (Война и мир in new-style orthography; in English War and Peace).[citation needed]

The 1805 manuscript was re-edited and annotated in Russia in 1893 and since has been translated into English, German, French, Spanish, Dutch, Swedish, Finnish, Albanian, Korean, and Czech.

Tolstoy was instrumental in bringing a new kind of consciousness to the novel. His narrative structure is noted for its "god-like" ability to hover over and within events, but also in the way it swiftly and seamlessly portrayed a particular character's point of view. His use of visual detail is often cinematic in scope, using the literary equivalents of panning, wide shots and close-ups. These devices, while not exclusive to Tolstoy, are part of the new style of the novel that arose in the mid-19th century and of which Tolstoy proved himself a master.

The standard Russian text of War and Peace is divided into four books (comprising fifteen parts) and an epilogue in two parts. Roughly the first half is concerned strictly with the fictional characters, whereas the latter parts, as well as the second part of the epilogue, increasingly consist of essays about the nature of war, power, history, and historiography. Tolstoy interspersed these essays into the story in a way that defies previous fictional convention. Certain abridged versions remove these essays entirely, while others, published even during Tolstoy's life, simply moved these essays into an appendix"""#load_doc(in_filename)
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = array(sequences)
X,y = sequences[:][:-1], sequences[:][-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))