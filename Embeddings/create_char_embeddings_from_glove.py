import numpy as np
import os

file_path = "../Resources/glove.840B.300d.txt"

vectors = {}
with open(file_path, 'rb') as f:
    for line in f:
        line_split = line.strip().split(b" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        for char in word:
            if char < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)

base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
with open(base_name, 'w') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(str(word) + " " + " ".join(str(x) for x in avg_vector) + "\n")