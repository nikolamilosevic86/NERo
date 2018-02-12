import unittest
from DataProcessors import CoNLL2003Processor

class Test_CoNLL2003Test(unittest.TestCase):
    def test_file_reading(self):
        conll=CoNLL2003Processor.CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
        self.assertEqual(conll.dataset[0],('Thousands','O'))
    def test_conll_isuppercase(self):
        conll = CoNLL2003Processor.CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
        self.assertEqual(conll.isCapitalized("Hello"),True)
        self.assertEqual(conll.isCapitalized("nothing"),False)
    def test_conll_advanced(self):
        conll = CoNLL2003Processor.CoNLL2003Processor("C:\\Users\\mbaxkhm4\\NERo\\Datasets\\CoNLL2003\\ner_dataset.csv")
        conll.addPoS()
        print(conll.dataset[0])
        self.assertEqual(conll.dataset[0], ('Thousands', 'NNS', 'capitalized', 'O'))
if __name__ == '__main__':
    unittest.main()