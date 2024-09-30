
import unittest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import BASE_TOKENIZER
import storynavigation.modules.util as util

class TestUtil(unittest.TestCase):

    def setUp(self):
        self.short_testdata = Corpus.from_file("tests/short-testdata.tab") 
        self.short_testdata = BASE_TOKENIZER(self.short_testdata)

    def test_tupelize_corpus(self):
        output = util.tupelize_corpus(self.short_testdata)
        self.assertEqual(len(output), len(self.short_testdata), "Does not return right number of stories")
        self.assertIsInstance(output[0], tuple, "Does not return tuples")
        self.assertEqual(output[0], ("Hello world1.", 0), "Wrong output for first story")

if __name__ == '__main__':
    unittest.main()

# tests written with the help of ChatGPT.
