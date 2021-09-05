from hw1 import *
#from compare_files import compare_files
from os import path
import pickle

import unittest

from contextlib import redirect_stdout
import io

#from bitstring import Bits
"""
Files required:
dict1.pkl
dict2.pkl
code.pkl
decode.pkl
"""



class TestAssignment1(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.string1 = 'the quick brown fox jumps over the lazy dog to steal the supper sneakily'
        self.string2 = 'the five boxing wizards jump quickly with such pizazz and strike the devils'

        #self.bitstring = '00110100101011010110100011110110101101001001111001000100000111010011011010000000001010111011011110111111110101100111001001010110011010011111100000010100101101111100011110100000000011010011011111111000101001001100110000000101011100100010110101110111011101110111110011011010000101110101100011010010101101110110101111010000111111111'
        self.bitstring = '01000011101011011110000001011101011010011010110110011100001101100011110011110001111111101011101011100101101011001110010010011110111010010111100000100000101001100100010110011110000100001111000101110010000001111011110100011111111011111111111110111010110110111110001001000111000001010101011001000011101011010111101001011000011000010'

        with open('dict1.pkl', 'rb') as fp:
            self.dict1 = pickle.load(fp)
        with open('dict2.pkl', 'rb') as fp:
            self.dict2 = pickle.load(fp)
        with open('code.pkl', 'rb') as fp:
            self.code = pickle.load(fp)
        with open('decode.pkl', 'rb') as fp:
            self.decode = pickle.load(fp)

    def test_entropy(self):
        test_dict = {'n': 0.05, ' ': 0.15, 'e': 0.1, 'h': 0.1, 'a': 0.05, 's': 0.15, 't': 0.2, 'o': 0.05, 'i': 0.1, 'r': 0.05}
        H_correct = 3.1464393446710153
        self.assertAlmostEqual(entropy(test_dict), H_correct)
        
    def test_expected_length(self):
        L_correct1 = 4.555555555555556
        L_correct2 = 4.386666666666667
        L_correct3 = 1.75
        code_a = {'a':'0', 'b':'11', 'c':'10'}
        dict_a = {'a':0.25, 'b':0.70, 'c':0.05} 
        self.assertAlmostEqual(L_correct1, expected_length(self.code, self.dict1))
        self.assertAlmostEqual(L_correct2, expected_length(self.code, self.dict2))
        self.assertAlmostEqual(L_correct3, expected_length(code_a, dict_a))

    def test_cross_entropy(self):
        dict_a = {'a':0.25, 'b':0.70, 'c':0.05} 
        dict_b = {'a':0.1, 'b':0.2, 'c':0.7} 
        self.assertAlmostEqual(2.481560348784482, cross_entropy(dict_a, dict_b))
        self.assertAlmostEqual(3.3282643009871054, cross_entropy(dict_b, dict_a))

    def test_kldiv(self):
        dict_a = {'a':0.25, 'b':0.70, 'c':0.05} 
        dict_b = {'a':0.1, 'b':0.2, 'c':0.7} 
        self.assertAlmostEqual(1.4052627230592831, kldiv(dict_a, dict_b))
        self.assertAlmostEqual(2.171484651540065, kldiv(dict_b, dict_a))        

    def test_create_frequency_dict(self):
        simple_dict = create_frequency_dict(self.string1)
        correct_dict = {'a': 0.041666666666666664, ' ': 0.18055555555555555, 'i': 0.027777777777777776, 'z': 0.013888888888888888, 'o': 0.06944444444444445, 'd': 0.013888888888888888,
            'h': 0.041666666666666664, 'r': 0.041666666666666664, 'e': 0.09722222222222222, 'b': 0.013888888888888888, 'm': 0.013888888888888888, 'y': 0.027777777777777776, 'c': 0.013888888888888888,
            'n': 0.027777777777777776, 'j': 0.013888888888888888, 't': 0.06944444444444445, 'q': 0.013888888888888888, 'l': 0.041666666666666664, 'k': 0.027777777777777776, 'x': 0.013888888888888888,
            'p': 0.041666666666666664, 'g': 0.013888888888888888, 'u': 0.041666666666666664, 's': 0.05555555555555555, 'f': 0.013888888888888888, 'w': 0.013888888888888888, 'v': 0.013888888888888888}
        self.assertEqual(simple_dict.keys(), correct_dict.keys())
        for key in simple_dict:
            self.assertAlmostEqual(simple_dict[key], correct_dict[key])
    
    def test_invert_dict(self):
        self.assertEqual({'a': 1, 'b':2, 'c':3}, invert_dict({1:'a', 2:'b', 3:'c'}))
        self.assertEqual(self.decode, invert_dict(self.code))

    '''
    def test_huffman_code(self):
        freqs = get_frequency_dict(self.simple_string)
        code = huffman_code(freqs)
        average_length = sum(freqs[ch] * len(code[ch]) for ch in code)
        self.assertAlmostEqual(4.409090909090909, average_length)
        freqs = get_frequency_dict(self.saguaro_text)
        code = huffman_code(freqs)
        average_length = sum(freqs[ch] * len(code[ch]) for ch in code)
        self.assertAlmostEqual(4.551745405205378, average_length)
    '''

    def test_encode(self):
        self.assertEqual(self.bitstring, encode(self.string2, self.code))

    def test_decode(self):
        self.assertEqual(self.string2, decode(self.bitstring, self.code))

    '''
    def test_pad_bitstring(self):
        bs = '0010111010'
        bs2 = '00100111'
        self.assertEqual('0000000000100111', pad_bitstring(bs2))
        self.assertEqual('000001100010111010000000', pad_bitstring(bs))
    

    def test_write_encoded_file(self):
        [a, b] = write_encoded_file(self.simple_string, 'simple_test')
        self.assertTrue(path.exists('simple_test.compressed'))
        self.assertTrue(path.exists('simple_test.code'))
        self.assertEqual(26, a)
        self.assertEqual(368, b)
        #with open('simple_test.code') as fp:
        #    written_dict = eval(fp.read())
        #    self.assertEqual(self.simple_code, written_dict)
        #self.assertTrue(compare_files('simple_test.code', 'simple_test_correct.code', read_bytes = False))

    def test_decode_encoded_file(self):
        string = decode_encoded_file('test_message')
        self.assertEqual("There's something I ought to tell you. I'm not left-handed either!", string)

    def test_main(self):
        with io.StringIO() as buf, redirect_stdout(buf):
            main()
            self.assertEqual('Length of input file: 8107\nInput file entropy: 4.524\nLength of compressed file: 4614\nLength of encoding dict: 1244\n',
                 buf.getvalue())
            #self.assertTrue(compare_files('saguaros_correct.compressed', 'saguaros.compressed', read_bytes = True))
            self.assertTrue(compare_files('poem.txt', 'poem_correct.txt'))
            #text = open('saguaros.txt')
            #write_encoded_file(self.saguaro_text, 'saguaros')
            string = decode_encoded_file('saguaros')
            self.assertEqual(string, self.saguaro_text)
    '''

test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment1)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')