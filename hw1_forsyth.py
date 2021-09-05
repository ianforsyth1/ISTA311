'''
Name: Ian Forsyth
Assignment: Programming HW 1
Course: ISTA 311
Collaborators: Nizan Howard
'''

# Imports: we need numpy for np.log2()
import numpy as np

# Functions

def entropy(dist):
    '''
    this function calculates the entropy of a given probability distribution.
    '''
    total = 0
    for key in dist:
        total += sum([dist[key] * (np.log2(1/dist[key]))])
    return total 
        
def expected_length(code, dist):
    '''
    this function calculates the expected length of code words
    using the given frequency distribution.
    '''
    total = 0
    for key in dist:
        total += sum([len(code[key]) * dist[key]])
    return total

def cross_entropy(distp, distq):
    '''
    this function calculates the cross entropy between two given probability distributions. 
    '''
    total = 0
    for key in distp:
        total += sum([distp[key] * (np.log2(1/distq[key]))])
    return total
    
def kldiv(distp, distq):
    '''
    this function calculates the Kullback-Liebler divergence from the
    first probability distribution to the second.
    '''
    total = 0
    for key in distp:
        total += sum([distp[key] * (np.log2(distp[key]) - np.log2(distq[key]))])
    return total
    
def create_frequency_dict(string):
    '''
    this function takes a string and returns a dictionary representing
    a probability distribution
    '''
    dict1 = {key:string.count(key) / len(string) for key in string}  
    return dict1
    
def invert_dict(code):
    '''
    this function inverts our dictionary's values with its keys.
    '''
    inverted = dict(map(reversed, code.items())) #*insert Top Gun reference*
    return inverted
    
def encode(message, code):
    '''
    this function takes a string and dictionary of symbol code to encode the string
    as a string of 0s and 1s and returns the string.
    '''
    encode = ''
    for char in message:
        encode += code[char]
    return encode

def decode(message, code):
    '''
    this function takes the string of 0s and 1s, and a dictionary that represents symbol code
    to decode the string and return the decoded value.
    '''
    inverse = invert_dict(code)
    out = ''
    buffer = ''
    while message:
        buffer += message[len(buffer)]
        if buffer in inverse:
            out += inverse[buffer]
            message = message[len(buffer):]
            buffer = ''
    return out