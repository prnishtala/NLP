# Team Members: Santu Hazra, Manu Chauhan, Ammar Adil and Prakash Nishtala

import os
import nltk
import string
from collections import Counter
from itertools import permutations, combinations, combinations_with_replacement

letters = string.ascii_lowercase


# write a python function to print pyramid pattern
def pyramid_pattern(symbol='*', count=4):
    for i in range(1, count + 1):
        print(' ' * (count - i) + symbol * i, end='')
        print(symbol * (i - 1) + ' ' * (count - i))


# write a python function to count the occurrence of a given word in a given file
def check_word_count(word, file):
    if not os.path.isfile(file):
        raise FileNotFoundError
    if not isinstance(word, str):
        raise TypeError

    with open(file, 'r') as f:
        lines = f.readlines()
        words = [l.strip().split(' ') for l in lines]
        words = [word for sublist in words for word in sublist]
        c = Counter(words)
    return c.get(word, 0)


# write a python function to make permutations from a list with given length
def get_permutations(data_list, l=2):
    return list(permutations(data_list, r=l))


# write a python program to get all possible permutations of size of the string in lexicographic sorted order.
def get_ordered_permutations(word, k):
    [print(''.join(x)) for x in sorted(list(permutations(word, int(k))))]


# write a python program to get all possible combinations, up to size of the string in lexicographic sorted order.
def get_ordered_combinations(string, k):
    [print(''.join(x)) for i in range(1, int(k) + 1) for x in combinations(sorted(string), i)]


# write a python function to get all possible size replacement combinations of the string in lexicographic sorted order.
def get_ordered_combinations_with_replacement(string, k):
    [print(''.join(x)) for x in combinations_with_replacement(sorted(string), int(k))]



# write a python function for Caesar Cipher, with given shift value and return the modified text
def caesar_cipher(text, shift=1):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return text.translate(table)


# write a python function for a string to swap the case of all letters.
def swap_case(s):
    return ''.join(x for x in (i.lower() if i.isupper() else i.upper() for i in s))


# write a python function to get symmetric difference between two sets from user.
def symmetric_diff_sets():
    M, m = input(), set(list(map(int, input().split())))
    N, n = input(), set(list(map(int, input().split())))
    s = sorted(list(m.difference(n)) + list(n.difference(m)))
    for i in s:
        print(i)


# write a python function to check if given set is subset or not
def check_subset():
    for _ in range(int(input())):
        x, a, z, b = input(), set(input().split()), input(), set(input().split())
    print(a.issubset(b))


# write a python program for basic HTML parser
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])


parser = MyHTMLParser()

for i in range(int(input())):
    parser.feed(input())


# write a python function for Named Entity Recognizer using NLTK
def ner_checker(texts):
    all_set = set()

    def nltk_ner_check(texts):
        for i, text in texts:
            for entity in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
                if isinstance(entity, nltk.tree.Tree):
                    etext = " ".join([word for word, tag in entity.leaves()])
                    # label = entity.label()
                    all_set.add(etext)

    nltk_ner_check(texts=texts)
    return all_set


#write a function to compress a given string. Suppose a character 'c' occurs consecutively X times in the string. Replace these consecutive occurrences of the character 'c' with  (X, c) in the string.
def compress(text):
    from itertools import groupby
    for k, g in groupby(text):
        print("({}, {})".format(len(list(g)), k), end=" ")


# write a python function to count 'a's in the repetition of a given string 'n' times.
def repeated_string(s, n):
    return s.count('a') * (n // len(s)) + s[:n % len(s)].count('a')


# write a python function to find all the substrings of given string that contains 2 or more vowels. Also, these substrings must lie in between 2 consonants and should contain vowels only.
def find_substr():
    import re
    v = "aeiou"
    c = "qwrtypsdfghjklzxcvbnm"
    m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c, v, c), input(), flags=re.I)
    print('\n'.join(m or ['-1']))


# write a python function that given five positive integers and find the minimum and maximum values that can be calculated by summing exactly four of the five integers.
def min_max():
    nums = [int(x) for x in input().strip().split(' ')]
    print(sum(nums) - max(nums), sum(nums) - min(nums))


# write a python function to find the number of (i, j) pairs where i<j and ar[i]+ar[j] is divisible by k in a data list
def divisible_sum_pairs(arr, k):
    count = 0
    n = len(arr)
    for i in range(n - 1):
        j = i + 1
        while j < n:
            if ((arr[i] + arr[j]) % k) == 0:
                count += 1
            j += 1
    return count


import math


# Write a python Class to calculate area of a circle and print the vale for a radius
class CircleArea:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius * self.radius


r = 2
obj = CircleArea(r)
print("Area of circle:", obj.area())


# Write a python function to count the number of Words in a Text File
def check_words():
    fname = input("file name: ")
    num_words = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    print("Number of words = ", num_words)


# Write a python function to Count the Number of Lines in a Text File
def check_lines():
    fname = input("file name: ")
    num_lines = 0
    with open(fname, 'r') as f:
        for line in f:
            num_lines += 1
    print("Number of lines = ", num_lines)


# Write a python function that Counts the Number of Blank Spaces in a Text File
def count_blank_space():
    fname = input("file name:")
    count = 0
    with open(fname, 'r') as f:
        for line in f:
            count += line.count(' ')
    return count


# Write a python function to check if 2 strings are anagrams or not
def anagram(s1, s2):
    if sorted(s1) == sorted(s2):
        return True
    else:
        return False


# Write a python function to remove the duplicate items from a List and return the modified data list
def remove_duplicates(data):
    c = Counter(data)
    s = set(data)
    for item in s:
        count = c.get(item)
        while count > 1:
            data.pop(item)
            count -= 1
    return data


# write a python function to get the most common word in text
def most_common(text):
    c = Counter(text)
    return c.most_common(1)


# write a python function to do bitwise multiplication on a given bin number by given shifts
def bit_mul(n, shift):
    return n << shift


# write a python function for bitwise division with given number of shifts
def bit_div(n, shift):
    return n >> shift


# write a python program to implement Queue
from collections import deque

class Queue():
    '''
    Thread-safe, memory-efficient, maximally-sized queue supporting queueing and
    dequeueing in worst-case O(1) time.
    '''


    def __init__(self, max_size = 10):
        '''
        Initialize this queue to the empty queue.

        Parameters
        ----------
        max_size : int
            Maximum number of items contained in this queue. Defaults to 10.
        '''

        self._queue = deque(maxlen=max_size)


    def enqueue(self, item):
        '''
        Queues the passed item (i.e., pushes this item onto the tail of this
        queue).

        If this queue is already full, the item at the head of this queue
        is silently removed from this queue *before* the passed item is
        queued.
        '''

        self._queue.append(item)


    def dequeue(self):
        '''
        Dequeues (i.e., removes) the item at the head of this queue *and*
        returns this item.

        Raises
        ----------
        IndexError
            If this queue is empty.
        '''

        return self._queue.pop()


# write a python function to get dot product between two lists of numbers
def dot_product(a, b):
    return sum(e[0] * e[1] for e in zip(a, b))


# write a python function to strip punctuations from a given string
def strip_punctuations(s):
    return s.translate(str.maketrans('', '', string.punctuation))


# write a python function that returns biggest character in a string
from functools import reduce


def biggest_char(string):
    if not isinstance(string, str):
        raise TypeError
    return reduce(lambda x, y: x if ord(x) > ord(y) else y, string)


# write a python function to Count the Number of Digits in a Number
def count_digits():
    n = int(input("Enter number:"))
    count = 0
    while n > 0:
        count = count + 1
        n = n // 10
    return count


# write a python function to count number of vowels in a string
def count_vowels(text):
    v = set('aeiou')
    for i in v:
        print(f'\n {i} occurs {text.count(i)} times')


# write a python function to check external IP address
def check_ip():
    import re
    import urllib.request as ur
    url = "http://checkip.dyndns.org"
    with ur.urlopen(url) as u:
        s = str(u.read())
        ip = re.findall(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", s)
        print("IP Address: ", ip[0])
        return ip[0]


# write a python function for some weird hypnosis text.
def weird():
    import random

    def getlength(script):
        return sum((i['length'] for i in script))

    def truncate(target_length, script):
        if getlength(script) > target_length:
            script = sorted(script, key=lambda k: (k['priority'], -k['length']))[:-1]
            return truncate(target_length, script)
        return sorted(script, key=lambda k: k['index'])

    def as_text(script):
        return "\n".join([i['text'] for i in script])

    priorities_and_sentences = [
        (1, "...now... sitting comfortably in the chair"),
        (2, "...with your feet still flat on the ground"),
        (3, "...back straight and head up right"),
        (2, "...make these adjustments now if you need to"),
        (3, "... pause.............................."),
        (1, "...your eyes ...still ...comfortably closed"),
        (2, "...nice and relaxed...comfortable and relaxed..."),
        (3, "... pause......................................."),
        (1, "...now...I want you to notice...how heavy your head is starting to feel..."),
        (1, "how heavy your head feels..."),
        (3, "... pause......................................."),
        (2, "really noticing the weight... of your head..."),
        (3,
         "and how much more ...comfortable...it will feel when you let your neck relaxes ...and your head begins to fall forward ...into a much more comfortable"),
    ]

    scriptlist = [{'priority': j[0], 'text': j[1], 'length': len(j[1]), 'index': i} for i, j in
                  enumerate(priorities_and_sentences)]

    print(as_text(truncate(500, scriptlist)))
    print(as_text(truncate(300, scriptlist)))
    print(as_text(truncate(200, scriptlist)))


# write a python function for dice roll asking user for input to continue and randomly give an output.
def dice():
    import random
    min = 1
    max = 6
    roll_again = 'y'

    while roll_again == "yes" or roll_again == "y":
        print("Rolling the dice...")
        print(random.randint(min, max))
        roll_again = input("Roll the dices again?")


from cryptography.fernet import Fernet


# write a python program to Encrypt and Decrypt features within 'Secure' class with key generation, using cryptography module
class Secure:
    def __init__(self):
        """
           Generates a key and save it into a file
        """
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)

    @staticmethod
    def load_key():
        """
        Load the previously generated key
        """
        return open("secret.key", "rb").read()

    def encrypt_message(self, message):
        """
        Encrypts a message
        """
        key = self.load_key()
        encoded_message = message.encode()
        f = Fernet(key)
        encrypted_message = f.encrypt(encoded_message)
        print("\nMessage has been encrypted: ", encrypted_message)
        return encrypted_message

    def decrypt_message(self, encrypted_message):
        """
        Decrypts an encrypted message
        """
        key = self.load_key()
        f = Fernet(key)
        decrypted_message = f.decrypt(encrypted_message)
        print("\nDecrypted message:", decrypted_message.decode())


s = Secure()
encrypted = s.encrypt_message("My deepest secret!")
s.decrypt_message(encrypted)


# write a python function to generate SHA256 for given text
def get_sha256(text):
    import hashlib
    return hashlib.sha256(text).hexdigest()


# write a python function to check if SHA256 hashed value is valid for given data or not
def check_sha256_hash(hashed, data):
    import hashlib
    return True if hashed == hashlib.sha256(data.encode()).hexdigest() else False


# write a python function to get HTML code for a given URL
def get_html(url="http://www.python.org"):
    import urllib.request

    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    print(mystr)


# write a python function to get Bitcoin prices after every given 'interval' seconds
def get_btc_price(interval=5):
    import requests
    import json
    from time import sleep

    def getBitcoinPrice():
        URL = "https://www.bitstamp.net/api/ticker/"
        try:
            r = requests.get(URL)
            priceFloat = float(json.loads(r.text)["last"])
            return priceFloat
        except requests.ConnectionError:
            print("Error querying Bitstamp API")

    while True:
        print("Bitstamp last price: US $ " + str(getBitcoinPrice()) + "/BTC")
        sleep(interval)


# write a python function to get stock prices for a company from 2015 to 2020-12
def get_stock_prices(tickerSymbol='TSLA'):
    import yfinance as yf

    # get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2015-1-1', end='2020-12-20')

    # see your data
    print(tickerDf)


# write a python function to get 10 best Artists playing on Apple iTunes
def get_artists():
    import requests
    url = 'https://itunes.apple.com/us/rss/topsongs/limit=10/json'
    response = requests.get(url)
    data = response.json()
    for artist_dict in data['feed']['entry']:
        artist_name = artist_dict['im:artist']['label']
        print(artist_name)


# write a python function to get prominent words from user test corpus using TFIDF vectorizer
def get_words(corpus, new_doc, top=2):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words='english')
    if not corpus:
        corpus = [
            'I would like to check this document',
            'How about one more document',
            'Aim is to capture the key words from the corpus',
            'frequency of words in a document is called term frequency'
        ]

    X = tfidf.fit_transform(corpus)
    feature_names = np.array(tfidf.get_feature_names())

    if not new_doc:
        new_doc = ['can key words in this new document be identified?',
                   'idf is the inverse document frequency calculated for each of the words']
    responses = tfidf.transform(new_doc)

    def get_top_tf_idf_words(response, top_n=top):
        sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
        return feature_names[response.indices[sorted_nzs]]

    print([get_top_tf_idf_words(response, 2) for response in responses])


# write a python function to generate wordcloud on given text or file
import os


def get_word(data):
    if not (isinstance(data, str) or os.path.isfile(data)):
        raise TypeError("Text must be string or a File object.")
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    stopwords = set(STOPWORDS)
    if os.path.isfile(data):
        with open(data, 'r') as f:
            data = f.read()

    data = ' '.join(data.lower().split(' '))
    wordcloud = WordCloud(width=400, height=400,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=15).generate(data)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


# get_word(data="./christmas_carol.txt")


# write a python function to sort each item in a data structure on one of the keys
def sort_list_with_key():
    animals = [
        {'type': 'lion', 'name': 'Mr. T', 'age': 7},
        {'type': 'tiger', 'name': 'scarface', 'age': 3},
        {'type': 'puma', 'name': 'Joe', 'age': 4}]
    print(sorted(animals, key=lambda animal: -animal['age']))


# write a python function with generator for an infinite sequence
def infinite_sequence():
    n = 0
    while True:
        yield n
        n += 1


import uuid


# write a python function to generate a Unique identifier across space and time in this cosmos.
def get_uuid():
    return uuid.uuid4()


import secrets


# write a python function to generate cryptographically strong pseudo-random data
def get_cryptographically_secure_data(n=101):
    return secrets.token_bytes(n), secrets.token_hex(n)


# write a python function to convert byte to UTF-8
def byte_to_utf8(data):
    return data.decode("utf-8")
print(byte_to_utf8(data=b'r\xc3\xa9sum\xc3\xa9'))


def hanoi(disks, source, auxiliary, target):
    if disks == 1:
        print('Move disk 1 from peg {} to peg {}.'.format(source, target))
        return
 
    hanoi(disks - 1, source, target, auxiliary)
    print('Move disk {} from peg {} to peg {}.'.format(disks, source, target))
    hanoi(disks - 1, auxiliary, source, target)


# Write a python program to implement a Stack using One Queue

class Stack:
    def __init__(self):
        self.q = Queue()
 
    def is_empty(self):
        return self.q.is_empty()
 
    def push(self, data):
        self.q.enqueue(data)
 
    def pop(self):
        for _ in range(self.q.get_size() - 1):
            dequeued = self.q.dequeue()
            self.q.enqueue(dequeued)
        return self.q.dequeue()
 
 
class Queue:
    def __init__(self):
        self.items = []
        self.size = 0
 
    def is_empty(self):
        return self.items == []
 
    def enqueue(self, data):
        self.size += 1
        self.items.append(data)
 
    def dequeue(self):
        self.size -= 1
        return self.items.pop(0)
 
    def get_size(self):
        return self.size
 
 
s = Stack()
 
print('Menu')
print('push <value>')
print('pop')
print('quit')
 
while True:
    do = input('What would you like to do? ').split()
 
    operation = do[0].strip().lower()
    if operation == 'push':
        s.push(int(do[1]))
    elif operation == 'pop':
        if s.is_empty():
            print('Stack is empty.')
        else:
            print('Popped value: ', s.pop())
    elif operation == 'quit':
        break

# Write a python program to implement Dequeue

class Dequeue:
    def __init__(self):
        self.items = []
 
    def is_empty(self):
        return self.items == []
 
    def append(self, data):
        self.items.append(data)
 
    def append_left(self, data):
        self.items.insert(0, data)
 
    def pop(self):
        return self.items.pop()
 
    def pop_left(self):
        return self.items.pop(0)
 
 
q = Dequeue()
print('Menu')
print('append <value>')
print('appendleft <value>')
print('pop')
print('popleft')
print('quit')
 
while True:
    do = input('What would you like to do? ').split()
 
    operation = do[0].strip().lower()
    if operation == 'append':
        q.append(int(do[1]))
    elif operation == 'appendleft':
        q.append_left(int(do[1]))
    elif operation == 'pop':
        if q.is_empty():
            print('Dequeue is empty.')
        else:
            print('Popped value from right: ', q.pop())
    elif operation == 'popleft':
        if q.is_empty():
            print('Dequeue is empty.')
        else:
            print('Popped value from left: ', q.pop_left())
    elif operation == 'quit':
        break

# Write a python program to Check and print if string is palindrome using Stack

class Stack:
    def __init__(self):
        self.items = []
 
    def is_empty(self):
        return self.items == []
 
    def push(self, data):
        self.items.append(data)
 
    def pop(self):
        return self.items.pop()
 
 
s = Stack()
text = "ABA"
 
for character in text:
    s.push(character)
 
reversed_text = ''
while not s.is_empty():
    reversed_text = reversed_text + s.pop()
 
if text == reversed_text:
    print('The string is a palindrome.')
else:
    print('The string is not a palindrome.')

# Write a python program to Check and print if Expression is Correctly Parenthesized using Stack

class Stack:
    def __init__(self):
        self.items = []
 
    def is_empty(self):
        return self.items == []
 
    def push(self, data):
        self.items.append(data)
 
    def pop(self):
        return self.items.pop()
 
 
s = Stack()
exp = "(x+y"
 
for c in exp:
    if c == '(':
        s.push(1)
    elif c == ')':
        if s.is_empty():
            is_balanced = False
            break
        s.pop()
else:
    if s.is_empty():
        is_balanced = True
    else:
        is_balanced = False
 
if is_balanced:
    print('Expression is correctly parenthesized.')
else:
    print('Expression is not correctly parenthesized.')

# Write a python program to Implement Linear Search and print the key element if found

def linear_search(alist, key):
    """Return index of key in alist. Return -1 if key not present."""
    for i in range(len(alist)):
        if alist[i] == key:
            return i
    return -1
 
 
alist = [2, 3, 5, 6, 4, 5]

key = 6
 
index = linear_search(alist, key)
if index < 0:
    print(f'{key} was not found.')
else:
    print(f'{key} was found at index {index}.')

# Write a python program to Implement Binary Search without Recursion and print the key element if found

def binary_search(alist, key):
    """Search key in alist[start... end - 1]."""
    start = 0
    end = len(alist)
    while start < end:
        mid = (start + end)//2
        if alist[mid] > key:
            end = mid
        elif alist[mid] < key:
            start = mid + 1
        else:
            return mid
    return -1
 
 
alist = [2, 3, 5, 6, 4, 5]

key = 6
 
index = binary_search(alist, key)
if index < 0:
    print(f'{key} was not found.')
else:
    print(f'{key} was found at index {index}.')


# Write a python program to Implement Binary Search with Recursion and print the key element if found

def binary_search_rec(alist, start, end, key):
    """Search key in alist[start... end - 1]."""
    if not start < end:
        return -1
 
    mid = (start + end)//2
    if alist[mid] < key:
        return binary_search_rec(alist, mid + 1, end, key)
    elif alist[mid] > key:
        return binary_search_rec(alist, start, mid, key)
    else:
        return mid
 
 
alist = [2, 3, 5, 6, 4, 5]

key = 6
 
index = binary_search_rec(alist, 0, len(alist), key)
if index < 0:
    print(f'{key} was not found.')
else:
    print(f'{key} was found at index {index}.')


# Write a python program to Implement Bubble sort and print the sorted list for the below list

def bubble_sort(alist):
    for i in range(len(alist) - 1, 0, -1):
        no_swap = True
        for j in range(0, i):
            if alist[j + 1] < alist[j]:
                alist[j], alist[j + 1] = alist[j + 1], alist[j]
                no_swap = False
        if no_swap:
            return
 
 
alist = [2, 3, 5, 6, 4, 5]
bubble_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Selection sort and print the sorted list for the below list

def selection_sort(alist):
    for i in range(0, len(alist) - 1):
        smallest = i
        for j in range(i + 1, len(alist)):
            if alist[j] < alist[smallest]:
                smallest = j
        alist[i], alist[smallest] = alist[smallest], alist[i]
 
 
alist = [2, 3, 5, 6, 4, 5]
selection_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Insertion sort and print the sorted list for the below list

def insertion_sort(alist):
    for i in range(1, len(alist)):
        temp = alist[i]
        j = i - 1
        while (j >= 0 and temp < alist[j]):
            alist[j + 1] = alist[j]
            j = j - 1
        alist[j + 1] = temp
 
 
alist = [2, 3, 5, 6, 4, 5]
insertion_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Merge sort and print the sorted list for the below list

def merge_sort(alist, start, end):
    '''Sorts the list from indexes start to end - 1 inclusive.'''
    if end - start > 1:
        mid = (start + end)//2
        merge_sort(alist, start, mid)
        merge_sort(alist, mid, end)
        merge_list(alist, start, mid, end)
 
def merge_list(alist, start, mid, end):
    left = alist[start:mid]
    right = alist[mid:end]
    k = start
    i = 0
    j = 0
    while (start + i < mid and mid + j < end):
        if (left[i] <= right[j]):
            alist[k] = left[i]
            i = i + 1
        else:
            alist[k] = right[j]
            j = j + 1
        k = k + 1
    if start + i < mid:
        while k < end:
            alist[k] = left[i]
            i = i + 1
            k = k + 1
    else:
        while k < end:
            alist[k] = right[j]
            j = j + 1
            k = k + 1
 
 
alist = [2, 3, 5, 6, 4, 5]
merge_sort(alist, 0, len(alist))
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Quicksort and print the sorted list for the below list

def quicksort(alist, start, end):
    '''Sorts the list from indexes start to end - 1 inclusive.'''
    if end - start > 1:
        p = partition(alist, start, end)
        quicksort(alist, start, p)
        quicksort(alist, p + 1, end)
 
 
def partition(alist, start, end):
    pivot = alist[start]
    i = start + 1
    j = end - 1
 
    while True:
        while (i <= j and alist[i] <= pivot):
            i = i + 1
        while (i <= j and alist[j] >= pivot):
            j = j - 1
 
        if i <= j:
            alist[i], alist[j] = alist[j], alist[i]
        else:
            alist[start], alist[j] = alist[j], alist[start]
            return j
 
 
alist = [2, 3, 5, 6, 4, 5]
quicksort(alist, 0, len(alist))
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Heapsort and print the sorted list for the below list

def heapsort(alist):
    build_max_heap(alist)
    for i in range(len(alist) - 1, 0, -1):
        alist[0], alist[i] = alist[i], alist[0]
        max_heapify(alist, index=0, size=i)
 
def parent(i):
    return (i - 1)//2
 
def left(i):
    return 2*i + 1
 
def right(i):
    return 2*i + 2
 
def build_max_heap(alist):
    length = len(alist)
    start = parent(length - 1)
    while start >= 0:
        max_heapify(alist, index=start, size=length)
        start = start - 1
 
def max_heapify(alist, index, size):
    l = left(index)
    r = right(index)
    if (l < size and alist[l] > alist[index]):
        largest = l
    else:
        largest = index
    if (r < size and alist[r] > alist[largest]):
        largest = r
    if (largest != index):
        alist[largest], alist[index] = alist[index], alist[largest]
        max_heapify(alist, largest, size)
 
 
alist = [2, 3, 5, 6, 4, 5]
heapsort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Counting sort and print the sorted list for the below list

def counting_sort(alist, largest):
    c = [0]*(largest + 1)
    for i in range(len(alist)):
        c[alist[i]] = c[alist[i]] + 1
 
    c[0] = c[0] - 1 
    for i in range(1, largest + 1):
        c[i] = c[i] + c[i - 1]
 
    result = [None]*len(alist)
 
    for x in reversed(alist):
        result[c[x]] = x
        c[x] = c[x] - 1
 
    return result
 
 
alist = [2, 3, 5, 6, 4, 5]
k = max(alist)
sorted_list = counting_sort(alist, k)
print('Sorted list: ', end='')
print(sorted_list)

# Write a python program to Implement Radix sort and print the sorted list for the below list

def radix_sort(alist, base=10):
    if alist == []:
        return
 
    def key_factory(digit, base):
        def key(alist, index):
            return ((alist[index]//(base**digit)) % base)
        return key
    largest = max(alist)
    exp = 0
    while base**exp <= largest:
        alist = counting_sort(alist, base - 1, key_factory(exp, base))
        exp = exp + 1
    return alist
 
def counting_sort(alist, largest, key):
    c = [0]*(largest + 1)
    for i in range(len(alist)):
        c[key(alist, i)] = c[key(alist, i)] + 1
 
    c[0] = c[0] - 1
    for i in range(1, largest + 1):
        c[i] = c[i] + c[i - 1]
 
    result = [None]*len(alist)
    for i in range(len(alist) - 1, -1, -1):
        result[c[key(alist, i)]] = alist[i]
        c[key(alist, i)] = c[key(alist, i)] - 1
 
    return result
 
alist = [2, 3, 5, 6, 4, 5]
sorted_list = radix_sort(alist)
print('Sorted list: ', end='')
print(sorted_list)

# Write a python program to Implement Bucket sort and print the sorted list for the below list

def bucket_sort(alist):
    largest = max(alist)
    length = len(alist)
    size = largest/length
 
    buckets = [[] for _ in range(length)]
    for i in range(length):
        j = int(alist[i]/size)
        if j != length:
            buckets[j].append(alist[i])
        else:
            buckets[length - 1].append(alist[i])
 
    for i in range(length):
        insertion_sort(buckets[i])
 
    result = []
    for i in range(length):
        result = result + buckets[i]
 
    return result
 
def insertion_sort(alist):
    for i in range(1, len(alist)):
        temp = alist[i]
        j = i - 1
        while (j >= 0 and temp < alist[j]):
            alist[j + 1] = alist[j]
            j = j - 1
        alist[j + 1] = temp
 
 
alist = [2, 3, 5, 6, 4, 5]
sorted_list = bucket_sort(alist)
print('Sorted list: ', end='')
print(sorted_list)

# Write a python program to Implement Gnome sort and print the sorted list for the below list

def gnome_sort(alist):
    for pos in range(1, len(alist)):
        while (pos != 0 and alist[pos] < alist[pos - 1]):
            alist[pos], alist[pos - 1] = alist[pos - 1], alist[pos]
            pos = pos - 1
 
 
alist = [2, 3, 5, 6, 4, 5]
gnome_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Cocktail Shaker sort and print the sorted list for the below list

def cocktail_shaker_sort(alist):
    def swap(i, j):
        alist[i], alist[j] = alist[j], alist[i]
 
    upper = len(alist) - 1
    lower = 0
 
    no_swap = False
    while (not no_swap and upper - lower > 1):
        no_swap = True
        for j in range(lower, upper):
            if alist[j + 1] < alist[j]:
                swap(j + 1, j)
                no_swap = False
        upper = upper - 1
 
        for j in range(upper, lower, -1):
            if alist[j - 1] > alist[j]:
                swap(j - 1, j)
                no_swap = False
        lower = lower + 1
 
 
alist = [2, 3, 5, 6, 4, 5]
cocktail_shaker_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Comb sort and print the sorted list for the below list

def comb_sort(alist):
    def swap(i, j):
        alist[i], alist[j] = alist[j], alist[i]
 
    gap = len(alist)
    shrink = 1.3
 
    no_swap = False
    while not no_swap:
        gap = int(gap/shrink)
 
        if gap < 1:
            gap = 1
            no_swap = True
        else:
            no_swap = False
 
        i = 0
        while i + gap < len(alist):
            if alist[i] > alist[i + gap]:
                swap(i, i + gap)
                no_swap = False
            i = i + 1
 
 
alist = [2, 3, 5, 6, 4, 5]
comb_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python program to Implement Shell sort and print the sorted list for the below list

def gaps(size):
    length = size.bit_length()
    for k in range(length - 1, 0, -1):
        yield 2**k - 1
 
 
def shell_sort(alist):
    def insertion_sort_with_gap(gap):
        for i in range(gap, len(alist)):
            temp = alist[i]
            j = i - gap
            while (j >= 0 and temp < alist[j]):
                alist[j + gap] = alist[j]
                j = j - gap
            alist[j + gap] = temp
 
    for g in gaps(len(alist)):
        insertion_sort_with_gap(g)
 
 
alist = [2, 3, 5, 6, 4, 5]
shell_sort(alist)
print('Sorted list: ', end='')
print(alist)

# Write a python Class to calculate area of a rectangle and print the area

class rectangle():
    def __init__(self,breadth,length):
        self.breadth=breadth
        self.length=length
    def area(self):
        return self.breadth*self.length
a=6
b=4
obj=rectangle(a,b)
print("Area of rectangle:",obj.area())

# Write a python Class to calculate area of a circle and print the vale for a radius

class CircleArea():
    def __init__(self,radius):
        self.radius=radius
    def area(self):
        return 3.14 * self.radius * self.radius
a=6
obj=CircleArea(a)
print("Area of rectangle:",obj.area())

# Write a python Class to calculate Perimeter of a circle and print the vale for a radius

class CirclePerimeter():
    def __init__(self,radius):
        self.radius=radius
    def perimeter(self):
        return 2 * 3.14 * self.radius
a=6
obj=CirclePerimeter(a)
print("Perimeter of rectangle:",obj.perimeter())

# Write a python Class to print All Possible Subsets from a Set of Distinct Integers

class sub:  
    def f1(self, s1):  
        return self.f2([], sorted(s1))  
 
    def f2(self, curr, s1):  
        if s1:  
            return self.f2(curr, s1[1:]) + self.f2(curr + [s1[0]], s1[1:])  
        return [curr]  
a=[2, 3, 5, 6, 4, 5]

print("Subsets: ")
print(sub().f1(a))

# Write a python program to Read and print the Contents of a File

a=str(input("Enter file name .txt extension:"))
file2=open(a,'r')
line=file2.readline()
while(line!=""):
    print(line)
    line=file2.readline()
file2.close()

# Write a python program to Count and print the Number of Words in a Text File

fname = input("Enter file name: ")
 
num_words = 0
 
with open(fname, 'r') as f:
    for line in f:
        words = line.split()
        num_words += len(words)
print("Number of words:")
print(num_words)

# Write a python program to Count the Number of Lines in a Text File

fname = input("Enter file name: ")
num_lines = 0
with open(fname, 'r') as f:
    for line in f:
        num_lines += 1
print("Number of lines:")
print(num_lines)

# Write a python program to Count the Occurrences of a Word in a Text File

fname = input("Enter file name: ")
word='the'
k = 0
 
with open(fname, 'r') as f:
    for line in f:
        words = line.split()
        for i in words:
            if(i==word):
                k=k+1
print(f"Frequency of Occurrences of the word {a} is:")
print(k)

# Write a python function to Copy the Contents of One File into Another

def copy(from_file, to_file):
    with open(from_file) as f:
        with open(to_file, "w") as f1:
            for line in f:
                f1.write(line)

# Write a python function that Counts the Number of Times a Certain Letter Appears in the Text File
def count_letter(fname, l):
    k = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            for i in words:
                for letter in i:
                    if(letter==l):
                        k=k+1
    return k

# Write a python function that Print all the Numbers Present in the Text File

def print_number(fname):
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            for i in words:
                for letter in i:
                    if(letter.isdigit()):
                        print(letter)


# Write a python function that Counts the Number of Blank Spaces in a Text File

def count_blank_space(fname):
    k = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            for i in words:
                for letter in i:
                    if(letter.isspace):
                        k=k+1
    return k

# Write a python function that Capitalize the First Letter of Every Word in the File

def capitalize(fname):
    with open(fname, 'r') as f:
        for line in f:
            l=line.title()
            print(l)

# Write a python function that prints the Contents of a File in Reverse Order

def reverse_content(filename):
    for line in reversed(list(open(filename))):
        print(line.rstrip())

# Write a python Program to Flatten and print a List

a=[[1,[[2]],[[[3]]]],[[4],5]]
flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]
print(flatten(a))

# Write a Python Program to print the LCM of Two Numbers

def lcm(a,b):
    lcm.multiple=lcm.multiple+b
    if((lcm.multiple % a == 0) and (lcm.multiple % b == 0)):
        return lcm.multiple
    else:
        lcm(a, b)
    return lcm.multiple
lcm.multiple=0
a=4
b=7
if(a>b):
    LCM=lcm(b,a)
else:
    LCM=lcm(a,b)

print(LCM)

# Write a Python function to print the GSD of Two Numbers

def gcd(a,b):
    if(b==0):
        return a
    else:
        return gcd(b,a%b)


# Write a Python function to Find if a Number is Prime or Not Prime

def check(n, div = None):
    if div is None:
        div = n - 1
    while div >= 2:
        if n % div == 0:
            print("Number not prime")
            return False
        else:
            return check(n, div-1)
    else:
        print("Number is prime")
        return 'True'

# Write a Python function to Find the Power of a Number Using Recursion

def power(base,exp):
    if(exp==1):
        return(base)
    if(exp!=1):
        return(base*power(base,exp-1))

# Write a Python function to Find the Total Sum of a Nested List Using Recursion

def sum1(lst):
    total = 0
    for element in lst:
        if (type(element) == type([])):
            total = total + sum1(element)
        else:
            total = total + element
    return total

# Write a Python function to Count and print the Number of Vowels Present in a String using Sets

def count_vowels(s):
    count = 0
    vowels = set("aeiou")
    for letter in s:
        if letter in vowels:
            count += 1
    return count

# Write a Python Program to prints Common Letters in Two Input Strings

s1='python'
s2='schoolofai'
a=list(set(s1)&set(s2))
print("The common letters are:")
for i in a:
    print(i)

# Write a Python Program that Prints which Letters are in the First String but not in the Second

s1='python'
s2='schoolofai'
a=list(set(s1)-set(s2))
print("The letters are:")
for i in a:
    print(i)

# Write a Python Program to Concatenate Two Dictionaries Into One

def concat_dic(d1, d2):
    return d1.update(d2)

# Write a Python Program to Multiply All the Items in a Dictionary

def mul_dict(d):
    tot=1
    for i in d:    
        tot=tot*d[i]
    return tot

# Write a Python Program to Remove the Given Key from a Dictionary

def remove_item_dict(d, key):
    if key in d: 
        del d[key]
    else:
        print("Key not found!")
        exit(0)

# Write a Python Program to Map Two Lists into a Dictionary

def map_dict(keys, values):
    return dict(zip(keys,values))

# Write a Python Program to Remove the nth Index Character from a Non-Empty String

def remove(string, n):  
      first = string[:n]   
      last = string[n+1:]  
      return first + last

# Write a Python Program to Detect if Two Strings are Anagrams

def anagram_check(s1, s2):
    if(sorted(s1)==sorted(s2)):
        return True
    else:
        return False

# Write a Python Program to Form a New String where the First Character and the Last Character have been Exchanged

def change(string):
      return string[-1:] + string[1:-1] + string[:1]

# Write a Python Program to Remove the Characters of Odd Index Values in a String

def modify(string):  
    final = ""   
    for i in range(len(string)):  
        if i % 2 == 0:  
            final = final + string[i]  
    return final

# Write a Python Program to Take in Two Strings and Print the Larger String

string1='python'
string2='theschoolofai'
count1=0
count2=0
for i in string1:
      count1=count1+1
for j in string2:
      count2=count2+1
if(count1<count2):
      print("Larger string is:")
      print(string2)
elif(count1==count2):
      print("Both strings are equal.")
else:
      print("Larger string is:")
      print(string1)

# Write a Python Program to Count and print Number of Lowercase Characters in a String

string='This is an Assignment'
count=0
for i in string:
      if(i.islower()):
            count=count+1
print("The number of lowercase characters is:")
print(count)

# Write a Python Program to Put Even and Odd elements in a List into Two Different Lists

a=[2, 3, 8, 9, 2, 4, 6]
even=[]
odd=[]
for j in a:
    if(j%2==0):
        even.append(j)
    else:
        odd.append(j)
print("The even list",even)
print("The odd list",odd)

# Write a Python Program to Sort the List According to the Second Element in Sublist

a=[['A',34],['B',21],['C',26]]
for i in range(0,len(a)):
    for j in range(0,len(a)-i-1):
        if(a[j][1]>a[j+1][1]):
            temp=a[j]
            a[j]=a[j+1]
            a[j+1]=temp


# Write a Python Program to Find the Second Largest Number in a List Using Bubble Sort

a=[2, 3, 8, 9, 2, 4, 6]
for i in range(0,len(a)):
    for j in range(0,len(a)-i-1):
        if(a[j]>a[j+1]):
            temp=a[j]
            a[j]=a[j+1]
            a[j+1]=temp 

# Write a Python Program to Find the Intersection of Two Lists

def main(alist, blist):
    def intersection(a, b):
        return list(set(a) & set(b))
    return intersection(alist, blist)

# Write a Python Program to Create a List of Tuples with the First Element as the Number and Second Element as the Square of the Number using list comprehension

l_range=2
u_range=5
a=[(x,x**2) for x in range(l_range,u_range+1)]


# Write a Python Program to print all Numbers in a Range which are Perfect Squares and Sum of all Digits in the Number is Less than 10

l=6
u=9
a=[x for x in range(l,u+1) if (int(x**0.5))**2==x and sum(list(map(int,str(x))))<10]
print(a)

# Write a Python Program to Swap the First and Last Value of a List

a=[2, 3, 8, 9, 2, 4, 6]
n = len(a)
temp=a[0]
a[0]=a[n-1]
a[n-1]=temp
print("New list is:")
print(a)

# Write a Python Program to Remove and print the Duplicate Items from a List

a=[2, 3, 8, 9, 2, 4, 6]
b = set()
unique = []
for x in a:
    if x not in b:
        unique.append(x)
        b.add(x)
print("Non-duplicate items:")
print(unique)

# Write a Python Program to Read a List of Words and Return the Length of the Longest One

a=['the', 'tsai', 'python']
max1=len(a[0])
temp=a[0]
for i in a:
    if(len(i)>max1):
       max1=len(i)
       temp=i
print("The word with the longest length is:")
print(temp)

# Write a Python Program to Remove the ith Occurrence of the Given Word in a List where Words can Repeat

a=['the', 'tsai', 'python' ,'a' ,'the', 'a']
c=[]
count=0
b='a'
n=3
for i in a:
    if(i==b):
        count=count+1
        if(count!=n):
            c.append(i)
    else:
        c.append(i)
if(count==0):
    print("Item not found ")
else: 
    print("The number of repetitions is: ",count)
    print("Updated list is: ",c)
    print("The distinct elements are: ",set(a))


# Write a Python function to Find Element Occurring Odd Number of Times in a List

def find_odd_occurring(alist):
    """Return the element that occurs odd number of times in alist.
 
    alist is a list in which all elements except one element occurs an even
    number of times.
    """
    ans = 0
 
    for element in alist:
        ans ^= element
 
    return ans

# Write a Python Program to Check if a Date is Valid and Print the Incremented Date if it is

date="20/04/2021"
dd,mm,yy=date.split('/')
dd=int(dd)
mm=int(mm)
yy=int(yy)
if(mm==1 or mm==3 or mm==5 or mm==7 or mm==8 or mm==10 or mm==12):
    max1=31
elif(mm==4 or mm==6 or mm==9 or mm==11):
    max1=30
elif(yy%4==0 and yy%100!=0 or yy%400==0):
    max1=29
else:
    max1=28
if(mm<1 or mm>12):
    print("Date is invalid.")
elif(dd<1 or dd>max1):
    print("Date is invalid.")
elif(dd==max1 and mm!=12):
    dd=1
    mm=mm+1
    print("The incremented date is: ",dd,mm,yy)
elif(dd==31 and mm==12):
    dd=1
    mm=1
    yy=yy+1
    print("The incremented date is: ",dd,mm,yy)
else:
    dd=dd+1
    print("The incremented date is: ",dd,mm,yy)


# Write a Python function to Check Whether a Given Year is a Leap Year

def leapyear_check(year):
    if(year%4==0 and year%100!=0 or year%400==0):
        return True
    else:
        return False

# Write a Python Program to print Prime Factors of an Integer

n=24
print("Factors are:")
i=1
while(i<=n):
    k=0
    if(n%i==0):
        j=1
        while(j<=i):
            if(i%j==0):
                k=k+1
            j=j+1
        if(k==2):
            print(i)
    i=i+1


# Write a Python Program to print all the Divisors of an Integer

n=60
print("The divisors of the number are:")
for i in range(1,n+1):
    if(n%i==0):
        print(i)


# Write a Python Program to Check if a Number is an Armstrong Number

def amstrong_check(n):
    a=list(map(int,str(n)))
    b=list(map(lambda x:x**3,a))
    if(sum(b)==n):
        return True
    else:
        return False

# Write a Python Program to Print the Pascal’s triangle for n number of rows given by the user

n=10
a=[]
for i in range(n):
    a.append([])
    a[i].append(1)
    for j in range(1,i):
        a[i].append(a[i-1][j-1]+a[i-1][j])
    if(n!=0):
        a[i].append(1)
for i in range(n):
    print("   "*(n-i),end=" ",sep=" ")
    for j in range(0,i+1):
        print('{0:6}'.format(a[i][j]),end=" ",sep=" ")
    print()

# Write a Python Program to Check if a Number is a Perfect Number

def perfect_no_check(n):
    sum1 = 0
    for i in range(1, n):
        if(n % i == 0):
            sum1 = sum1 + i
    if (sum1 == n):
        return True
    else:
        return False

# Write a Python Program to Check if a Number is a Strong Number

def strong_no_check(num):
    sum1=0
    temp=num
    while(num):
        i=1
        f=1
        r=num%10
        while(i<=r):
            f=f*i
            i=i+1
        sum1=sum1+f
        num=num//10
    if(sum1==temp):
        return True
    else:
        return False

# Write a Python Program to Check If Two Numbers are Amicable Numbers

def amicable_no_check(x, y):
    sum1=0
    sum2=0
    for i in range(1,x):
        if x%i==0:
            sum1+=i
    for j in range(1,y):
        if y%j==0:
            sum2+=j
    if(sum1==y and sum2==x):
        return True
    else:
        return False

# Write a Python Program to Check if a Number is a Prime Number

def prime_no_check(a):
    k=0
    for i in range(2,a//2+1):
        if(a%i==0):
            k=k+1
    if(k<=0):
        return True
    else:
        return False

# Write a Python Program to print the Sum of First N Natural Numbers

n=7
sum1 = 0
while(n > 0):
    sum1=sum1+n
    n=n-1
print("The sum of first n natural numbers is",sum1)

# Write a Python Program to Print all Pythagorean Triplets in the Range

limit=10
c=0
m=2
while(c<limit):
    for n in range(1,m+1):
        a=m*m-n*n
        b=2*m*n
        c=m*m+n*n
        if(c>limit):
            break
        if(a==0 or b==0 or c==0):
            break
        print(a,b,c)
    m=m+1

# Write a Python Program to print the Number of Times a Particular Number Occurs in a List

a=[2, 3, 8, 9, 2, 4, 6]
k=0
num=int(input("Enter the number to be counted:"))
for j in a:
    if(j==num):
        k=k+1
print("Number of times",num,"appears is",k)

# Write a Python Program to test and print Collatz Conjecture for a Given Number

def collatz(n):
    while n > 1:
        print(n, end=' ')
        if (n % 2):
            # n is odd
            n = 3*n + 1
        else:
            # n is even
            n = n//2
    print(1, end='')

# Write a Python function to Count Set Bits in a Number

def count_set_bits(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# Write a Python Program to Generate Gray Codes using Recursion

def get_gray_codes(n):
    """Return n-bit Gray code in a list."""
    if n == 0:
        return ['']
    first_half = get_gray_codes(n - 1)
    second_half = first_half.copy()
 
    first_half = ['0' + code for code in first_half]
    second_half = ['1' + code for code in reversed(second_half)]
 
    return first_half + second_half

# Write a Python Program to Convert Gray Code to Binary

def gray_to_binary(n):
    """Convert Gray codeword to binary and return it."""
    n = int(n, 2)
 
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask

    return bin(n)[2:]

# Write a Python Program to Convert Binary to Gray Code

def binary_to_gray(n):
    """Convert Binary to Gray codeword and return it."""
    n = int(n, 2)
    n ^= (n >> 1)

    return bin(n)[2:]

# Write a Python Program to print the Reverse a Given Number

n=1023
rev=0
while(n>0):
    dig=n%10
    rev=rev*10+dig
    n=n//10
print("Reverse of the number:",rev)

# Write a Python Program to Accept Three Digits and Print all Possible Combinations from the Digits

a=2
b=9
c=5
d=[]
d.append(a)
d.append(b)
d.append(c)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            if(i!=j&j!=k&k!=i):
                print(d[i],d[j],d[k])

# Write a Python function to Print an Identity Matrix

def print_identity_matrix(n):
    for i in range(0,n):
        for j in range(0,n):
            if(i==j):
                print("1",sep=" ",end=" ")
            else:
                print("0",sep=" ",end=" ")
        print()

# Write a Python Program Print Restaurant Menu using Class given menu and cost as list

class Food(object):
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def getprice(self):
        return self.price
    
    def __str__(self):
        return self.name + ' : ' + str(self.getprice())
    
def buildmenu(names, costs):
    menu = []
    for i in range(len(names)):
        menu.append(Food(names[i], costs[i]))
    return menu

names = ['Coffee', 'Tea', 'Pizza', 'Burger', 'Fries', 'Apple', 'Donut', 'Cake']

costs = [250, 150, 180, 70, 65, 55, 120, 350]

Foods = buildmenu(names, costs)

n = 1
for el in Foods:
    print(n,'. ', el)
    n = n + 1

# Write a Python Program to print a list of fibonacci series for a given no using closer

def fib():
    cache = {1:1, 2:1}

    def calc_fib(n):
        if n not in cache:
            print(f'Calculating fib({n})')
            cache[n] = calc_fib(n - 1) + calc_fib(n - 2)
        return cache[n]
    return calc_fib

# Write a Python Program to print a list of fibonacci series for a given no using class

class Fib:
    def __init__(self):
        self.cache = {1:1, 2:1}

    def fib(self, n):
        if n not in self.cache:
            print(f'Calculating fib({n})')
            self.cache[n] = self.fib(n-1) + self.fib(n-2)
        return self.cache[n]

# Write a Python function to calculate factorial of a given no using closer

def fact():
    cache = {0:1, 1:1}

    def calc_fib(n):
        if n not in cache:
            print(f'Calculating fact({n})')
            cache[n] = calc_fib(n - 1) * n
        return cache[n]
    return calc_fib

# Write a Python function to calculate factorial of a given no using class

class Fact:
    def __init__(self):
        self.cache = {0:1, 1:1}

    def fact(self, n):
        if n not in self.cache:
            self.cache[n] = self.fact(n-1) * n
        return self.cache[n]

# Write a Python function to calculate dot product of two given sequence

def dot_product(a, b):
    return sum( e[0]*e[1] for e in zip(a, b))

# Write a Python function to Find the Sum of Sine Series

import math
def sin(x,n):
    sine = 0
    for i in range(n):
        sign = (-1)**i
        pi=22/7
        y=x*(pi/180)
        sine = sine + ((y**(2.0*i+1))/math.factorial(2*i+1))*sign
    return sine

# Write a Python function to Find the Sum of Cosine Series

def cosine(x,n):
    cosx = 1
    sign = -1
    for i in range(2, n, 2):
        pi=22/7
        y=x*(pi/180)
        cosx = cosx + (sign*(y**i))/math.factorial(i)
        sign = -sign
    return cosx

# Write a Python function to strip vowels from a string

def vowel_stripping(string):
    '''This function takes a string as an input strips out vowels and returns stripted out string'''
    return "".join([x for x in string if x not in('a','e','i','o','u')])

# Write a Python function that shifts the character of strings

def char_shift(string, shift_count):
    '''This function takes a string as an input and shifts each character by 5 and returns shifted string'''
    return "".join([chr(ord(x)+shift_count) if (ord(x)+shift_count) <= 122 else chr(96 + (ord(x)+shift_count) - 122) for x in string])

# Write a Python function that returns biggest character in a string

from functools import reduce
def biggest_char(string):
    '''This function takes an input as a string and returns the biggest output character in the string'''
    biggest_chr = lambda x, y: x if ord(x) > ord(y) else y
    return reduce(biggest_chr, string)

# Write a Python function that calculate interior angle of a equilateral polygon

def interior_angle(no_of_sides):
    return (no_of_sides - 2) * 180 / no_of_sides

# Write a Python function that calculate side length of a equilateral polygon

import math
def side_length(no_of_sides, circumradius):
    return 2 * circumradius * math.sin(math.pi / no_of_sides)

# Write a Python function that calculate area of a equilateral polygon

import math
def area(no_of_sides, circumradius):
    side_length = 2 * circumradius * math.sin(math.pi / no_of_sides)
    apothem = circumradius * math.cos(math.pi / no_of_sides)
    return no_of_sides / 2 * side_length * apothem