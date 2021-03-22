END_Phase_1_CapstoneProject.ipynb is the single file containing all the steps taken starting from Data Cleansing to Model Training and Evaluation.

The objective of this Capstone project was to generate a fully executable Python Code from English text.

# Data Cleansing

About the Dataset:
 - Firstly, we leveraged the public dataset available at this [link](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view) which contained some 4600+ examples
 -  The dataset is a huge file with English sentences at the beginning of a program in a comment style and the Python code followed that.
 -  The dataset was read line by line with an intention to generate a key-value pair between English sentences and the corresponding Python code blocks.
 - However, given the fact that even the underlying Python codes had comments, made it difficult to separate the English sentences from the genuine comments in the code.

# Data Preparation
 - Given the aforestated challenges with the dataset, the data cleansing was one of the toughest parts of this exercise.
 - We leveraged RegEx heavily to remove unwanted data/comments. 
	 - Driver Code Comments: "^\s*#\s+driver"
	 - Unwanted empty lines: "^\s*\n+\s*$"
	 - Certain comments with the pattern In[12]: "^\s*#\s+in\[[0-9]+\]"
	 - Certain comments with just some numbers eg. #23 : "^\s*#+\s*[0-9]*\s*\n"
- Post this, we removed some other unwanted data maually for primarily two reasons: a) the dataset size was manageable for manual intervention, b) didn't want to over-engineer for less frequent patterns
- Finally, we leverage Spacy for Tokenization


# Model Architecture and Salient features

We used a Transformer based model architecture with multi-headed attention which is covered very elaborately [here](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb).

Below is the overall network architecture

![](https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/9479fcb532214ad26fd4bda9fcf081a05e1aaf4e/assets/transformer1.png)

A distinct feature of this was the usage of "Multi-Head Attention Layer" which is represented below:
![](https://github.com/bentrevett/pytorch-seq2seq/raw/49df8404d938a6edbf729876405558cc2c2b3013/assets/transformer-attention.png)

# Loss function

We used Cross Entropy loss function as we've been able to achieve great results using the same in the past. 
However, we wish to revisit this project using other loss functions in future.

# Python Code embedding strategy

- We used Gensim Word2Vec model to generated the 256 dimensional word embeddings for the Python code syntaxes and keywords

# Evaluation Metrics

We used Perplexity as a measure of model's performance and we able to achieve a training perplexity of 1.4 within 75 epochs. Each epoch took about 3 secs on Colab GPU.

# Moment of Truth: Model auto-generates Python Code Snippets for 25 randomly asked questions
#1
Input:
Write a python program that checks if a string is a pallindrome

Output:
```
def is_palindrome(st):
    st = st.lower()
    rev_st = st[::-1]
    try:
        assert rev_st == st
        return true
    except assertionerror:
        return false

st = "nitin"
print(is_palindrome(st))
```
#2
Input:   
Write a python function to generate random number between 2 integers

Output:
```
def random_number(a, b):
    import random
    return random.randint(a, b)
```

#3
Input:
Write a python class to print all possible subsets from a set of distinct integers

Output:
```
class sub:
    def f1(self, s1):
        return self.f2([], sorted(s1))
    def f2(self, curr, s1):
        if s1:
            return self.f2(curr, s1[1:]) + self.f2(curr + [s1[0]], s1[1:])
        return [curr]
a=[2, 3, 5, 6, 4, 5]
print("subsets: ")
print(sub().f1(a))
```

#4
Input:
Write a python function to get prominent words from user test corpus using tfidf vectorizer

Output:
```
def get_words(corpus, new_doc, top=2):
    import numpy as np
    from sklearn.feature_extraction.text import tfidfvectorizer
    tfidf = tfidfvectorizer(stop_words='english')
    if not corpus:
        corpus = [
            'i would like to check this document',
            'how about one more document',
            'aim is to capture the key words from the corpus',
            'frequency of words in a document is called term frequency'
        ]
    x = tfidf.fit_transform(corpus)
    feature_names = np.array(tfidf.get_feature_names())
    if not new_doc:
        new_doc = ['can key words in this new document be identified?',
                   'idf is the inverse document frequency calculated for each of the words']
    responses = tfidf.transform(new_doc)
    def get_top_tf_idf_words(response, top_n=top):
        sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
        return feature_names[response.indices[sorted_nzs]]
    print([get_top_tf_idf_words(response, 2) for response in responses])
```
#5
Input:
Write a python function to count number of lists in a list of lists

Output:
```
def countlist(lst):
    count = 0
    for el in lst:
        if type(el)== type([]):
            count+= 1
    return count
```
#6
Input:
Write a python program to prints common letters in two input strings

Output:
```
s1='python'
s2='schoolofai'
a=list(set(s1)&set(s2))
print("the common letters are:")
for i in a:
    print(i)
```
#7
Input:
42 write a function to subtract two matrices in python

Output:
```
matrix1 = [[0, 1, 2],
           [3, 5, 5],
           [6, 7, 8]]
matrix2 = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]
def subtractthematrix(matrix1, matrix2):
    matrix1rows = len(matrix1)
    matrix2rows = len(matrix2)
    matrix1col = len(matrix1[0])
    matrix2col = len(matrix2[0])
```    
#8
Input:  
Write a function to calculate the kinetic energy of an object of mass m and velocity v

Output:
```
def cal_ke(mass:float,velocity:float)->float: 
	return (mass*(velocity)**2)/2
```
#9
Input:
Write a python function to return minimum sum of factors of a number

Output:
```
def findminsum(num):
    sum = 0
    i = 2
    while(i * i <= num):
        while(num % i == 0):
            sum += i
            num /= i
        i += 1
    sum += num
    return sum
```

#10
Input:  
Write a function to compress a given string . suppose a character ' c ' occurs consecutively x times in the string . replace these consecutive occurrences of the character ' c ' with   ( x , c ) in the string .

Output:
```
def compress(text):
    from itertools import groupby
    for k, g in groupby(text):
        print("({}, {})".format(len(list(g)), k), end=" ")
```

#11
Input:
Write a boolean python function to check if a given string matches a given pattern

Output:
```
import re
def match(pattern, string):
    if re.match(pattern, string):
        return true
    return false
```

#12
Input:
Write a python function to remove falsy values from a list

Output:
```
def newlist(lst):
  return list(filter(none, lst))
```
#13
Input:
Write a function to return the time taken by a given of moving object based of distance travelled in given time

Output:
```
def cal_time(distance:float,speed:float)->float:
    return distance/speed
```
#14
Input:
Write a python program to convert binary to gray code

Output:
```
def gray_to_binary(n):
    """convert gray codeword to binary and return it."""
    n = int(n, 2)
    n ^= (n >> 1)
    return bin(n)[2:]
        mask >>= 1
        n ^= mask
    return bin(n)[2:]
```
#15
Input:
Write a function to calculate and return electricity bill . units used are given . price per unit is fixed and is increased after 750 units

Output:
```
def calc_elect_bill(units):
    if units > 0:
        if units <= 750:
            return 5*units
        else:
            return 5*(750) + 7*(units-750)
    else:
        return -1
```

#16
Input: 
Write python program to create a dictionary with key as first character and value as words starting with that character

Output:
```
string_input = '''geeksforgeeks is a computer science portal for geeks.
    it contains well written, well thought and well explained
    computer science and programming articles, quizzes etc.'''
words = string_input.split()
dictionary = {}
for word in words:
    if (word[0].lower() not in dictionary.keys()):
        dictionary[word[0].lower()] = []
        dictionary[word[0].lower()].append(word)
    else:
        if (word not in dictionary[word[0].lower()]):
            dictionary[word[0].lower()].append(word)
print(dictionary)
```
#17
Input:
Write a python program to delete an element from a list.

Output:
```
list = ['a', 'bc', 'd', 'e']
element = 'bc'
list.remove(element)
```

#18
Input:
Write a Python function to return woodall numbers

Output:
```
def woodall_number(n):
    if n >= 0:
        return n * 2 ** n - 1
```

#19
Input:
Write a python program that takes height in centimeters as user input and return height in feet and inches

Output:
```
cm=int(input("enter the height in centimeters:"))
inches=0.394*cm
feet=0.0328*cm
print("the length in inches",round(inches,2))
print("the length in feet",round(feet,2))
```
#20
Input:
Write a python function that capitalize the first letter of every word in the file

Output:
```
def capitalize(fname):
    with open(fname, 'r') as f:
        for line in f:
            l=line.title()
            print(l)
```

#21
Input:
Write a function program to reverse the digits of an integer

Output:
```
def reverse_integer(x):
        sign = -1 if x < 0 else 1
        x *= sign       
```
#22
Input:
Write a python program to typecast an integer to string and print it

Output:
```
x = 2
y = str(x)
print(y)
```

#23
Input:
Write a python program to print the following floating numbers with no decimal places

Output:
```
x = 3.1415926
y = -12.9999
print("\noriginal number: ", x)
print("formatted number with no decimal places: "+"{:.0f}".format(x))
print("original number: ", y)
print("formatted number with no decimal places: "+"{:.0f}".format(y))
```

#24
Input:
Write a Python function to return star numbers

Output:
```
def star_number(n):
    return 6*n*(n-1)+1
```
#25
Input:
Frequency of words in text

Output:
```
line = input()
for word in line.split(): 
	freq[word] = freq.get(word,0)+1 
	words = freq.keys()
	words.sort()
for w in words: 
	print("%s:%d" % (w,freq[w]))
```
