import nltk
import sys
import os
import string
import numpy as np
import functools
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

 
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

   
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    
    query = set(tokenize(input("Query: ")))

    
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    
    idfs = compute_idfs(sentences)

    
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    
   
    res = dict()
    filenames = os.listdir(directory)
    for filename in filenames:
        path = os.path.join(directory, filename)
        sentence = ""
        with open(path, "r", encoding='UTF-8') as file:
            for f in file.readlines():
                sentence += f.strip() + " "
        res[filename] = sentence
        
    return res
    


def tokenize(document):
   
    tmp = nltk.word_tokenize(document)
    words = [t.lower() for t in tmp]
    res = []
    for word in words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            res.append(word)
    
    return res


def compute_idfs(documents):
   
    res = dict()
    n = len(documents)
    tmp_documents = dict()
    all_words = set()
    for filename in documents:
        tmp_documents[filename] = set(documents[filename])
        all_words.update(tmp_documents[filename])

    for word in all_words:
        cnt = 0
        for filename in tmp_documents:
            if word in tmp_documents[filename]:
                cnt += 1
        res[word] = np.log(n / cnt)
        
    return res

def compute_tf(word, words):
    cnt = 0
    for w in words:
        if w == word:
            cnt += 1
    
    return cnt


def top_files(query, files, idfs, n):
    
   
    tmp = []
    for filename in files:
        score = 0
        for word in query:
            if word in idfs:
                score += compute_tf(word, files[filename]) * idfs[word]
        tmp.append((filename, score))
        
    tmp.sort(key=lambda x: -x[1])
    res = []
    for i in range(n):
        res.append(tmp[i][0])
    
    return res

def compute_cnt(word, sentence):
    score = 0
    for s in sentence:
        if s == word:
            score += 1
    
    return score / len(sentence)


def cmp(a, b):
    if a[1] != b[1]:
        return b[1] - a[1]
    else:
        return b[2] - a[2]

    


def top_sentences(query, sentences, idfs, n):
    
    tmp = []
    for sentence in sentences:
        s1 = 0
        s2 = 0
        for word in query:
            if word in sentences[sentence]:
                s1 += idfs[word]
            s2 += compute_cnt(word, sentences[sentence])
        tmp.append((sentence, s1, s2))
        
    tmp = sorted(tmp, key=functools.cmp_to_key(cmp))
    
    res = []
    for i in range(n):
        res.append(tmp[i][0])
    
    return res
    


if __name__ == "__main__":
    main()
