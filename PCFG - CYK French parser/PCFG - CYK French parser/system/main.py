#!/usr/bin/env python
# coding: utf-8

# Keep this
import os
import pickle 
import numpy as np
import argparse
import  nltk 
import time
from nltk import Tree, word_tokenize
import numpy as np
from nltk.grammar import Production
import re
from tqdm import tqdm
import random
import string
from collections import Counter , defaultdict
from itertools import product 
from PYEVALB import parser as evalbparser
# Here compute all util functions
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from PYEVALB import parser as evalbparser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import multiprocessing as mp
Pool = mp.Pool

import nltk
nltk.download('punkt') # in requirements


parser = argparse.ArgumentParser(description='Start parser NLP2')

#PCFG args
parser.add_argument('--corpus-file', type=str, default='sequoia-corpus+fct.mrg_strict',help="Data file to parse.")
parser.add_argument('--polyglot-file', type=str, default='polyglot-fr.pkl',help="Get polyglot for embeddings")

parser.add_argument('--train-size', type=float, default=0.8, help='train-size')
parser.add_argument('--test-size', type=float, default=0.1, help='test-size')
parser.add_argument('--bigram-coef', type=float, default=0.2, help='Unigram_bigram coefficient for the oov module')
parser.add_argument('--text-path', type=str, default='', help='new_sentence to test the parser')
parser.add_argument('--output-path', type=str, default='evaluation_data.parser_output', help='the output file for test corpus')
parser.add_argument('--text-output-path', type=str, default='new_sentence.parser_output', help='the output file for the new_sentence')

argparser= parser.parse_args()


train_size,test_size,coef = argparser.train_size,argparser.train_size, argparser.bigram_coef
path_corpus  = argparser.corpus_file
path_polyglot = argparser.polyglot_file
output_path = argparser.output_path

text_path = argparser.text_path
text_ouput_path = argparser.text_output_path




# Helpers function 

### Get corpus functions
def get_all_corpus(path= 'sequoia-corpus+fct.mrg_strict'):
    """ Extract the input corpus of sentences and remove hyphen tags """
    full_path = path
    corpus = {}
    with open(full_path,encoding='utf-8') as f:
        for i , line in enumerate(f):
            sent =line.rstrip().split(" ")
            sent=[word.split("-")[0] if word[0]=='(' else word for word in sent]
            corpus[i]=" ".join(sent)
    return corpus
def split_train_test(corpus, train_size=0.8, val_size=0.1, test_size=0.1):
    """ Split the corpus into : train (80%),val(10%), test(10%) """
    
    n = len(corpus)
    train_idx = int(n * train_size)
    val_idx = int(n*(val_size+train_size))
    train = corpus [:train_idx]
    val = corpus [train_idx:val_idx]
    test = corpus [val_idx:]
    return train, val, test

def get_vocabulary (train_corpus):
    vocab = []
    for sentence in train_corpus : 
         vocab.extend(sentence)
    vocab = np.unique(vocab)
    return vocab

def get_fr_word_embedding(vocab,path = 'polyglot-fr.pkl'):
    """ From the Fr plyglot lexicon, extrcat words and embeddings"""
    full_path = path
    with open(full_path, 'rb') as f:
        polyglot = pickle._Unpickler(f)
        polyglot.encoding = 'latin1'
        words, embeddings =   polyglot.load()
        w2embed = {word:embedding for word, embedding in zip(words, embeddings)}
        
    return w2embed




## In main
print('Process corpus =====')
start =time.time()
corpus = get_all_corpus(path=path_corpus)
print('Corpus size =',len(list(corpus.keys())))
tokenized_corpus = [Tree.fromstring(sentence).leaves() for sentence in corpus.values()]
train, val, test = split_train_test(tokenized_corpus, train_size=0.8, val_size=0.1, test_size=0.1)
# Here the corpus contains tags 
train_corpus, val_corpus , test_corpus = split_train_test(list(corpus.values()), train_size=0.8, val_size=0.1, test_size=0.1)
print( 'Train =',len(train),'Val =',len(val),'Test =',len(test))
end = time.time()
print('Finihed in :',end-start)
print('Build vocabulary =====')
start =time.time()
vocab = get_vocabulary (train)
print('Training Vocabulary size = ',len(vocab))
word2id = {word: idx for (idx, word) in enumerate(vocab)}
id2word = dict(enumerate(vocab))
end = time.time()
print('Finihed in :',end-start)

### Distances definition and neighbors

def cosine_similarity(embed_vec,w2embed):
    """
    Calculate cosine similarity between embed_vec(current embedding word)
    """
    embeddings =list(w2embed.values())
    inner_embed = np.inner(embeddings,embed_vec)
    sim_score = inner_embed / (np.linalg.norm(embed_vec)*np.linalg.norm(embeddings,axis=1))
    return sim_score

def closest_word_embed(word, w2embed, id2word_plyglot, embed_neigh=10):
    if word in w2embed.keys():
        vector = w2embed[word]
        similar_vectors  = cosine_similarity(vector,w2embed)
        candidates_id = np.flip(np.argsort(similar_vectors)[-embed_neigh:])
        embed_candidates = [id2word_plyglot[idx] for idx in candidates_id ]
        return embed_candidates
      
    else:
        return []

def levenstein_distance(sent1,sent2):
    """
    Computes the levenstein distance between two sentences
    """
    n,p = len(sent1)+1 , len(sent2)+1
    m = np.zeros((n,p))
    m[:,0] = np.arange(n)
    m[0,:] = np.arange(p)
    for i in range(1,n):
        for j in range(1,p):
            if  sent1[i-1] == sent2[j-1]:
                m[i,j] = min([m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]])
            else : 
                m[i,j] = min([m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]+1])
    
    return m[n-1,p-1]

def levenstein_candidates(word,vocab,lev_neigh = 5):
    """ 
    Find the corespondant word for each oov word
    """  
    lev =np.vectorize(lambda w : levenstein_distance(word.lower(),w.lower()))
    voba_lev_distances = lev(vocab)
    result = [new_word[1] for new_word in sorted(zip(voba_lev_distances,vocab))[:lev_neigh]]
    return result

def update_vocab_embeddings(word2id, w2embed):
    inter = set(word2id.keys()).intersection(set(w2embed.keys()))
    inter_word2id = {w : word2id[w] for w in inter}
    size_vocab = len(list(inter_word2id.keys()))
    inter_w2embed = {w : w2embed[w] for w in inter}
    size_embed = len(list(inter_w2embed.keys()))
    print('Intersection between vocab and embeddings', 'new_vocab_size=',size_vocab, 'new_embed_size=',size_embed)
    return inter_word2id, inter_w2embed
    

# !!! takes ~1-2min to run !!!
print('Get embedding from polyglot Fr =====')
start =time.time()
w2embed = get_fr_word_embedding(vocab,path = path_polyglot)
id2word_plyglot = dict(enumerate(w2embed.keys()))
end =time.time()
print('Finihed in :',end-start)

inter_word2id, inter_w2embed =update_vocab_embeddings(word2id, w2embed)





def uni_bi_grams(vocab, sentences, word2id):
    n= len(vocab)
    bigrams= np.ones((n,n))
    unigrams= np.zeros(n)
    print('Build Unigrams from train ===')
    for sentence in sentences:
        for word in sentence:
            unigrams[word2id[word]] +=1
    norm_uni = np.sum(unigrams)
    unigrams/=norm_uni
    
    print('Build Bigrams from train ===')
    for sentence in sentences:
        for i,word in enumerate(sentence):
            bigrams[word2id[sentence[i-1]],word2id[word]] +=1
            
    norm_bi = np.sum(bigrams, axis = 1)[:, None]
    bigrams/=norm_bi
    return unigrams, bigrams 

def score_context(idx, word, sentence,word2id,unigrams, bigrams,coef =coef): # Process_word function
    
    # We take the log to avoid overflow
    if idx ==0:
        return np.log(unigrams[word2id[word]])
    else: 
        previous_word = sentence[-1]
        score = coef* unigrams[word2id[word]] + (1-coef)*bigrams[word2id[previous_word],word2id[word]]
        
        return np.log(score)
    
def get_new_words(word,vocab,w2embed,word2id,unigrams, bigrams,id2word_plyglot,lev_neigh=10,embed_neigh=20):
    candidates =[]
    max_iter = 20
    i=0
    while len(candidates)==0 and i<max_iter :
        lev_list = levenstein_candidates(word,vocab,lev_neigh)
        embed_list = closest_word_embed(word, w2embed, id2word_plyglot, embed_neigh)
        candidates = set(embed_list).intersection(set(lev_list)) 
        lev_neigh+=1
        i+=1
    return candidates
    
### OOV module
print('Incorporating context: bigrams  =====')
unigrams, bigrams = uni_bi_grams(vocab, train, word2id)

print('length of unigrams/bigrams:',len(unigrams))


# In helpers
def OOV(sentence , vocab, w2embed,word2id,unigrams, bigrams,id2word_plyglot,lev_neigh=10,embed_neigh=20):
    score =0
    replacement = []
    #tokens = Tree.fromstring(sentence).leaves()
    tokens = sentence
    for (idx,word) in enumerate(tokens):
            
            if word in vocab:
                score += score_context(idx, word, replacement,word2id,unigrams,bigrams)
                replacement.append(word)   
                
            else:   
                correction =[]
                candidates = get_new_words(word,vocab,w2embed,word2id,unigrams, bigrams,id2word_plyglot,lev_neigh=10,embed_neigh=20)
                for new_word in candidates :
                    correction.append([new_word,score_context(idx, new_word, replacement,word2id,unigrams, bigrams)])
                
                if len(correction)>0:
                    best_combination = sorted(correction)[-1]
                    score+=  best_combination[1] 
                    replacement.append(best_combination[0])
    return " ".join(replacement)


def get_probabilities_lexicon(lexicon_list):
    unique_pos, distinct = np.unique(lexicon_list, return_counts=True)
    unique_pos = np.array([[pos.lhs(), pos.rhs()[0].lower()]  for pos in unique_pos])
    pos_matrix = np.hstack((unique_pos,distinct.astype(np.float64).reshape(-1,1)))
    
    NT_l, x = np.unique(pos_matrix[:, 0], return_inverse=True)
    NT_r, y = np.unique(pos_matrix[:, 1], return_inverse=True)
    
    l_size, r_size = len(NT_l),len(NT_r)
    probabilities_lexicon= np.zeros((l_size, r_size))
    probabilities_lexicon[x, y] = pos_matrix[:, 2]
    probabilities_lexicon = probabilities_lexicon / np.sum(probabilities_lexicon,axis=1).reshape(-1,1)
    return NT_l,NT_r,probabilities_lexicon


def extract_lexicon(train_corpus):
    lexical_grammar =defaultdict(set)
    axioms = set()
    lexicon_list = []
    start = time.time()
    for sentence in train_corpus:
        tree = Tree.fromstring(sentence, remove_empty_top_bracketing=True)
        tree.chomsky_normal_form(horzMarkov=2)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        prods = tree.productions()
        axioms.add(prods[0].lhs().symbol())
        lexicon_list.extend([prod for prod in prods if prod.is_lexical()])
        
    lexicon_grammar= Counter(lexicon_list) 
    rules_distinct = [[pos.lhs().symbol(),pos.rhs()[0].lower()] for pos in list(lexicon_grammar.keys())]
    rules_count_distincts =  list(lexicon_grammar.values())
    for rule in rules_distinct:
        lexical_grammar[rule[0]].add(rule[1])
    
    NT_l,NT_r,probabilities_lexicon = get_probabilities_lexicon(lexicon_list)
    
    pos2id = {pos.symbol() : idx for (idx,pos) in enumerate(NT_l)}
    word_ref ={word : idx for (idx,word) in enumerate(NT_r)}
    return lexical_grammar, probabilities_lexicon,pos2id,word_ref 
                
                
def PCFG_model(train_corpus)  :
    pcfg_grammar_dict =defaultdict(set)
    axioms = set()
    pcfg_list = []
    start = time.time()
    for sentence in train_corpus:
        tree = Tree.fromstring(sentence, remove_empty_top_bracketing=True)
        tree.chomsky_normal_form(horzMarkov=2)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        prods = tree.productions()
        axioms.add(prods[0].lhs().symbol())
        pcfg_list.extend([prod for prod in prods if prod.is_nonlexical()])
    pcfg_grammar= Counter(pcfg_list)
    rules_distinct = [[pos.lhs().symbol(),pos.rhs()] for pos in list(pcfg_grammar.keys())]
    rules_count_distincts =  list(pcfg_grammar.values())
    for rule in rules_distinct:
        
        if len(rule[1]) ==2 : 
            pcfg_grammar_dict[rule[0]].add((rule[1][0].symbol(),rule[1][1].symbol()))
        else:  
            pcfg_grammar_dict[rule[0]].add(rule[1].symbol())   
    unique_rules, distinct = np.unique(pcfg_list, return_counts=True)
    unique_rules = np.array([[rule.lhs(), rule.rhs()]  for rule in unique_rules])
    rules_matrix = np.hstack((unique_rules,distinct.astype(np.float64).reshape(-1,1)))
    NT_l, x = np.unique(rules_matrix[:, 0], return_inverse=True)
    NT_r, y = np.unique(rules_matrix[:, 1], return_inverse=True)
    l_size, r_size = len(NT_l),len(NT_r)
    pcfg= np.zeros((l_size, r_size))
    pcfg[x, y] = rules_matrix[:, 2]
    pcfg = pcfg / np.sum(pcfg,axis=1).reshape(-1,1)
    NT_lhs= {NT.symbol():idx for (idx,NT) in enumerate(NT_l)}
    NT_rhs = {(NT[0].symbol(),NT[1].symbol()):idx for idx,NT in enumerate(NT_r) if len(NT)>1}
    for (idx,NT) in enumerate(NT_r):
        if len(NT)==1:
            NT_rhs[NT[0].symbol()]=idx
    return pcfg_grammar,pcfg_grammar_dict,axioms ,pcfg,NT_lhs,NT_rhs
    



# In main
start = time.time()
lexical_grammar, probabilities_lexicon,pos2id,word_ref = extract_lexicon(train_corpus) 
pcfg_grammar,pcfg_grammar_dict,axioms,pcfg,NT_lhs,NT_rhs  = PCFG_model(train_corpus) 
end = time.time()
print('PCFG model finished in', end-start)


# helpers for CYK
def build_binaries_unaries(pcfg_grammar_dict,NT_l,NT_r,pos2id):
    binaries = {}
    for lhs in pcfg_grammar_dict.keys() :
        for rhs in pcfg_grammar_dict[lhs] :
            if not rhs in binaries.keys() : binaries[rhs] = set()
            binaries[rhs].add(lhs)
    left_binary = set([bi[0] for bi in binaries.keys()])
    right_binary = set([bi[1] for bi in binaries.keys()])
    
    unaries_target = set([target for target in NT_r.keys() if np.ndim(target)==0])
    binaries_target = set(NT_r.keys()) - set(unaries_target)
    binaries_init = defaultdict(set)
    binaries_inv = defaultdict(set)
    
    for NT,target in pcfg_grammar_dict.items():
        #print('target = ',target)
        #print('bin target = ',binaries_target)
        binaries_tg = target & binaries_target
        #print(binaries_tg)
        if binaries_tg:
            binaries_init.update({NT:binaries_tg})
   #print('binaries',binaries_init.items())
    for NT,targets in binaries_init.items():
        for target in targets:
            binaries_inv[target].add(NT)
            
    ### reformat Non-terminal words
    new_nt_l =NT_l.copy()
    new_nt_l.update({nt:i+len(NT_l) for (nt,i) in pos2id.items()})
    new_nt_l_inv = {i:nt for (nt,i) in new_nt_l.items()}
    return   binaries_inv,new_nt_l

def get_word_tag_dict(lexical_grammar):
    word2pos = defaultdict(set)
    for pos,tokens in lexical_grammar.items():
        for word in tokens:
            word2pos[word.lower()].add(pos)
    return word2pos
    
def process_sentence(sentence,probabilities_lexicon,pos2id,word_ref,word2pos,binaries_inv):
    #sent =sentence.split(" ")
    #sent=sentence
    #sent=[word.split("-")[0] if word[0]=='(' else word for word in sent]
    #new =" ".join(sent)
    #tokens = Tree.fromstring(new).leaves()
    tokens= sentence
    p=len(tokens)
    score = [[{} for i in range(p+1)] for j in range(p+1)]
    score_left = [[set() for i in range(p+1)] for j in range(p+1)]
    score_right = [[set() for i in range(p+1)] for j in range(p+1)]
    left_dict = set()
    right_dict = set()
        
    for binary in binaries_inv:
        left_dict.add(binary[0])
        right_dict.add(binary[1])
        
    for idx, word in enumerate(tokens):
        for pos in word2pos[word.lower()]:
            score[idx][idx+1][pos] = probabilities_lexicon[pos2id[pos],word_ref[word.lower()]] 
            if pos in left_dict: 
                score_left[idx][idx+1].add(pos)
            if pos in right_dict :
                score_right[idx][idx+1].add(pos)          
    return score, score_left,score_right,left_dict, right_dict  

def failure_msg(sentence):
    msg = '(SENT '
    
    for word in sentence[:-1]:
        msg+= '(NULL '+word+')'
    msg+= '(NULL '+sentence[-1]+')'+')'
    return msg

def reconstruct_tree(back_tags, start,end, tokens,axioms,score,NT,n):
    """
    Use dynamic programming to track back the tree : recursive implementation
    """
    if n==1:
        candidates = [score[start][end].get(c,0) for c in axioms]
        NT = axioms[np.argmax(np.array(candidates))]
        if 'SENT' not in NT: return failure_msg(tokens)
        msg = '(' + NT + ' ' + tokens[start] + ')'
        return msg
    
    if end == start +1:
        msg = '(' + NT + ' ' + tokens[start] + ')'
        return msg
       
        
    if end == n+start:
        candidates =np.array([c for c in back_tags[start][end].keys() if c in axioms])
        if len(candidates)==0: return failure_msg(tokens)
        best_axiom = candidates[np.argmax([score[start][end][k] for k in candidates])]
        limit,lhs, rhs = back_tags[start][end][best_axiom]
    
    else:
        limit,lhs, rhs = back_tags[start][end][NT]
     
    left_result = '(' + NT + ' ' + reconstruct_tree(back_tags, start,limit, tokens,axioms,score,lhs,n) 
    right_result = reconstruct_tree(back_tags, limit, end, tokens,axioms,score,rhs,n) + ')'
    msg = left_result + ' '+right_result 
    return msg
            
    
def unchomsky(parsing):
    tree = Tree.fromstring(parsing)
    tree.un_chomsky_normal_form()
    unchomsky_result = ' '.join(tree.pformat().split())
    return unchomsky_result
    



def CYK2(sentence,axioms, probabilities_lexicon,pos2id,word_ref,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict):
    
    word2pos =  get_word_tag_dict(lexical_grammar)
    binaries_inv, nt_dict = build_binaries_unaries(pcfg_grammar_dict,NT_lhs,NT_rhs,pos2id)
    score, score_left,score_right,left_dict, right_dict   = process_sentence(sentence,probabilities_lexicon,pos2id,word_ref,word2pos,binaries_inv)
    n= len(sentence)
    back_tags =[[dict() for i in range(n+1)] for k in range(n+1)]
    for w in range(2,n+1):
        for start in range(n+1-w):
            end= start+w
            for limit in range(start+1,end):
                ## O(n^3)
                left_rule_set = score[start][limit]
                right_rule_set = score[limit][end]
                intersection_rules = set(product(score_left[start][limit], score_right[limit][end])) & set(binaries_inv)
                for (B,C) in intersection_rules:
                    for A in binaries_inv[(B,C)] :
                        proba = left_rule_set[B] * right_rule_set[C] * pcfg[NT_lhs[A]][NT_rhs[(B,C)]] 
                        if proba > score[start][end].get(A, 0.):
                            score[start][end][A] = proba
                            if A in left_dict : 
                                score_left[start][end].add(A)
                            if A in right_dict : 
                                score_right[start][end].add(A)
                            back_tags[start][end][A] = (limit,B,C) 
    
    start ,end,NT =0,n,'SENT'
    result = reconstruct_tree(back_tags, start,end, sentence,axioms,score,NT,n)
    normalized_result = unchomsky(result)
    return normalized_result          
                            
                 
    
                                


def display_results( vocab, w2embed,word2id,word_ref,unigrams, bigrams,id2word_plyglot,axioms, probabilities_lexicon,pos2id,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict,sentence=None,testset=None,lev_neigh=10,embed_neigh=20):
    
    parsed_list=[]
    if sentence:
        try:
            start =time.time() 
            print('sentence',sentence)
            replacement = OOV(sentence , vocab, w2embed,word2id,unigrams, bigrams,id2word_plyglot,lev_neigh,embed_neigh)
            print('oov replaced by',replacement)
            end = time.time()
            print('oov in =====',end-start)
            tokens =replacement.split(' ')
            start =time.time()
            parsed = CYK2(tokens,axioms, probabilities_lexicon,pos2id,word_ref,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict)
            end=time.time()
            print('CYK in =====',end-start)
            print('Parsed',parsed)
            parsed_list.append(parsed)

        except:
            print('Unable to parse')
    
    if testset:
        for sentence in tqdm(testset):
            try:
                start =time.time() 
                print('sentence',sentence)
                replacement = OOV(sentence , vocab, w2embed,word2id,unigrams, bigrams,id2word_plyglot,lev_neigh,embed_neigh)
                print('oov replaced by',replacement)
                end = time.time()
                print('oov in =====',end-start)
                tokens =replacement.split(' ')
                start =time.time()
                parsed = CYK2(tokens,axioms, probabilities_lexicon,pos2id,word_ref,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict)
                end=time.time()
                print('CYK in =====',end-start)
                parsed_list.append(parsed) 
                print('Parsed',parsed)
            except:
                print('Unable to parse')
            
    return parsed_list    




#parsed_list = display_results( vocab, w2embed,word2id,word_ref,unigrams, bigrams,id2word_plyglot,axioms, probabilities_lexicon,pos2id,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict,sentence=test[12],testset=None)





def send_results(parsed_list, output_file = 'evaluation_data.parser_output'):
    with open(output_file, 'w', encoding='utf-8') as output :
        n = len(parsed_list)
        for i in range(n) : 
            if len(parsed_list[i])>0:
                if not i : output.write(parsed_list[i][0])
                else : output.write('\n' + parsed_list[i][0])
        print('Saved')

def get_single_result(sentence):
    parsed_list = display_results( vocab, w2embed,word2id,word_ref,unigrams, bigrams,id2word_plyglot,axioms, probabilities_lexicon,pos2id,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict,sentence=sentence,testset=None)
    return parsed_list

def multiprocess_func(func, n_jobs, arg):
    if n_jobs == -1: 
        n_jobs = mp.cpu_count()
    start = time.time()
    with Pool(n_jobs) as p: 
        result = p.map(func, arg)
    print("Finished parsing in %.2f seconds"%(time.time() - start))
    return result





    
if text_path =='':
    print('Parsing the testset using multiprocessing')
    parsed_list = multiprocess_func(get_single_result, -1, test[14:25])
    pickle.dump(parsed_list, open( "results.p", "wb" ) )    
    send_results(parsed_list, output_file = output_path)
else:
    print('Parsing the new sentence')
    with open(text_path,encoding='utf-8') as f:
        for i , line in enumerate(f):
            sent =line.rstrip().split(" ")
            sent=[word.split("-")[0] if word[0]=='(' else word for word in sent]
            #sentence=" ".join(sent)
            sentence=sent
    parsed_list = display_results( vocab, w2embed,word2id,word_ref,unigrams, bigrams,id2word_plyglot,axioms, probabilities_lexicon,pos2id,lexical_grammar,pcfg,NT_lhs,NT_rhs,pcfg_grammar_dict,sentence=sentence,testset=None)
    send_results(parsed_list, output_file = text_ouput_path)



