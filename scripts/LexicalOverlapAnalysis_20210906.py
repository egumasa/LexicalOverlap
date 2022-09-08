#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:11:14 2020

@author: Masaki Eguchi

Acknowledgement: 
The basic ideas of the NLP pipeline have been adapted from:
    - Kyle, K. (2020). The relationship between features of source text 
       use and integrated writing quality. Assessing Writing.
    - See Github repo: https://github.com/kristopherkyle/CRAT-two-source-AW

Description:
    The current code calculates the lexical overlap between source text and 
    summary texts with regard to n-grams (contiguous sequence of n words). 
    It takes a text file for the source text and a directory which contains 
    summary raw texts (.txt). The output of the program is a csv file, containing
    the results of the lexical overlap analysis for each summary text.

"""

from collections import Counter
#import copy
import csv
import glob  # used to grab file paths for input texts.
#import json
from operator import itemgetter  #for sorting
import re
import os

import spacy  # spacy is used as the main NLP pipeline (tokenization, lemmatization)
# the version of the spacy model used for the development was ver 2.
# Spacy ver 3 should work, but if not, please use ver 2.
nlp = spacy.load("en_core_web_md")  #see how to download the model

skip_list = [",", ".", "?", ":", ";", "\n", "\n\n"]

pron_dict = {
    "i": "I",
    "my": "I",
    "me": "I",
    "mine": "I",
    'myself': 'I',
    'we': 'we',
    'us': 'we',
    'our': 'we',
    'ours': 'we',
    'ourselves': 'our',
    "you": "you",
    "your": "you",
    "yours": "you",
    'yourself': 'you',
    "she": "she",
    "her": 'she',
    "hers": "she",
    'herself': 'she',
    "he": 'he',
    "his": 'he',
    "him": 'he',
    'himself': 'he',
    'they': 'they',
    'them': 'they',
    'their': 'they',
    'theirs': 'them',
    'themselves': 'them',
    'it': 'it',
    'its': 'it',
    'itself': 'it',
    'one': 'one',
    'oneself': 'one',
}


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


## for accuracy test
def test_learner_tokenization(text_name, sent_boundary=True):
    text = open(text_name).read().strip()

    make_dir('tokenization_accuracy')
    make_dir('tokenization_accuracy/individual_output')

    with open(
            "tokenization_accuracy/individual_output/" + text_name +
            "_period.txt", 'w') as indout:
        for idx, sent in enumerate(text.split(".")):
            indout.write("\n" + str(idx) + "\n" + sent + "\n")
            sent_number = idx
    doc = nlp(text)  #parse the text

    with open(
            "tokenization_accuracy/individual_output/" + text_name +
            "_spacy.txt", 'w') as indout:

        for idx, sent in enumerate(doc.sents):  #iterate through sentences
            indout.write("\n" + str(idx))
            sent_holder = []
            for token in sent:  #iterate token within sentence
                sent_holder.append(token.text)
            indout.write('\n' + " ".join(sent_holder))

            spacy_sent_number = idx

    with open(
            "tokenization_accuracy/individual_output/" + text_name +
            "_simple.txt", 'w') as indout:
        sents = re.split(r"\.|\?|\!", text)

        for idx, sent in enumerate(sents):  #iterate through sentences
            doc = nlp(sent)  #parse the text
            indout.write("\n" + str(idx))
            sent_holder = []
            for token in doc:  #iterate token within sentence
                sent_holder.append(token.text)
            indout.write('\n' + " ".join(sent_holder))

            simple_sent_number = idx
    return sent_number, spacy_sent_number, simple_sent_number


# This is commented out because it is a test function.
# =============================================================================
# with open("tokenization_accuracy_{}.csv".format(sourse_text_ver), "w") as outf:
# 	outf.write("filename,splited_by_period,spacy,simple")
# 	for text in learner_data:
# 		sent_number, spacy_sent_number, simple_sent_number  = test_learner_tokenization(text, sent_boundary = True)
# 		outf.write('\n' + ",".join([text, str(sent_number), str(spacy_sent_number), str(simple_sent_number)]))
#
# =============================================================================


### Function for lexical overlap analysis starts here
def extract_lemma(parsed_doc):
    '''
    Parameters
    ----------
    parsed_doc : Spacy doc or sents
        spacy parsed sentences

    Returns
    -------
    list
        token holder and lemma holders

    '''
    tokens = []  #holder for type
    lemmas = []  #holder for lemma

    #print("\n" + str(idx))
    for token in parsed_doc:  #iterate token within sentence
        if token.text not in skip_list:  #skip any tokes in skip list
            tokens.append(token.text.lower())
            #print(token.text, token.tag_, token.i)
            if token.tag_ in ['PRP', 'PRP$']:  #pronouns should be swapped
                lemmas.append(pron_dict[token.text.lower()])
            else:
                lemmas.append(token.lemma_.lower())

    return [tokens, lemmas]


## ngram list
### split into sentences and parse each sentence.
def lemmatize(text: str, sent_boundary=True, spacy=False):
    '''Implements sentence tokenization and lemmatization
    
    Parameters
    ----------
    text : str
        The texts to analyse
    sent_boundary : Boolean
        Whether to extract n-gram within the sentence only. The default is True.
    spacy : Boolean, optional
        Whether to use spacy for the sentence tokenization. The default is False.

    Returns
    -------
    holder : TYPE
        The dictionary holds lest of types and lemmas.
        It also passes the sent_boundary parameter (whether ngrams were
        extracted only within the sentence) to the next function.

    '''
    holder = {'type': [], 'lemma': [], "sent_boundary": sent_boundary}

    if spacy == True:
        doc = nlp(text)  #sentence tokenize with spacy
        for idx, sent in enumerate(doc.sents):  #iterate through sentences
            lemmatized = extract_lemma(sent)
            if sent_boundary == True:
                holder['type'].append(lemmatized[0])
                holder['lemma'].append(lemmatized[1])

            else:
                holder['type'].extend(lemmatized[0])
                holder['lemma'].extend(lemmatized[1])

    if spacy == False:  #simple regex sent tokenization is used when spacy == False.
        sents = re.split(r"\.|\?|\!",
                         text)  #simple sentence tokenize by punctuations

        for text in sents:
            doc = nlp(text.strip())  #parse the text
            lemmatized = extract_lemma(doc)
            if sent_boundary == True:
                holder['type'].append(lemmatized[0])
                holder['lemma'].append(lemmatized[1])

            else:
                holder['type'].extend(lemmatized[0])
                holder['lemma'].extend(lemmatized[1])

    return holder


def ngrammer(token_list: list, n: int, lemma=True):
    '''Actual implementation of ngram extraction code
    

    Parameters
    ----------
    token_list : list
        List of tokens for ngram extraction
    n : int
        Lengths of ngram.
    lemma : Boolean, optional
        . The default is True.

    Returns
    -------
    holder : TYPE
        DESCRIPTION.

    '''
    holder = []

    text_len = len(token_list)

    for idx, token in enumerate(token_list):
        if n + idx > text_len:
            continue
        else:
            holder.append("__".join(token_list[idx:idx + n]))
    return holder


def ngram_extractor(token_dict: dict,
                    n=2,
                    lemma=[True, False],
                    token=[False, True]):
    ''' Extract ngrams from token dict and return ngram holder
    
    Parameters
    ----------
    token_dict : Dict
        Result of lemmatize(); {'type':[], 'lemma': [], "sent_boundary": sent_boundary}.
    n : int, optional
        N-lengths for ngram extraction. The default is 2.
    lemma : Boolean, optional
        whether we use lemmatized token in extraction. The default is [True, False].
    token : Boolean, optional
        whether we count the same item as many as it occurs or only count once. 
        The default is [False, True].

    Returns
    -------
    ngram_holder : list
        list of extracted ngrams.

    '''
    ngram_holder = []
    sent_boundary = token_dict[
        'sent_boundary']  #retrieve the parameter setting True or False

    if lemma == True:
        token_list = token_dict['lemma']
    else:
        token_list = token_dict['type']

    if sent_boundary == True:
        for sent in token_list:
            #print(sent)
            ngram_holder.extend(ngrammer(sent, n=n))
    else:
        ngram_holder.extend(ngrammer(token_list, n=n))
    #print(ngram_holder)

    if token == False:  #for type analysis, discard any duplicates from the list
        ngram_holder = list(dict.fromkeys(ngram_holder))
    return ngram_holder


def counter(source_ngrams: list, learner_ngrams: list):
    '''Count the number of overlapping items between two lists
    
    Parameters
    ----------
    source_ngrams : list
        List of ngrams from source text.
    learner_ngrams : list
        List of ngrams from learner text.

    Returns
    -------
    score : dict
        dictionary that contains:
            - total number of ngrams
            - the number of hits
            - a list of hit ngrams.

    '''
    score = {"total": len(learner_ngrams), "hit": 0, "hit_ngrams": []}

    for ngram in learner_ngrams:  # iterate learner-produced ngrams
        if ngram in source_ngrams:  # check whether it is in the source text
            score["hit"] += 1  #increment the count
            score['hit_ngrams'].append(
                ngram)  #store the item for qualitative analysis

    return score


def overlap_score(source_text: str, learner_text: str, spacy: bool = False):
    ''' This function takes raw texts and produces a series of overlap indices
        with different settings. 
    
    Parameters
    ----------
    source_text : str
        Source text in string.
    learner_text : str
        Summary (learner) text in string
    spacy : boolean, optional
        Whether to use spacy pipeline for tokenization and lemmatization

    Returns
    -------
    score_dict : dict
        Dictionary that contains the lexical overlapping results for one learner text 
        with different settings. The key is unique combinations of 
        {sent_boundary}_{lemmasetting}_{ngram}

    '''
    score_dict = {}  # storage for the outcome of this function
    #settings
    sent_boundary = [True, False]  #both settings are active
    lemma = [True, False]  #both settings are active
    ngram_lengths = [1, 2, 3, 4, 5, 6]  #compute from 1 to 6 grams

    for sent_setting in sent_boundary:  #iterate two choices

        if sent_setting == True:
            index1 = "sent"  #set index name
        else:
            index1 = "wholetext"

        for lemma_setting in lemma:
            if lemma_setting == True:
                index2 = "lemma"  #set index name
            else:
                index2 = "wdform"

            for n_len in ngram_lengths:
                index3 = "{}gram".format(str(n_len))  #set index name
                ### extract ngram here ###
                source_ngrams = ngram_extractor(lemmatize(
                    source_text, sent_boundary=sent_setting, spacy=spacy),
                                                n=n_len,
                                                lemma=lemma_setting,
                                                token=False)
                learner_ngrams = ngram_extractor(lemmatize(
                    learner_text, sent_boundary=sent_setting, spacy=spacy),
                                                 n=n_len,
                                                 lemma=lemma_setting,
                                                 token=False)

                index = "_".join([index2, index3,
                                  index1])  #concatenate index name
                score_dict.update({
                    index: counter(source_ngrams, learner_ngrams)
                })  # implements countere here
    return score_dict


def update_hit_ngram_holder(hit_ngram_holder_index: dict, score: dict,
                            index: str):
    '''
    
    Parameters
    ----------
    hit_ngram_holder : dict
        This is the holder to generate the frequency list, See main function.
    hit_ngrams : list
        THis is the list of extracted hit Ngrams in each learner text.
    index : str
        index string to put the hit ngrams in appropriate location of the hit_ngram_holder

    Returns
    -------
    None: this function updates hit_ngram_holder as it goes, so no results to return directly.

    '''
    hit_ngram_dict = {}
    hit_ngram_dict = Counter(score['hit_ngrams'])

    #print(index, hit_ngram_dict)
    for item, freq in hit_ngram_dict.items():
        #print("To {}, appending {}".format(index, item))
        if item not in hit_ngram_holder_index:
            hit_ngram_holder_index[item] = {'freq': freq, 'n': 1}
        elif item in hit_ngram_holder_index:
            hit_ngram_holder_index[item]['freq'] += freq
            hit_ngram_holder_index[item]['n'] += 1


## output for frequency list
def write_frequency_list(freq_dict: dict, source_text_ver: str):
    ''' takes dictionary of overlapping ngrams across the whole learner corpus
        write a tsv file with ngrams, frequency and range. 
        Note that freq will be equal to range in the type analysis.
        This is because the same ngrams are tallied only once.
    
    Parameters
    ----------
    freq_dict : dict
        dictionary that contain overlapping ngrams.
    source_text_ver : str
        This is the name of the source text used when writing the tsv file

    Returns
    -------
    None.

    '''
    make_dir("frequency_lists")

    for index, item_info in freq_dict.items():
        with open(
                'frequency_lists/freq_list_{}_{}.txt'.format(
                    source_text_ver, index), 'w') as outf:
            tsv_writer = csv.writer(outf, delimiter='\t')

            tsv_writer.writerow([index, 'freq', 'n_text'])
            holder = []
            for ngram, freq_dict in item_info.items():
                holder.append([ngram, int(freq_dict['freq']), freq_dict['n']])

            holder = sorted(holder, key=itemgetter(1), reverse=True)  #sorting
            tsv_writer.writerows(holder)


def run_overlapanalysis(source_text,
                        l_texts,
                        source_text_ver: str,
                        itemoutput=False,
                        freq_list=False):

    make_dir("item_output")

    # for number of files
    counter = 1

    # for frequency dict
    hit_ngram_holder = {}

    ## when you refactor next time, add components to parse the source text here for effectiveness
    #source_ngrams = ngram_extractor(lemmatize(source_text, sent_boundary = sent_setting, spacy = spacy), n = n_len, lemma = lemma_setting, token = False)

    with open(out_filename, 'w') as outf:  #open the output file
        index_list = ['filename']  # this is the index list

        for idx, l_text in enumerate(
                l_texts):  #iterate each text in the learner corpus
            print("Processing {} out of {} files...".format(
                counter, len(l_texts)))
            filename = l_text.split('/')  #split filename
            counter += 1
            scores = overlap_score(
                source_text, open(l_text).read(),
                spacy=False)  #calculating overlap score here
            #print(scores)
            ## Output process from here
            #write the index names on the first row
            if idx == 0:  #for the first line, we will write header for index name
                indices = list(scores.keys())
                #hit_ngram_holder = dict.fromkeys(indices, []) #initialize the holder with dictionary
                #print(hit_ngram_holder)
                for indexname in indices:
                    index_list.extend(
                        list(map(lambda x: x + indexname, ['total_', 'hit_'])))
                outf.write(",".join(index_list))

            ## Writing overlap score from here
            outf.write("\n" + filename[-1])  #first cell is the filename
            #print(indices)
            for index in indices:  #iterate overlap-score dict
                outf.write("," +
                           str(scores[index]['total']))  #write total score
                outf.write(',' + str(scores[index]['hit']))  #write hit score

            if itemoutput == True:  # if True this will output overlap items.
                with open("item_output/" + l_text + "indout.txt", 'w') as f:
                    for index in indices:
                        f.write("\n" + str(index))
                        f.write("\n" + "\n".join(scores[index]['hit_ngrams']))
                        f.write("\n")

            #for frequency list
            if freq_list:
                for i, score in scores.items():
                    if i not in hit_ngram_holder:
                        hit_ngram_holder[i] = {}

                    update_hit_ngram_holder(hit_ngram_holder[i], score, i)
                    #hit_ngram_holder[i].extend(score['hit_ngrams'])
        if freq_list:
            write_frequency_list(hit_ngram_holder, source_text_ver)

    return hit_ngram_holder


def main(source_text_ver: str = "BR"):
    source_text_name = "InputText/{}_n-gram.txt".format(source_text_ver)
    DIR_data = 'RtS_{}_Revised/*.txt'.format(source_text_ver)

    learner_data = glob.glob(DIR_data)
    source_text = open(source_text_name).read()
    out_filename = 'LexicalOverlapScore_{}_type2_20210722.csv'.format(
        source_text_ver)
    freq_dict = run_overlapanalysis(source_text, learner_data, source_text_ver,
                                    True, True)


## Running with BR text
# =============================================================================
# source_text_ver = "BR"
# source_text_name = "InputText/{}_n-gram.txt".format(source_text_ver)
# DIR_data = 'RtS_{}_Revised/*.txt'.format(source_text_ver)
#
# learner_data = glob.glob(DIR_data)
#
# source_text = open(source_text_name).read()
#
# out_filename = 'LexicalOverlapScore_{}_type2_20210722.csv'.format(source_text_ver)
# freq_dict = run_overlapanalysis(source_text, learner_data, source_text_ver, True, True)
#
# =============================================================================

## Running with ICRC text
# =============================================================================
# source_text_ver = "ICRC"
# source_text_name = "InputText/{}_n-gram.txt".format(source_text_ver)
# DIR_data = 'RtS_{}_Revised/*.txt'.format(source_text_ver)
#
# learner_data = glob.glob(DIR_data)
#
# source_text = open(source_text_name).read()
#
# out_filename = 'LexicalOverlapScore_{}_type2_20210722.csv'.format(source_text_ver)
# freq_dict = run_overlapanalysis(source_text, learner_data, source_text_ver, True, True)
#
# =============================================================================
