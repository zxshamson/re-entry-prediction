# encoding=utf-8
import sys
import random
import numpy as np


class Corpus:

    def __init__(self, filename, maxwordnum, pred_percentage):

        self.convNum = 0            # Number of conversations
        self.convIDs = {}           # Dictionary that maps conversations to integer IDs
        self.r_convIDs = {}         # Inverse of last dictionary
        self.userNum = 0            # Number of users
        self.userIDs = {'null': -1}           # Dictionary that maps users to integer IDs
        self.r_userIDs = {-1: 'null'}         # Inverse of last dictionary
        self.wordNum = 0            # Number of words
        self.wordIDs = {}           # Dictionary that maps words to integers
        self.r_wordIDs = {}         # Inverse of last dictionary

        self.convs = {}        # Each conv is a list of turns, each turn is a tuple of (userID, turnID, p_turnID, words)
        self.user_history = {}  # Each user's history is a list of turns, each turn is (convID, turnID, p_turnID, words)
        self.pred_reply = {}   # Store the userIDs that reply the conv after the predicted percentage
        self.time = {}         # Record each turn's arriving time of each conv (if no time information, store turn nums)
        wordCount = {}  # The count every word appears
        msg_temp = {"null": -1}   # Temporarily store the turn num of msgs for each convs
        conv_temp = {}  # Temporarily store the turn sum for each convs

        # The first time reading, figure out the user, conv and word nums
        with open(filename, 'r') as f:
            readNum = 0
            for line in f:
                readNum += 1
                if readNum % 500000 == 0:
                    print "Data reading.... Line ", readNum
                msgs = line.strip().split('\t')
                try:
                    temp = self.convIDs[msgs[0]]
                except KeyError:
                    self.convIDs[msgs[0]] = self.convNum
                    self.r_convIDs[self.convNum] = msgs[0]
                    self.convNum += 1
                try:
                    temp = self.userIDs[msgs[5]]
                except KeyError:
                    self.userIDs[msgs[5]] = self.userNum
                    self.r_userIDs[self.userNum] = msgs[5]
                    self.userNum += 1
                for word in msgs[4].split(' '):
                    try:
                        wordCount[word] += 1
                    except KeyError:
                        wordCount[word] = 1
                try:
                    conv_temp[self.convIDs[msgs[0]]] += 1
                except KeyError:
                    conv_temp[self.convIDs[msgs[0]]] = 0
                msg_temp[msgs[1]] = conv_temp[self.convIDs[msgs[0]]]

        sortedword = sorted(wordCount.keys(), key=lambda x: wordCount[x], reverse=True)
        # If the parameter equals to -1, it means there is no word counts limit
        if maxwordnum == -1 or maxwordnum > len(sortedword):
            maxwn = len(sortedword) + 2
        else:
            maxwn = maxwordnum
        self.wordNum = maxwn
        for w in xrange(1, maxwn-1):  # 0 for padding
            self.wordIDs[sortedword[w-1]] = w
            self.r_wordIDs[w] = sortedword[w-1]
        self.wordIDs['<NoWords>'] = maxwn-1
        self.r_wordIDs[maxwn-1] = '<NoWords>'

        # Find the turn num that will be the predicting part
        for key in conv_temp.keys():
            if pred_percentage <= 1:
                pred_temp = (conv_temp[key] + 1) * pred_percentage
            else:
                pred_temp = pred_percentage
            conv_temp[key] = (1 if (pred_temp < 1) else int(pred_temp))

        # The second time reading, store the reply messages
        with open(filename, 'r') as f:
            readNum = 0
            for line in f:
                readNum += 1
                if readNum % 500000 == 0:
                    print "Data reading(2nd).... Line ", readNum
                msgs = line.strip().split('\t')
                words = []
                u = self.userIDs[msgs[5]]
                i = self.convIDs[msgs[0]]
                for word in msgs[4].split(' '):
                    try:
                        words.append(self.wordIDs[word])
                    except KeyError:
                        sys.exc_clear()
                sent_len = len(words)
                if sent_len == 0:
                    words.append(maxwn-1)
                if msg_temp[msgs[1]] < conv_temp[i]:  # Be the training part
                    if u != -1:
                        try:
                            self.user_history[u].append(words)
                        except KeyError:
                            self.user_history[u] = []
                            self.user_history[u].append(words)
                    current_turn = [u, msg_temp[msgs[1]], msg_temp[msgs[2]]]
                    current_turn.extend(words)
                    try:
                        self.convs[i].append(current_turn)
                        # each turn of convs is a list of [user, turn_id, parent_turn_id, words]
                    except KeyError:
                        self.convs[i] = [current_turn]
                    time_info = int(msgs[6]) if len(msgs) >= 7 else msg_temp[msgs[1]]
                    try:
                        self.time[i].append((u, time_info))
                    except KeyError:
                        self.time[i] = [(u, time_info)]
                else:  # Be the predicting part
                    try:
                        self.pred_reply[i].append(u)
                    except KeyError:
                        self.pred_reply[i] = [u]

        for key in self.convs.keys():
            try:
                set_temp = set(self.pred_reply[key])
                self.pred_reply[key] = [u for u in set_temp]
            except KeyError:
                self.pred_reply[key] = []
        print "Corpus process over! UserNum: ", self.userNum, "ConvNum: ", self.convNum, "MsgNum: ", readNum


if __name__ == '__main__':

    corp = Corpus("twitter.data", -1, 1)
    cids = corp.convs.keys()
    print cids




