import torch
import torchtext
import numpy as np
from pprint import pprint
import conllu

sentenceLens = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
CUT_OFF = 40
BATCH_SIZE = 32


class Data(torch.utils.data.Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.device = device
        self.cutoff = CUT_OFF

        self.tags = tags

        # zip em together
        self.sentences = [list(zip(sentence, tag)) for sentence, tag in zip(self.sentences, self.tags)]

        self.Ssentences = [sentence for sentence in self.sentences if self.cutoff > len(sentence) > 0]
        self.Lsentences = [sentence for sentence in self.sentences if len(sentence) >= self.cutoff]
        # split long sentences into 2
        self.sentences = []
        for sentence in self.Lsentences:
            half = len(sentence) // 2
            self.sentences.append(sentence[:half])
            self.sentences.append(sentence[half:])
        self.sentences += self.Ssentences
        # print(self.sentences)

        self.sentences = [[("<bos>", "<bot>")] + sentence + [("<eos>", "<eot>")] for sentence in self.sentences if
                          len(sentence) <= self.cutoff]

        # print(self.sentences)
        # print(len(self.sentences))
        # exit(1)
        self.vocab = set()
        self.tagVocab = set()
        self.mxSentSize = 0
        # self.mxSentSize = 20
        for sentence in self.sentences:
            if len(sentence) not in sentenceLens:
                sentenceLens[len(sentence)] = 1
            else:
                sentenceLens[len(sentence)] += 1
            for token in sentence:
                self.vocab.add(token[0])
                self.tagVocab.add(token[1])

            if len(sentence) > self.mxSentSize:
                self.mxSentSize = len(sentence)

        self.vocab = list(self.vocab)
        self.tagVocab = list(self.tagVocab)

        # add padding token
        self.vocab.append("<pad>")
        self.tagVocab.append("<pad>")
        # add Unknown
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")

        if "<unk>" not in self.tagVocab:
            self.tagVocab.append("<unk>")

        if "<bos>" not in self.vocab:
            self.vocab.append("<bos>")

        if "<bot>" not in self.tagVocab:
            self.tagVocab.append("<bot>")

        if "<eos>" not in self.vocab:
            self.vocab.append("<eos>")

        if "<eot>" not in self.tagVocab:
            self.tagVocab.append("<eot>")

        self.vocabSet = set(self.vocab)
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # print(self.idx2w)

        self.tagVocabSet = set(self.tagVocab)
        self.tagW2idx = {w: i for i, w in enumerate(self.tagVocab)}
        self.tagIdx2w = {i: w for i, w in enumerate(self.tagVocab)}

        # pad each sentence to 40
        for i in range(len(self.sentences)):
            # print(type(self.sentences[i]))
            self.sentences[i] = self.sentences[i] + [("<pad>", "<pad>")] * (self.mxSentSize - len(self.sentences[i]))
        self.sentencesIdx = torch.tensor([[self.w2idx[token[0]] for token in sentence] for sentence in self.sentences],
                                         device=self.device)
        self.tagSentencesIdx = torch.tensor(
            [[self.tagW2idx[token[1]] for token in sentence] for sentence in self.sentences],
            device=self.device)
        self.padIdx = self.w2idx["<pad>"]
        self.tagPadIdx = self.tagW2idx["<pad>"]

    def handle_unknowns(self, vocab_set, vocab):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j][0] not in vocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j][0] in self.vocab:
                        self.vocab.remove(self.sentences[i][j][0])
                    if self.sentences[i][j][0] in self.vocabSet:
                        self.vocabSet.remove(self.sentences[i][j][0])
                    self.sentences[i][j] = "<unk>"
        self.w2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2w = {i: w for i, w in enumerate(vocab)}
        self.sentencesIdx = torch.tensor([[self.w2idx[token[0]] for token in sentence] for sentence in self.sentences],
                                         device=self.device)

    def __len__(self):
        return len(self.sentencesIdx)

    def __getitem__(self, idx):
        # sentence, last word
        return self.sentencesIdx[idx], self.tagSentencesIdx[idx]


input = open('./UD_English-Atis/en_atis-ud-train.conllu', 'r', encoding='utf-8')
data = conllu.parse(input.read())
tags = []
sentences = []
for sentence in data:
    unitTag = []
    unitSentence = []
    for token in sentence:
        unitTag.append(token["upos"])
        unitSentence.append(token["form"])

    tags.append(unitTag)
    sentences.append(unitSentence)

trainData = Data(sentences, tags)
for sentence, tag in trainData:
    # convert back to word from idx
    s = [(trainData.idx2w[int(idx)], trainData.tagIdx2w[int(tag)]) for idx, tag in zip(sentence, tag)]
    pprint(s)
