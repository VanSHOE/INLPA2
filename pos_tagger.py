import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pprint import pprint
import conllu
import random
import time
from sklearn.metrics import classification_report as cr
import os

random.seed(time.time())

sentenceLens = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
CUT_OFF = 40
BATCH_SIZE = 32
MODEL = "Test"


class Data(torch.utils.data.Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.device = device
        self.cutoff = CUT_OFF

        self.tags = tags

        # lowercase everything
        self.sentences = [[token.lower() for token in sentence] for sentence in self.sentences]
        self.tags = [[token.lower() for token in sentence] for sentence in self.tags]

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
        # add padding token
        self.vocab.add("<pad>")
        self.tagVocab.add("<pad>")
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

        self.bosIdx = self.w2idx["<bos>"]
        self.botIdx = self.tagW2idx["<bot>"]

        self.eosIdx = self.w2idx["<eos>"]
        self.eotIdx = self.tagW2idx["<eot>"]

    def handle_unknowns(self, vocab_set, vocab, tagVocab_set, tagVocab):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j][0] not in vocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j][0] in self.vocab:
                        self.vocab.remove(self.sentences[i][j][0])
                    if self.sentences[i][j][0] in self.vocabSet:
                        self.vocabSet.remove(self.sentences[i][j][0])
                    t = list(self.sentences[i][j])
                    t[0] = "<unk>"
                    self.sentences[i][j] = tuple(t)
        self.w2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2w = {i: w for i, w in enumerate(vocab)}
        self.sentencesIdx = torch.tensor([[self.w2idx[token[0]] for token in sentence] for sentence in self.sentences],
                                         device=self.device)

        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j][1] not in tagVocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j][1] in self.tagVocab:
                        self.tagVocab.remove(self.sentences[i][j][1])
                    if self.sentences[i][j][1] in self.tagVocabSet:
                        self.tagVocabSet.remove(self.sentences[i][j][1])
                    t = list(self.sentences[i][j])
                    t[1] = "<unk>"
                    self.sentences[i][j] = tuple(t)

        self.tagW2idx = {w: i for i, w in enumerate(tagVocab)}
        self.tagIdx2w = {i: w for i, w in enumerate(tagVocab)}
        self.tagSentencesIdx = torch.tensor(
            [[self.tagW2idx[token[1]] for token in sentence] for sentence in self.sentences],
            device=self.device)

    def rem_low_freq(self, threshold):
        # remove words with frequency less than threshold
        freq = {}
        for sentence in self.sentences:
            for token in sentence:
                if token[0] not in freq:
                    freq[token[0]] = 1
                else:
                    freq[token[0]] += 1

        self.handle_unknowns(self.vocabSet, self.vocab, self.tagVocabSet, self.tagVocab)

    def __len__(self):
        return len(self.sentencesIdx)

    def __getitem__(self, idx):
        # sentence, last word
        return self.sentencesIdx[idx], self.tagSentencesIdx[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, outputSize):
        super(LSTM, self).__init__()  # call the init function of the parent class
        self.device = device
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_size = hidden_size  # size of LSTM hidden state
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.decoder = nn.Linear(hidden_size, outputSize)  # linear layer to map the hidden state to output classes
        self.train_data = None

        self.elayer = nn.Embedding(vocab_size, input_size)

        self.to(self.device)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        embeddings = self.elayer(x)

        # Forward propagate through the LSTM layer
        out, _ = self.lstm(embeddings)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return self.decoder(out)


def getLossDataset(data: Data, model):
    model.eval()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    criterionL = nn.CrossEntropyLoss()  # ignore_index=data.tagPadIdx)
    loss = 0

    for i, (x, y) in enumerate(dataL):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        loss += criterionL(output, y)

    return loss / len(dataL)


def train(model, data, optimizer, criterion, valDat, maxPat=5):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    lossDec = True
    prevLoss = 10000000
    prevValLoss = 10000000
    epoch = 0
    es_patience = maxPat
    model.train_data = data
    while lossDec:
        epoch_loss = 0
        for i, (x, y) in enumerate(dataL):
            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # print loss every 100 batches
            # if i % 100 == 0:
            #     print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        validationLoss = getLossDataset(valDat, model)
        print(f"Validation loss: {validationLoss}")
        if validationLoss > prevValLoss:
            print("Validation loss increased")
            if es_patience > 0:
                es_patience -= 1
            else:  # early stopping
                print("Early stopping")
                # model = torch.load(f"{MODEL}.pt")
                model.load_state_dict(torch.load(f"{MODEL}.pt"))
                lossDec = False
        else:
            # torch.save(model, f"{MODEL}.pt")
            torch.save(model.state_dict(), f"{MODEL}.pt")
            es_patience = maxPat
        prevValLoss = validationLoss
        model.train()
        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1


def accuracy(model, data):
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        _, predicted = torch.max(output, 1)
        # print x as words
        # print(x)
        # print([model.train_data.idx2w[i] for i in x.view(-1).tolist()])
        # print(x.view(-1).size(0))
        # print(y.size(0))
        # exit(22)
        # only those are suitable in which x is not bos eos or pad
        suitableIdx = [i for i in range(x.view(-1).size(0)) if x.view(-1)[i] != model.train_data.padIdx]
        # test = x.view(-1)[suitableIdx].tolist()
        # print([model.train_data.idx2w[i] for i in test])
        # exit(0)
        y_masked = y[suitableIdx]
        total += y_masked.size(0)
        correct += (predicted[suitableIdx] == y_masked).sum().item()
    return correct / total


def runSkMetric(model, data):
    model.eval()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])
        suitableIdx = [i for i in range(x.view(-1).size(0)) if x.view(-1)[i] != model.train_data.padIdx]
        _, predicted = torch.max(output, 1)
        y_masked = y[suitableIdx]
        y_pred_masked = predicted[suitableIdx]
        y_true.extend(y_masked.tolist())
        y_pred.extend(y_pred_masked.tolist())

    y_trueTag = [model.train_data.tagIdx2w[i] for i in y_true]
    y_predTag = [model.train_data.tagIdx2w[i] for i in y_pred]
    return cr(y_trueTag, y_predTag)


inputT = open('./UD_English-Atis/en_atis-ud-train.conllu', 'r', encoding='utf-8')
data = conllu.parse(inputT.read())
val = open('./UD_English-Atis/en_atis-ud-dev.conllu', 'r', encoding='utf-8')
valData = conllu.parse(val.read())
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

trainData = Data(sentences.copy(), tags.copy())

if os.path.exists("Final.pt"):
    print("Loading model...")
    model = torch.load("Final.pt")
else:
    model = LSTM(300, 300, 1, len(trainData.vocab), len(trainData.tagVocab))
    model.train_data = trainData
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, trainData, optimizer, criterion, valData, 4)
    torch.save(model, "Final.pt")

tags = []
sentences = []
for sentence in valData:
    unitTag = []
    unitSentence = []
    for token in sentence:
        unitTag.append(token["upos"])
        unitSentence.append(token["form"])

    tags.append(unitTag)
    sentences.append(unitSentence)

valData = Data(sentences.copy(), tags.copy())
valData.handle_unknowns(model.train_data.vocabSet, model.train_data.vocab, model.train_data.tagVocabSet,
                        model.train_data.tagVocab)

model.eval()
test = open('./UD_English-Atis/en_atis-ud-test.conllu', 'r', encoding='utf-8')
testData = conllu.parse(test.read())
tags = []
sentences = []
for sentence in testData:
    unitTag = []
    unitSentence = []
    for token in sentence:
        unitTag.append(token["upos"])
        unitSentence.append(token["form"])

    tags.append(unitTag)
    sentences.append(unitSentence)

testData = Data(sentences.copy(), tags.copy())
testData.handle_unknowns(model.train_data.vocabSet, model.train_data.vocab, model.train_data.tagVocabSet,
                         model.train_data.tagVocab)
# print(f"Training Accuracy: {accuracy(model, model.train_data)}")
#
# print(f"Validation Accuracy: {accuracy(model, valData)}")

print(f"Testing Accuracy: {accuracy(model, testData)}")
# exit(0)
result = runSkMetric(model, testData)
# save to file
with open("result.txt", "w") as f:
    f.write(result)
print(result)
# exit(0)

sent = str(input("input sentence: ")).split()
orig = sent.copy()

# convert to idx
for i in range(len(sent)):
    if sent[i] in model.train_data.vocab:
        sent[i] = model.train_data.w2idx[sent[i]]
    else:
        sent[i] = model.train_data.w2idx["<unk>"]

sent = torch.tensor(sent).to(model.device)
output = model(sent)
# softmax and output tag
output = torch.nn.functional.softmax(output, dim=1)
_, predicted = torch.max(output, 1)

for i in range(len(sent)):
    print(f"{orig[i]}\t{model.train_data.tagIdx2w[predicted[i].item()]}")
