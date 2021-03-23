import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
import torchtext

import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm


class Vocabulary:
    def __init__(self):
        self.word2index = {"[SOS]": 0, "[EOS]": 1, "[PAD]": 2}
        self.index2word = {0: "[SOS]", 1: "[EOS]", 2: "[PAD]"}

        self.tokenize_func = torchtext.data.get_tokenizer("basic_english")
        self.num_words = len(self.word2index)
        self.max_seq_len = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    def add_sentence(self, sentence):
        tokens = self.tokenize_func(sentence)
        for token in tokens:
            self.add_word(token)

        sentence_len = len(tokens)
        if sentence_len > self.max_seq_len:
            self.max_seq_len = sentence_len


class ImageCaptionDataset(Dataset):
    def __init__(self, path=None, train=False):
        super(ImageCaptionDataset, self).__init__()
        self.vocab = Vocabulary()
        self.path = path
        self.train = train

        image_names = os.listdir(path)

        caption_dict = dict()

        caption_json = open("captions.json")
        captions = json.load(caption_json)['images']

        for caption in captions:
            caption_dict[caption['file_name']] = caption['captions']
            for c in caption['captions']:
                self.vocab.add_sentence(c)

        self.caption_dict = caption_dict

        self.image_names = image_names
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(30),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(self.path+'/'+image_name)

        if self.train:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor()
            ])(image)

        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        if not self.train:
            return image, image_name

        num_caption = len(self.caption_dict[image_name])
        caption = self.caption_dict[image_name][np.random.randint(num_caption)]
        caption = [self.vocab.word2index[token] for token in self.vocab.tokenize_func(caption)]
        caption.append(1)  # EOS token

        if len(caption) - 2 < self.vocab.max_seq_len:
            caption.extend([2] * (self.vocab.max_seq_len - len(caption) + 2))  # pad PAD token

        caption = torch.Tensor(caption).long()

        return image, caption

    def __len__(self):
        return len(self.image_names)


class Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoder, self).__init__()
        if pretrain:
            inception = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        else:
            inception = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
        self.features = nn.Sequential(*list(inception.children())[:-4])
        self.fc = nn.Linear(2048, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = x.reshape((x.size(0), 64, -1))
        x = self.fc(x)
        x = self.relu(x)
        return x


class Attention(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024):
        super(Attention, self).__init__()
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        hidden = hidden.transpose(0, 1)
        score = self.tanh(self.w1(x) + self.w2(hidden))

        attention_weights = self.softmax(self.v(score))

        context_vector = attention_weights * x
        context_vector = context_vector.sum(axis=1)

        return context_vector


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.gru = nn.GRU(1024, 1024, batch_first=True)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, features, hidden):
        context_vector = self.attention(features, hidden)

        x = self.embedding(x).view(x.size(0), 1, -1)
        context_vector = context_vector.unsqueeze(1)

        x = torch.cat((context_vector, x), dim=2)

        x, hidden = self.gru(x, hidden)

        x = self.fc1(x)
        x = x.reshape(-1, x.size(2))
        x = self.softmax(self.fc2(x))

        return x, hidden


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, y):
        batch_size = y.size(0)
        target_len = y.size(1)
        vocab_size = self.decoder.vocab_size

        x = x.to(self.device)
        y = y.to(self.device)

        # encoder
        features = self.encoder(x)

        outputs = torch.zeros((batch_size, target_len, vocab_size)).to(self.device)
        dec_input = torch.Tensor([[0]]*batch_size).long().to(self.device)  # [SOS]: 0

        dec_hidden = torch.zeros([1, batch_size, 1024]).to(self.device)

        # decoder
        for t in range(target_len):
            dec_output, dec_hidden = self.decoder(dec_input, features, dec_hidden)

            outputs[:, t, :] = dec_output

            dec_input = y[:, t]

        return outputs

    def predict(self, x, dataset):
        batch_size = x.size(0)

        x = x.to(self.device)

        features = self.encoder(x)

        dec_input = torch.Tensor([[0]]*batch_size).long().to(self.device)
        dec_hidden = torch.zeros([1, batch_size, 1024]).to(self.device)

        captions = []
        for t in range(dataset.vocab.max_seq_len):
            dec_output, dec_hidden = self.decoder(dec_input, features, dec_hidden)
            _, top_index = dec_output.data.topk(1)

            if top_index.item() == 1:
                break
            else:
                captions.append(dataset.vocab.index2word[top_index.item()])

            dec_input = top_index.detach()

        return captions
