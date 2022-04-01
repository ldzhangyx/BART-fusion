import numpy as np
import torch
import torch.nn as nn
import torchaudio
import os
import random

from attention_modules import BertConfig, BertEncoder, BertPooler


class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class CNNSA(nn.Module):
    '''
    Won et al. 2019
    Toward interpretable music tagging with self-attention.
    Feature extraction with CNN + temporal summary with Transformer encoder.
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50):
        super(CNNSA, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer7 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))

        # Transformer encoder
        bert_config = BertConfig(vocab_size=256,
                                 hidden_size=256,
                                 num_hidden_layers=2,
                                 num_attention_heads=8,
                                 intermediate_size=1024,
                                 hidden_act="gelu",
                                 hidden_dropout_prob=0.4,
                                 max_position_embeddings=700,
                                 attention_probs_dropout_prob=0.5)
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.vec_cls = self.get_cls(256)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_class)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def get_spec(self, ids, audio_length=15*16000, allow_random=False):

        wav_list = list()

        for id in ids:
            audio_path = os.path.join("/import/c4dm-datasets/Music4All/music4all/audios", id + '.mp3')
            (wav, sample_rate) = torchaudio.backend.sox_io_backend.load(audio_path)

            # to mono
            mono_wav = torch.mean(wav, dim=0)

            # cut length
            if allow_random:
                random_index = random.randint(0, len(mono_wav) - audio_length - 1)
            else:
                random_index = 0
            mono_wav_cut = mono_wav[random_index: random_index + audio_length]

            wav_list.append(mono_wav_cut)

        # merge wav to (bs, length)
        data = torch.stack(wav_list, dim=0)

        # to spectrogram
        spectrogram = self.spec(data.cuda())

        return spectrogram

    def forward(self, ids):
        # Spectrogram
        # for batch
        spec = self.get_spec(ids)
        spec_db = self.to_db(spec)
        x = spec_db.unsqueeze(1) # add channel dim
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Get [CLS] token
        x = x.permute(0, 2, 1)
        x = self.append_cls(x)

        # Transformer encoder
        x = self.encoder(x)
        x = x[-1] # last layer
        # x = self.pooler(x)
        #
        # # Dense
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = nn.Sigmoid()(x)

        return x # return the last layer. Shape: (length, 256)


# test code
# model = CNNSA()
# model.load_state_dict(torch.load("best_model.pth"))
# id = ["wlIcjSZkgW0cgWrm", "wlIcjSZkgW0cgWrm"]
# output = model(id)