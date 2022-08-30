import sys
sys.path.append('..')

from torch.utils.data import Dataset
import pickle
import random
from . import LyricsCommentData

class LyricsCommentsDataset(Dataset):

    def __init__(self, random=False):
        super(LyricsCommentsDataset, self).__init__()
        self.random = random
        with open("dataset.pkl", "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        lyrics = self.data[item].lyrics
        # if random:
        #     comment = random.choice(self.data[item].comments)
        # else:
        comment = self.data[item].comments[0]
        # the longest?
        for i, (tmp_item, _) in enumerate(self.data[item].comments):
            if len(tmp_item) > len(comment[0]):
                comment = self.data[item].comments[i]

        comment = comment[0] # keep comments w/o rating

        return [lyrics, comment]


class LyricsCommentsDatasetClean(Dataset):

    def __init__(self, random=False):
        super(LyricsCommentsDatasetClean, self).__init__()
        self.random = random
        with open("cleaned_dataset.pkl", "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        lyrics = self.data[item].lyrics
        comment = self.data[item].comment

        return [lyrics, comment]


class LyricsCommentsDatasetPsuedo(Dataset):

    def __init__(self, dataset_path, random=False):
        super(LyricsCommentsDatasetPsuedo, self).__init__()
        self.random = random
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        lyrics = self.data[item].lyrics.replace('\n', ';')
        comment = self.data[item].comment

        return [lyrics, comment]


class LyricsCommentsDatasetPsuedo_fusion(Dataset):

    def __init__(self, dataset_path):
        super(LyricsCommentsDatasetPsuedo_fusion, self).__init__()
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        lyrics = self.data[item].lyrics.replace('\n', ';')
        comment = self.data[item].comment
        music_id = self.data[item].music4all_id

        return [lyrics, comment, music_id]


from torch.utils.data import Dataset, DataLoader
import torch
from MusicData import MusicData
import csv
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
from scipy import signal
import numpy as np
import torchaudio
import transformers
import nltk


class Music4AllDataset(Dataset):
    def __init__(self,
                 mel_bins,
                 audio_length,
                 pad_length,
                 tag_file_path=r"Music4All/music4all/id_genres.csv",
                 augment=True):
        self.tag_file_path = tag_file_path
        self.allow_cache = True
        self.mel_bins = mel_bins
        self.audio_length = audio_length
        self.pad_length = pad_length
        self.augment = augment
        # read all tags
        tags_file = open(tag_file_path, 'r', encoding='utf-8')
        self.tags_reader = list(csv.reader(tags_file, delimiter='\t'))[1:]
        tags_file.close()
        if self.augment:
            self.data_augmentation()

    def data_augmentation(self):
        pass

    def __len__(self):
        return len(self.tags_reader)

    def __getitem__(self, item):
        """

        :param item: index
        :return: tags and mel-spectrogram.
        """
        id = self.tags_reader[item][0]
        tags = self.tags_reader[item][1] #.split(',')

        # pad tags
        # if len(tags) >= self.pad_length:
        #     tags = tags[:self.pad_length]
        # else:
        #     for i in range(self.pad_length - len(tags)):
        #         tags.append("[PAD]")

        spec_path = os.path.join("Music4All/temp_data/specs/data_cache/", id + ".npy")
        exist_cache = os.path.isfile(spec_path)
        # search cache
        # if exist cache, load
        if self.allow_cache and exist_cache:
            spectrogram = torch.Tensor(np.load(spec_path))
        # if does not exist, calculate and save
        else:
            audio_path = os.path.join("Music4All/music4all/audios",
                                      id + '.mp3'
                                      )
            (data, sample_rate) = torchaudio.backend.sox_io_backend.load(audio_path)
            spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.mel_bins,
                                                               n_fft=512,
                                                               sample_rate=sample_rate,
                                                               f_max=8000.0,
                                                               f_min=0.0,
                                                               )(torch.Tensor(data))
            # TODO: There is a huge bug!
            # cut length
            if self.audio_length is not None:
                spectrogram = spectrogram[:, :, :self.audio_length]
            # to mono
            spectrogram = spectrogram[0, :, :].unsqueeze(0)

            if self.allow_cache:
                np.save(spec_path, spectrogram.numpy())

        return tags, spectrogram


class MusCapsDataset(Dataset):
    def __init__(self,
                 mel_bins,
                 audio_length,
                 pad_length,
                 tag_file_path=r"Music4All/music4all/id_genres.csv",
                 augment=True):
        self.tag_file_path = tag_file_path
        self.allow_cache = True
        self.mel_bins = mel_bins
        self.audio_length = audio_length
        self.pad_length = pad_length
        self.augment = augment
        # read all tags
        tags_file = open(tag_file_path, 'r', encoding='utf-8')
        self.tags_reader = list(csv.reader(tags_file, delimiter='\t'))[1:]
        tags_file.close()
        if self.augment:
            self.data_augmentation()

    def data_augmentation(self):
        pass

    def __len__(self):
        return len(self.tags_reader)

    def __getitem__(self, item):
        """

        :param item: index
        :return: tags and mel-spectrogram.
        """
        id = self.tags_reader[item][0]
        tags = self.tags_reader[item][1] #.split(',')

        # pad tags
        # if len(tags) >= self.pad_length:
        #     tags = tags[:self.pad_length]
        # else:
        #     for i in range(self.pad_length - len(tags)):
        #         tags.append("[PAD]")

        spec_path = os.path.join("Music4All/temp_data/specs/data_cache/", id + ".npy")
        exist_cache = os.path.isfile(spec_path)
        # search cache
        # if exist cache, load
        if self.allow_cache and exist_cache:
            spectrogram = torch.Tensor(np.load(spec_path))
        # if does not exist, calculate and save
        else:
            audio_path = os.path.join("Music4All/music4all/audios",
                                      id + '.mp3'
                                      )
            (data, sample_rate) = torchaudio.backend.sox_io_backend.load(audio_path)
            spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.mel_bins,
                                                               n_fft=512,
                                                               sample_rate=sample_rate,
                                                               f_max=8000.0,
                                                               f_min=0.0,
                                                               )(torch.Tensor(data))
            # cut length
            if self.audio_length is not None:
                spectrogram = spectrogram[:, :, :self.audio_length]
            # to mono
            spectrogram = spectrogram[0, :, :].unsqueeze(0)
            np.save(spec_path, spectrogram.numpy())

        return tags, spectrogram

class GTZANDataset(Dataset):
    def __init__(self, raw_dataset, is_augment=True, window=1366):
        self.raw = raw_dataset
        self.data = list()
        self.mel_bins = 96
        self.gtzan_genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]
        self.is_augment = is_augment
        self.window = window
        self.init()

    def init(self):
        for i, (waveform, sample_rate, label) in enumerate(self.raw):
            spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.mel_bins)(torch.Tensor(waveform))
            if self.is_augment:
                self.augment(spectrogram, label)
            else:
                self.data.append((spectrogram[:,:,:self.window], label))

    def augment(self, spectrogram, label):
        length = spectrogram.shape[-1] # length
        # augment audio with sliding window
        hop_length = 250
        slices = (length - self.window) // hop_length
        for i in range(slices):
            self.data.append((spectrogram[:, :, i * hop_length:self.window + i*hop_length], label))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spectrogram, label = self.data[index]
        label = self.gtzan_genres.index(label)
        return spectrogram, label



