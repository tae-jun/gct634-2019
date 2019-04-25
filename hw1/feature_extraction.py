# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import os
import numpy as np
import librosa
import essentia
import essentia.standard as es

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 20
FEATURES = [
  'lowlevel.average_loudness',
  'lowlevel.dissonance.mean',
  'lowlevel.dynamic_complexity',
  'lowlevel.pitch_salience.mean',
  'lowlevel.silence_rate_20dB.mean',
  'lowlevel.silence_rate_30dB.mean',
  'lowlevel.silence_rate_60dB.mean',
  'lowlevel.spectral_centroid.mean',
  'lowlevel.spectral_complexity.mean',
  'lowlevel.zerocrossingrate.mean'
]


def extract_mfcc(dataset='train'):
  f = open(data_path + dataset + '_list.txt', 'r')

  i = 0
  for file_name in f:
    # progress check
    i = i + 1
    if not (i % 10):
      print(i)

    # load audio file
    file_name = file_name.rstrip('\n')
    file_path = data_path + file_name
    # print file_path

    features, features_frames = es.MusicExtractor(lowlevelStats=['mean'],
                                                  rhythmStats=['mean'],
                                                  tonalStats=['mean'])(file_path)

    mfcc = features_frames['lowlevel.mfcc']
    mfcc0, mfcc1 = mfcc[:44].mean(axis=0), mfcc[44:88].mean(axis=0)
    mfcc2, mfcc3 = mfcc[88:132].mean(axis=0), mfcc[132:].mean(axis=0)

    mel = features_frames['lowlevel.melbands']
    mel0, mel1 = mel[:44].mean(axis=0), mel[44:88].mean(axis=0)
    mel2, mel3 = mel[88:132].mean(axis=0), mel[132:].mean(axis=0)

    selected = np.concatenate([mfcc0, mfcc1, mfcc2, mfcc3, mel0, mel1, mel2, mel3,
                               np.array([features[f] for f in FEATURES])])

    # save mfcc as a file
    file_name = file_name.replace('.wav', '.npy')
    save_file = mfcc_path + file_name

    if not os.path.exists(os.path.dirname(save_file)):
      os.makedirs(os.path.dirname(save_file))
    np.save(save_file, selected)

  f.close()


if __name__ == '__main__':
  extract_mfcc(dataset='train')
  extract_mfcc(dataset='valid')
  extract_mfcc(dataset='test')
