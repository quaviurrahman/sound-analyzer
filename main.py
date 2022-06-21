import os, sklearn.cluster
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis.audioSegmentation import labels_to_segments
from pyAudioAnalysis.audioTrainTest import normalize_features
import numpy as np
import scipy.io.wavfile as wavfile
import IPython

# read signal and get normalized segment feature statistics:
input_file = "data/another_cup_of_coffee.wav"
fs, x = read_audio_file(input_file)
mt_size, mt_step, st_win = 5, 0.5, 0.1
[mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
round(fs * st_win), round(fs * st_win * 0.5))
(mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
mt_feats_norm = mt_feats_norm[0].T

# perform clustering
n_clusters = 5
x_clusters = [np.zeros((fs, )) for i in range(n_clusters)]
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(mt_feats_norm.T)
cls = k_means.labels_

# save clusters to concatenated wav files
segs, c = labels_to_segments(cls, mt_step)

# convert flags to segment limits
for sp in range(n_clusters):                
    count_cl = 0
    for i in range(len(c)):     

# for each segment in each cluster (>2 secs long)
        if c[i] == sp and segs[i, 1]-segs[i, 0] > 2:
            count_cl += 1
          
# get the signal and append it to the cluster's signal (followed by some silence)
            cur_x = x[int(segs[i, 0] * fs): int(segs[i, 1] * fs)]
            x_clusters[sp] = np.append(x_clusters[sp], cur_x)
            x_clusters[sp] = np.append(x_clusters[sp], np.zeros((fs,)))

# write cluster's signal into a WAV file
    print(f'cluster {sp}: {count_cl} segments {len(x_clusters[sp])/float(fs)} sec total dur')        
    wavfile.write(f'cluster_{sp}.wav', fs, np.int16(x_clusters[sp]))
    IPython.display.display(IPython.display.Audio(f'cluster_{sp}.wav'))
