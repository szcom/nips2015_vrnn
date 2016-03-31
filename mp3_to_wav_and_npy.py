import os
import scipy.io.wavfile as wav
import numpy as np
import fnmatch

def convert(data_path):
    data_matches = []
    for root, dir_names, file_names in os.walk(data_path):
        for filename in fnmatch.filter(file_names, '*.mp3'):
            data_matches.append(os.path.join(root, filename))
            fname = os.path.join(root, filename)
            oname = os.path.join(root, filename.replace('mp3', 'wav'))
            cmd = 'gst-launch-1.0 filesrc location={0} \! decodebin \!audioconvert \! audioresample \! "audio/x-raw,channels=1,rate=16000,format=S16LE" \! wavenc \! filesink location={1}'.format(fname,oname )
            print(cmd)
            os.system(cmd)
            data = wav.read(oname)
            npy = os.path.join(root, filename.replace('mp3', 'npy'))
            np.save(npy, data[1])

if __name__ == "__main__":
    data_path = "./blizzard/train/unsegmented/"
    convert(data_path)
