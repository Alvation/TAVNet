import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import soundfile


def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")



def process_audio(audio_path):
    waveform, sample = soundfile.read(audio_path)
    return waveform


def process_video(video_path: str=None):
    feats = load_video(video_path)
    return feats


def save_data(item_list,split):
    for i,item in enumerate(tqdm(item_list, leave=False, desc=f"creat list of {split}", ncols=75)):
        video_feature = process_video(video_path = item["video_path"])
        # audio_feature = process_audio(item["audio_path"])
        video_npy_file = os.path.splitext(item["video_path"])[0]+'.npy'
        # audio_npy_file = os.path.splitext(item["audio_path"])[0]+'.npy'
        np.save(video_npy_file, video_feature)
        # np.save(audio_npy_file, audio_feature)




def read_file(tsv_path,wrd_path):
    print(f"tsv_path is {tsv_path}")
    print(f"wrd_path is {wrd_path}")
    item_list=[]
    with open(tsv_path,'r') as tsv, open(wrd_path,'r') as wrd:
        for index, (tsv_item, wrd_item) in enumerate(zip(tsv.readlines()[1:],wrd.readlines())):
            tmp_dict = {}
            items = tsv_item.strip().split("\t")
            tmp_dict["id"] = items[0]
            tmp_dict["video_path"] = items[1]
            tmp_dict["audio_path"] = items[2]
            tmp_dict["video_frames"] = items[3]
            tmp_dict["video_frames"] = items[4]
            tmp_dict["text"] = wrd_item.strip()
            item_list.append(tmp_dict)
    return item_list





def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 preprocess pretrain dir', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='tsv and wrd file root dir')
    args = parser.parse_args()

    
    split_list = ["train", "valid", "test"]

    for split in split_list:
        tsv_path = os.path.join(args.lrs3, split+".tsv")
        wrd_path = os.path.join(args.lrs3, split+".wrd")
        item_list = read_file(tsv_path, wrd_path)
        save_data(item_list, split)

        

if __name__ == "__main__":
    main()


    

