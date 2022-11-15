import os
import tqdm
import argparse
import numpy as np

from collections import defaultdict

import pandas as pd
from pycocotools.coco import COCO

from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
gulim = fm.FontProperties(fname='gulim.ttc')

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--score_csv', default="scores.csv", help="best model dir")
    parser.add_argument('--target_csv', default="target.csv", help="target csv")

    return parser.parse_args()

def _get_metrics(targets, preds):
    conf_matrix = confusion_matrix(targets, preds, normalize='true') 
    accuracy = sum(np.array(targets) == np.array(preds)) / len(targets)
    
    return accuracy, conf_matrix

if __name__ == "__main__":
    
    # 우선순위 순서로
    # 바꾸면 우선순위 바뀌어서 적용됨
    names = ["정상", "홍계", "배꼽", "피부손상F", "피부손상C", "피부손상S", "골절C", "가슴멍", "날개멍", "다리멍"]
    
    # threshold
    # 전부다 0.5로 설정하면 기존과 같음
    class_score_thresh_dict = {}
    class_score_thresh_dict["정상"] = 0.5
    class_score_thresh_dict["홍계"] = 0.5
    class_score_thresh_dict["배꼽"] = 0.5
    class_score_thresh_dict["피부손상F"] = 0.5
    class_score_thresh_dict["피부손상C"] = 0.5
    class_score_thresh_dict["피부손상S"] = 0.5
    class_score_thresh_dict["골절C"] = 0.5
    class_score_thresh_dict["가슴멍"] = 0.5
    class_score_thresh_dict["날개멍"] = 0.5
    class_score_thresh_dict["다리멍"] = 0.5
    print(class_score_thresh_dict)
    
    args = get_args()
    
    score_dict = pd.read_csv(args.score_csv, index_col='image_path').T.to_dict()
    target_dict = pd.read_csv(args.target_csv, index_col='image_path').T.to_dict()
    
    tuned_dict = {}
    for img_path in target_dict.keys():
        new_label = 10
        for name in names:
            if new_label == 10 and score_dict[img_path][name] > class_score_thresh_dict[name]:
                new_label = names.index(name)+1
        
        tuned_dict[img_path] = {}
        tuned_dict[img_path]['predict'] = names[new_label-1]
        tuned_dict[img_path]['category_id'] = new_label
    
    preds = []
    targets = []
    for img_path in target_dict.keys():
        if img_path not in tuned_dict or img_path not in target_dict:
            print(f"{img_path} is missing")
        preds.append(names.index(tuned_dict[img_path]['predict']))
        targets.append(names.index(target_dict[img_path]['predict']))
    
    accuracy, confusion_matrix = _get_metrics(targets, preds)
    print("mean recall : ", sum(map(lambda i: confusion_matrix[i][i], range(len(confusion_matrix))))/len(confusion_matrix))
    print(f"accuracy : {accuracy*100:.2f} %")
    
    
    # 시각화
    fig, ax = plt.subplots()
    plt.title(f"accuracy : {accuracy*100:.2f} %", fontproperties = gulim, fontsize=15)
    plt.viridis()
    
    im = ax.imshow(np.array(confusion_matrix))
    
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names,fontproperties = gulim, rotation=45)
    ax.set_yticklabels(names,fontproperties = gulim)
    plt.xlabel("predict")
    plt.ylabel("target")

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{confusion_matrix[i][j]:.2f}", ha="center", va="center", color="w")

    fig.tight_layout()
    plt.colorbar(im,ax=ax)
    plt.savefig(f"tuned_conf_mat.png")