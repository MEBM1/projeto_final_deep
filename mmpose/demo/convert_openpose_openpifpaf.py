import json
import mmengine
from tqdm import tqdm
import os

root_folder = "./results_folder/"
origin_folder = "results"

all_files = []
# get all files from root dir
for path, subdirs, files in os.walk(root_folder+origin_folder):
    for name in files:
        if name[-4:] == 'json':
            all_files.append(os.path.join(path, name))

#file = "/set01/video_0003/00143.png.predictions.json"

for file in tqdm(all_files):

    file_path = file
    out_file = root_folder + "results_convert/" + file[24:]
    #out_file = root_folder + "results_convert/" + file[30:]

    with open(file_path, "r") as f:
        data = json.load(f)

    out_folder = out_file[:51]
    #out_folder = out_file[:56]

    out = []
    for j, i in enumerate(data['instance_info']):
        instance_dict = {}
        keypoints = []
        for l, k in enumerate(i['keypoints']):
            keypoints.extend(k)
            keypoints.append(i['keypoint_scores'][l])
        instance_dict['keypoints'] = keypoints
        instance_dict['bbox'] = i['bbox'][0]
        instance_dict['category_id'] = 1
        instance_dict['score'] = i['bbox_score']
        out.append(instance_dict)

    mmengine.mkdir_or_exist(out_folder)
    with open(out_file, 'w') as f:
        json.dump(out, f)