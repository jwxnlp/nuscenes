# -*- coding: utf-8 -*-
# @author: Jiang Wei
# @date: 2024/08/24
#**************************************************************************************
import os, argparse, tqdm, cv2, time
import numpy as np

import multiprocessing

from nuscenes import NuScenes
from nuscenes.utils import splits
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Visualize Raw GTs of BEV Lane",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #----------------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        "--data_dir", type=str, default=r"/mnt/luci-nas/jiangwei/projects/nuscenes/data/nuscenes", help=""
    )
    parser.add_argument(
        "--version", type=str, default=r"v1.0-mini", help=""
    )
    parser.add_argument(
        "--save_dir", type=str, default=r"/mnt/luci-nas/jiangwei/projects/nuscenes/data", help=""
    )
    
    # multi_processing
    parser.add_argument(
        "--n_proc", type=int, default=1, help="number of processing"
    )
    return parser.parse_args()

# functions
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_scenes(nusc):
    """"""
    scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        scenes.append(scene)
    return scenes


def split_scenes(version):
    """
    params:

    return: 
        scene's names
    """
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError(f"ERROR: [ {version} ]: unknown")
    
    print(f"--- number of scenes: {len(train_scenes)} training scenes, {len(val_scenes)} validation scenes! ")
    return train_scenes, val_scenes


def process(proc_ordinal, args, scene_recs, nusc):
    """"""
    CAMS = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", 
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
    ]
    for scene_rec in tqdm.tqdm(scene_recs):
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        
        scene_name = scene_rec["name"]
        video_path = os.path.join(args.save_dir, f"{scene_name}.mp4")
        
        fps = 10
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        if os.path.exists(video_path):
            os.remove(video_path)
        resolution = (1600*3, 900*2)
        writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)
        
        N = 0
        while True:
            # process lidar
            sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            # print(f"--- {lidar_path}")
            
            # process vision
            #--------------------------------------------------------------------------------------
            img_dict = dict()
            for cam_name in CAMS:
                sd_rec = nusc.get('sample_data', sample_rec['data'][cam_name])
                img_path, _, _ = nusc.get_sample_data(sd_rec['token'])
                
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"ERROR: [ {img_path} ]: Not Exist!")
                img = cv2.imread(img_path)
                if img is None:
                    raise Exception(f"ERROR: [ {img_path} ]: Fail to Decode!")
                # print(f"--- {cam_name} {img.shape} {img_path}")
                img_dict[cam_name] = img
            # prepare single frame
            row1 = cv2.hconcat([
                img_dict["CAM_FRONT_LEFT"], img_dict["CAM_FRONT"], img_dict["CAM_FRONT_RIGHT"]])
            row2 = cv2.hconcat([
                img_dict["CAM_BACK_LEFT"], img_dict["CAM_BACK"], img_dict["CAM_BACK_RIGHT"]])
            frame = cv2.vconcat([row1, row2])
            
            
            writer.write(frame)
            
            N += 1
            
            if sample_rec['next'] != '':
                sample_rec = nusc.get('sample', sample_rec['next'])
            else:
                break
        
        writer.release()
        
        assert N == scene_rec["nbr_samples"], \
            f"ERROR: [ {scene_name} ]: wrong number of samples!"
        
        exit(-1)
            
    return
    
    
# main function  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(args):
    """"""
    
    # initialize NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.data_dir, verbose=True)
    
    
    
    
    
    print(f"--- total scene num: {len(nusc.scene)}")
    
    scenes = check_scenes(nusc)
    
    print(f"--- exist scene num: {len(scenes)}")
    
    print(scenes[0])
    
    # prepare related folder
    timestamp = time.strftime("%Y%m%d", time.localtime())
    args.save_dir = os.path.join(args.save_dir, f"{args.version}_vis_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # multiprocessing: consuming
    #----------------------------------------------------------------------
    p_list = []
    for i in range(args.n_proc):
        p = multiprocessing.Process(
            target=process,
            args=(i, args, scenes[i::args.n_proc], nusc))
        p.start()
        print(f"ID of process p{i}: {p.pid}, {len(scenes[i::args.n_proc])}")
        p_list.append(p)

    for p in p_list:
        p.join()
    
    return

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    main(parse_args())