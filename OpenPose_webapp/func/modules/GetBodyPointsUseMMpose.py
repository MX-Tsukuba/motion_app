import logging
import os
import csv
from argparse import ArgumentParser

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

def GetBodyPointsUseMMpose(frame_):

    model = init_model(
        r'OpenPose_webapp\mmPoseModels\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
        r'OpenPose_webapp\mmPoseModels\td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
        device='cuda:0',
        cfg_options=None)
        
    
    batch_results = inference_topdown(model, frame_)
    results = merge_data_samples(batch_results)

    return results.pred_instances.keypoints[0],results.pred_instances