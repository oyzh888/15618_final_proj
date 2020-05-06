from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
from collections import OrderedDict
import json_tricks as json
import time

import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform


YD_KEYPOINT_INDEXES = {
    0: 'Nose',
    1: 'Neck',
    2: 'RShoulder',
    3: 'RElbow',
    4: 'RWrist',
    5: 'LShoulder',
    6: 'LElbow',
    7: 'LWrist',
    8: 'RHip',
    9: 'RKnee',
    10: 'RAnkle',
    11: 'LHip',
    12: 'LKnee',
    13: 'LAnkle',
    14: 'REye',
    15: 'LEye',
    16: 'REar',
    17: 'LEar'
}

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

MPII_KEYPOINT_INDEXES = {
    0: 'rankle',
    1: 'rknee',
    2: 'rhip',
    3: 'lhip',
    4: 'lknee',
    5: 'lankle',
    6: 'pelvis',
    7: 'throax',
    8: 'upper_neck',
    9: 'head_top',
    10: 'rwrist',
    11: 'relbow',
    12: 'rshoulder',
    13: 'lshoulder',
    14: 'lelbow',
    15: 'lwrist'
}

YD2COCO = [0,-1,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3] 

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    transformed_img = transformed_img.cuda()
    pred = model([transformed_img])  # Pass the image to the model
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        model_input = model_input.cuda()
        output = pose_model(model_input)
        preds, confidences = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))
        return preds, confidences


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5
    
    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200
    
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='output/'):
    #pose_dir = prefix+'poses/'
    #box_dir = prefix+'boxes/'
    pose_dir = os.path.join(prefix, 'poses')
    box_dir = os.path.join(prefix, 'boxes')
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    if os.path.exists(box_dir) and os.path.isdir(box_dir):
        shutil.rmtree(box_dir)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    return pose_dir, box_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='output/example_video/')
    parser.add_argument('--inferenceFps', type=int, default=30)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    video_name = args.videoFile.split("/")[-1].strip(".mp4")
    print(video_name)

    pose_dir, box_dir = prepare_output_dirs(args.outputDir)
    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    json_output_filename = os.path.join(args.outputDir, '{}.json'.format(video_name))
    csv_output_rows = []
    video_output_json = OrderedDict()
    #video_name = args.videoFile.split(".")[0]
    
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    box_model.eval()

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    #pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS).cuda()
    pose_model = pose_model.cuda()

    # Loading an video
    
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)
    
    #if fps < args.inferenceFps:
    #    print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
    #    exit()
    
    every_nth_frame = round(fps/args.inferenceFps)

    success, image_bgr = vidcap.read()
    count = 0

    person_detection_time = []
    pose_estimation_time = []
    while success:
        
        print("Processing frame %d" % count)
        
        #if count % every_nth_frame != 0:
        #    success, image_bgr = vidcap.read()
        #    count += 1
        #    continue
        
        image = image_bgr[:, :, [2, 1, 0]]
        count_str = str(count).zfill(4)
        
        t1 = time.time()
        # object detection box
        pred_boxes = get_person_detection_boxes(box_model, image, threshold=0.7)
        t2 = time.time()
        if args.writeBoxFrames:
            image_bgr_box = image_bgr.copy()
            for box in pred_boxes:
                cv2.rectangle(image_bgr_box, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates
            cv2.imwrite(os.path.join(box_dir, 'box%s.jpg' % count_str), image_bgr_box)
        if not pred_boxes:
            success, image_bgr = vidcap.read()
            count += 1
            continue
        
        t3 = time.time()
        # pose estimation
        box = pred_boxes[0]  # assume there is only 1 person
        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
        pose_preds, pose_confidences = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
        t4 = time.time()

        new_csv_row = [count]
        frame_keypoints = []
        frame_score = 0.0
        for pose_c, mat in zip(pose_confidences[0], pose_preds[0]):
            x_coord, y_coord = int(mat[0]), int(mat[1])
            cv2.circle(image_bgr, (x_coord, y_coord), 4, (255, 0, 0), 2)
            new_csv_row.extend([x_coord, y_coord])
            #frame_keypoints.extend([x_coord, y_coord, pose_c[0]])
            #frame_score += pose_c[0]

        for YD_index, COCO_index in enumerate(YD2COCO):
            if COCO_index != -1:
                mat = pose_preds[0, COCO_index]
                pose_c = pose_confidences[0, COCO_index, 0]
                x_coord, y_coord = int(mat[0]), int(mat[1])
                frame_keypoints.extend([x_coord, y_coord, pose_c])
                frame_score += pose_c
            else:
                frame_keypoints.extend([-1, -1, -1])

        frame_score /= (np.array(YD2COCO) != -1).sum()
        video_output_json["_".join(["frame", str(count)])] = [{'score': frame_score, "keypoints": frame_keypoints}]
        csv_output_rows.append(new_csv_row)
        cv2.imwrite(os.path.join(pose_dir, 'pose%s.jpg' % count_str), image_bgr)

        
        # get next frame
        success, image_bgr = vidcap.read()
        count += 1
        
        person_detection_time.append(t2-t1)
        pose_estimation_time.append(t4-t3)


    print(sum(person_detection_time) / len(person_detection_time))
    print(sum(pose_estimation_time) / len(pose_estimation_time))    
    
    # write csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)
    
    json.dump(video_output_json, json_output_filename)
    
    
    s = "/data/hezibin/ffmpeg/ffmpeg -y -r "\
              + str(args.inferenceFps)\
              + " -pattern_type glob -i '"\
              + pose_dir\
              + "/*.jpg' -c:v libx264 -vf fps="\
              + str(args.inferenceFps)+" -pix_fmt yuv420p {} 2>/dev/null".format(os.path.join(args.outputDir, "test.mp4"))
    os.system(s)
    


if __name__ == '__main__':
    main()
