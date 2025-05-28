import pdb
import cv2
import numpy as np
import supervision as sv
import pdb
import os
from tqdm import tqdm
import torch
import torchvision
from groundingdino.util.inference import Model
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor

def label2name(type,label):
    if type=='airplane':
        if label ==1:
            return 'body'
        elif label ==2:
            return 'wing'
        elif label ==3:
            return 'tail'
        elif label ==4:
            return 'engine'
    if type=='chair':
        if label ==1:
            return 'back'
        if label ==2:
            return 'seat'
        if label ==3:
            return 'leg'
        if label==4:
            return 'arm'
    if type=='table':
        if label==1:
            return 'top'
        if label==2:
            return 'leg'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Predict classes and hyper-param for GroundingDINO
CLASSES=None
CLASSES_to_label=None

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

def get_seg(source_image_path, res_path):
    # load image
    image = cv2.imread(source_image_path)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    # pdb.set_trace()
    # labels = [
    #     f"\n{CLASSES[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _ 
    #     in detections]
    # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    detections_tosave = {'boxes': detections.xyxy, 'scores': detections.confidence, 'labels':[CLASSES_to_label[i] for i in detections.class_id]}
    np.save(res_path.replace('png', 'npy'), detections_tosave)
    print(f"After NMS: {len(detections.xyxy)} boxes")
    
    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    np.save(res_path.replace('.png', '_mask.npy'), detections.mask)
    
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    # labels = [ f"{CLASSES_to_label[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ ,_  in detections]
    labels = [f"{CLASSES_to_label[class_id]} {confidence:0.2f}" 
              for confidence, class_id in zip(detections.confidence, detections.class_id)]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    cv2.imwrite(res_path, annotated_image)

def get_seg_dir(gen_path, output_path):
    for f in os.listdir(gen_path):
        if 'gen' in f and '.png' in f:
            get_seg(os.path.join(gen_path, f), os.path.join(output_path, f))
import json
#TBD: use config
# config_path = "/home/cli7/yhc_Workspace/SPA/real_exp/config.json"
# # instance_name = 'bird_general'
# with open(config_path) as f:
#     config_data = json.load(f)
# instance_name = config_data['world_info']['current_instance']
# data_path = "/home/cli7/yhc_Workspace/SPA/real_exp/depth/bird_general.scad"  
# data_path = "../examples/stage1/working_dir/bike_with_holder.scad"
# res_dir = "../examples/stage3"
dataname = 'simplecar'
data_path = f"../outputs/{dataname}/stage1/working_dir/{dataname}_general_allnodes.scad"
res_dir = f"../outputs/{dataname}/stage3"
cubeid = f"{dataname}_general_allnodes"

# CLASSES_to_label = ["wheel", "frame", "seat", "handlebar"]
# CLASSES_to_label = ["Frame", "Front Wheel", "Rear Wheel", "Pedal", "Handlebar", "Saddle"]

# CLASSES_to_label = ["locomotive", "wheels", "car", "coupler", "smokestack"] # train
# CLASSES_to_label = ["head", "body", "eyes", "beak", "wing", "tail", "feet", "hat", "tie", "jacket"] # wintertux
# CLASSES_to_label = ["Head", "Body", "Eyes", "Nose", "Mouth", "Arms", "Buttons", "Hat", "Scarf", "Broom"] # snowman
# CLASSES_to_label = ["Trunk", "Branches", "Ornaments", "Stand", "Leaves", "Top"] # christmastree
# CLASSES_to_label = ["Head", "Body", "Eyes", "Legs", "Mouth"] # creeper
# CLASSES_to_label = ["Trunk", "Ears", "Tusks", "Legs", "Feet", "Tail", "Eyes", "Body"] # elephant
# CLASSES_to_label = ["Blade", "Base"] # fan
# CLASSES_to_label = ["Seat", "Backrest", "Legs"] # lowchair
# CLASSES_to_label = ["Head", "Body", "Eyes", "Beak", "Wings", "Tail"] # rubberduck
CLASSES_to_label = ["Body", "Wheels", "Windows", "Doors", "Lights"] # simplecar
typediscribe = dataname
CLASSES=[f'{name} of {typediscribe}' for name in CLASSES_to_label]
# CLASSES=[f'{name} ' for name in CLASSES_to_label]
# pdb.set_trace()
for ry in [i*36 for i in range(10)]:
    gen_path = f'{data_path}/rz={ry}'
    output_path = f'{res_dir}/{os.path.basename(data_path)}/rz=f{ry}'
    os.makedirs(gen_path, exist_ok=True)
    os.makedirs(f'{res_dir}/{os.path.basename(data_path)}',exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    get_seg_dir(gen_path, output_path)