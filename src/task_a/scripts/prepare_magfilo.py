import os
import json
import cv2
import numpy as np
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm

def process_split(data_dir, split_name):
    # Construct paths
    json_path = os.path.join(data_dir, split_name, f"MAGFiLO_1.0_Annotations_kaggle2026_{split_name}.json")
    if not os.path.exists(json_path):
        print(f"Skipping {split_name} - no annotations found at {json_path}")
        return

    print(f"Loading annotations for {split_name} from {json_path}...")
    coco = COCO(json_path)
    
    img_dir = os.path.join(data_dir, split_name, f"{split_name}_images")
    mask_dir = os.path.join(data_dir, split_name, f"{split_name}_masks")
    spine_dir = os.path.join(data_dir, split_name, f"{split_name}_spines")
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(spine_dir, exist_ok=True)
    
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images in {split_name}. Rasterizing masks and spines...")
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        h, w = img_info.get('height', 1024), img_info.get('width', 1024)
        
        # Initialize blank masks
        mask = np.zeros((h, w), dtype=np.uint8)
        spine = np.zeros((h, w), dtype=np.uint8)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            if 'segmentation' in ann and type(ann['segmentation']) == list:
                # Polygon format
                for seg in ann['segmentation']:
                    if len(seg) >= 6: # Need at least 3 points
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255)
            
            # Look for spines encoded as keypoints (skeleton) in standard COCO:
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                kpts = np.array(ann['keypoints']).reshape((-1, 3))
                # Only use visible/labeled keypoints (v > 0)
                valid_kpts = [k[:2].astype(np.int32) for k in kpts if k[2] > 0]
                if len(valid_kpts) > 1:
                    cv2.polylines(spine, [np.array(valid_kpts)], isClosed=False, color=255, thickness=1)
            
            # Look for spines provided as an extra custom segmentation field:
            if 'spine' in ann and type(ann['spine']) == list:
                for seg in ann['spine']:
                    if len(seg) >= 4:
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.polylines(spine, [poly], isClosed=False, color=255, thickness=1)
                        
        # Save the generated masks
        base_name = os.path.splitext(img_info['file_name'])[0]
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}.png"), mask)
        cv2.imwrite(os.path.join(spine_dir, f"{base_name}.png"), spine)
        
    print(f"Finished processing {split_name}. Masks saved to {mask_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess MAGFiLO COCO JSON into PNG masks and spines")
    parser.add_argument("--data_dir", type=str, default="data/raw/MAGFiLO/MAGFiLO_1.0_Kaggle_2026")
    args = parser.parse_args()
    
    for split in ['train', 'test']:
        process_split(args.data_dir, split)

if __name__ == "__main__":
    main()
