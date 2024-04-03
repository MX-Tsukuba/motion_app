import cv2
import numpy as np

def adjust_keypoints(keypoints, desired_right_foot_pos=(200, 900), desired_neck_pos=(300, 100),MODE_="MOT16"):
    if MODE_ == "MOT16":
        right_foot_index=11 
        neck_index=0
    elif MODE_ == "COCO":
        right_foot_index=10
        neck_index=1

    # Extract the original positions of the right foot and neck
    right_foot_pos = keypoints[right_foot_index] if keypoints[right_foot_index] is not None else (0, 0)
    neck_pos = keypoints[neck_index] if keypoints[neck_index] is not None else (0, 0)
    
    if right_foot_pos == (0, 0) or neck_pos == (0, 0):
        # raise ValueError("Required keypoints (neck or right foot) not found.")
        return "skip"
    
    # Calculate the necessary translation for the right foot
    translation_x_foot = desired_right_foot_pos[0] - right_foot_pos[0]
    translation_y_foot = desired_right_foot_pos[1] - right_foot_pos[1]
    
    # Apply the translation to all keypoints
    translated_keypoints = []
    for point in keypoints:
        if point is not None:
            # pointがNoneでない場合、xとyをアンパック
            x, y = point
            # 計算した平行移動を適用
            translated_x = x + translation_x_foot
            translated_y = y + translation_y_foot
            translated_keypoints.append((translated_x, translated_y))
        else:
            translated_keypoints.append((None, None))
        # Recalculate the position of the neck after translation
        translated_neck_pos = translated_keypoints[0]
        if translated_neck_pos[1] == None:
            return "skip"
    
    # Calculate the scale factor based on the neck's desired and current position
    # This assumes that the x scale factor is the same as the y scale factor, to maintain the aspect ratio
    
    scale_factor = (desired_neck_pos[1] - desired_right_foot_pos[1]) / (translated_neck_pos[1] - desired_right_foot_pos[1])
    
    # Apply the scale factor to all keypoints
    adjusted_keypoints = [(desired_right_foot_pos[0] + (x - desired_right_foot_pos[0]) * scale_factor, 
                           desired_right_foot_pos[1] + (y - desired_right_foot_pos[1]) * scale_factor) 
                          if x is not None and y is not None else (None, None) 
                          for x, y in translated_keypoints]
    
    return adjusted_keypoints


# Normalization.py
def calculate_scale_and_translation(keypoints, desired_right_foot_pos=(200, 900), desired_neck_pos=(300, 100), MODE_="MOT16"):
    if MODE_ == "MOT16":
        right_foot_index=11 
        neck_index=0
    elif MODE_ == "COCO":
        right_foot_index=10
        neck_index=1

    # Extract the original positions of the right foot and neck
    right_foot_pos = keypoints[right_foot_index] if keypoints[right_foot_index] is not None else (0, 0)
    neck_pos = keypoints[neck_index] if keypoints[neck_index] is not None else (0, 0)
    
    if right_foot_pos == (0, 0) or neck_pos == (0, 0):
        # raise ValueError("Required keypoints (neck or right foot) not found.")
        return "skip"
    
    # Calculate the necessary translation for the right foot
    translation_x_foot = desired_right_foot_pos[0] - right_foot_pos[0]
    translation_y_foot = desired_right_foot_pos[1] - right_foot_pos[1]

    translation = (translation_x_foot,translation_y_foot)
        
    if keypoints[0] is not None:
        x,y = keypoints[0]
        translated_x = x + translation_x_foot
        translated_y = y + translation_y_foot
        translated_neck_pos = (translated_x,translated_y)
        if translated_neck_pos[1] == None:
            return "skip"
    else:
        return "skip"
    
    scale_factor = (desired_neck_pos[1] - desired_right_foot_pos[1]) / (translated_neck_pos[1] - desired_right_foot_pos[1])

    return scale_factor, translation
        

def adjust_keypoints_with_scale_and_translation(keypoints, scale_factor, translation, desired_right_foot_pos=(200, 900)):
    # Apply the translation to all keypoints
    translated_keypoints = []
    for point in keypoints:
        if point is not None:
            # pointがNoneでない場合、xとyをアンパック
            x, y = point
            # 計算した平行移動を適用
            translated_x = x + translation[0]
            translated_y = y + translation[1]
            translated_keypoints.append((translated_x, translated_y))
        else:
            translated_keypoints.append((None, None))
        # Recalculate the position of the neck after translation
        translated_neck_pos = translated_keypoints[0]
        if translated_neck_pos[1] == None:
            return "skip"
        
    adjusted_keypoints = [(desired_right_foot_pos[0] + (x - desired_right_foot_pos[0]) * scale_factor, 
                           desired_right_foot_pos[1] + (y - desired_right_foot_pos[1]) * scale_factor) 
                          if x is not None and y is not None else (None, None) 
                          for x, y in translated_keypoints]
    
    return adjusted_keypoints


def plot_transformed_image_and_keypoints(image, keypoints, adjusted_keypoints, output_size=(600, 1000), MODE_="MOT16", usePoints=True):
    if MODE_ == "MOT16":
        right_foot_index = 11 
        neck_index = 0
    elif MODE_ == "COCO":
        right_foot_index = 10
        neck_index = 1

    transformed_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # キーポイントがNoneかどうかのチェックを追加
    if None not in keypoints[neck_index]  and None not in adjusted_keypoints[neck_index] and None not in adjusted_keypoints[right_foot_index]:
        right_foot_orig = keypoints[right_foot_index]
        right_foot_adjusted = adjusted_keypoints[right_foot_index]
        
        # 計算されたscale_factorとtranslationを使用して変換を適用する
        scale_factor = np.linalg.norm(np.subtract(adjusted_keypoints[neck_index], right_foot_adjusted)) / np.linalg.norm(np.subtract(keypoints[neck_index], right_foot_orig))
        translation = np.subtract(right_foot_adjusted, np.multiply(right_foot_orig, scale_factor))

        M = np.float32([[scale_factor, 0, translation[0]], [0, scale_factor, translation[1]]])
        transformed_image = cv2.warpAffine(image, M, (output_size[0], output_size[1]))

    if (usePoints):
        for point in adjusted_keypoints:
            if None not in point:
                x, y = point
                cv2.circle(transformed_image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)

    return transformed_image


def calculate_trim_coordinates(keypoints, marginTL, marginTT, marginBR, marginBB, MODE_="MOT16"):
    if MODE_ == "MOT16":
        right_hip_index = 11
        neck_index = 0
    elif MODE_ == "COCO":
        right_hip_index = 10
        neck_index = 1

    # Extract the required keypoints
    right_hip = keypoints[right_hip_index] if keypoints[right_hip_index] is not None else (0, 0)
    neck = keypoints[neck_index] if keypoints[neck_index] is not None else (0, 0)
    
    # Make sure we have valid coordinates for right hip and neck
    if right_hip == (0, 0) or neck == (0, 0):
        return None, None

    # Calculate the bounding box that includes the whole body based on the model of keypoints
    top_left_x = min(right_hip[0], neck[0]) - marginTL
    top_left_y = min(right_hip[1], neck[1]) - marginTT
    bottom_right_x = max(right_hip[0], neck[0]) + marginBR
    bottom_right_y = max(right_hip[1], neck[1]) + marginBB

    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

def trim_image(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def trim_image(image, top_left, bottom_right):
    """指定された座標に基づいて画像をトリミングする。"""
    # トリミング範囲を指定して画像を切り出す

    trimmed_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return trimmed_image
