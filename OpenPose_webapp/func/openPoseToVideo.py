# 動画をのパスを入力として骨格推定を行うコード

import cv2
import os
from datetime import datetime
from func.modules import GetBodyPoints, FrameAddSkeleton,  Normalization, saveFunc

def SkeletalEstimation(video_path_,TL_position,BR_position,MODE_="MOT16"):

    # 現在の日時を取得
    now = datetime.now()

    # 文字列に変換するためのフォーマットを指定
    # 例: '2024-04-03 14:59:00'
    date_string = now.strftime('%Y-%m-%d-%H-%M-%S')
    
    landmarks_data = []
    probs_data = []
    frames_with_skelton_data = []
    edit_raw_frames_data = []

    # 動画からフレーム画像を取得
    cap = cv2.VideoCapture(video_path_)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            trimmed_image = Normalization.trim_image(frame, TL_position,BR_position)

            copied_image = trimmed_image.copy()

            points,probs = GetBodyPoints.GetBodyPoints(trimmed_image, MODE_=MODE_)

            skeleton_frame = FrameAddSkeleton.FrameAddSkeleton(trimmed_image, points, MODE_=MODE_)

            edit_raw_frames_data.append(copied_image)
            landmarks_data.append(points)
            frames_with_skelton_data.append(skeleton_frame)
            probs_data.append(probs)
        else: 
            # ここ、breakにしないとwhile分から抜け出せなくなる。
            break
        


    # 取得されたデータ(座標データとフレーム画像)を保存
    base_dir = r"OpenPose_webapp\static\datas"

    save_dir = os.path.join(base_dir, date_string) #現在の時刻を名前としてdatas直下に作成するディレクトリのパス
    frames_dir = os.path.join(save_dir, "frames_with_skelton") #撮影された画像群を保存するディレクトリのパス
    raw_frames_dir = os.path.join(save_dir, "raw_frames") #撮影された画像群を保存するディレクトリのパス
    csv_path = os.path.join(save_dir, "landmarks.csv")
    probs_path = os.path.join(save_dir, "probs.csv")

    saveFunc.save_landmarks_to_csv(csv_path, landmarks_data)
    saveFunc.save_frames_to_directory(frames_dir, frames_with_skelton_data)
    saveFunc.save_frames_to_directory(raw_frames_dir, edit_raw_frames_data)
    saveFunc.save_probs_to_csv(probs_path, probs_data)

    # 作業完了後、キャプチャを解放しウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

    return date_string