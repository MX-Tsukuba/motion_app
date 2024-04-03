import cv2
import os
import csv
import datetime

def SaveFrameDatas(base_dir_,file_name_,landmarks_data_,frames_data_):
    save_dir = os.path.join(base_dir_, file_name_) #現在の時刻を名前としてDBディレクトリ直下に作成するディレクトリのパス
    frames_dir = os.path.join(save_dir, "frames") #撮影された画像群を保存するディレクトリのパス
    dir_exists = os.path.exists(frames_dir)

    # if (dir_exists == False):
    os.makedirs(frames_dir, exist_ok=True) #ディレクトリの作成
    csv_path = os.path.join(save_dir, "landmarks.csv") #取得された座標群を保存するパス

    # CSVファイルにランドマークデータを保存
    with open(csv_path, "w", newline="") as f:
        # CSVライターオブジェクトの作成
        writer = csv.writer(f)
        # ヘッダー行の書き込み
        header = [f'{xy}{i}' for i in range(25) for xy in ['x', 'y']]
        writer.writerow(header)
        # データ行の書き込み
        for row in landmarks_data_:
            # 各キーポイントペアの"x", "y"をフラットなリストに変換して書き込み
            flat_row = [coordinate for pair in row for coordinate in pair]
            writer.writerow(flat_row)

    # フレームデータをファイルに保存
    for i, frame in enumerate(frames_data_):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    print(f"Data saved to {save_dir}")

def save_landmarks_to_csv(csv_path, landmarks_data):
    # ファイルが保存されるディレクトリを取得
    directory = os.path.dirname(csv_path)
    # ディレクトリが存在しない場合は作成
    os.makedirs(directory, exist_ok=True)
    
    # CSVファイルにランドマークデータを保存
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # ヘッダー行の書き込み
        header = [f'{xy}{i}' for i in range(25) for xy in ['x', 'y']]
        writer.writerow(header)
        # データ行の書き込み
        for row in landmarks_data:
            # 各キーポイントペアの"x", "y"をフラットなリストに変換して書き込み
            flat_row = [coordinate for pair in row for coordinate in pair]
            writer.writerow(flat_row)

def save_landmarks_to_csv_not_pair(csv_path, landmarks_data):
    # ファイルが保存されるディレクトリを取得
    directory = os.path.dirname(csv_path)
    # ディレクトリが存在しない場合は作成
    os.makedirs(directory, exist_ok=True)
    
    # CSVファイルにランドマークデータを保存
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # ヘッダー行の書き込み
        header = [f'{xy}{i}' for i in range(25) for xy in ['x', 'y']]
        writer.writerow(header)
        # データ行の書き込み
        for row in landmarks_data:
            writer.writerow(row)

def save_probs_to_csv(csv_path, probs_data):
    # ファイルが保存されるディレクトリを取得
    directory = os.path.dirname(csv_path)
    # ディレクトリが存在しない場合は作成
    os.makedirs(directory, exist_ok=True)
    # print(probs_data)
    
    # CSVファイルにランドマークデータを保存
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # ヘッダー行の書き込み
        header = [f'probs{i}' for i in range(25)]
        writer.writerow(header)
        # データ行の書き込み
        for row in probs_data:
            writer.writerow(row)
        
def save_frames_to_directory(frames_dir, frames_data):
    # ディレクトリが存在しない場合は作成
    os.makedirs(frames_dir, exist_ok=True)
    
    # フレームデータをファイルに保存
    for i, frame in enumerate(frames_data):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    print(f"Frames saved to {frames_dir}")
