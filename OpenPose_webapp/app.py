from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
import tempfile
import csv
from datetime import datetime
import json
# 外部ファイルからの関数をインポート（例: process_video関数）
from func import openPoseToVideo

app = Flask(__name__, static_folder='static')

def load_keypoints(csv_file_path):
    df = pd.read_csv(csv_file_path)
    keypoints_list = []
    for index, row in df.iterrows():
        keypoints = []
        for i in range(0, len(row), 2):
            if not pd.isna(row[i]) and not pd.isna(row[i + 1]):
                keypoints.append((row[i], row[i + 1]))
            else:
                keypoints.append((None, None))
        keypoints_list.append(keypoints)
    return keypoints_list

def load_probs(csv_file_path):
    df = pd.read_csv(csv_file_path, skiprows=1)
    selected_columns = [i for i in range(15)]
    df_selected = df.iloc[:, selected_columns]
    return df_selected

def get_image_paths(image_dir):
    image_files = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, file) for file in image_files]
    return image_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        model_name = request.form['model_name']  # モデル名を取得

        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, 'temp_video.mp4')
        video_file.save(video_path)

        x1 = int(float(request.form['x1']))
        y1 = int(float(request.form['y1']))
        x2 = int(float(request.form['x2']))
        y2 = int(float(request.form['y2']))
        
        timestamp = openPoseToVideo.SkeletalEstimation(video_path, (x1, y1), (x2, y2), model_name)

        redirect_url = url_for('imputation', directory_name=timestamp)
        return jsonify({'redirect_url': redirect_url})
    
    return render_template('index.html')

@app.route('/select-directory', methods=['GET', 'POST'])
def select_directory():
    if request.method == 'POST':
        selected_directory = request.form.get('directory_name')
        # ユーザーが選択したディレクトリにリダイレクト
        return redirect(url_for('imputation', directory_name=selected_directory))

    # static/datas直下のディレクトリ一覧を取得
    data_dir = os.path.join(app.static_folder, 'datas')
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    return render_template('select_directory.html', directories=directories)

@app.route('/imputation/<directory_name>')
def imputation(directory_name):
    model_name = request.args.get('model_name', 'openPose')
    data_dir = os.path.join(app.static_folder, 'datas', directory_name)
    fixed_keypoints_file_path = os.path.join(data_dir, 'fixed_keypoints.csv')
    if os.path.exists(fixed_keypoints_file_path):
        keypoints_csv_file_path = fixed_keypoints_file_path
    else:
        keypoints_csv_file_path = os.path.join(data_dir, 'landmarks.csv')
    probs_csv_file_path = os.path.join(data_dir, 'probs.csv')
    image_dir = os.path.join(data_dir, 'raw_frames')

    POSE_PAIRS_MAP = {
        "openPose": [[2,3],[1,2],[1,5],[5,6],[3,4],[6,7],[1,8],[9,10],[10,11],[2,9],[5,12],[9,8],[8,12],[12,13],[13,14],[0,1]],
        "YOLO": [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    }

    keypoints = load_keypoints(keypoints_csv_file_path)
    probs = load_probs(probs_csv_file_path)
    image_paths = get_image_paths(image_dir)
    threshold = 0.5
    below_threshold_indices = []

    for i, row in enumerate(probs):
        if any(float(num) <= threshold for num in row):
            below_threshold_indices.append(i)

    image_urls = [os.path.join(app.static_url_path, 'datas', directory_name, 'raw_frames', os.path.basename(path)) for path in image_paths]

    return render_template('imputation.html', image_paths=image_urls, keypoints=keypoints, pose_pairs=POSE_PAIRS_MAP[model_name], below_threshold_indices=below_threshold_indices)

@app.route('/save-keypoints/<directory_name>', methods=['POST'])
def save_keypoints(directory_name):
    keypoints = request.json  # JSON形式でキーポイントデータを受け取る

    csv_dir = os.path.join(app.static_folder, 'datas', directory_name, "fixed_keypoints.csv")
    json_dir = os.path.join(app.static_folder, 'datas', directory_name, "keypoints.json")
   
    header = [f'{xy}{i}' for i in range(31) for xy in ['x', 'y']]  # 31個のキーポイント

    # CSVファイルの保存
    with open(csv_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for frame_keypoints in keypoints:
            # フラット化を安全に行う
            flat_list = []
            for sublist in frame_keypoints:
                if isinstance(sublist, list):
                    flat_list.extend(sublist)  # sublistがリストの場合、拡張
                else:
                    flat_list.append(sublist)  # sublistが数値の場合、追加
            writer.writerow(flat_list)
    
    # JSONファイルの保存
    formatted_json = []
    for frame in keypoints:
        frame_dict = {}
        for i, (x, y) in enumerate(zip(frame[::2], frame[1::2])):  # x, y値のペアを抽出
            frame_dict[f'x{i}'] = x
            frame_dict[f'y{i}'] = y
        formatted_json.append(frame_dict)

    with open(json_dir, 'w') as jsonfile:
        json.dump(formatted_json, jsonfile, indent=4)
    
    return jsonify({'message': 'Keypoints saved successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
