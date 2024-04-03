from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
import tempfile
import csv
from datetime import datetime
# 外部ファイルからの関数をインポート（例: process_video関数）
from func import openPoseToVideo

app = Flask(__name__, static_folder='static')

def load_keypoints(csv_file_path):
    df = pd.read_csv(csv_file_path, skiprows=1)
    exclude_indices = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    keypoints_list = []
    for index, row in df.iterrows():
        keypoints = []
        for i in range(0, len(row) - 1, 2):
            if i//2 not in exclude_indices:
                if not pd.isna(row[i]) and not pd.isna(row[i+1]):
                    keypoints.append((row[i], row[i+1]))
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
        # 一時ファイルを作成
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, 'temp_video.mp4')
        # アップロードされたファイルを一時ファイルに保存
        video_file.save(video_path)

        x1 = int(float(request.form['x1']))
        y1 = int(float(request.form['y1']))
        x2 = int(float(request.form['x2']))
        y2 = int(float(request.form['y2']))
        
        # process_video関数を呼び出し、時刻を取得
        timestamp = openPoseToVideo.SkeletalEstimation(video_path,(x1,y1),(x2,y2))

        # リダイレクトでページを更新
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
    data_dir = os.path.join(app.static_folder, 'datas', directory_name)
    keypoints_csv_file_path = os.path.join(data_dir, 'landmarks.csv')
    probs_csv_file_path = os.path.join(data_dir, 'probs.csv')
    image_dir = os.path.join(data_dir, 'raw_frames')

    POSE_PAIRS = [[2,3],[1,2],[1,5],[5,6],[3,4],[6,7],[1,8],[9,10],[10,11],[2,9],[5,12],[9,8],[8,12],[12,13],[13,14],[0,1]]
    
    keypoints = load_keypoints(keypoints_csv_file_path)
    probs = load_probs(probs_csv_file_path)
    image_paths = get_image_paths(image_dir)
    threshold = 0.5
    below_threshold_indices = []
    
    for i, row in enumerate(probs):
        if any(float(num) <= threshold for num in row):
            below_threshold_indices.append(i)

    image_urls = [os.path.join(app.static_url_path, 'datas', directory_name, 'raw_frames', os.path.basename(path)) for path in image_paths]

    return render_template('imputation.html', image_paths=image_urls[1:-1], keypoints=keypoints, pose_pairs=POSE_PAIRS, below_threshold_indices=below_threshold_indices)

@app.route('/save-keypoints/<directory_name>', methods=['POST'])
def save_keypoints(directory_name):
    keypoints = request.json
    data_dir = os.path.join(app.static_folder, 'datas', directory_name,"fixed_keypoints.csv")
    with open(data_dir, 'w', newline='') as csvfile:
        header = [f'{xy}{i}' for i in range(25) for xy in ['x', 'y']]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for frame_keypoints in keypoints:
            # Flatten the keypoints array and handle None values
            flat_keypoints = [coord for kp in frame_keypoints for coord in (kp or [None, None])]
            writer.writerow(flat_keypoints)
    return jsonify({'message': 'Keypoints saved successfully!'})


if __name__ == '__main__':
    app.run(debug=True)
