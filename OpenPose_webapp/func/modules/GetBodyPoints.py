import cv2
import numpy as np
import time

def GetBodyPoints(frame_,MODE_="MOT16"):

    if MODE_ == "COCO":
        protoFile = "OpenPoseModels\coco\pose_deploy_linevec.prototxt"  # ネットワークのテキスト記述が含まれる.prototxtファイルへのパス。このファイルには、ネットワークの各レイヤーの定義やそれらのレイヤー間の接続など、ネットワークのアーキテクチャに関する情報が含まれている。
        weightsFile = "OpenPose_webapp\OpenPoseModels\coco\pose_iter_440000.caffemodel"  # 学習済みネットワークが含まれる.caffemodelファイルへのパス
        nPoints = 18  # キーポイントの数
    elif MODE_ == "MPI":
        protoFile = "OpenPoseModels\mpi\pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "OpenPoseModels\mpi\pose_iter_160000.caffemodel"
        nPoints = 15
    # 以下は未完成
    elif MODE_ == "MOT16":
        protoFile = r"OpenPose_webapp\OpenPoseModels\mot16\pose_deploy.prototxt"
        weightsFile = r"OpenPose_webapp\OpenPoseModels\mot16\pose_iter_584000.caffemodel"
        nPoints = 25

    # 画像に関するデータを取得
    frameCopy = np.copy(frame_)  # キーポイントを描画するための画像のコピーを作成
    frameWidth = frame_.shape[1]  # 画像の幅
    frameHeight = frame_.shape[0]  # 画像の高さ
    threshold = 0.1  # キーポイントを検出するための閾値

    # ネットワークをロードする
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # GPUが使用可能かどうかをチェック
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # ネットワーク用に入力画像を準備する
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame_, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)  # 準備されたblobをネットワークの入力として設定する

    t = time.time()  # 実行時間を測定するための開始時間
    output = net.forward()  # 出力を取得するために前方パスを実行する
    print("time taken by network : {:.3f}".format(time.time() - t))  # 実行時間を出力

    H = output.shape[2]  # 出力層の高さ
    W = output.shape[3]  # 出力層の幅

    # 検出されたキーポイントを格納するための空のリスト
    points = []
    probs = []

    # すべてのキーポイントに対してループ
    for i in range(nPoints):
        # 現在のキーポイントのための信頼度マップを取得する
        probMap = output[0, i, :, :]
        # 信頼度マップの最大値を見つける
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # ポイントを元の画像に合わせてスケーリングする
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        probs.append(prob)

        if prob > threshold:  # 閾値より大きい場合
            # 画像上にキーポイントを描画する
            cv2.circle(frameCopy, (int(x), int(y)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            
            points.append((int(x), int(y)))  # リストにポイントを追加
        else:
            points.append((None,None))  # キーポイントが検出されない場合はNoneを追加

    return points,probs

def find_highest_prob_on_circle(prob_map_resized, center, radius):
    max_prob = -999 # いったん、確率を使わずに値を出す。
    max_point = (None, None)

    # 円周上の点をサンプリング
    for angle in np.linspace(0, 2 * np.pi, num=360):
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        
        # ヒートマップ上の確率を確認
        if x < 0 or x >= prob_map_resized.shape[1] or y < 0 or y >= prob_map_resized.shape[0]:
            continue  # インデックスが範囲外の場合はスキップ
        
        prob = prob_map_resized[y, x]
        
        # max_prob = prob
        # max_point = (x, y)
        if prob > max_prob:
            max_prob = prob
            max_point = (x, y)
    
    return max_prob, max_point

def distance_between_points(point1, point2):
    from math import sqrt
    dx = point1[0] - point2[0]  # Calculate the difference in the x-coordinates
    dy = point1[1] - point2[1]  # Calculate the difference in the y-coordinates
    distance = sqrt(dx**2 + dy**2)  # Calculate the Euclidean distance
    return distance

def GetBodyPoints_with_alg(frame_,neck_point,neck_right_length=None,neck_left_length=None,right_S_E=None,left_S_E=None,right_E_H=None,left_E_H=None,MODE_="MOT16"):

    if MODE_ == "COCO":
        protoFile = "OpenPoseModels\coco\pose_deploy_linevec.prototxt"  # ネットワークのテキスト記述が含まれる.prototxtファイルへのパス。このファイルには、ネットワークの各レイヤーの定義やそれらのレイヤー間の接続など、ネットワークのアーキテクチャに関する情報が含まれている。
        weightsFile = "OpenPoseModels\coco\pose_iter_440000.caffemodel"  # 学習済みネットワークが含まれる.caffemodelファイルへのパス
        nPoints = 18  # キーポイントの数
    elif MODE_ == "MPI":
        protoFile = "OpenPoseModels\mpi\pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "OpenPoseModels\mpi\pose_iter_160000.caffemodel"
        nPoints = 15
    # 以下は未完成
    elif MODE_ == "MOT16":
        protoFile = "OpenPoseModels\mot16\pose_deploy.prototxt"
        weightsFile = "OpenPoseModels\mot16\pose_iter_584000.caffemodel"
        nPoints = 25

    # 画像に関するデータを取得
    frameCopy = np.copy(frame_)  # キーポイントを描画するための画像のコピーを作成
    frameWidth = frame_.shape[1]  # 画像の幅
    frameHeight = frame_.shape[0]  # 画像の高さ
    threshold = 0.1  # キーポイントを検出するための閾値

    # ネットワークをロードする
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # GPUが使用可能かどうかをチェック
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # ネットワーク用に入力画像を準備する
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame_, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)  # 準備されたblobをネットワークの入力として設定する

    t = time.time()  # 実行時間を測定するための開始時間
    output = net.forward()  # 出力を取得するために前方パスを実行する
    print("time taken by network : {:.3f}".format(time.time() - t))  # 実行時間を出力

    H = output.shape[2]  # 出力層の高さ
    W = output.shape[3]  # 出力層の幅

    # 検出されたキーポイントを格納するための空のリスト
    points = []
    probs = []

    # すべてのキーポイントに対してループ
    for i in range(nPoints):
        if (i == 1):
            points.append(neck_point)
            probs.append(1)
            continue
        elif (i == 2) and neck_right_length:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, neck_point, neck_right_length)
            points.append(point)
            probs.append(1)
            print(point)
            continue
        elif (i == 5) and neck_left_length:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, neck_point, neck_left_length)
            points.append(point)
            probs.append(1)
            continue
        elif (i == 3) and right_S_E:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, points[2], right_S_E)
            points.append(point)
            probs.append(1)
            continue
        elif (i == 6) and left_S_E:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, points[5], left_S_E)
            points.append(point)
            probs.append(1)
            continue
        elif (i == 4) and right_E_H:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, points[3], right_E_H)
            points.append(point)
            probs.append(1)
            continue
        elif (i == 7) and left_E_H:
            probMap = output[0, i, :, :]
            probMap_resized = cv2.resize(probMap, (frameWidth, frameHeight))
            prob, point = find_highest_prob_on_circle(probMap_resized, points[6], left_E_H)
            # print(point)
            points.append(point)
            probs.append(1)
            continue
        
        # 現在のキーポイントのための信頼度マップを取得する
        probMap = output[0, i, :, :]
        # 信頼度マップの最大値を見つける
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # ポイントを元の画像に合わせてスケーリングする
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        probs.append(prob)

        if prob > threshold:  # 閾値より大きい場合
            # 画像上にキーポイントを描画する
            cv2.circle(frameCopy, (int(x), int(y)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            
            points.append((int(x), int(y)))  # リストにポイントを追加
        else:
            points.append((None,None))  # キーポイントが検出されない場合はNoneを追加

    return points,probs