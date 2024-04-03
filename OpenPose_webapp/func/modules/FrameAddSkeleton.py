import cv2

def FrameAddSkeleton(frame_,points_,MODE_="MOT16"):
    # 選択したモードに基づいてファイルパスと設定を設定する
    if MODE_ == "COCO":
        POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    elif MODE_ == "MPI":
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
    # 以下は未完成
    elif MODE_ == "MOT16":
        POSE_PAIRS = [[2,3],[1,2],[1,5],[5,6],[3,4],[6,7],[1,8],[9,10],[10,11],[2,9],[5,12],[9,8],[8,12],[11,23],[11,24],[23,22],[24,22],[12,13],[13,14],[14,21],[21,20],[21,20],[21,19],[20,19],[1,0],[0,16],[0,15],[15,17],[16,18]]  # MOT16用に適切なペアを設定する必要があります

    # スケルトンを描画する
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points_[partA] and points_[partB] and None not in points_[partA] and None not in points_[partB] :  # 両方のポイントが存在する場合
            # 検出されたキーポイント間にスケルトンを形成するための線を描画する
            cv2.line(frame_, points_[partA], points_[partB], (0, 255, 255), 2)
            cv2.circle(frame_, points_[partA], 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    return frame_