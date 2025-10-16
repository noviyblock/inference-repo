import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations as tf_trans
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, BoundingBox3D
from geometry_msgs.msg import Pose, Vector3

def camera_info_to_K(camera_info: CameraInfo) -> np.ndarray:
    """
    returns 3x3 camera intrinsic matrix from CameraInfo
    """
    K = np.array(camera_info.k, dtype=np.float32).reshape((3, 3))
    return K

def lidar_to_numpy(lidar_msg, num_points=200000):
    """
    msg: sensor_msgs/PointCloud2
    returns float32 numpy with shape [1, M, C]
    """
    points_numpy = pc2.read_points_numpy(lidar_msg, skip_nans=True) # [M, C]

    M = points_numpy.shape[0]
    C = points_numpy.shape[1]
    if C != 5:
        raise ValueError(f"Unexpected number of point fields: {C}")

    if M >= num_points:
        points_numpy =  points_numpy[:num_points, :].astype(np.float32)
    else:
        pad = np.zeros((num_points - M, C), dtype=np.float32)
        points_numpy =  np.vstack([points_numpy.astype(np.float32), pad])

    return np.expand_dims(points_numpy, axis=0) # [1, M, C]

def preprocess_image(img, target_hw=(256, 704)):
    H, W = target_hw

    im = cv2.resize(img, (W, H))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = np.transpose(im, (2, 0, 1))  # C H W

    return im.astype(np.float32)

def transform_to_matrix(transform):
    """
    transform: geometry_msgs/Transform
    return float32 numpy with shape [4, 4]
    """
    translation = [transform.translation.x, transform.translation.y, transform.translation.z]
    
    rotation = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    
    matrix = tf_trans.quaternion_matrix(rotation)
    matrix[:3, 3] = translation
    
    return matrix.astype(np.float32)

def process_bevfusion_outputs(outputs, output_names):

    boxes = None
    scores = None
    labels = None

    for i in range(len(output_names)):
        name = output_names[i].lower()

        if 'box' in name:
            boxes = outputs[i]     #  [N, 7|9]
        elif 'score' in name:
            scores = outputs[i]    # [N]
        elif 'label' in name:
            labels = outputs[i]    # [N]
        elif i == 0 and boxes is None:
            boxes = outputs[i]

    return boxes, scores, labels

def create_detection3d_array(boxes, scores, labels, frame_id, stamp, confidence_threshold=0.3):
    """
    Создание vision_msgs/Detection3DArray из выходов BEV-Fusion
    """

    det_array = Detection3DArray()
    det_array.header.stamp = stamp
    det_array.header.frame_id = frame_id
    
    if boxes is None or len(boxes) == 0:
        return det_array
    
    valid_indices = scores >= confidence_threshold
    
    boxes = boxes[valid_indices]
    scores = scores[valid_indices] 
    labels = labels[valid_indices]
    
    for i in range(len(boxes)):
        box = boxes[i]
        score = float(scores[i])
        label = int(labels[i])
        
        # Создаем Detection3D
        detection = Detection3D()
        detection.header = det_array.header
        
        # Object hypothesis
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(label)
        hypothesis.hypothesis.score = score
        detection.results.append(hypothesis)
        
        # Bounding box
        bbox = BoundingBox3D()
        
        # Центр бокса
        center = Pose()
        center.position.x = float(box[0])
        center.position.y = float(box[1]) 
        center.position.z = float(box[2])
        
        # Ориентация (yaw угол)
        qx, qy, qz, qw = tf_trans.quaternion_from_euler(0, 0, float(box[6]))
        center.orientation.x = qx
        center.orientation.y = qy
        center.orientation.z = qz
        center.orientation.w = qw
        
        bbox.center = center
        
        # Размеры бокса
        size = Vector3()
        size.x = float(box[3])  # length
        size.y = float(box[4])  # width  
        size.z = float(box[5])  # height
        bbox.size = size
        
        detection.bbox = bbox
        det_array.detections.append(detection)
    
    return det_array

def lidar_to_dense_grid(lidar_msg, 
                         x_range=(-100, 100), 
                         y_range=(-100, 100), 
                         z_range=(-5, 20), 
                         grid_size=(41, 1440, 1440),
                         max_points_per_voxel=10):

    points_numpy = pc2.read_points_numpy(lidar_msg, skip_nans=True)

    Z, X, Y = grid_size
    dense_grid = np.zeros((5, Z, X, Y), dtype=np.float32)
    count_grid = np.zeros((Z, X, Y), dtype=np.int32)  # для усреднения

    x = points_numpy[:, 0]
    y = points_numpy[:, 1]
    z = points_numpy[:, 2]
    intensity = points_numpy[:, 3]
    obj_tag = points_numpy[:, 4]

    # Индексы в сетке
    x_idx = np.floor((x - x_range[0]) / (x_range[1] - x_range[0]) * (X-1)).astype(np.int32)
    y_idx = np.floor((y - y_range[0]) / (y_range[1] - y_range[0]) * (Y-1)).astype(np.int32)
    z_idx = np.floor((z - z_range[0]) / (z_range[1] - z_range[0]) * (Z-1)).astype(np.int32)

    # Ограничение индексов в пределах сетки
    x_idx = np.clip(x_idx, 0, X-1)
    y_idx = np.clip(y_idx, 0, Y-1)
    z_idx = np.clip(z_idx, 0, Z-1)

    # Усреднение точек, попавших в один воксель
    for i in range(points_numpy.shape[0]):
        zi, xi, yi = z_idx[i], x_idx[i], y_idx[i]
        cnt = count_grid[zi, xi, yi]
        if cnt < max_points_per_voxel:
            dense_grid[0, zi, xi, yi] = (dense_grid[0, zi, xi, yi] * cnt + x[i]) / (cnt + 1)
            dense_grid[1, zi, xi, yi] = (dense_grid[1, zi, xi, yi] * cnt + y[i]) / (cnt + 1)
            dense_grid[2, zi, xi, yi] = (dense_grid[2, zi, xi, yi] * cnt + z[i]) / (cnt + 1)
            dense_grid[3, zi, xi, yi] = (dense_grid[3, zi, xi, yi] * cnt + intensity[i]) / (cnt + 1)
            dense_grid[4, zi, xi, yi] = (dense_grid[4, zi, xi, yi] * cnt + obj_tag[i]) / (cnt + 1)
            count_grid[zi, xi, yi] += 1

    # Добавляем batch dimension
    dense_grid = dense_grid[None, :, :, :, :]  # [1, 5, Z, X, Y]
    return dense_grid
