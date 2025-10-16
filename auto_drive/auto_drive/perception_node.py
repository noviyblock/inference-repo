import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
from vision_msgs.msg import Detection3DArray

import numpy as np
import onnxruntime as ort
import threading
import traceback

import os
from ament_index_python import get_package_share_directory

from cv_bridge import CvBridge
import message_filters

import tf2_ros

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


from .utils import (
    lidar_to_numpy,
    camera_info_to_K,
    transform_to_matrix,
    process_bevfusion_outputs,
    create_detection3d_array,
    lidar_to_dense_grid,
    preprocess_image
)

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # parameters (tweakable)
        self.declare_parameter('cameras', [
            '/CAM_FRONT',
            '/CAM_FRONT_LEFT',
            '/CAM_FRONT_RIGHT',
            '/CAM_BACK_LEFT',
            '/CAM_BACK_RIGHT',
            '/CAM_BACK',
        ])

        self.declare_parameter('lidar_topic', '/LIDAR_TOP')
        self.declare_parameter('onnx_path', '/path/to/model.onnx')
        self.declare_parameter('use_sensor_time', True)
        self.declare_parameter('use_cuda', False)
        self.declare_parameter('input_keys', ['image', 'dense_grid'])

        self.camera_topics = [camera + '/image_rect_compressed' for camera in self.get_parameter('cameras').value]
        self.camera_info_topics = [camera + '/camera_info' for camera in self.get_parameter('cameras').value]
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        onnx_path = self.get_parameter('onnx_path').get_parameter_value().string_value
        self.time_slop = 0.5
        self.use_sensor_time = bool(self.get_parameter('use_sensor_time').get_parameter_value().bool_value)
        use_cuda = bool(self.get_parameter('use_cuda').get_parameter_value().bool_value)
        self.input_keys = self.get_parameter('input_keys').value

        # qos
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.KEEP_LAST

        # cv bridge
        self._bridge = CvBridge()

        # camera subscribers
        self.camera_subs = []
        for t in self.camera_topics:
            sub = message_filters.Subscriber(self, CompressedImage, t, qos_profile=qos)
            self.camera_subs.append(sub)

        # lidar subscriber 
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic, qos_profile=qos)

        # camera_info subscribers
        self.camera_info = {}
        for t in self.camera_info_topics:
            self.create_subscription(CameraInfo, t, lambda msg, top=t: self._camera_info_cb(msg, top), 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # synchronization
        sync_inputs = self.camera_subs + [self.lidar_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(sync_inputs, queue_size=10, slop=self.time_slop)
        self.sync.registerCallback(self._synced_cb)

        # publishers
        self.detections_pub = self.create_publisher(Detection3DArray, '/bevfusion/detections', 10)
        
        # Load ONNX model
        providers = ['CPUExecutionProvider']
        if use_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.get_logger().info(f'Loaded ONNX model: {onnx_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load ONNX model: {e}')
            self.session = None

        for i, input_info in enumerate(self.session.get_inputs()):
            self.get_logger().debug(f"Input {i}: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
        for i, output_info in enumerate(self.session.get_outputs()):
            self.get_logger().debug(f"Output {i}: name='{output_info.name}', shape={output_info.shape}, type={output_info.type}")

        """
        # Lidar replay
        self.lidar_replay_pub = self.create_publisher(PointCloud2, '/LIDAR_TOP_replay', 10)

        # Cameras replay
        self.camera_replay_pubs = []
        for cam in self.camera_topics:
            pub = self.create_publisher(CompressedImage, cam + '_replay', 10)
            self.camera_replay_pubs.append(pub)

        # transform replay
        self.tf_replay_pub = self.create_publisher(TFMessage, '/tf_replay', 10)
        """

        self.lock = threading.Lock()
        self.get_logger().info('PerceptionNode initialized')

    def _camera_info_cb(self, msg: CameraInfo, topic_name):
        # store by topic_name so we can link to camera topics
        self.camera_info[topic_name] = msg
        self.get_logger().debug(f"Store info for {topic_name}")

    def _try_lookup_transform(self, target_frame, source_frame, time):
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
            return t
        except Exception as e:
            return None

    def _synced_cb(self, *msgs):

        if self.session is None:
            self.get_logger().warning('ONNX session not available')
            return

        try:
            with self.lock:
                if len(msgs) < 2:
                    return
                imgs_msgs = msgs[:-1]
                lidar_msg = msgs[-1]

                lidar_stamp = lidar_msg.header.stamp

                """
                # tf for replays
                tf_msgs = []

                # map -> base_link
                try:
                    map_to_base = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
                    tf_msgs.append(TransformStamped(
                        header=map_to_base.header,
                        child_frame_id=map_to_base.child_frame_id,
                        transform=map_to_base.transform
                    ))
                except tf2_ros.TransformException as e:
                    self.get_logger().warning(f'Failed lidar TF lookup: {e}')

                # LIDAR_TOP -> base_link
                try:
                    lidar_to_base = self.tf_buffer.lookup_transform('LIDAR_TOP', 'base_link', rclpy.time.Time())
                    tf_msgs.append(TransformStamped(
                        header=lidar_to_base.header,
                        child_frame_id=lidar_to_base.child_frame_id,
                        transform=lidar_to_base.transform
                    ))
                except tf2_ros.TransformException as e:
                    self.get_logger().warning(f'Failed lidar TF lookup: {e}')
                """
                
                inputs = {}

                # Check input keys
                for key in self.input_keys:
                    if key == 'image':
                        cv_img = self._bridge.compressed_imgmsg_to_cv2(imgs_msgs[0], "bgr8")
                        img_in = preprocess_image(cv_img)

                        inputs['image'] = np.expand_dims(img_in, 0)                   # [1, 3, H, W]

                    elif key == 'images':
                        imgs = []
                        for im in imgs_msgs:
                            cv_img = self._bridge.compressed_imgmsg_to_cv2(im, "bgr8")
                            img = preprocess_image(cv_img)
                            imgs.append(img)

                        imgs_in = np.stack(imgs, axis=0)

                        inputs['images'] = np.expand_dims(imgs_in, 0)                  # [1, N, 3, H, W]

                    elif key == 'points':
                        inputs['points'] = lidar_to_numpy(lidar_msg)                   # [1, M, 5] 

                    elif key == 'dense_grid':
                        inputs['dense_grid'] = lidar_to_dense_grid(lidar_msg,
                                                                   x_range=(-50, 50),
                                                                   y_range=(-50, 50))  # [1, 5, Z, X, Y]

                    elif key not in ('intrinsics', 'extrinsics', 'lidar2cam'):
                        self.get_logger().warning(f"Unknown input key '{key}' — skipping.")

                # Check camera properties in keys
                if any(k in self.input_keys for k in ('intrinsics', 'extrinsics', 'lidar2cam')):
                    cam_intrinsics = []
                    cam_extrinsics = []
                    lidar2cam = []

                    # get lidar transform
                    try:
                        lidar_tf = self.tf_buffer.lookup_transform(
                            'LIDAR_TOP', 'base_link', lidar_stamp)
                        lidar_tf_matrix = transform_to_matrix(lidar_tf.transform)
                    except tf2_ros.TransformException as e:
                        self.get_logger().warning(f'Failed lidar TF lookup: {e}')
                        lidar_tf_matrix = np.eye(4, dtype=np.float32)


                    for ci_topic in self.camera_info_topics:
                        camera_name = ci_topic.replace('/camera_info', '').replace('/', '')

                        # get intrisics
                        ci = self.camera_info.get(ci_topic, None)
                        if ci is not None:
                            K = camera_info_to_K(ci)
                        else:
                            self.get_logger().warn(f'CameraInfo missing for {ci_topic}, using identity K')
                            K = np.eye(3, dtype=np.float32)
                        cam_intrinsics.append(K)

                        try:
                            # get extrinsics
                            cam_extrinsic_tf = self.tf_buffer.lookup_transform(
                                camera_name,
                                'base_link',
                                lidar_stamp
                            )
                            cam_extrinsic_matrix = transform_to_matrix(cam_extrinsic_tf.transform)
                            cam_extrinsics.append(np.linalg.inv(cam_extrinsic_matrix))

                            lidar2cam.append(np.dot(np.linalg.inv(lidar_tf_matrix), cam_extrinsic_matrix))

                        except tf2_ros.TransformException as e:
                            self.get_logger().warning(f'Transform error for {camera_name}: {e}')
                            # Use identity matrices as fallback
                            cam_extrinsics.append(np.eye(4, dtype=np.float32))
                            lidar2cam.append(np.eye(4, dtype=np.float32))

                    inputs['intrinsics'] = np.expand_dims(np.stack(cam_intrinsics, axis=1), 0)  # [1, N, 3, 3]
                    inputs['extrinsics'] = np.expand_dims(np.stack(cam_extrinsics, axis=1), 0)  # [1, N, 4, 4]
                    inputs['lidar2cam']  = np.expand_dims(np.stack(lidar2cam, axis=1), 0)       # [1, N, 4, 4]

                # run inference
                outputs = self.session.run(None, inputs)
                output_names = [out.name for out in self.session.get_outputs()]
                #for i, name in enumerate(output_names):
                #    self.get_logger().info(f"Name: {name}, Shape: {outputs[i].shape}")

                stamp = lidar_msg.header.stamp if self.use_sensor_time else self.get_clock().now().to_msg()
                boxes, scores, labels = process_bevfusion_outputs(outputs, output_names)

                det_array_msg = create_detection3d_array(boxes, scores, labels, lidar_msg.header.frame_id, stamp, confidence_threshold=0.0)
                self.detections_pub.publish(det_array_msg)

                """
                # Publish replay lidar
                lidar_replay = PointCloud2()
                lidar_replay = lidar_msg
                lidar_replay.header.stamp = stamp
                self.lidar_replay_pub.publish(lidar_replay)

                # Publish replay cameras
                for i, im_msg in enumerate(imgs_msgs):
                    img_replay = CompressedImage()
                    img_replay = im_msg
                    img_replay.header.stamp = stamp
                    self.camera_replay_pubs[i].publish(img_replay)

                # Publish replay tf
                if tf_msgs:
                    self.tf_replay_pub.publish(TFMessage(transforms=tf_msgs))
                """

                self.get_logger().info('Detections Published')

        except Exception as e:
            self.get_logger().error('Exception in timer_cb: ' + str(e))
            self.get_logger().debug(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
