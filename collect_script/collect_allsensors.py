import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from lgsvl_msgs.msg import DetectedRadarObjectArray
import message_filters
import cv2
import csv
import time
from datetime import datetime
import numpy as np
import os
import math
from train.utils import mkdir_p, CSV_PATH, IMG_PATH

IMAGE_DIM = (480,270)

class Collect(Node):
	def __init__(self):
		super().__init__('collect', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
		self.get_logger().info(f'[{self.get_name()}] Initializing the collector node...')
		
		mkdir_p(CSV_PATH)
		mkdir_p(IMG_PATH)

		sub_centre_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/center/image/compressed')
		sub_depth_centre = message_filters.Subscriber(self,CompressedImage, '/simulator/sensor/depth_camera/center/image/compressed')
		sub_seg_centre = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/segmentation_camera')

		sub_left90_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/left90/image/compressed')
		sub_depth_left = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/depth_camera/left90/image/compressed')
		
		sub_right90_camera = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/camera/right90/image/compressed')
		sub_depth_right = message_filters.Subscriber(self, CompressedImage, '/simulator/sensor/depth_camera/right90/image/compressed')
		
		sub_radar = message_filters.Subscriber(self, DetectedRadarObjectArray, '/simulator/sensor/radar')
		sub_control = message_filters.Subscriber(self, TwistStamped, '/simulator/control/command')

		ts = message_filters.ApproximateTimeSynchronizer(
			[sub_centre_camera, sub_depth_centre, sub_seg_centre, sub_left90_camera, sub_depth_left, sub_right90_camera, sub_depth_right, sub_radar, sub_control], 10000, 0.1)
		ts.registerCallback(self.callback)
		self.get_logger().info(f'[{self.get_name()}] Up and running the collector node...')
		


	def callback(self, centre_camera, depth_centre, seg_centre, left90_camera, 
			depth_left, right90_camera, depth_right, radar, control):
		
		ts_sec = centre_camera.header.stamp.sec
		ts_nsec = centre_camera.header.stamp.nanosec
		linear_accel = control.twist.linear.x
		steer_cmd = control.twist.angular.x
		brake_cmd = control.twist.linear.y
		throttle = control.twist.linear.z
		velocity = control.twist.angular.y
		radarobj= []
		msg_id = str(datetime.now().isoformat()) 

		self.get_logger().info(f"[{ts_sec}.{ts_nsec}] Format:{centre_camera.format}, Speed/Velocity:{velocity}")
		self.get_logger().info(f"[{ts_sec}.{ts_nsec}] Format:{centre_camera.format}, Linear_accel:{linear_accel}")
		self.get_logger().info(f"[{ts_sec}.{ts_nsec}] Format:{centre_camera.format}, Throttle:{throttle}")
		self.get_logger().info(f"[{ts_sec}.{ts_nsec}] Format:{centre_camera.format}, Steering_cmd:{steer_cmd}")
		self.get_logger().info(f"[{ts_sec}.{ts_nsec}] Format:{centre_camera.format}, Braking:{brake_cmd}")
		for obj in radar.objects:
			eucledian_distance = math.sqrt(obj.object_relative_position.x**2 + obj.object_relative_position.y**2)
			radarobj.append(eucledian_distance)
			#self.get_logger().info(f"distance to objects: {eucledian_distance}")
		
		no_of_objects = len(radar.objects)
		radarobj.sort()
		
		if len(radar.objects) == 0:
			cmd_list = [linear_accel, throttle, steer_cmd, brake_cmd, velocity, 0, 0] 
		else:    
			cmd_list = [linear_accel, throttle, steer_cmd, brake_cmd, velocity, no_of_objects, radarobj]
		
		#self.get_logger().info(f"saving images...")
		self.save_image(centre_camera, depth_centre, seg_centre, left90_camera, 
			depth_left, right90_camera, depth_right, msg_id)
		self.save_csv(msg_id, cmd_list)
		
	def save_image(self, centre_camera, depth_centre, seg_centre, left90_camera, 
			depth_left, right90_camera, depth_right, msg_id):

		centre_img_np_arr = np.fromstring(bytes(centre_camera.data), np.uint8)
		depth_centre_np_arr = np.fromstring(bytes(depth_centre.data), np.uint8)
		segc_img_np_arr = np.fromstring(bytes(seg_centre.data), np.uint8)
		left90_img_np_arr = np.fromstring(bytes(left90_camera.data), np.uint8)
		depth_left_np_arr = np.fromstring(bytes(depth_left.data), np.uint8)
		right90_img_np_arr = np.fromstring(bytes(right90_camera.data), np.uint8)
		depth_right_np_arr = np.fromstring(bytes(depth_right.data), np.uint8)
			
		centre_img_cv = cv2.imdecode(centre_img_np_arr, cv2.IMREAD_COLOR)
		centre_img_cv = cv2.resize(centre_img_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		depth_centre_cv = cv2.imdecode(depth_centre_np_arr, cv2.IMREAD_GRAYSCALE)
		depth_centre_cv = cv2.resize(depth_centre_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		seg_img_cv = cv2.imdecode(segc_img_np_arr, cv2.IMREAD_COLOR)
		seg_img_cv = cv2.resize(seg_img_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		
		left90_img_cv = cv2.imdecode(left90_img_np_arr, cv2.IMREAD_COLOR)
		left90_img_cv = cv2.resize(left90_img_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		depth_left_cv = cv2.imdecode(depth_left_np_arr, cv2.IMREAD_GRAYSCALE)
		depth_left_cv = cv2.resize(depth_left_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		
		right90_img_cv = cv2.imdecode(right90_img_np_arr, cv2.IMREAD_COLOR)
		right90_img_cv = cv2.resize(right90_img_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		depth_right_cv = cv2.imdecode(depth_right_np_arr, cv2.IMREAD_GRAYSCALE)
		depth_right_cv = cv2.resize(depth_right_cv, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		
		cv2.imwrite(os.path.join(IMG_PATH, f'centre-{msg_id}.jpg'), centre_img_cv)
		cv2.imwrite(os.path.join(IMG_PATH, f'depth_c-{msg_id}.jpg'), depth_centre_cv)
		cv2.imwrite(os.path.join(IMG_PATH, f'seg_c-{msg_id}.jpg'), seg_img_cv)

		cv2.imwrite(os.path.join(IMG_PATH, f'left-{msg_id}.jpg'), left90_img_cv)
		cv2.imwrite(os.path.join(IMG_PATH, f'depth_l-{msg_id}.jpg'), depth_left_cv)

		cv2.imwrite(os.path.join(IMG_PATH, f'right-{msg_id}.jpg'), right90_img_cv)
		cv2.imwrite(os.path.join(IMG_PATH, f'depth_r-{msg_id}.jpg'), depth_right_cv)

	def save_csv(self, msg_id, cmd_list):
		with open(os.path.join(CSV_PATH, 'training_data.csv'), 'a+') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow([msg_id, cmd_list])

def main(args=None):
	rclpy.init(args=args)
	collect = Collect()
	rclpy.spin(collect)


if __name__ == '__main__':
	main()	