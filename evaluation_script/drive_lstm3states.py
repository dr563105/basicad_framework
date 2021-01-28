import rclpy
from rclpy.node import Node
from rclpy.timer import WallTimer
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CompressedImage
import message_filters
import threading
import numpy as np
import cv2
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from train.utils import preprocess_image
from lgsvl_msgs.msg import DetectedRadarObjectArray
import math
import time
import collections

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IMAGE_DIM = (160, 70)


class Drive(Node):
    def __init__(self):
        super().__init__('drive', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.log = self.get_logger()
        self.log.info('Starting Drive node...')

        self.image_lock = threading.RLock()

        # ROS topics
        self.center_camera_topic = self.get_param('center_camera_topic')
        self.control_topic = self.get_param('control_topic')

        self.log.info('Camera topic: {}'.format(self.center_camera_topic))
        self.log.info('Control topic: {}'.format(self.control_topic))

        # ROS communications
        self.image_sub = self.create_subscription(CompressedImage, self.center_camera_topic, self.image_callback)
        self.control_pub = self.create_publisher(TwistStamped, self.control_topic)

        # ROS timer
        #self.publish_period = .02  # seconds
        #self.check_timer = self.create_timer(1, self.check_camera_topic)

        # ROS parameters
        self.enable_visualization = self.get_param('visualization', False)
        self.model_path = self.get_param('model_path')

        # Model parameters
        self.model = self.get_model(self.model_path)
        self.img = None
        self.img_gray = None
        self.steering = 0.
        self.acceleration = 0.
        self.time_steps = 15
        self.queue_images = collections.deque(maxlen=self.time_steps)#FIFO queue
        self.inference_time = 0.

        # For visualizations
        self.steer_ratio = self.get_param('steer_ratio', 1.)
        self.steering_wheel_single_direction_max = self.get_param('steering_wheel_single_direction_max', 360.)  # in degrees
        self.wheel_base = self.get_param('wheel_base', 2.836747)  # in meters

        # FPS
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.

    def image_callback(self, img):
        self.get_fps()
        if self.image_lock.acquire(True):
            self.img = img
            if self.model is None:
                self.model = self.get_model(self.model_path)
            t0 = time.time()
            self.preprocess_img(img)
            self.queue_images.appendleft(self.img_gray)
            #print("queue images shape: ", np.array(self.queue_images).shape) #(15, 70, 160, 1)
            if (len(self.queue_images)  == self.time_steps):
                #queuedimages = np.array(self.queue_images)
                queuedimages = np.expand_dims(self.queue_images, axis=0) 
                #print("queued images shape: ", np.array(queuedimages).shape)
                cmds, st = self.predict(self.model, queuedimages)
                self.log.info(f'Predicted commands: "{cmds[0][0], cmds[0][1], cmds[0][2], st[0][0]}"')
                if (cmds[0][0] > cmds[0][2]) and (cmds[0][0] > cmds[0][1]):
                    self.acceleration = cmds[0][0]
                elif (cmds[0][1] > cmds[0][2]) and ((cmds[0][1]) > (cmds[0][0])): #and (0.0 < (cmds[0][1]) < 1.0):
                    self.acceleration = -cmds[0][1]              
                else: 
                    self.acceleration = 0.0

                self.steering = st[0][0]
                t1 = time.time()
                self.publish_steering()
                self.inference_time = t1 - t0
                if self.enable_visualization:
                    self.visualize(self.img, self.steering)
            self.image_lock.release()

    def preprocess_img(self, img):
        image_list = []
        c = np.fromstring(bytes(img.data), np.uint8)
        image = cv2.imdecode(c, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (480,270), interpolation=cv2.INTER_AREA)
        image = image[130:, :410]
        image = cv2.resize(image, IMAGE_DIM, interpolation=cv2.INTER_AREA) 
        image_list.append(image)
        image = np.dstack(image_list)
        #image = np.expand_dims(image, axis=0)
        self.img_gray = image

    def publish_steering(self):
        if self.img is None:
            return
        message = TwistStamped()
        message.twist.linear.x = (float(self.acceleration))
        message.twist.angular.x =  (float(self.steering))
        self.control_pub.publish(message)
        #self.log.info('[{:.3f}] Predicted acceleration command: "{}"'.format(time.time(), message.twist.linear.x))
        #self.log.info('[{:.3f}] Predicted steering command: "{}"'.format(time.time(), message.twist.angular.x))

    def get_model(self, model_path):
        self.log.info('Loading model from {}'.format(model_path))
        model = load_model(model_path, compile = False)
        self.log.info('Model loaded!')

        return model

    def predict(self, model, img):
        cmds,st  = self.model.predict(img)
        return cmds,st 

    def visualize(self, img, steering):
        c = np.fromstring(bytes(img.data), np.uint8)
        image = cv2.imdecode(c, cv2.IMREAD_COLOR)

        steering_wheel_angle_deg = steering * self.steering_wheel_single_direction_max
        wheel_angle_deg = steering_wheel_angle_deg / self.steer_ratio  # wheel angle in degree [-29.375, 29.375]
        curvature_radius = self.wheel_base / (2 - 2 * math.cos(2 * steering_wheel_angle_deg / self.steer_ratio)) ** 2

        kappa = 1 / curvature_radius
        curvature = int(kappa * 50)
        if steering < 0:  # Turn left
            x = -curvature
            ra = 0
            rb = -70
        else:  # Turn right
            x = curvature
            ra = -110
            rb = -180

        cv2.ellipse(image, (960 + x, image.shape[0]), (curvature, 500), 0, ra, rb, (0, 255, 0), 2)

        cv2.putText(image, "Prediction: %f.7" % (steering), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Steering wheel angle: %.3f degrees" % steering_wheel_angle_deg, (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Wheel angle: %.3f degrees" % wheel_angle_deg, (30, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Prediction time: %d ms" % (self.inference_time * 1000), (30, 220), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Frame speed: %d fps" % (self.fps), (30, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        image = cv2.resize(image, (round(image.shape[1] / 2), round(image.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow('LGSVL End-to-End Lane Following', image)
        cv2.waitKey(1)

    def get_fps(self):
        self.frames += 1
        now = time.time()
        if now >= self.last_time + 1.0:
            delta = now - self.last_time
            self.last_time = now
            self.fps = self.frames / delta
            self.frames = 0

    def get_param(self, key, default=None):
        val = self.get_parameter(key).value
        if val is None:
            val = default
        return val

    def check_camera_topic(self):
        if self.img is None:
            self.log.info('Waiting a camera image from ROS topic: {}'.format(self.center_camera_topic))
        else:
            self.log.info('Received a camera image from ROS topic: {}'.format(self.center_camera_topic))
            #self.destroy_timer(self.check_timer)
            #self.create_timer(self.publish_period, self.publish_steering)


def main(args=None):
    rclpy.init(args=args)
    drive = Drive()
    rclpy.spin(drive)


if __name__ == '__main__':
    main()
