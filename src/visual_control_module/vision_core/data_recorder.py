import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import csv
from datetime import datetime

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')
        self.cvb = CvBridge()
        
        self.dataset_dir = os.path.expanduser('~/jderobot_gsoc26/dataset')
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.dataset_dir, 'driving_log.csv')
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(['timestamp', 'image_path', 'linear_v', 'angular_w'])

        self.current_v = 0.0
        self.current_w = 0.0
        self.image_count = 0

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.img_sub = self.create_subscription(Image, '/cam_f1_left/image_raw', self.img_cb, 10)
        
        self.get_logger().info(f"Recorder Armed! Saving data to {self.dataset_dir}")

    def cmd_cb(self, msg):
        self.current_v = msg.linear.x
        self.current_w = msg.angular.z

    def img_cb(self, msg):
        try:
            cv_image = self.cvb.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"img_{timestamp}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)

        cv2.imwrite(img_path, cv_image)

        self.csv_writer.writerow([timestamp, img_filename, self.current_v, self.current_w])
        self.csv_file.flush() 
        
        self.image_count += 1
        if self.image_count % 50 == 0:
            self.get_logger().info(f"Successfully saved {self.image_count} images to dataset...")

def main(args=None):
    rclpy.init(args=args)
    recorder = DataRecorder()
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.csv_file.close()
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()