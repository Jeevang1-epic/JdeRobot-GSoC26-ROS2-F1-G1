import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np

from vision_core.network import VisualDriveNet
class VisualNode(Node):

    def __init__(self):
        super().__init__('visual_node')
        self.sub = self.create_subscription(Image, '/cam_f1_left/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10) 
        self.cvb = CvBridge()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = VisualDriveNet().to(self.dev)
        self.net.eval()
        
    def cb(self, msg):
        img = self.cvb.imgmsg_to_cv2(msg, 'bgr8')
        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.dev)
        
        with torch.no_grad():
            out = self.net(img_t)
            
        v = Twist()
        v.linear.x = float(out[0][0])
        v.angular.z = float(out[0][1])
        
        self.pub.publish(v)

def main(args=None):
    rclpy.init(args=args)
    n = VisualNode()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()