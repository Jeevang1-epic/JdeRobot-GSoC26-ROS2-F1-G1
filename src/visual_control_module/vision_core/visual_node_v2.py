import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import VisualDriveNet

class VisualNodeV2(Node):
    def __init__(self):
        super().__init__('visual_node_v2')
        self.cvb = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_sub = self.create_subscription(Image, '/cam_f1_left/image_raw', self.img_cb, 10)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VisualDriveNet().to(self.device)
        
        model_path = os.path.expanduser('~/jderobot_gsoc26/src/visual_control_module/models/robust_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.predictions = []
        self.timestamps = []
        self.start_time = time.time()
        self.run_duration = 180.0  # 3 minutes
        self.shutdown_initiated = False
        
        self.get_logger().info(f"AI Engine Active. Auto-shutdown in 3 minutes.")
        self.timer = self.create_timer(1.0, self.check_timer)

    def check_timer(self):
        if self.shutdown_initiated:
            return
            
        current_time = time.time() - self.start_time
        if current_time >= self.run_duration:
            self.shutdown_initiated = True
            self.get_logger().info("3 minutes reached. Commencing auto-shutdown sequence...")
            
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
            
            self.generate_telemetry_plots()
            
            self.get_logger().info("Node shutting down safely...")
            
            rclpy.shutdown()
            sys.exit(0)

    def img_cb(self, msg):
        if self.shutdown_initiated:
            return
            
        try:
            cv_image = self.cvb.imgmsg_to_cv2(msg, 'bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                steering_tensor = self.model(input_tensor)
                steering = steering_tensor.flatten()[0].item() * 1.5 
            
            twist_msg = Twist()
            twist_msg.linear.x = 0.1
            twist_msg.angular.z = float(steering)
            self.cmd_pub.publish(twist_msg)
            
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            self.predictions.append(steering)
            
        except Exception as e:
            self.get_logger().error(f"vision error: {e}")

    def generate_telemetry_plots(self):
        self.get_logger().info("Rendering final telemetry visuals...")
        models_dir = os.path.expanduser('~/jderobot_gsoc26/src/visual_control_module/models')
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(self.timestamps, self.predictions, color='#2ecc71', linewidth=2)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.title('Robust Model: Autonomous Steering Telemetry', fontsize=12)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Steering Angle (rad/s)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(models_dir, 'robust_inference_scatter.png'))
            plt.close()
            
            brittle_left = 367
            brittle_right = 748
            robust_left = 808
            robust_right = 813

            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(2)

            ax.bar(index, [brittle_left, robust_left], bar_width, label='Left Corrections', color='#3498db')
            ax.bar(index + bar_width, [brittle_right, robust_right], bar_width, label='Right Corrections', color='#e74c3c')

            ax.set_title('Brittle vs Robust Architecture: Dataset Balance')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(['Brittle Engine', 'Robust Engine'])
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(models_dir, 'model_comparison_bar.png'))
            plt.close()
                
            self.get_logger().info("SUCCESS: Visuals saved to models directory!")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    node = VisualNodeV2()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass 
    except Exception as e:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()