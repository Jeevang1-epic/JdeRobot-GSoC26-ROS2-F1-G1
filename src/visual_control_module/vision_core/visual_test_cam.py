import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import VisualDriveNet

class ActivationVisualizer(Node):
    def __init__(self):
        super().__init__('activation_visualizer')
        self.cvb = CvBridge()
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
        
        self.get_logger().info(f"visual interpretability engine online.")

    def img_cb(self, msg):
        try:
            cv_image = self.cvb.imgmsg_to_cv2(msg, 'bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                x = F.elu(self.model.conv1(input_tensor))
                x = F.elu(self.model.conv2(x))
                x = F.elu(self.model.conv3(x))
                x = F.elu(self.model.conv4(x))
                features = F.elu(self.model.conv5(x))
                
                x_flat = self.model.flatten(features)
                x_fc = F.elu(self.model.fc1(x_flat))
                x_fc = F.elu(self.model.fc2(x_fc))
                x_fc = F.elu(self.model.fc3(x_fc))
                steering = self.model.fc4(x_fc).item()

            activation = torch.mean(features, dim=1).squeeze().cpu().numpy()
            activation = np.maximum(activation, 0)
            if np.max(activation) > 0:
                activation /= np.max(activation)
            
            heatmap = cv2.resize(activation, (cv_image.shape[1], cv_image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            overlay = cv2.addWeighted(cv_image, 0.6, heatmap, 0.4, 0)
            
            cv2.putText(overlay, f"Steering: {steering:.4f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Robust AI: Live Activation Map", overlay)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"stream error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ActivationVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()