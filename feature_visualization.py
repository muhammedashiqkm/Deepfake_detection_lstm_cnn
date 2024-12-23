import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import seaborn as sns
from scipy.ndimage import zoom
import logging
from deepfake_detector import DeepfakeDetector, VideoFrameDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureVisualizer:
    def __init__(self, model, device, save_dir='visualizations'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Register hooks for feature extraction
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for EfficientNet layers
        for name, module in self.model.efficient_net.named_children():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(get_features(f'conv_{name}'))
    
    def visualize_feature_maps(self, input_tensor, layer_name, num_features=8):
        """Visualize feature maps from a specific layer"""
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor.unsqueeze(0).to(self.device))
        
        # Get feature maps
        feature_maps = self.features[layer_name][0].cpu()
        
        # Create grid of feature maps
        num_maps = min(num_features, feature_maps.size(0))
        fig, axes = plt.subplots(2, num_maps//2, figsize=(15, 6))
        axes = axes.ravel()
        
        for idx in range(num_maps):
            feature_map = feature_maps[idx].numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            axes[idx].imshow(feature_map, cmap='viridis')
            axes[idx].axis('off')
            axes[idx].set_title(f'Filter {idx}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{layer_name}_feature_maps.png'))
        plt.close()
    
    def visualize_activation_maps(self, input_tensor, original_image):
        """Generate class activation maps (CAM)"""
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor.unsqueeze(0).to(self.device))
            
        # Get the weights from the last fully connected layer
        fc_weights = self.model.fc[-1].weight.data
        
        # Get the feature maps from the last convolutional layer
        feature_maps = self.features[list(self.features.keys())[-1]][0]
        
        # Create class activation map
        batch_size, num_channels, height, width = feature_maps.size()
        cam = torch.zeros(height, width).to(self.device)
        
        # Use the weights to combine feature maps
        for idx, weight in enumerate(fc_weights[1]):  # Use weights for fake class
            cam += weight * feature_maps[idx]
        
        # Normalize CAM
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Resize CAM to match input image size
        cam = zoom(cam, original_image.shape[:2] / np.array(cam.shape))
        
        # Create heatmap
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Overlay heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title('Activation Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'activation_map.png'))
        plt.close()
    
    def visualize_lstm_attention(self, input_tensor):
        """Visualize LSTM temporal attention"""
        self.model.eval()
        
        # Forward pass to get LSTM outputs
        with torch.no_grad():
            batch_size, sequence_length, c, h, w = input_tensor.size()
            
            # Get CNN features
            cnn_input = input_tensor.view(-1, c, h, w).to(self.device)
            cnn_features = self.model.efficient_net(cnn_input)
            cnn_features = cnn_features.view(batch_size, sequence_length, -1)
            
            # Get LSTM outputs
            lstm_out, (hidden, cell) = self.model.lstm(cnn_features)
            
            # Calculate attention weights using the last hidden state
            attention_weights = torch.bmm(lstm_out, hidden[-1].unsqueeze(2))
            attention_weights = torch.softmax(attention_weights.squeeze(2), dim=1)
        
        # Plot attention weights
        plt.figure(figsize=(10, 4))
        sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis')
        plt.xlabel('Frame')
        plt.ylabel('Attention Weight')
        plt.title('LSTM Temporal Attention')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'lstm_attention.png'))
        plt.close()

def visualize_sample(model_path, sample_video_path, device='cuda'):
    """Visualize features for a sample video"""
    # Load model
    model = DeepfakeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize visualizer
    visualizer = FeatureVisualizer(model, device)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load video frames
    dataset = VideoFrameDataset([sample_video_path], [0], transform=transform)
    frames_tensor = dataset[0][0]  # Get first item (frames)
    
    # Get original frame for visualization
    cap = cv2.VideoCapture(sample_video_path)
    ret, original_frame = cap.read()
    cap.release()
    if ret:
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Could not read video frame")
    
    # Visualize features
    logger.info("Generating feature visualizations...")
    
    # Visualize feature maps from different layers
    for layer_name in visualizer.features.keys():
        visualizer.visualize_feature_maps(frames_tensor[0], layer_name)
        logger.info(f"Generated feature maps for layer {layer_name}")
    
    # Visualize activation maps
    visualizer.visualize_activation_maps(frames_tensor[0], original_frame)
    logger.info("Generated activation maps")
    
    # Visualize LSTM attention
    visualizer.visualize_lstm_attention(frames_tensor.unsqueeze(0))
    logger.info("Generated LSTM attention visualization")

if __name__ == "__main__":
    # Example usage
    model_path = "metrics/best_model.pth"  # Path to your trained model
    sample_video_path = "path_to_sample_video.mp4"  # Path to a sample video
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_sample(model_path, sample_video_path, device)
