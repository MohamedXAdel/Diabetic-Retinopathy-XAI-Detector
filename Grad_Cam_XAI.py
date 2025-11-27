import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Capture the first element of the tuple
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        #Forward Pass
        # Ensure we are tracking gradients for XAI, even in eval mode
        with torch.set_grad_enabled(True):
            output = self.model(x)
            
            # Handle Binary Model 
            if output.shape[1] == 1:
                # Binary case: We always visualize "Evidence for DR" (Class 0 in the tensor is the only class)
                target_index = 0
            else:
                # Multiclass case
                if class_idx is None:
                    target_index = output.argmax(dim=1).item()
                else:
                    target_index = class_idx

            #  Backward Pass
            self.model.zero_grad()
            
            # Create one-hot target
            one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(x.device)
            one_hot_output[0][target_index] = 1
            
            output.backward(gradient=one_hot_output, retain_graph=True)
            
            # Generate Map
            if self.gradients is None or self.activations is None:
                print("Error: Hooks did not fire. Check model architecture.")
                return np.zeros((224, 224)), 0

            # Global Average Pooling of gradients
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            
            # Weight the activations
            activations = self.activations.detach().clone()
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            #Average the channels to get the heatmap
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            # ReLU to keep only positive features
            heatmap = F.relu(heatmap)
            
            # Normalize
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
            
            return heatmap.cpu().numpy(), target_index



def overlay_heatmap(img_path, heatmap):
    """ Helper to overlay heatmap on original image """
    img = cv2.imread(img_path)
    if img is None: return None, None
    img = cv2.resize(img, (224, 224)) # IMG_SIZE
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return img[:,:,::-1], superimposed_img[:,:,::-1]