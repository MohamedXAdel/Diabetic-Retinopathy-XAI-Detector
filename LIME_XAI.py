import numpy as np
import torch
from torchvision.transforms.functional import normalize
from lime import lime_image
from skimage.segmentation import mark_boundaries
from src.preprocess import preprocess_image


class LIME_Explainer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.model.eval()
        self.device = device

        # LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images):
        """
        LIME provides numpy images (H,W,3), so we:
        - convert to tensor
        - normalize
        - run through model
        - return softmax probabilities
        """

        # Convert to tensor: (N, H, W, 3) → (N, 3, H, W)
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.tensor(images, dtype=torch.float32).to(self.device)

        # Apply normalization like validation transforms
        images = np.transpose(images, (0, 3, 1, 2)) 
        images = torch.tensor(images, dtype=torch.float32, device=self.device)

        # scale to [0,1]
        images = images / 255.0
        
        # normalization
        images = normalize(
            images,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )



        with torch.no_grad():
            logits = self.model(images)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs = torch.cat([1 - probs, probs], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
        return probs

    def explain(self, image_np, top_label=None, num_samples=1000):
        """
        image_np must be RGB (H,W,3)
        """

        explanation = self.explainer.explain_instance(
            image_np,
            self.predict_fn,
            top_labels=1 if top_label is None else [top_label],
            hide_color=0,
            num_samples=num_samples,
        )

        label = explanation.top_labels[0] if top_label is None else top_label

        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=7,
            hide_rest=False,
        )

        lime_img = mark_boundaries(temp / 255.0, mask)
        return lime_img, label
