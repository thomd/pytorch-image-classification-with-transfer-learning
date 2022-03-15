import math
import numpy as np
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import cv2

def create_image_grid(tensor, true_labels, pred_labels, path, nrow=8, limit=None, pad=12):
    tensor = tensor.cpu()
    if limit is not None:
        tensor = tensor[:limit, ::]
        true_labels = true_labels[:limit]
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + pad), int(tensor.size(3) + pad)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + pad, width * xmaps + pad), 1)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            t = tensor[k]
            img = cv2.UMat(np.asarray(np.transpose(t.numpy(), (1, 2, 0)) * 255).astype('uint8'))
            text = f'{str(pred_labels[k])}'
            image = cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 12, cv2.LINE_AA) # outline
            color = (0, 255, 0) if pred_labels[k] == true_labels[k] else (255, 0, 0)
            image = cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, color, 3, cv2.LINE_AA)
            t = transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + pad, height - pad).narrow(2, x * width + pad, width - pad).copy_(t)
            k += 1

    npimg = grid.numpy()
    plt.figure(figsize=(xmaps * 2.4, ymaps * 2.4), dpi=100)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
