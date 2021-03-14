from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def ImageLoader(image_path, size=(300, 480)):
    h, w = size
    image = Image.open(image_path).resize((w, h))
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image

def ImageShow(tensor, title, size=(10, 8), save=False):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)    
    image = transforms.ToPILImage()(image)
    if save:
        image.save(title+".jpg")
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.title(title)
    plt.show()