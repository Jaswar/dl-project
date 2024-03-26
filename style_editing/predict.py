import torch as th
from model import Model
from torchvision.transforms import transforms
import numpy as np
import os
from PIL import Image


def main():
    path = 'checkpoints/model.pth'
    images_path = 'data_eg3d/eg3d_gen'
    out_path = 'predictions'

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = Model(1)
    model.load_state_dict(th.load(path))
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    latents = []
    y = []
    with th.no_grad():
        for i, file in enumerate(os.listdir(images_path)):
            if file.endswith('.png'):
                seed = int(file.split('.')[0].replace('seed', ''))
                image_path = os.path.join(images_path, file)
                latent_path = os.path.join(images_path, f'seed{seed:04d}.pt')

                img = transform(Image.open(image_path)).to(device)
                latent = th.load(latent_path).squeeze(0)

                y_hat = model(img.unsqueeze(0)).item()
                y_hat = 1 if y_hat > 0.5 else 0
                y.append(y_hat)
                latents.append(latent.cpu().numpy())

            if i % 100 == 0:
                print(f'Processed {i + 1} images')
        y = np.array(y)
        latents = np.array(latents)
        print(y.shape, latents.shape)
        print(np.mean(y))
        np.save(os.path.join(out_path, 'y.npy'), y)
        np.save(os.path.join(out_path, 'latents.npy'), latents)



if __name__ == '__main__':
    main()
