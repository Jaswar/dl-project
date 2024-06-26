# Style editing: eyeglasses

This folder contains the code required to perform style editing from scratch for the eyeglasses feature. Please
note that additional libraries are required for this folder. These are `torchvision`, `sklearn`, `pandas`, `cv2`.

Below is the overview of files:

 - `model.py`: Contains the ResNet model used for classification whether a person wears glasses or not.
 - `prepare_data.py`: Helper script that organizes the CelebA dataset to be valid with the ImageFolder from PyTorch. 
The dataset can be downloaded from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).
 - `main.py`: Trains the model from `model.py` on the CelebA dataset.
 - `utils.py`: Training and evaluation subroutines used by `main.py`.
 - `predict.py`: Runs the model trained from above on the images generated by eg3d. Generates a training dataset for the SVM.
 - `svm.py`: Trains an SVM on the dataset generated by `predict.py`. Saves the resulting normal vector to a file.

Recreating the eyeglasses feature from scratch requires the steps as outlined the steps from the blogpost. This can
be achieved by running the files from this directory. Make sure that the working directories are correct in each file.
