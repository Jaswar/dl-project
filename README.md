# Reproducibility of "Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D)"

This repository contains our implementation for the reproducibility of the paper "Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D)" by Chan et al. 

**Authors:** Maurits Kienhuis, Petar Petrov, Niels van der Voort, Jan Warchocki (Group 29)

**Blogpost:** https://hackmd.io/@jaswar/dl-group-29

**Original paper:** https://nvlabs.github.io/eg3d/

## Running the code 

### Feature 1: Style editing

The normal vector for the eyeglasses and gender feature have been computed and are stored in the `style_editing/svm_coefs` folder.
To run the code for the eyeglasses feature, run the following command in the `eg3d` folder:

```bash
python gen_samples.py --outdir=out --trunc=0.7 --shapes=false --seeds=0-5 --network=networks/ffhq512-128.pkl --z-path=../style_editing/svm_coefs/coef.npy --alpha=1.0
```

The `z-path` variable controls where the normal vector can be found. The `alpha` variable controls the strength of the feature. 
Due to the training procedure, positive alpha will remove glasses while negative alpha will add them. Similarly, to run on
the gender feature, run:

```bash
python gen_samples.py --outdir=out --trunc=0.7 --shapes=false --seeds=0-5 --network=networks/ffhq512-128.pkl --z-path=../style_editing/svm_coefs/gender_coef.pt --alpha=1.0
```

Positive alpha will make the generated images more masculine while negative alpha will make them more feminine.

The code for the whole training procedure for the eyeglasses feature can be found in the `style_editing` folder. The 
code for the gender feature can be found in the `3dgen.ipynb` notebook.

### Feature 2: Background disappearance

Here it is sufficient to run the default command, which is as follows:

```bash
python gen_samples.py --outdir=out --trunc=0.7 --shapes=false --seeds=0-5 --network=networks/ffhq512-128.pkl
```

This will generate the faces with the extreme angles. The example used in the blogpost is seed 1.

### Feature 3: Eye extraction

The command to generate images with the eyes extracted is as follows:

```bash
python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0 --network=networks/ffhq512-128.pkl --eyes=true
```

This will generate a depth image, called `seed0000-depth0.png` and the corresponding `.mrc` file with only the eyes extracted.


