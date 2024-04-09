# [FaceRefiner: High-Fidelity Facial Texture Refinement with Differentiable Rendering-based Style Transfer](https://ieeexplore.ieee.org/document/10443565)

## Overview

<p align="center"><img width="100%" src="figures/overview.png" style="background-color:white;" /></p>
The overview of our proposed FaceRefiner. The inputs of FaceRefiner include the face image I, the 3D face reconstruction results (3D model M and camera pose P, sampled texture I<sub>S</sub>) and the initial imperfect texture I<sub>C</sub> produced by an existing facial texture generation method. The differentiable rendering-based style transfer is adopted to improve the quality of I<sub>C</sub>. The differentiable renderer is employed to produce rendered image I<sub>R</sub> of the inputted camera pose P. Then the rendering loss is calculated to measure the inconsistency between rendered and inputted image, and the gradients are back-propagated to a classical style transfer module containing style and content loss to optimize the facial texture I<sub>X</sub>.
<br/>

## Requirements
**This implementation is tested under Ubuntu 22.04 environment with Nvidia GPUs 3090**

## Installation
### 1. Clone the repository and set up a conda environment as follows:
```
git clone https://github.com/HarshWinterBytes/FaceRefiner
cd FaceRefiner
conda env create -f environment.yml
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda activate face_refiner
```

### 2. Installation of Deep3DFaceRecon_pytorch
- **2.a.** Install Nvdiffrast library:
```
cd external/deep3dfacerecon/    
git clone https://github.com/NVlabs/nvdiffrast.git
pip install .
```
- **2.b.** Install Arcface Pytorch:
```
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch/ ./models/
```

- **2.c.** Prepare prerequisite models: Deep3DFaceRecon_pytorch method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat" and "BFM_model_front.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
FaceRefiner
│
└─── external
     │
     └─── deep3dfacerecon_pytorch
          │
          └─── BFM
              │
              └─── 01_MorphableModel.mat
              │
              └─── BFM_model_front.mat
              │
              └─── Exp_Pca.bin
              |
              └─── ...
```
- **2.d.** Deep3DFaceRecon_pytorch provides a model trained on a combination of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 
[LFW](http://vis-www.cs.umass.edu/lfw/), [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
[IJB-A](https://www.nist.gov/programs-projects/face-challenges), [LS3D-W](https://www.adrianbulat.com/face-alignment), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. Download the pre-trained model using this [link (google drive)](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing) and organize the directory into the following structure:
```
FaceRefiner
│
└─── external
     │
     └─── deep3dfacerecon_pytorch
          │
          └─── checkpoints
               │
               └─── face_recon
                   │
                   └─── epoch_latest.pth

```

- **2.e.** Download the pre-trained model from Arcface using this [link](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3). By default, we use the resnet50 backbone ([ms1mv3_arcface_r50_fp16](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)), organize the download files into the following structure:
```
FaceRefiner
│
└─── external
     │
     └─── deep3dfacerecon_pytorch
          │
          └─── checkpoints
               │
               └─── recog_model
                    │
                    └─── ms1mv3_arcface_r50_fp16
                         |
                         └─── backbone.pth
```

### 3. Installation of face3d

```
cd external/face3d/mesh/cython
python setup.py build_ext -i 
```

## Usage
- Run
 ```
 sh run.sh
 ```

## License
- The source code shared here is protected under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License which does **NOT** allow commercial  use. To view a copy of this license, see LICENSE


## Acknowledgement
- Our projection relies on futscdav's [STROTSS](https://github.com/futscdav/strotss)
- Thanks [OSTEC](https://github.com/barisgecer/OSTeC) for providing face visibility maps and content images
- Thanks [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for providing 3d face reconstruction and content images
- We use [MTCNN](https://github.com/ipazc/mtcnn) for face detection
- We use [face3d](https://github.com/yfeng95/face3d) for uv face rendering

## Citation
If you find this work is useful for your research, please cite our paper: 

```
@ARTICLE{10443565,
  author={Li, Chengyang and Cheng, Baoping and Cheng, Yao and Zhang, Haocheng and Liu, Renshuai and Zheng, Yinglin and Liao, Jing and Cheng, Xuan},
  journal={IEEE Transactions on Multimedia}, 
  title={FaceRefiner: High-Fidelity Facial Texture Refinement with Differentiable Rendering-based Style Transfer}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Faces;Three-dimensional displays;Rendering (computer graphics);Image reconstruction;Face recognition;Solid modeling;Cameras;facial texture generation;3D face reconstruction;style transfer},
  doi={10.1109/TMM.2024.3361728}}

```
<br/>
