# Computer Vision Project

This is a project for the course: **Computer Vision** of the University of Trento.

Students:
- Giovanni Lorenzini    223715
- Davide Dalla Stella   223727

## Results

### Chair model

Image of the chair mesh generated from the photos using nerf_pl:

![chair](https://user-images.githubusercontent.com/8526483/206315283-b1f43f68-09bd-4550-90e6-407fd7cea46d.png)

### Human-object interaction

Video of the capture in Motive software:

![motive](https://user-images.githubusercontent.com/8526483/206319338-88c03508-4641-463e-8120-c18c95f45338.gif)

Video of the final result in Unreal Engine:

![unreal](https://user-images.githubusercontent.com/8526483/206318766-1908d13d-21c7-4e8c-9822-8b90391f7152.gif)

## Instructions

If you want to reproduce the results, follow the instructions below.

Clone the repository:
```bash
git clone --recursive https://github.com/lorenzinigiovanni/cv-project.git
```

### Model generation

Extract the 3D model of the chair from some photos using [nerf_pl](https://github.com/kwea123/nerf_pl/tree/dev).

Procedure:
- take circa 50 photos of the chair from different angles;
- use [COLMAP](https://github.com/colmap/colmap) taking in input the photos to generate the sparse model through sparse reconstruction;
- using [LLFF](https://github.com/Fyusion/LLFF) prepare the dataset for the training using the following command:
```bash
python img2poses.py chair
```
- use [nerf_pl](https://github.com/kwea123/nerf_pl/tree/dev) to train the model using the dataset generated in the previous step:
```bash
python train.py  --dataset_name llff --root_dir chair --N_importance 64 --img_wh 504 378 --num_epochs 30 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5  --exp_name chair --spheric
```
- optional: use [nerf_pl](https://github.com/kwea123/nerf_pl/tree/dev) to generate a gif of the views:
```bash
python eval.py --root_dir chair --dataset_name llff --scene_name chair --img_wh 400 400 --N_importance 64 --ckpt_path ckpts/chair/epoch\=17.ckpt --spheric_poses
```
- extract the mesh using `extract_mesh.py` script modifying it to reflect the correct paths and using the last `ckpt`.
 
Look at the `dataset` folder for an example of the dataset.

### Human-object interaction

Make the model interacts with a human model through [OptiTrack](https://optitrack.com) and [Unreal Engine](https://www.unrealengine.com).

For a better understanding of the following procedure, look at this [tutorial](https://docs.optitrack.com/plugins/optitrack-unreal-engine-plugin/unreal-engine-motionbuilder-workflow).

Procedure for OptiTrack:
- attach some markers to the chair and to the human model;
- in Motive software:
    - create a rigid body for the chair;
    - create a skeleton for the human model;
    - enable the streaming of the data.

Procedure in MotionBuilder:
- use both OptiTrack and Unreal Engine plugins;
- associate the skeleton to the human model;
- export the animation to Unreal Engine using the LiveLink plugin.

Procedure in Unreal Engine:
- in Quixel Bridge download a Metahuman model that you like;
- create an animation blueprint for the human model that uses the LiveLink streaming from MotionBuilder;
- import the chair model and add the LiveLink plugin to it;
- start the simulation.
