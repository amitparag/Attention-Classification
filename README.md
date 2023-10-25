### Attention-Classification
*Slip detection* with [Franka Emika](https://www.franka.de/) and [GelSight Sensors](https://www.gelsight.com/gelsightmini/) .

**Author** - [Amit Parag](https://scholar.google.com/citations?user=wsRIfL4AAAAJ&hl=en)

**Instructor** - [Ekrem Misimi](https://www.sintef.no/en/all-employees/employee/ekrem.misimi/)

### Précis
The aim of the experiments is to learn the difference between slip and wriggle through videos by training a Video-Vision Transformer model.

![Screenshot from 2023-09-01 12-02-11](https://github.com/amitparag/Attention-Classification/assets/19486899/be3a25a3-36e6-43ac-a242-4a00f55a82d1)

Video Vision Tranformers were initially proposed in this [paper](https://arxiv.org/abs/2103.15691). 

We use the first variant - spatial transformer followed by a temporal one - in our experiments. 

The training dataset were collected by performing the wriggling motion.

We define "wriggle" as a sequence of motions that involve 

    lifting an object, 
    
    rotationally shaking it 
    
    followed by tangential shake, vertical shake and perpendicular shake. 
    
    The object is then put back on the table.

The objects used for experiments are listed in [object_info.txt](https://github.com/amitparag/Attention-Classification/blob/main/objects_info.txt)

Two examples are shown below :



https://github.com/amitparag/Attention-Classification/assets/19486899/5bfce6da-073d-45e8-86b1-123f00ec70f9




https://github.com/amitparag/Attention-Classification/assets/19486899/f70485a8-5e0e-4f24-bf60-bc692965dd3f




The occurence of slip is usually characterized by the properties of object in question such  as its weight, elasticity, orientation of grip. 

One example of slip is shown below.


https://user-images.githubusercontent.com/19486899/265270077-bfd973d2-94ac-4395-aab3-194bfe0f313c.mp4







This motion is repeated for 30 objects. 


The resulting (slip) video (from one of the experiments) from the sensor attached to the gripper is shown below.


https://github.com/amitparag/Attention-Classification/assets/19486899/ffdab1a0-f602-4898-9168-f3077726b9d3


https://github.com/amitparag/Attention-Classification/assets/19486899/9fb5e856-824f-4e60-9db5-2d167f1b0cc8

An example of wriggle is



https://github.com/amitparag/Attention-Classification/assets/19486899/11a95897-6bb7-4f0d-ac93-71ddb5483f3f


After the data has been collected, we augment the data by adding noise and swapping channels in each video


A transformed video of 5 frames would look like:


https://github.com/amitparag/Attention-Classification/assets/19486899/53fcfa95-814f-4430-8d32-452754c32f6b


Data from 25 objects were kept aside for training. After data transformation the new augmented dataset contained 110 slip cases and 408 wriggle cases.




### Training

  For training, the data folder needs to be arranged like so -

      root_dir/
      
        ├── train/
    
         ├── slip/
      
             ├── video1.avi
      
             ├── video2.avi
      
             └── ...
       
        └── wriggle/
         
            ├── video1.avi
      
            ├── video2.avi
      
            └── ...
  
        
      ├── test/
    
         ├── slip/
      
             ├── video1.avi
      
             ├── video2.avi
      
             └── ...
       
        └── wriggle/
         
            ├── video1.avi
      
            ├── video2.avi
      
            └── ...
  
  
              
      ├── validation/
    
         ├── slip/
      
             ├── video1.avi
      
             ├── video2.avi
      
             └── ...
       
        └── wriggle/
         
            ├── video1.avi
      
            ├── video2.avi
      
            └── ...
    





### Model Architecture

    • image_size         =  (240,320), # image size
    
    • frames             =   450, # number of frames
    
    • image_patch_size   =   (80,80), # image patch size
    
    • frame_patch_size   =   45, # frame patch size
    
    • num_classes        =   2,
    
    • dim                =   64,
    
    • spatial_depth      =   3, # depth of the spatial transformer
    
    • temporal_depth     =   3, # depth of the temporal transformer
    
    • heads              =   4,
    
    • mlp_dim            =  128


Training a bigger model on 16 or 32 Gb RAM leads to the script getting automatically killed. 
So, if you want to try it, make sure you have access to compute clusters and adapt the code for gpu. Should be fairly straightforward. 
This architecture took 17.35 hours to train for 250 epochs.


### Certain problems you may face

    1: Installing real time kernel. See requirements below.

    1. Marker Tracking
  
        Marker tracking algorithms may fail to converge or ends up computing absurd vector fields. We experimented with marker tracking but ended up not using them.
         
  
    2. Sensors
  
        The Gelsight sensors are susceptible to damage. After a few experiments, the gel pad on one the sensors started to leak gel while second one somehow got scrapped off.
        We initially started with 2 sensors, but then discarded the data from one of the sensors. 
        
        

<div style="display: flex; justify-content: center;">
    <img src="assets/Screenshot from 2023-09-03 15-12-14.png" alt="Image 1" style="width: 400px; margin-right: 10px;">
    <img src="assets/Screenshot from 2023-09-03 15-12-44.png" alt="Image 2" style="width: 400px;">
</div>

      


  The resulting data is unusable

  https://github.com/amitparag/Attention-Classification/assets/19486899/df3ed124-8a7d-4620-bab5-38bbe121bc3f

    Also note that the regular 3D printed grippers can develop cracks and break. 
    We initially used a normal 3D printer and then eventually a more "fancy" one, for instance, in the video "Coil of Wires", different grippers are used.
    It should also be noted that the usb-c cabel connected to the GelSight sensors gets disconnected a lot in the middle of experiments. So you will have to redo the same experiment multiple times - frustrating but c'est la vie.
    The pins of the mini sensor is a bit dodgy.
    
    
    4. Low Batch Size

        The training script uses a batch size of 4. While it is generally preferable to have a higher batch size, restrictions due to compute capabilities still apply.
  
    5. Minor Convergence issues in the initial epochs

        Sometimes, the network gets stuck in local minima. Either restart the experiment with different learning rate or let it run for a few more epochs.
        For example, in one of the experiments, the network was trapped in a local minima - the validation accuracy score remained unchanged for 100 epochs for learning rate of 1e-3.
        The usual irritating local minima stuff - change some parameter slightly. 
        
  
    6. OpenCV issues

        There a a few encoding issues with opencv something to do with how it compresses and encodes data.


### Requirements

See [requirements.txt](https://github.com/amitparag/Attention-Classification/blob/main/requirements.txt)
    
Numpy, preferably 1.20.0. Higher versions have changed numpy.bool to bool. Might lead to clashes.

Use [Pyav](https://pypi.org/project/av/), [Imgaug](https://github.com/aleju/imgaug)
and [Imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) for processing and augmenting the dataset.

See [notes](https://github.com/amitparag/Attention-Classification/tree/main/notes) for instructions on installing real time kernel and libfranka. 
    
#### Acknowledgements

