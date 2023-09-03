## Attention-Classification
*Slip detection* with [Franka Emika](https://www.franka.de/) and [GelSight Sensors](https://www.gelsight.com/gelsightmini/) .

**Author** - [Amit Parag](https://scholar.google.com/citations?user=wsRIfL4AAAAJ&hl=en)

**Instructor** - [Ekrem Misimi](https://www.sintef.no/en/all-employees/employee/ekrem.misimi/)

## Précis
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

Two examples are shown below :


https://github.com/amitparag/Attention-Classification/assets/19486899/5bfce6da-073d-45e8-86b1-123f00ec70f9




https://github.com/amitparag/Attention-Classification/assets/19486899/f70485a8-5e0e-4f24-bf60-bc692965dd3f




The occurence of slip is usually characterized by the properties of object in question such  as its weight, elasticity, orientation of grip. 

One example of slip is shown below.



https://github.com/amitparag/Attention-Classification/assets/19486899/bfd973d2-94ac-4395-aab3-194bfe0f313c



This motion is repeated for 30 objects. 


The resulting (slip) video (from one of the experiments) from the sensor attached to the gripper is shown below. This is the dataset used for training.  



https://github.com/amitparag/Attention-Classification/assets/19486899/9fb5e856-824f-4e60-9db5-2d167f1b0cc8

After the data has been collected, we augment the data by adding 

            from imgaug import augmenters as iaa

            iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur(k=(3, 11)),
        ]),
        # Strengthen or weaken the contrast in each image.
        sometimes(
            iaa.LinearContrast((0.75, 1.5))
        ),
        sometimes(
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)


For data augmentation, see [1 - Transforming and Augmenting the dataset.ipynb](https://github.com/amitparag/Attention-Classification/blob/main/1%20-%20Transforming%20and%20Augmenting%20the%20dataset.ipynb).

After this, we use [2 - Dataset.ipynb](https://github.com/amitparag/Attention-Classification/blob/main/2%20-%20Dataset.ipynb) to create train, test, validation directories (See below)

The third step is training the model. Code is provided in [training.py](https://github.com/amitparag/Attention-Classification/blob/main/training.py) 

After that validation and plotting results.


## Training

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
    





## Model Architecture

    • image_size         =  (240,320), # image size
    
    • frames             =   450, # number of frames
    
    • image_patch_size   =   (80,80), # image patch size
    
    • frame_patch_size   =   45, # frame patch size
    
    • num_classes        =   2,
    
    • dim                =   64,
    
    • spatial_depth      =   3, # depth of the spatial transformer
    
    • temporal_depth     =   3, # depth of the temporal transformer
    
    • heads              =   4,
    
    • mlp_dim            =  126


Training a bigger model on 16 or 32 Gb RAM leads to the script getting automically killed. 
So, if you want to try it, make sure you have access to compute clusters and adapt the code for gpu. Should be fairly straightforward. 


## Certain problems you may face

    1. Marker Tracking
  
        Marker tracking algorithms may fail to converge or ends up computing absurd vector fields. We experimented with marker tracking but ended up not using them.
         
  
    2. Sensors
  
        The Gelsight sensors are susceptible to damage. After a few experiments, the gel pad on one the sensors started to leak gel while second one somehow got scrapped off.
        We initially started with 2 sensors, but then discarded the data from one of the sensors.

<div style="display: flex; justify-content: center;">
    <img src="assets/Screenshot from 2023-09-03 15-18-07.png" alt="Image 1" style="width: 200px; margin-right: 10px;">
    <img src="assets/Screenshot from 2023-09-03 15-17-53.png" alt="Image 2" style="width: 200px;">
</div>

      


  The resulting data is unusable

  https://github.com/amitparag/Attention-Classification/assets/19486899/df3ed124-8a7d-4620-bab5-38bbe121bc3f

    Also note that the regular 3D printed grippers can develop cracks and break. We initially used a normal 3D printer and then eventually a more "fancy" one.
  
    4. Low Batch Size

        The training script uses a batch size of 4. While it is generally preferable to have a higher batch size, restrictions due to compute capabilities still apply.
  
    5. Minor Convergence issues in the initial epochs

        Sometimes, the network gets stuck in local minima. Either restart the experiment with different learning rate or let it run for a few more epochs.
        For example, in one of the experiments, the network was trapped in a local minima - the validation accuracy score remained unchanged for 100 epochs for learning rate of 1e-3.
        The usual irritating local minima stuff - change some parameter slightly. 
        
  
    6. OpenCV issues

        There a a few encoding issues with opencv something to do with how it compresses and encodes data. Use ![PyAV](https://pypi.org/project/av/), ![Imgaug](https://github.com/aleju/imgaug)
        and ![ImageIO-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) for processing and augmenting the dataset.


## Requirements

See [requirements.txt](https://github.com/amitparag/Attention-Classification/blob/main/requirements.txt)
    
Numpy, preferably 1.20.0. Higher versions have changed numpy.bool to bool. Might lead to clashes.

Use [Pyav](https://pypi.org/project/av/), [Imgaug](https://github.com/aleju/imgaug)
and [Imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) for processing and augmenting the dataset.


    
### Acknowledgements

We thank [Prof Edward Adelson](https://bcs.mit.edu/directory/edward-adelson) for useful discussions and providing us with newer softer Gelsight Sensors. 
