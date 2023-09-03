## Attention-Classification
*Slip detection* with [Franka Emika](https://www.franka.de/) and [GelSight Sensors](https://www.gelsight.com/gelsightmini/) .

## Précis
The aim of the experiments is to learn the difference between slip and wriggle through videos by training a Video-Vision Transformer model.

![Screenshot from 2023-09-01 12-02-11](https://github.com/amitparag/Attention-Classification/assets/19486899/be3a25a3-36e6-43ac-a242-4a00f55a82d1)

Video Vision Tranformers were initially proposed in this [paper](https://arxiv.org/abs/2103.15691). We use the the first variant - spatial transformer followed by a temporal one - in our experiments. 

## Requirements

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

      Marker tracking algorithms may fail to converge or compute absurd vector fields. We experimented with marker tracking but ended up not using them.
       

  2. Sensors

      The Gelsight sensors are susceptible to damage.
    
  

  4. Low Batch Size

  5. Minor Convergence issues in the initial epochs

  6. OpenCV issues

### Acknowledgements
