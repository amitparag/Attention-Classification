## Attention-Classification
*Slip detection* with [Franka Emika](https://www.franka.de/) and [GelSight Sensors](https://www.gelsight.com/gelsightmini/) .

## Précis
![Screenshot from 2023-09-01 12-02-11](https://github.com/amitparag/Attention-Classification/assets/19486899/be3a25a3-36e6-43ac-a242-4a00f55a82d1)



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


## Some Problems you may face

  1. Marker Tracking

  2. Sensors

  3. Low Batch Size

  4. Minor Convergence issues in the initial epochs

  5. OpenCV issues
