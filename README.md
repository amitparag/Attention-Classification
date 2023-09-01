## Attention-Classification
Slip detection with Franka Emika and GelSight Sensors.

## Précis

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
