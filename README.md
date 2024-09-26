  

## Compile image

    cd docker
    bash run.sh --with-cuda --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"

  It will run the simple_interface.py, use all data inside the images_in folder and outputs data into output folder.
  

## runs a pre-compiled image and it's up to the user know what should be ran


    cd (dust3r path)
    docker run -it -v ./:/dust3r -p 8888:8888 --gpus all docker-dust3r-demo:latest bash
## Output Strucure

```
output
│   out.csv
│   1.csv    
│   2.csv ...
```
### out.csv columns

 - 1-3 columns -> XYZ
 - 4-6 cols. -> RGB values normalized - min:0, max:1
 - 7 col -> confidence value 
 
 ### %i.csv
  - 4x4 homogenous transformation for each camara in scene

