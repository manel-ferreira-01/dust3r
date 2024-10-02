  

## Compile image

``` bash
cd docker/files
docker build -f ./cpu.Dockerfile -t docker-dust3r .
```
  
change for  ```cuda.dockerfile``` if using nvida+cuda

  

## launch container from image

  
``` bash
cd (dust3r path)
xhost +local:* && docker run -it -v ./:/dust3r --network host \
				 -e DISPLAY=$DISPLAY \
				 --gpus all \
				 docker-dust3r \
				 bash
```
Pass gpus accordingly with your setup.

## simple_interface.py
Reads images from a folder and returns point clouds.

### arguments:
- -h, --help           show this help message and exit
- --outdir OUTDIR      Output directory e.g. ./output
- --device DEVICE      Device to use (cpu or cuda)
- --filelist FILELIST  Path to the filelist containing input images, e.g ./images_in


## Output Strucure

  

```
output
│ out.csv
│ 1.csv
│ 2.csv ...
```

### out.csv columns
- 1-3 columns -> XYZ

- 4-6 cols. -> RGB values normalized - min:0, max:1

- 7 col -> confidence value

### %i.csv

- 4x4 homogenous transformation for each camara in scene