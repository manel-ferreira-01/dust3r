build docker:

cd docker
bash run.sh --with-cuda --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"


docker run -it -v ./:/dust3r -p 8888:8888 --gpus all docker-dust3r-demo:latest bash