# Build docker image:

```bash
docker build --build-arg SAGE_STORE_URL=${SAGE_STORE_URL} --build-arg SAGE_USER_TOKEN=${SAGE_USER_TOKEN} --build-arg BUCKET_ID_MODEL=${BUCKET_ID_MODEL} -t iperezx/training-smokedetect:0.1.0 .
```

# Run docker image:

```bash
docker run -v ${PWD}:/src --gpus all -it -p 8888:8888 gao/lightening:1.0 bash
```

docker run --gpus all gao/lightening:1.0 nvidia-smi

# Attach to container and run jupyter notebook:

```bash
docker attach iperezx/training-smokedetect:0.1.0
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

docker run --rm -it --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 -v nvidia_driver_367.57:/usr/local/nvidia:ro --name $CONTAINER_NAME -p 3000:3000 $CONTAINER_IMG:$CONTAINER_VERSION $CMD

# Access the notebook through your desktops browser on http://localhost:8888

http://127.0.0.1:8888/lab?token=61f0e62dfa9615e5920b1c6bf9220d32f30ca0149878c06e
