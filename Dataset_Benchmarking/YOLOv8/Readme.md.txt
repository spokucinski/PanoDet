Benchmarking with use of YOLOv8 is based on the official Docker image of ready-to-use container.

To correctly start the container run the following command:

```
docker run --ipc=host -it -v D:\YOLOv8Mount\datasets:/usr/src/datasets --name YOLOv8Container --gpus all ultralytics/ultralytics:latest
```