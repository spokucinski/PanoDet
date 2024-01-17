Benchmarking with use of YOLOv5 is based on the official Docker image of ready-to-use container.

To correctly start the container run the following command:

```
docker run --ipc=host -it -v D:\YOLOv5Mount\datasets:/usr/src/datasets --name YOLOv5Container --gpus all ultralytics/yolov5:latest
```