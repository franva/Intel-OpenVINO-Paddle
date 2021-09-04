# Intel-OpenVINO-Paddle

[[English]](README.md) | [[简体中文]](README_CN.md)

Note: The work presented in this repository is for demo and preview purposes only. 

This repository is a companion repo which is meant to be used with a tutorial I wrote on the Intel.cn AI Developer Community Forum for deploying Paddle model to the Intel Movidius Myraid X VPU.

The tutorial will include all details about how to convert Paddle models to ONNX, then from there converting to the Intel OpenVINO IR model. It then shows you how to verify the converted model to make sure you get the correct and functional model before moving to the next step. 
At the end, it also shows how to compile the OpenVINO IR model to a smaller size model with an extension `.blob`. You will get your model deployed onto the edge devices, in my case, it's the OAK-D camera which has Intel Movidius Myriad X VPU on board. This means, the code will also work if you use the VPU directly.

I will update the readme to provide the link of the tutorial in a few days time.