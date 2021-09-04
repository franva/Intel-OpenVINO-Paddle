# Intel-OpenVINO-Paddle
[[English]](README.md) | [[简体中文]](README_CN.md)

本代码库是我发布在Intel AI中国开发者社区上的一篇关于如何转换Paddle模型到Intel OpenVINO格式的文章的伴随代码库. 里面集成了展示代码,用来帮助读者成功配置转换Paddle, ONNX模型到Intel OpenVINO的IR格式,并且进一步展示如何编译IR格式的模型 从来进一步缩小模型体积.

上文提到的教程会包括所有信息,比如如何转百度Paddle模型到ONNX模型, 然后在进入到下一个模型转换环节之前 会展示如果验证已转换好的ONNX模型 来测试它是否能够正常工作.
接下来展示如何把ONNX模型转换到Intel OpenVINO的IR格式.最终会手把手展示如何编译IR模型,使你拥有更小的模型体积.
最后的最后,教程还会展示如何把这个模型部署到边缘设备上,在此教程里 我们会用到OAK-D相机,它里面集成了Intel的Movidius Myriad X VPU.所以如果你直接用Intel的VPU也会同样管用.

如何转换Paddle模型到Intel OpenVINO格式的文章的链接会在接下来几天更新.