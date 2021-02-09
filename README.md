# training ensemble nets

ensemble nets are a single pass way of representing M models in a neural
net ensemble. see this [blog post](http://matpalm.com/blog/ensemble_nets)

note: blog post was based on a vmap version run on my local single GPU machine.
the code for this first version is under tag v1.

the code as is now is a port of the model to run on a pod tpu slice using haiku