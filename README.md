# training ensemble nets

ensemble nets are a single pass way of representing M models in a neural
net ensemble. see this [blog post](http://matpalm.com/blog/ensemble_nets)


```
# to reproduce the base line tuning
python3 tune_with_ax.py --mode siso
```

```
# to reproduce the single_input case
python3 tune_with_ax.py --mode simo
```

```
# to reproduce the multi_input case
python3 tune_with_ax.py --mode mimo
```

```
# to reproduce the single_input case with logit dropout
python3 tune_with_ax.py --mode simo_ld
```

see the notebooks under the blog/ folder to reproduce additional
figures in the blog post
