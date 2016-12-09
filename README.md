Pointer Networks in Tensorflow
===============


### Introduction
This repo has sample code for pointer networks (Vinyals 2015) over a toy problem indexing a high-value segment embedded between two low-value segments.

See https://medium.com/@devnag/ for the relevant blog post.


### Running
Run the sample code by typing:


```
./example_pointer_network.py
```

...and you'll train a simple pointer network on a low/high/low sequence indexing task, then test is on a separate data set with different segment lengths. The loss should drop to below 1% after about 2000 training steps (restarting if it stagnates), and will then attempt the test.
