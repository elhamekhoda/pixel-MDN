# Mixture Density Netowrk (MDN) to spllit merged pixel clusters

----

### MDN training code
These scripts are relevant for only the training part. There are other scripts (not included here) for producing the training inputs.

#### software environment
* Python 3.5.1
* keras 2.0.4
* theano 0.9.0 (as backend)
* ROOT 6.08/02
* gcc 4.9.3

#### Training inputs (total 60)
* 7x7 pixel charge map
* length-7 vector with pixel y-pitches
* layer information (0,1,2,3)
* barrel or endcap (barrel = 0, endcap = -2 or +2 )
* track theta
* track phi

#### Training outputs
Parameters of the mixture model. In this case the mean and std of the Gaussian kernels.
* Estimated position = Gaussian $`\mu`$
* Estimated uncertainty = Gaussian $`\sigma`$


