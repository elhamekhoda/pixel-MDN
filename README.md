# Mixture Density Netowrk (MDN) to spllit merged pixel clusters

----

### MDN training code
These scripts are relevant for only the training part. There are other scripts (not included here) for producing the training inputs.
* `train-MDN.py` is the main trainign script. It can be used for all three networks. The inputs should be in h5 format
* `genconfig.py` creates the network configuration
* `evalMDN.py` evalues the trained networks on the test set. It stores the outputs in a SQL data base
* `gensql_posErr.py` creates the SQL query to read the data base
* `residual.cxx` used while reading the SQL data base for ceating histograms
* `test-driver` executable. Runs SQL query and the residual.cxx script
* `root2json_MDNvars.py` used to convert the MDN variables (from a root file) into json format for `lwtnn`

For the MDN, the training sample is produced with 
[L-G's code](https://gitlab.cern.ch/Atlas-Inner-Tracking/pixel-NN), however,
before we convert the training sample from root to h5 file, we need to normalize all the input feature(totally 60) in the root file first. 
Here for the "normalize", the mean is shifted to be zero and the std to be 1. 
To normalize the sample, first calculate the mean and rms for each input feature:

python makemean.py --input training.root 

python makerms.py --input training.root

They will generate a "mean.txt" and "rms.txt".

Then run the normalization script:

root -l training.root

.L normalize.C 

normalize* nor = new normalize(NNinput)

nor->Loop()

This will generate "normalized.root". Use TBrowser to check that for each input feature the mean is 0, the rms is 1.

Next, convert the root file "normalized.root" to h5 file:

toh5.py --input normalized.root --output training.h5 --type pos1(pos2 or pos3)

Now the training.h5 is ready for the MDN training.
####How to run the code:
1. Training script:

```python
python train-MDN.py --training_input PATH-TO-INPUT-FILE(.h5) --training_output OUTPUT-PATH --outFile OUTPUT-SUFFIX --network_type NETWORK-TYPE (1particle, 2particle, 3particle) --config <(python $PWD/genconfig.py --type TYPE (pos1,po2,po3))
```
`--type`: takes values `pos1` for 1 particle network, `pos2` for 2 particle network and `pos3` for 3 particle network

`--network_type`: possible values = 1particle, 2particle, 3particle

`train-MDN_2p.py` is an example code for 2-particle training

Make sure you are using the correct hyperparameter. For 1-particle, the structure is [100,50,50], for 2/3-particle, the structure is [100,80,50].

2. Testing script:

```python
python eval-MDN.py --input TEST_INPUT  --output OUTPUT_DATABASE_FILE_PATH --normalization PATH_TO_NORMALIZATION_FILE --network_type NETWORK-TYPE (1particle, 2particle, 3particle) --model_file TRAINED_MODEL_NAME (h5 file) --config <(python genconfig.py --type pos1/2/3)
```
`TEST_INPUT` should be in ROOT format

`OUTPUT_DATABASE_FILE_PATH` should be .db file name with path

`PATH_TO_NORMALIZATION_FILE` txt file path where normalizations are stored

`TRAINED_MODEL_NAME` training output h5 file

3. Reading SQL data base and storing histograms(example for 1-particle network)

```bash
PATH=$PWD:$PATH bash test-driver pos1 OUTPUT_DATABSE_FILE OUTPUT_ROOT_FILE
```
`OUTPUT_ROOT_FILE` File name with .root extension


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
Parameters of the mixture model. In this case the _mean_($`\mu`$) and _precision_ ($`\beta`$) of the Gaussian kernels. Precision is defined as the inverse of variance.
* Estimated position = $`\mu`$
* Estimated uncertainty = $`\frac{1}{\sqrt{\beta}}`$


