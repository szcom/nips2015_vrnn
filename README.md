# Variational Recurrent Neural Networks and Importance Weights Sampling
This is an experiment to train ["A Recurrent Latent Variable Model for Sequential Data"](http://arxiv.org/abs/1506.02216) using the lower bound proposed by ["Importance Weighted Autoencoders"](http://arxiv.org/abs/1509.00519) 

Dependencies
------------
Most of the script files are written as pure Theano code, modules are implemented in a more general framework.
You can find the code at http://github.com/jych/cle.

Notice
------
The original Blizzard dataset should be downloaded by each user due to the license.<br>
http://www.synsig.org/index.php/Blizzard_Challenge_2013<br>
The original wave files have been read by numpy and saved into '.npz' format.
There is a function that reads the numpy formatted files and generate a hdf5 format file.
Here is the [tool to convert mp3 to wav and npy, 16kHz, 16bits LE](./mp3_to_wav_and_npy.py)
 
IWAE
----
The task was to predict 0.5 second long audio sequence. 8000 raw audio samples were organized in 40 timesteps of 200 samples(40x200=8000 amplitudes). The training data had 2M sequences or about 283 hours.
At every timestep there was a VAE with 200 inputs. Unlike standard VAE with univariate Gaussian prior in this case the prior over the latent variables came from RNN's hidden state. 

Importance Weighted Autoencoder adds multiple stochastic layers and novel objective function to the traditional VAE. In order to use IWAE approach in VRNN each sequence is repeated K times during the training. The hidden state is sampled K times from the same parameters. The objective is evaluated over 1..K hidden samples and then either averaged over K(VAE_K) or multiplied by its "importance weights"(IWAE_K). These two variants are compared to the performance of the original VRNN objective in terms of log likelihood on the held out data. IWAE with single stochastic layer and K=5 have beed evaluated with the following results for the estimate of the variational upper bound on NLL:
![VRNN vs IWAE_K vs VAE_K](https://github.com/szcom/nips2015_vrnn/raw/master/recon.png)

VRNN | VAE_K | IWAE_K
----|-----|------
[vrnn_gauss.py](./models/blizzard/vrnn_gauss.py) | [vrnn_gauss_alt_nll.py](./models/blizzard/vrnn_gauss_alt_nll.py) | [vrnn_gauss_iwae.py](./models/blizzard/vrnn_gauss_iwae.py)

The audio sequence duration and RNN layer size were cut short to 0.5s and 1000 hidden units respectively. 
IWAE implementation used in this experiment comes from Yuri Burda and can be found [here](https://github.com/yburda/iwae/blob/master/iwae.py)

Remarks
----------------
Lower bound suggested by IWAE_K did not improve VRNN performance when compared after the same number of sequences(but not epochs due to K factor). Would be interesting to see if pretrained as VRNN net can be improved by training it as IWAE. On MNIST VAE_K was outperfomed by IWAE_K with K=5 by almost 1 nat (86.47 vs 85.54) but the opposite happened in this VRNN experiment with audio data.
