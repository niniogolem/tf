import numpy as np
import keras
from keras import layers
from keras import ops
import sionna.phy

batch_size = 1024
n = 1000 # codeword length
k = 500 # information bits per codeword
m = 4 # bits per symbol
snr = 10

c = sionna.phy.mapping.Constellation("qam", m)
b = sionna.phy.mapping.BinarySource()([batch_size, k])
u = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n)(b)
x = sionna.phy.mapping.Mapper(constellation = c)(u)
y = sionna.phy.channel.AWGN()([x, 1/snr])
llr = sionna.phy.mapping.Demapper("app", constellation = c)([y, 1/snr])
b_hat = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(u)(llr)
print(b_hat)