import tensorflow as tf
import os
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



if tf.test.is_gpu_available():

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  print(tf.__version__)

#test the gpu compute time

  tf.debugging.set_log_device_placement(True)

  # Place tensors on the GPU
  npTmp = np.random.random((4096, 4096)).astype(np.float32)
  npMat1 = np.stack([npTmp,npTmp],axis=2)
  npMat2 = npMat1

  a = tf.constant(npMat1)
  b = tf.constant(npMat2)


  start_time = time.time()

  # Run on the GPU
  c = a*b

  print("CUDA --- %s seconds ---" % (time.time() - start_time))




  with tf.device('/cpu:0'):
    npTmp = np.random.random((1024, 1024)).astype(np.float32)
    npMat1 = np.stack([npTmp,npTmp],axis=2)
    npMat2 = npMat1

    x = tf.constant(npMat1)
    y = tf.constant(npMat2)
    start_time = time.time()
    c = x*y #run on the cpu

  print("CPU --- %s seconds ---" % (time.time() - start_time))
else:
  print("CUDA is not availble! Check the environment setting!")
