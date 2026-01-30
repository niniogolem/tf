import tensorflow as tf
import sionna
import sys
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"✅ Python Version: {sys.version.split()[0]} (Target: 3.12.x)")
print(f"✅ TensorFlow Version: {tf.__version__} (Target: 2.19.x)")
print(f"✅ Sionna Version: {sionna.__version__}")

# Check for GPU and CUDA
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"✅ GPU Detected: {details.get('device_name', 'Unknown GPU')}")
    # TF 2.19 often prints detailed CUDA info in the logs
else:
    print("❌ ERROR: No GPU detected.")

print(f"✅ All libraries loaded successfully.")
print(f"   NumPy Version: {np.__version__}")
print(f"   Pandas Version: {pd.__version__}")
print(f"   Sionna Version: {sionna.__version__}")