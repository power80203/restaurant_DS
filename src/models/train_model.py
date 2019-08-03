import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import config
from src.features import fe


# read data
df_main = fe.mergred_store_and_user()