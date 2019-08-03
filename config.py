
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))

# raw data
raw_data_path = r"./data/raw"

# cv1 = pd.read_csv('%s/geoplaces2.csv'%raw_data_path, encoding = "latin1")
# cv1 = cv1.drop('name', axis =1)

# export data

interim_data_path = r"./data/interim"
output_data_path = r"./data/processed"

fig_report_path = r"./reports/figures"



if __name__ == "__main__":
    pass