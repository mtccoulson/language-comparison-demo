#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:53:51 2020

@author: morleycoulson
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


if __name__ == '__main__':
    #Create the fake dataset
    fake_data = make_classification(
        n_samples = 1000, 
        n_features = 10,
        n_informative = 5,
        class_sep = 2
    )
    
    #convert to a pandas dataframe
    df = pd.DataFrame(fake_data[0])
    df.columns = ['col_' + str(i) for i in range(0, len(df.columns))]
    df['constant'] = 1
    df['target'] = fake_data[1]
    
    #save for usage
    df.to_csv('fake_data.csv', index = False)