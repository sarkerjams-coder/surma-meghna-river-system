# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 06:20:16 2026

@author: JAMESHUVO
"""

import pandas as pd
import glob
import os

folder = r"D:/Tusher/AA Thesis task/Imputed data/XGB_LGBM/XGB_LGBM_Performance_result"

dfs = []

for file in glob.glob(os.path.join(folder, "*.xlsx")):
    df = pd.read_excel(file)

    # Optional: keep the filename
    df["Source"] = os.path.splitext(os.path.basename(file))[0]

    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

# Put Source as the first column
cols = ["Source"] + [c for c in merged.columns if c != "Source"]
merged = merged[cols]

merged.to_excel(os.path.join(folder, "Merged_XGB_LGBM.xlsx"), index=False)

print(merged)