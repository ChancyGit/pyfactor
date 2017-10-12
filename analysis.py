import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')
from datetime import datetime

from performance import cal_perf_detail

ret = pd.read_csv('pct.csv')
#import pdb;pdb.set_trace()
comb_nv = (1+ret[['comb_nv']]/100).cumprod()
# bench_nv = ret[['SH000905']]
bench_nv = (1+ret[['SH000905']]/100).cumprod()
bench_nv.columns = ['AShare']
date_list = ret['date'].values.tolist()
comb_nv.index = date_list
bench_nv.index = date_list
#import pdb;pdb.set_trace()
cal_perf_detail(comb_nv,bench_nv,date_list,by_year=1,plot=1)
