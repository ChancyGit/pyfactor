import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')
from datetime import datetime
from dateutil.parser import parse

import time
import datetime
import os.path
import sys


#from data import (	get_tc_td,
# 					get_index_px,
# 					get_fin,
# 					load_zzhy_all,
# 					load_indexweight_all)
#from util import (	cal_score_factors_all,
# 					stk_filter)
#from performance import cal_perf_detail
#from config import *
#factor_data_dict 	= {}
#sector_stks_dict 	= {}
#date_list 		= []



factor_list = [	
					['1/PB','毛利/总资产','Delta毛利率','DeltaATO','最新一季营业利润增长率','Delta预期当年净利润增长率2']



					]	



index = ''
begt = 20081220
endt = 20170901
cycle = '月线'
SECTOR = '500' 			#or 800ew or 500 or ''
stk_pool = 'aShare'     		#or zz800 or aShare
target_stock_num 	= 80
target_percent 		= ''
sector 				= '500'				
bench_nv 			= []


sector_factor_dict 	= {}

sector_weighted 	= 0	#根据行业权重配置股票
is_save_fin 		= 0			#全复制中证金融行业
by_year 			= 1
if_plot 			= 1

time_start=time.time();

if not (len(factor_list)>0)^(len(sector_factor_dict)>0):
	print('one and only one between factor_list and sector_factor_dict should be input')
	exit(0)

if len(sector_factor_dict) or len(sector):
	sector_stks_dict  = load_zzhy_all(DATA_PATH)

if sector_weighted:
	sector_wt_df = load_indexweight_all(DATA_PATH)


if not(len(date_list)):
	date_list = get_tc_td(begt,endt)
#import pdb;pdb.set_trace()

#for different industry stocks in a pool , using their mean returns as benchmark
if not len(index):
	merged_bench = ['']*len(date_list)
	merged_bench[0] = 0

comb = ['']*len(date_list)
comb[0] = 0
comb_num = ['']*(len(date_list)-1)

last_comb = []
turnover = []

#import pdb;pdb.set_trace()

if not(len(factor_data_dict)):
	print('loading data')

	single_factor_list = []
	if len(factor_list):
		for x in [item for sublist in factor_list for item in sublist]:
			if len(x.split(';')) == 1:
				single_factor_list.append(x)
			else:
				for y in x.split(';'):
					single_factor_list.append(y)
	
	if len(sector_factor_dict):
		for hy, f_list in sector_factor_dict.items():
			for x in [item for sublist in f_list for item in sublist]:
				if len(x.split(';')) == 1 and x not in single_factor_list:
					single_factor_list.append(x)
				else:
					for y in x.split(';'):
						if y not in single_factor_list:
							single_factor_list.append(y)
				#import pdb;pdb.set_trace()


	factor_data_dict = cal_score_factors_all(date_list,
											stk_pool = stk_pool,
											sector=sector,
											sector_stks_dict=sector_stks_dict,
											gauss_factors=VALUE+PROFIT+PROFIT_IMPROVE+TRADE+GROWTH_,
											linear_factors=GROWTH,
											factor_list = single_factor_list)



print('backtesting')
for x in range(0,len(date_list)-1):
	INTdate = int(date_list[x].replace('-',''))
	#print(INTdate)
	factor = factor_data_dict[INTdate]

	if not len(index):
		merged_bench[x+1] = factor['下一个月涨幅'].mean()/100

	#factors = factor[VALUE+PROFIT+PROFIT_IMPROVE+TRADE+GROWTH+GROWTH_].dropna()
	factors = factor[single_factor_list].dropna()
	
	#import pdb;pdb.set_trace()
	mycomb,mycomb_hy = stk_filter(factors,
						date=INTdate,
						target_percent=target_percent,
						stk_num=target_stock_num,
						factor_list=factor_list,
						sector_factor_dict=sector_factor_dict,
						sector_stks_dict=sector_stks_dict,
						if_save_fin=is_save_fin)
	

	# if not len(last_comb):
	# 	last_comb = mycomb
	# else:
	# 	turnover.append(len(mycomb.difference(last_comb))/len(last_comb))
	
	ret = factor[['下一个月涨幅']]
	#import pdb;pdb.set_trace()
	rets = ret.ix[mycomb]
	#ret = ret.dropna()
	if not sector_weighted:
		comb[x+1] = (rets['下一个月涨幅'].mean()/100)
	else:
		comb_stks = pd.DataFrame(np.array([mycomb,mycomb_hy]).T,columns=['code','hy'])
		comb_stks['wt'] = np.nan
		
		for hy, gp in comb_stks.groupby('hy'):
			#import pdb;pdb.set_trace()
			hy_wt = 0.
			for ind in hy.split(';'):
				hy_wt += round(sector_wt_df[ind].ix[date_list[x]],4)
			comb_stks.loc[comb_stks['hy']==hy,'wt'] = round(hy_wt/len(gp),2)
			#import pdb;pdb.set_trace()
		
		weight = comb_stks.set_index('code')['wt']/100
		#import pdb;pdb.set_trace()
		fut_ret = rets['下一个月涨幅']
		
		print(weight.sum())

		comb[x+1] = weight.dot(fut_ret)/100
		#import pdb;pdb.set_trace()
	comb_num[x] = (len(rets))
	#bench.append(factor['下一个月涨幅'].mean()/100)


#import pdb;pdb.set_trace()		
df = pd.DataFrame(np.array([comb]).T,index=date_list,columns=['comb_pct'])
df['comb_nv'] = (1+df['comb_pct']).cumprod()
#df.to_csv('comb.csv')
if not len(bench_nv):
	if len(index):
		bench_nv = get_index_px(index,date_list[0],date_list[-1])
	elif index == '':
		bench_df = pd.DataFrame(np.array(merged_bench).T,index=date_list,columns=['bench_pct'])
		bench_df['bench_nv'] = (1+bench_df['bench_pct']).cumprod()
		bench_nv = bench_df[['bench_nv']]
	else:
		pass
time_end=time.time();
#import pdb;pdb.set_trace()
ret,sh,maxdd,winratio = cal_perf_detail(df[['comb_nv']],bench_nv,date_list,by_year=by_year,plot=if_plot)
print('*************************')
print('mean stock number {0}'.format(np.array(comb_num).mean()))
# print('mean stock turnover {0}'.format(np.array(turnover).mean()))
print('running time {0}'.format(time_end-time_start))


