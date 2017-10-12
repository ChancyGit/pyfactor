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


from data import (	get_tc_td,
					get_index_px,
					get_fin,
					load_zzhy_all,
					load_indexweight_all)
from util import (	cal_score_factors_all,
					stk_filter)
from performance import cal_perf_detail
from config import *


def backtest_syn(index,begt,endt,cycle,**kargs):
	'''
	test stock selection by factors
	what this function can do:
	(1) test factor combination for all stocks in 800ew or 300
		inputs: factor list

	(2) test factor combination for all stocks in zhongzhen industry indexs
		inputs: factor list

	(3) test factor combination ,different industry different factors
		inputs: sector_factor_dict
	'''
	stk_pool			= kargs.get('stk_pool','zz800')
	target_stock_num 	= kargs.get('stk_num',80)
	target_percent 		= kargs.get('target_percent','')
	sector 				= kargs.get('sector','')				#中证行业分类
	bench_nv 			= kargs.get('bench_nv',[])
	date_list 			= kargs.get('date_list',[])
	factor_list 		= kargs.get('factor_list',[])
	sector_factor_dict 	= kargs.get('sector_factor_dict',{})
	factor_data_dict 	= kargs.get('factor_data_dict',{})	
	sector_stks_dict 	= kargs.get('sector_stks_dict',{})	
	sector_weighted 	= kargs.get('sector_weighted',0)		#根据行业权重配置股票
	is_save_fin 		= kargs.get('is_save_fin',0)			#全复制中证金融行业
	by_year 			= kargs.get('by_year',1)
	if_plot 			= kargs.get('if_plot',1)
	
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
	print('loading data')
	if not(len(factor_data_dict)):


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

		factors = factor[single_factor_list].dropna()

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

	return ret,sh,maxdd,winratio



def opt_factor_list(index,begt,endt,cycle,**kargs):
	syn_value = ['1/PFCF;1/PE;总市值','1/PFCF;1/POP;总市值','1/PE;总市值',
				'1/POP;总市值','1/PE;1/POP;总市值','S/EV;1/PE;总市值',
				'S/EV;1/POP;总市值','1/PE;1/PS;总市值','1/PS;1/POP;总市值',
				'1/PFCF;EBITDA/EV;总市值','EBITDA/EV;总市值','1/PE;1/PB;总市值',
				'EBITDA/EV;1/POP;总市值','EBITDA/EV;1/PE;总市值','1/PB;1/POP;总市值',
				'1/PFCF;1/PB;总市值','1/PFCF;1/PS;总市值','1/PFCF;S/EV;总市值','EBITDA/EV;S/EV;总市值',
				'1/PB;总市值','EBITDA/EV;1/PS;总市值','S/EV;总市值','1/PB;1/PS;总市值',
				'S/EV;1/PB;总市值','S/EV;1/PS;总市值','EBITDA/EV;1/PB;总市值','1/PFCF;总市值','1/PS;总市值','1/PB','总市值']
	syn_profit = ['deltaROE;ROIC','delta毛利率;deltaROE;ROIC1','deltaROE;EP;毛利率',
					'delta毛利率;deltaROE;ROE','deltaROE;ROIC1','delta毛利率;deltaROE',
					'deltaROE;EBITDA-资本支出/IC','deltaROE;ROE','deltaROE;ROIC1;毛利率','ROIC1','ROE','毛利率']
	syn_growth = ['最新一季营业利润增长率','delta存货周转率;TTMFCF增长率;最新一季营业利润增长率',
					'delta应收账款周转率;TTMFCF增长率;最新一季营业利润增长率',
					'delta存货周转率;delta应收账款周转率;最新一季营业利润增长率',
					'delta毛利率;delta存货周转率;最新一季营业利润增长率']
	syn_trade = ['近一个月跌幅;ILLIQ;波动率','SKEW;换手率;ILLIQ',
					'近一个月跌幅;换手率;ILLIQ','SKEW;ILLIQ;波动率',
					'换手率;ILLIQ','近一个月跌幅;SKEW;换手率','近一个月跌幅;SKEW;ILLIQ',
					'近一个月跌幅;波动率','近一个月跌幅;换手率','近一个月跌幅;换手率;波动率','ILLIQ','近一个月跌幅']


	print('loading data')
	date_list = get_tc_td(begt,endt)
	bench_nv = get_800ew(date_list[0],date_list[-1])
	factor_data_dict = {}
	for x in range(0,len(date_list)-1):
		INTdate = int(date_list[x].replace('-',''))
		factor = get_fin(INTdate)
		factor = factor.set_index('代码')
		factor = score_factors(factor,VALUE+PROFIT+PROFIT_IMPROVE+TRADE+GROWTH_,GROWTH)
		factor_data_dict[INTdate] = factor
	#import pdb;pdb.set_trace()
	print('testing')
	txtName = 'goodfactors.txt'
	
	
	result_dict = {}
	#for v1 in syn_value+syn_profit+syn_growth+syn_trade:VALUE+PROFIT+PROFIT_IMPROVE+TRADE+GROWTH_,GROWTH
	for v1 in syn_value+syn_profit+syn_growth+syn_trade:
		#for v2 in syn_value:
			#for p in syn_profit:
				#for g in syn_growth:
		f_list =['总市值;1/PB;ROIC1;最新一季营业利润增长率',v1]
		f_name = '*'.join(f_list)
		result_dict[f_name] = [np.nan,np.nan,np.nan,np.nan]
		ret,sh,maxDD,calmar = backtest_syn(index,
										begt,
										endt,
										cycle,
										bench_nv=bench_nv,
										date_list=date_list,
										factor_data_dict=factor_data_dict,
										factor_list=f_list,
										by_year=0,
										if_plot=0)
		#import pdb;pdb.set_trace()
		result_dict[f_name][0] = ret.values[0]
		result_dict[f_name][1] = sh.values[0]
		result_dict[f_name][2] = maxDD.values[0]
		result_dict[f_name][3] = calmar.values[0]
		
		if result_dict[f_name][3] > 4:
			print(f_name)
			print(ret.values[0],calmar.values[0])
			f = open(txtName,'a+')
			info = f_name+'___'+str(result_dict[f_name][0])+'___'+str(result_dict[f_name][1])+'___'+str(result_dict[f_name][2])+'\n'
			f.write(info)
			f.close()
	
	#import pdb;pdb.set_trace()
	result = pd.DataFrame(result_dict)
	result = result.T
	result.columns = ['ret','sh','maxDD','calmar']
	result.to_csv('result.csv')
	return result



def demo_1():
	#等权800内选股不分行业

	
	# factor_list = [						
	# 				['1/PB'],
	# 				['毛利/总资产'],
	# 				['delta营业利润率'],
	# 				['delta总资产周转率'],
	# 				['最新一季营业利润增长率'],
	# 				['Delta预期当年净利润增长率2'],
	# 				['delta股东人数','近一个月跌幅'],
	# 				]		

	# factor_list = [	['1/PB'],
	# 				['毛利/总资产'],
	# 				['Delta毛利率'],
	# 				['DeltaATO'],
	# 				['最新一季营业利润增长率'],
	# 				['Delta预期当年净利润增长率2'],
	# 				['股东人数变化率','近一个月跌幅']
	# 			]	

	# factor_list = [	
	# 				['总市值','1/PB','毛利/总资产','DeltaBK','Delta毛利率','DeltaATO','最新一季营业利润增长率','Delta预期当年净利润增长率2']



	# 				]	
	factor_list = [	
					['总市值','1/PB','毛利/总资产','最新一季营业利润增长率']



					]	

	# factor_list = [	
	# 				['毛利/总资产','最新一季营业利润增长率','OPMstab_5y;OPMcagr_5y;OPMvar_5y','delta毛利率','delta总资产周转率','delta股东人数;近一个月跌幅',],
	# 				['1/PB'],
	# 				['总市值'],
	# 				# ['Delta预期当年净利润增长率2']
	# 				]	

	BegT = 20081220
	EndT = 20170901
	cycle = '月线'
	SECTOR = '500' 			#or 800ew or 500 or ''
	STK_POOL = 'aShare'     		#or zz800 or aShare
	index = ''

	backtest_syn(index,BegT,EndT,cycle,
				stk_pool = STK_POOL,
				factor_list=factor_list,
				sector=SECTOR,
				target_percent='',
				stk_num=80)

def demo_2():
	#中证800工业内选股

	factor_list  =[
					['1/PB','1/PS'],
					['ROIC'],
					['delta毛利率','delta总资产周转率'],
					['OPMvar_5y','最新一季营业利润增长率'],

					]	

	# factor_list = [	

	# 				['1/PB'],
	# 				['毛利/总资产'],
	# 				['delta毛利率','delta总资产周转率'],
	# 				['最新一季营业利润增长率'],
	# 				['Delta预期当年净利润增长率2',],
	# 				['delta股东人数','近一个月跌幅'],
	# 				]		

	BegT = 20090720
	EndT = 20170901
	cycle = '月线'

	SECTOR = '中证工业'
	index = hy_dict[SECTOR]
	#index = 'SH000842'

	backtest_syn(index,BegT,EndT,cycle,
				factor_list=factor_list,
				sector=SECTOR,
				target_percent=0.1)
	return 0

def demo_3():

	factor_list = [	['1/PB'],
					['毛利/总资产'],
					['Delta毛利率'],
					['DeltaATO'],
					['最新一季营业利润增长率'],
					['Delta预期当年净利润增长率2'],
					['股东人数变化率','近一个月跌幅']


				]	
	BegT = 20081220
	EndT = 20170901
	cycle = '月线'
	#SECTOR = '中证可选'
	#SECTOR = '中证消费;中证可选;中证医药;中证信息;中证电信'
	#SECTOR = '中证材料;中证能源'
	SECTOR = '中证工业'
	#SECTOR = '申万金融'
	#SECTOR = '申万金融'
	#index = 'SH000931'
	index = ''

	backtest_syn(index,BegT,EndT,cycle,
				factor_list=factor_list,
				sector=SECTOR,
				target_percent=0.1)
	return 0

def demo_4():
	#等权800内选股分行业
	
	# hy_factor_dict = {		'中证工业':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;最新一季净利润增长率'],
	# 						'中证消费':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;最新一季营业收入增长率'],
	# 						'中证材料;中证公用;中证电信;中证能源;中证可选':['Delta预期当年净利润增长率2','总市值;1/PB;ROIC1;最新一季营业利润增长率'],
	# 						'中证金融':['Delta预期当年净利润增长率2','总市值;1/PB;1/PE'],
	# 						'中证信息':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;Accrual2NI;最新一季营业利润增长率'],		
	# 						'中证医药':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;销售现金比;最新一季营业利润增长率']
	# 				}	
	
	# hy_factor_dict = {		'中证工业':['1/PB','ROIC1','delta营业利润增速','最新一季净利润增长率'],
	# 						'中证材料':['毛利率','1/PS','ROA','delta营业利润增速','最新一季营业利润增长率'],
	# 						'中证可选':['营业利润率','1/PS','ROA','delta营业利润增速','最新一季营业利润增长率'],
	# 						'中证信息':['1/PS','经营现金流/净资产','最新一季营业利润增长率','delta营业利润增速','股东人数变化'],
	# 						'中证消费':['Delta预期当年净利润增长率2;总市值;1/PB;ROE;最新一季营业收入增长率'],
	# 						'中证公用;中证电信;中证能源;中证地产':['Delta预期当年净利润增长率2;总市值;1/PB;ROIC1;最新一季营业利润增长率'],
	# 						'申万金融':['Delta预期当年净利润增长率2;总市值;1/PB;1/PE'],
	# 						'中证医药':['Delta预期当年净利润增长率2;总市值;1/PB;ROE;销售现金比;最新一季营业利润增长率']
	# 				}
	
	# hy_factor_dict = {		'中证材料':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;最新一季营业利润增长率'],
	# 						'中证消费':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;最新一季营业收入增长率'],
	# 						'中证公用;中证电信':['Delta预期当年净利润增长率2','总市值;1/PB;ROIC1;最新一季营业利润增长率'],
	# 						'中证信息;中证能源;中证可选':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;Accrual2NI;最新一季营业利润增长率'],
	# 						'中证金融':['Delta预期当年净利润增长率2','总市值;1/PB;1/PE'],							
	# 						'中证医药':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;销售现金比;最新一季营业利润增长率'],
	# 						'中证工业':['Delta预期当年净利润增长率2','总市值;1/PB;ROE;最新一季净利润增长率']
	# 				}	
	

	hy_factor_dict = {		'中证工业': [	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
											
										],
							'中证材料':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
										],
							'中证可选':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
										],
							'中证信息':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
										],
							'中证消费':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
										],
							'中证公用;中证电信;中证能源;中证地产':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
											],
							'申万金融':[['总市值;1/PB;1/PE']],
							'中证医药':[	['1/PB','1/PS;1/POP',],
											['毛利/总资产',],
											['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
											['最新一季营业利润增长率'],
											['delta股东人数','近一个月跌幅','ILLIQ'],
										],

					}
	BegT = 20090720
	EndT = 20170901
	cycle = '月线'
	SECTOR = '800ew'
	index = hy_dict[SECTOR]

	backtest_syn(index,BegT,EndT,cycle,
				sector_factor_dict=hy_factor_dict,
				sector=SECTOR,
				target_percent=0.11)
	return 0

def demo_5():
	#等权800内选股分行业分行业权重对冲300

	# key = '1/PB;ROA;deltaROE;'
	# hy_factor_dict = {		'中证工业':['Delta预期当年净利润增长率2',key+'最新一季净利润增长率'],
	# 						'中证消费':['Delta预期当年净利润增长率2',key+'最新一季营业收入增长率'],
	# 						'中证材料;中证公用;中证电信;中证能源;中证可选':['Delta预期当年净利润增长率2',key+'最新一季营业利润增长率'],
	# 						'中证金融':['Delta预期当年净利润增长率2','1/PB;1/PE'],
	# 						'中证信息':['Delta预期当年净利润增长率2',key+'Accrual2NI;最新一季营业利润增长率'],		
	# 						'中证医药':['Delta预期当年净利润增长率2',key+'销售现金比;最新一季营业利润增长率']
	# 				}
	


	# key = '1/PS;1/PB;ROE;deltaROE;'
	# hy_factor_dict = {		'中证工业':['Delta预期当年净利润增长率2',key+'最新一季净利润增长率'],
	# 						'中证消费':['Delta预期当年净利润增长率2',key+'最新一季营业收入增长率'],
	# 						'中证材料;中证公用;中证电信;中证能源;中证可选':['Delta预期当年净利润增长率2',key+'最新一季营业利润增长率'],
	# 						'申万金融;中证地产':['Delta预期当年净利润增长率2','1/PB;Delta预期当年净利润增长率2'],
	# 						'中证信息':['Delta预期当年净利润增长率2',key+'Accrual2NI;最新一季营业利润增长率'],		
	# 						'中证医药':['Delta预期当年净利润增长率2',key+'销售现金比;最新一季营业利润增长率']
	# 				}
	
	# BegT = 20090720
	# EndT = 20170630
	# cycle = '月线'
	# SECTOR = '300'
	# index = hy_dict[SECTOR]

	# backtest_syn(index,BegT,EndT,cycle,
	# 			sector_factor_dict=hy_factor_dict,
	# 			sector=SECTOR,
	# 			sector_weighted=1,
	# 			target_percent=0.12,
	# 			is_save_fin=0)

	# hy_factor_dict = {		'中证工业':['Delta预期当年净利润增长率2;1/PB;deltaROE;最新一季净利润增长率'],
	# 						'中证公用':['Delta预期当年净利润增长率2;1/PB;deltaROE;最新一季净利润增长率'],			
	# 						'中证可选;中证消费;中证医药;中证信息;中证电信':['Delta预期当年净利润增长率2;1/PB;毛利/净资产;delta存货周转率;最新一季营业利润增长率'],
	# 						'申万金融':['Delta预期当年净利润增长率2;1/PB;1/PE;Accrual2NI;最新一季营业利润增长率'],
	# 						'中证材料;中证能源':['Delta预期当年净利润增长率2;1/PB;毛利率;delta毛利率;最新一季营业利润增长率'],		
	# 						'中证地产':['Delta预期当年净利润增长率2;1/PB;1/PS;Accrual2NI;ROA']
	# 				}

	# hy_factor_dict = {	'中证工业':['1/PB;1/PE'],
	# 					'中证公用':['1/PB;FCF/营业收入;deltaROE;最新一季净利润增长率'],
	# 					'中证可选;中证消费;中证医药;中证信息;中证电信':['1/PE','ROIC1','毛利率','delta营业利润增速','最新一季营业利润增长率','近一个月跌幅', 'ILLIQ'],
	# 					'申万金融':['1/PB;1/PE'],
	# 					'中证材料;中证能源':['1/PS;1/PB;ROE;deltaROE;最新一季营业利润增长率'],		
	# 					'中证地产':['1/PB;1/PS;Accrual2NI;ROA']
	# 			}

	
	# hy_factor_dict = {		'中证工业':['1/PB','ROIC1','delta营业利润增速','最新一季净利润增长率'],
	# 						'中证材料':['毛利率','1/PS','ROA','delta营业利润增速','最新一季营业利润增长率'],
	# 						'中证可选':['营业利润率','1/PS','ROA','delta营业利润增速','最新一季营业利润增长率'],
	# 						'中证信息':['1/PS','经营现金流/净资产','最新一季营业利润增长率','delta营业利润增速','股东人数变化'],
	# 						'中证消费':['1/PB','ROIC1','delta营业利润增速','最新一季营业利润增长率'],
	# 						'中证公用;中证电信;中证能源;中证地产':['1/PB','ROIC1','delta营业利润增速','最新一季营业利润增长率'],
	# 						'申万金融':['1/PB;1/PE'],
	# 						'中证医药':['1/PB','ROIC1','delta营业利润增速','最新一季营业利润增长率']
	# 				}
	

	hy_factor_dict = {		'中证工业;中证公用;中证材料;中证能源;中证消费;中证可选;中证医药;中证信息;中证电信;中证地产': 
													[	
															['delta毛利率','delta总资产周转率',],
															['1/PB'],
															['毛利/总资产'],																			
															['最新一季营业利润增长率','OPMstab_5y;OPMvar_5y;OPMcagr_5y'],
															['Delta预期当年净利润增长率2'],
															['delta股东人数','近一个月跌幅;波动率'],
												],


							'申万金融':[	['1/PB',],
											['ROE',],
											['deltaBK'],
											['预期PE',],
										],
						}
	
	# 				}
	# hy_factor_dict = {		'中证工业': [	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
											
	# 									],
							# '中证材料':[	['1/PB','1/PS;1/POP',],
							# 				['毛利/总资产',],
							# 				['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
							# 				['最新一季营业利润增长率'],
							# 				['delta股东人数','近一个月跌幅','ILLIQ'],
	# 									],
	# 						'中证可选':[	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
	# 									],
	# 						'中证信息':[	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
	# 									],
	# 						'中证消费':[	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
	# 									],
	# 						'中证公用;中证电信;中证能源;中证地产':[	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
	# 										],
	# 						'申万金融':[['总市值;1/PB;1/PE']],
	# 						'中证医药':[	['1/PB','1/PS;1/POP',],
	# 										['毛利/总资产',],
	# 										['delta营业利润率','delta毛利率','delta总资产周转率','delta运营资产周转率',],
	# 										['最新一季营业利润增长率'],
	# 										['delta股东人数','近一个月跌幅','ILLIQ'],
	# 									],

	# 				}

	BegT = 20090720
	EndT = 20170630
	cycle = '月线'
	SECTOR = '300'
	index = hy_dict[SECTOR]

	backtest_syn(index,BegT,EndT,cycle,
				sector_factor_dict=hy_factor_dict,
				sector=SECTOR,
				sector_weighted=1,
				target_percent=0.06,
				is_save_fin=1)
	
	return 0

if __name__=='__main__':

	hy_dict = {		'中证消费':'SH000932',
					'中证可选':'SH000931',
					'中证金融':'SH000934',
					'中证工业':'SH000930',
					'中证公用':'SH000937',
					'中证材料':'SH000929',
					'中证医药':'SH000933',
					'中证信息':'SH000935',
					'中证能源':'SH000928',
					'中证电信':'SH000936',
					'800ew'	  :'SH000842',
					'300'	  :'SH000300',
					'500'	  :'SH000905'
					}

	
	demo_1()
	