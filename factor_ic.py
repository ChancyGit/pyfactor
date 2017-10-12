from __future__ import (absolute_import,
						division,
						print_function,
						unicode_literals)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')

from sklearn.decomposition import PCA
from datetime import datetime
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

import time
import datetime
import os.path
import sys

TS_API_PATH = 'E:\\PyWork\\tsltest\\'
sys.path.append(TS_API_PATH)
from ts_api import  (get_close_all,
					get_ts_td,
					get_ts_close,
					get_high_all,
					get_low_all,
					get_financials,
					get_indexweight,
					get_intraday_prices,
					get_my)

from data import (	get_tc_td,
					get_fin,
					load_zzhy_all)
from util import score_factors
from performance import cal_perf_detail
from config import DATA_PATH

'''
#中证可选;中证消费;中证医药;中证信息;中证电信
VALUE = [  
		'1/PB','1/PE','1/PS','EBITDA/EV','总资产','IC']

PROFIT = 	['ROIC1','Accrual2NI','ROE','ROA','毛利/净资产','delta存货周转率','deltaROE','毛利率','销售现金比','delta营业利润增速']

GROWTH = [	'最新一季营业利润增长率','最新一季营业收入增长率']

#GROWTH_ = ['Delta预期当年净利润增长率2']
GROWTH_ = []
'''
#中证工业
hcg = ['毛利/净资产','毛利/总资产','营业利润率','营运报酬率','资本报酬率']

VALUE = [  '1/PB','1/PS']

PROFIT = 	['ROIC1','ROE','ROA']

GROWTH = [	'最新一季营业利润增长率','最新一季销售毛利润增长率','最新一季净利润增长率']

GROWTH_2 = [	'delta营业利润增速','delta净利润增速',	'delta毛利润增速']

OTHER = ['Accrual','存货/资产总计','Accrual2NI']

#GROWTH_ = ['Delta预期当年净利润增长率2']
GROWTH_ = []

'''
#中证工业;中证公用
VALUE = [  '1/PB','1/PE','1/PS']

PROFIT = 	['ROIC1','ROE','EBITDA-资本支出/IC','delta毛利率','deltaROE','CROIC','FCF/营业收入','Accrual2NI','delta应收账款周转率']

GROWTH = [	'TTM净利润增长率','最新一季营业利润增长率','最新一季净利润增长率','PSG3']

GROWTH_ = ['Delta预期当年净利润增长率2']
'''


'''
#中证材料;中证能源
VALUE = [  
		'1/PB','1/PE','EBITDA/EV']

PROFIT = 	['ROIC1','销售现金比','ROE','ROA','毛利率','delta毛利率','Accrual2NI','deltaROE']

GROWTH = ['最新一季营业利润增长率','最新一季净利润增长率']

GROWTH_ = ['Delta预期当年净利润增长率2','预期PEG']
'''

'''

#申万金融
VALUE = [  
		'1/PB','1/PE']

PROFIT = 	['ROIC1','Accrual2NI','ROE','ROA','现金营业收入比','delta存货周转率']

GROWTH = ['最新一季营业利润增长率','最近五年销售毛利']

GROWTH_ = ['Delta预期当年净利润增长率2','预期PE']
'''
'''
#中证地产
VALUE = [  
		'1/PB','1/PE','1/PS']

PROFIT = 	['ROIC1','Accrual2NI','ROE','ROA','现金营业收入比','delta存货周转率']

GROWTH = ['最新一季营业利润增长率','最近五年归母利润']

GROWTH_ = ['Delta预期当年净利润增长率2']
'''



'''
VALUE = ['1/PFCF', 'EBITDA/EV', 'S/EV', '1/PE',
		'1/PB',	'1/PS',	'1/POP','总市值']

PROFIT = ['CROIC',	'ROIC',	'ROIC1','EP',
		'EBITDA-资本支出/IC',	'ROE',	'ROE(扣除)',
		'ROA',	'FCF/营业收入','毛利/净资产',
		'毛利率',	'净利率', '销售现金比',
		'现金营业收入比'	,'delta毛利率',
		'deltaROE',		
		'delta存货周转率','delta应收账款周转率',
		'负债/投入资本','外部融资额/总资产',
		'资产负债率','有形净值债务率','经营现金净流入/资本支出',
		'折旧/投入资本','长期负债变化率','近两年平均资本支出增长率',
		'Accrual2NI']


GROWTH = ['TTMFCF增长率','TTM净利润增长率','TTM营业利润增长率',
			'TTM营业收入增长率', '最新一季净利润增长率', '最新一季营业利润增长率',
			'最新一季营业收入增长率']

GROWTH_ = ['目标收益率','预期增长率','预期PE','预期PEG','Delta预期增长率','Delta预期增长率2','Delta预期当年净利润增长率2',
			'最近三年主营业务利润率','最近五年主营业务利润率','最近三年销售毛利','最近五年销售毛利',
			'最近三年归母利润','最近五年归母利润','PSG3','PSG5','NPG3','NPG5']
TRADE = ['近一个月跌幅', 'SKEW', '5/60','换手率','ILLIQ',
		'波动率']

'''

'''
VALUE = ['1/PFCF', 'EBITDA/EV', 'S/EV', '1/PE',
		'1/PB',	'1/PS',	'1/POP']

PROFIT = ['ROIC1',	'ROE',
		'ROA',	
		'毛利率',
		'现金营业收入比'	,'delta毛利率',
		'deltaROE',	
		'Accrual2NI']


GROWTH = ['最新一季营业利润增长率']

GROWTH_ = []
'''


def cal_factor_corr(index,begt,endt,kind):

	kind_dict={'估值':VALUE,'盈利':PROFIT,'成长':GROWTH,'交易':TRADE, 
				'ALL':VALUE+PROFIT+GROWTH+_GROWTH+TRADE}
	date_list = [parse(str(x)).strftime('%Y-%m-%d') for x in get_ts_td(begt,endt,cycle)]
	corr_dict = {}
	corr_mean_dict = {}
	corr_t_dict = {}
	for x in range(0,len(date_list)):
		INTdate = int(date_list[x].replace('-',''))
		print(INTdate)
		factor = get_fin(INTdate)
		factor = factor.set_index('代码')
		del factor['日期']
		corr_dict[date_list[x]] = factor[kind_dict[kind]].corr(method='spearman')
		#import pdb;pdb.set_trace()
	pdata = pd.Panel(corr_dict)
	#import pdb;pdb.set_trace()
	for x in kind_dict[kind]:
		mean = pdata[:,:,x].mean(axis=1)
		t = mean/pdata[:,:,x].std(axis=1)
		corr_mean_dict[x] = mean
		corr_t_dict[x] = t
	#import pdb;pdb.set_trace()
	sorted_columns = ['1/PFCF', 'EBITDA/EV', 'S/EV', '1/PE',
					'1/PB',	'1/PS',	'1/POP','CROIC',	'ROIC',	'ROIC1',	'EP',
					'EBITDA-资本支出/IC',	'ROE',	'ROE(扣除)',
					'ROA',	'FCF/营业收入', '毛利/净资产',
					'毛利率',	'净利率', '销售现金比',
					'现金营业收入比',	'delta毛利率',
					'deltaROE',	'负债/投入资本','外部融资额/总资产',
					'资产负债率','有形净值债务率','经营现金净流入/资本支出',
					'折旧/投入资本','长期负债变化率','近两年平均资本支出增长率',
					'delta存货周转率','delta应收账款周转率','TTMFCF增长率','TTM净利润增长率','TTM营业利润增长率',
					'TTM营业收入增长率', '最新一季净利润增长率', '最新一季营业利润增长率',
					'最新一季营业收入增长率','近一个月跌幅', 'SKEW', '5/60','换手率','ILLIQ',
					'波动率','总市值']
	print(len(sorted_columns))

	corr_mean = pd.DataFrame(corr_mean_dict)
	corr_mean = corr_mean.reindex_axis(sorted_columns, axis=1)
	corr_mean = corr_mean.reindex(corr_mean.columns)
	corr_t = pd.DataFrame(corr_t_dict)
	corr_t = corr_t.reindex_axis(sorted_columns, axis=1)
	corr_t = corr_t.reindex(corr_t.columns)
	
	mean_writer = pd.ExcelWriter('相关系数均值_{0}.xlsx'.format(kind))
	corr_mean.to_excel(mean_writer,kind)
	mean_writer.save()
	t_writer = pd.ExcelWriter('相关系数稳定性_{0}.xlsx'.format(kind))
	corr_t.to_excel(t_writer,kind)
	t_writer.save()
	corr_mean.replace(1,0).stack().sort_values().drop_duplicates().to_csv('corr_sort.csv')
	
	import pdb;pdb.set_trace()
	return 0

def analysis_factor_corr(factor_list):
	path = 'E:\\factor_comb\\corr\\'
	corr = pd.read_excel(path+'相关系数均值_ALL_spearman.xlsx')
	comb_corr = corr[factor_list].ix[factor_list]
	pca = PCA(n_components=1)
	reduced = pca.fit_transform(comb_corr)
	#print(reduced)
	#print(pca.components_)
	wts = reduced +abs(reduced.min())+0.5
	wts = wts/wts.sum()
	print(wts,factor_list)
	import pdb;pdb.set_trace()
	comb_corr.to_csv(';'.join(factor_list).replace('/','%')+'.csv')
	comb_corr = comb_corr.replace(1,0)
	print(comb_corr)
	print('*************')
	print(comb_corr.mean().mean())

	return comb_corr

def sort_factor_corr():

	corr = pd.read_excel('相关系数均值_ALL_spearman.xlsx')
	for x in corr.columns:
		single_corr = corr[[x]].sort_values(by=x)[:-2]
		single_corr.to_csv(x.replace('/','%')+'_corrwith.csv')
		#import pdb;pdb.set_trace()
	return 0

def cal_factorIC_corr(sector):
	path = 'E://factor_comb//data//factorIC//'

	def cal_corr(path,xlsx_fn):
		ic_df = pd.read_excel(path+'{0}.xlsx'.format(xlsx_fn))
		value = [	'1/PFCF', 'EBITDA/EV', 'S/EV', '1/PE',
					'1/PB',	'1/PS',	'1/POP'
				]
		profit = [	'CROIC',	'ROIC',	'ROIC1',	'EP',
					'EBITDA-资本支出/IC',	'ROE',	'ROE(扣除)',
					'ROA',	'FCF/营业收入', '毛利/净资产',
					'毛利率',	'净利率', '销售现金比',
					'现金营业收入比','delta毛利率',	'deltaROE',
				]
		quality = [	'负债/投入资本','外部融资额/总资产',
					'资产负债率','有形净值债务率','经营现金净流入/资本支出',
					'折旧/投入资本','长期负债变化率','近两年平均资本支出增长率',
					'delta存货周转率','delta应收账款周转率'
					]
		growth = [	'TTMFCF增长率','TTM净利润增长率','TTM营业利润增长率',
					'TTM营业收入增长率', '最新一季净利润增长率', 
					'最新一季营业利润增长率',	'最新一季营业收入增长率',
					'最新一季销售毛利润增长率'
					]
		moment = [	'最近6个月动量',	'最近3个月动量',	'最近一年动量',
					'最近三年动量',		'Accrual2NI', 'Accrual', 'Accrua2Asset'
					]
		trade = [	'近一个月跌幅', 'SKEW', '5/60','换手率',
					'ILLIQ','波动率'
					]
		predicted = ['目标收益率','预期增长率','预期PE','预期PEG',
					'Delta预期增长率','Delta预期增长率2',
					'Delta预期当年净利润增长率2'
					]
		profit_related = ['ROE_var_3y','ROE_var_5y','GM_var_3y','GM_var_5y',
						'OPM_var_3y','OPM_var_5y','delta营业利润增速',
						'delta营业利润增速1','delta净利润增速','delta营业收入增速',
						'delta毛利润增速','delta毛利润增速1','毛利/总资产',
						'营运报酬率','资本报酬率','存货增长率','存货/资产总计',
						'主营业务利润率','营业利润率','总资产周转率',
						'delta总资产周转率','权益乘数','股东人数',
						'股东人数变化','经营现金流/净资产','总资产','营业总收入','IC']

		sorted_columns = value+profit+quality+growth+moment+trade+predicted+profit_related
		rank_ic = ic_df.T.iloc[len(sorted_columns):2*len(sorted_columns)].T
		rank_ic.columns = sorted_columns
		corr = rank_ic.corr(method='spearman')
		return corr
	raw = cal_corr(path, sector+'因子_0')
	size_regress = cal_corr(path, sector+'因子_1')

	#import pdb;pdb.set_trace()

	writer = pd.ExcelWriter('{0}_相关系数.xlsx'.format(sector))
	raw.to_excel(writer,'无市值回归')
	#import pdb;pdb.set_trace()
	sorted_raw = pd.DataFrame(raw.stack().sort_values().drop_duplicates())
	sorted_raw.to_excel(writer,'无市值回归相关系数排序')
	size_regress.to_excel(writer,'市值回归')
	#import pdb;pdb.set_trace()
	sorted_regress = pd.DataFrame(size_regress.stack().sort_values().drop_duplicates())
	sorted_regress.to_excel(writer,'市值回归相关系数排序')
	#import pdb;pdb.set_trace()
	return 0



def cal_ic(index,begt,endt,cycle,kind,**kargs):
	chosen_ = kargs.get('chosen',[])
	sector = kargs.get('sector','')
	mode = kargs.get('mode','all')

	#date_list = [parse(str(x)).strftime('%Y-%m-%d') for x in get_ts_td(begt,endt,cycle)]
	date_list = [parse(str(x)).strftime('%Y-%m-%d') for x in get_tc_td(begt,endt)]
	#import pdb;pdb.set_trace()
	ic_dict = {}

	if len(sector):
		sector_stks_dict  = load_zzhy_all(DATA_PATH)
	
	small_comb_num = 5
	big_comb_num = 6
	for i in range(small_comb_num,big_comb_num):
		if not len(chosen_):
			chosen = [';'.join(x) for x in list(combinations(kind,i))]
		else:
			chosen = chosen_
		for factors in list(combinations(kind,i)):
			bbb = ';'.join(factors)
			if bbb.split(';') not in [x.split(';') for x in chosen]:
				continue
			for f in factors:
				if i>1:
					ic_dict[';'.join(factors)]= ['']*(len(date_list)-1)
				else:
					ic_dict[f] = ['']*(len(date_list)-1)
	
	for x in range(0,len(date_list)-1):
		INTdate = int(date_list[x].replace('-',''))
		print(INTdate)
		'''
		if not len(sector):
			factor = get_fin(INTdate)
		else:
			factor = get_fin(INTdate,sector=sector)
		'''
		factor = get_fin(INTdate,sector=sector,sector_stks_dict=sector_stks_dict)
		factor = factor.set_index('代码')
		#import pdb;pdb.set_trace()
		#factor = score_factors(factor,VALUE+PROFIT+TRADE+GROWTH_,GROWTH)
		factor = score_factors(factor,VALUE+PROFIT+GROWTH_,GROWTH)
		#import pdb;pdb.set_trace()
		for i in range(small_comb_num,big_comb_num):
			if not len(chosen_):
				chosen = [';'.join(x) for x in list(combinations(kind,i))]
			else:
				chosen = chosen_
			for factors in list(combinations(kind,i)):
				bbb = ';'.join(factors)
				if bbb.split(';') not in [x.split(';') for x in chosen]:
					continue
				
				score = []
				for f in factors:
					f_score = factor[[f]]
					if not len(score):
						score = f_score
					else:
						#import pdb;pdb.set_trace()
						score = score.join(f_score)

				score = score.mean(axis=1)
				score = score.dropna()
				score.name='f'
				ret = factor[['下一个月涨幅']]
				ret = ret.join(score)
				ret = ret.dropna()
				#import pdb;pdb.set_trace()
				fut_ret_rank = pd.cut(ret['下一个月涨幅'],10,labels=False)+1
				factor_rank = pd.cut(ret['f'],10,labels=False)+1
				ic = stats.spearmanr(fut_ret_rank,factor_rank)[0]
				#import pdb;pdb.set_trace()

				if i>1:
					ic_dict[';'.join(factors)][x]=ic
				else:
					#import pdb;pdb.set_trace()
					ic_dict[f][x] = ic
				#import pdb;pdb.set_trace()

	ic_df = pd.DataFrame(ic_dict)
	t_stat, p_value = stats.ttest_1samp(ic_df, 0)
	#import pdb;pdb.set_trace()

	sorted_columns = ['1/PFCF', 'EBITDA/EV', 'S/EV', '1/PE',
					'1/PB',	'1/PS',	'1/POP','CROIC',	'ROIC',	'ROIC1',	'EP',
					'EBITDA-资本支出/IC',	'ROE',	'ROE(扣除)',
					'ROA',	'FCF/营业收入', '毛利/净资产',
					'毛利率',	'净利率', '销售现金比',
					'现金营业收入比',	'delta毛利率',
					'deltaROE',	'负债/投入资本','外部融资额/总资产',
					'资产负债率','有形净值债务率','经营现金净流入/资本支出',
					'折旧/投入资本','长期负债变化率','近两年平均资本支出增长率',
					'delta存货周转率','delta应收账款周转率','TTMFCF增长率','TTM净利润增长率','TTM营业利润增长率',
					'TTM营业收入增长率', '最新一季净利润增长率', '最新一季营业利润增长率',
					'最新一季营业收入增长率','近一个月跌幅', 'SKEW', '5/60','换手率','ILLIQ',
					'波动率','目标收益率','预期增长率','预期PE','预期PEG','Delta预期增长率','Delta预期增长率2','Delta预期当年净利润增长率2','总市值']
	#import pdb;pdb.set_trace()
	IC = pd.DataFrame(ic_df.mean(),columns=['IC'])
	IC['STD'] = ic_df.std()
	IC['IR'] = IC['IC']/ic_df.std()
	#import pdb;pdb.set_trace()
	IC['T'] = t_stat
	IC['P'] = p_value
	
	if not len(mode):
		alpha_factor = IC.sort_values(by='T',ascending=False)
		selected = alpha_factor
		selected.to_csv('IC_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(begt,endt,index,'AllbyT',str(small_comb_num),str(big_comb_num),sector))
	else:
		IC = IC.reindex_axis(kind)
		selected = IC
		selected.to_csv('IC_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(begt,endt,index,'all',str(small_comb_num),str(big_comb_num),sector))
	return selected


def analysis_sector_ic(sector,begt,endt,**kargs):
	latest_year = kargs.get('latest_year',2017)

	def get_excel_data(datapath,sector,begt,endt):
		
		def gen_df(df,columns,index):
			df.columns = columns
			df.index = index
			return df
		value = [	'1/PFCF', 'EBITDA/EV', 'S/EV', 
					'1/PE',	'1/PB',	'1/PS',	'1/POP',
					'1/PIC','1/PBE'
				]
		profit = [	'CROIC',	'ROIC',		'ROIC1',	
					'EP',		'EBITDA-资本支出/IC',	
					'ROE',	'ROE(扣除)',	'ROA',	
					'毛利率',			'营业利润率', 
					'净利率',			'营运报酬率',
					'资本报酬率',		'毛利/净资产',	
					'毛利/总资产',		'营业利润/净资产',	
					'毛利/EV',	
					'经营现金流/净资产','FCF/营业收入',	
					'销售现金比',		'现金营业收入比'
				]
		lev = [		'负债/投入资本','外部融资额/总资产',
					'资产负债率',	'有形净值债务率',
					'经营现金净流入/资本支出',	
					'折旧/投入资本',
					'长期负债变化率',
					'近两年平均资本支出增长率'
					]
		growth = [	'TTMFCF增长率',
					'TTM净利润增长率',
					'TTM营业利润增长率',
					'TTM营业收入增长率', 
					'最新一季净利润增长率', 
					'最新一季营业利润增长率',	
					'最新一季营业收入增长率',
					'最新一季销售毛利润增长率'
					]

		growth_moment = [	'GMvar_3y',		'GMvar_5y',		'OPMvar_3y',
							'OPMvar_5y',	'ROEvar_3y',	'ROEvar_5y',
							'GMstab_3y',	'GMstab_5y',	'OPMstab_3y',
							'OPMstab_5y',	'ROEstab_3y',	'ROEstab_5y',
							'GMcagr_3y',	'GMcagr_5y',	'OPMcagr_3y',	
							'OPMcagr_5y',	'ROEcagr_3y',	'ROEcagr_5y',
							'delta毛利率',	'delta营业利润率',	'deltaROE',
							'delta应收账款周转率',
							'delta总资产周转率',	'delta运营资产周转率',
							'deltaTA',	'deltaBK',	'deltaSALES',	'delta股东人数',
							'NOA',	'Accrual2NI',	'Accrual', '总资产'
							
						]
		trade = [	'近一个月跌幅', 'SKEW', '5/60','换手率',
					'ILLIQ','波动率','最近3个月动量', '最近6个月动量','最近一年动量'
					]

		predicted = ['目标收益率','预期增长率','预期PE','预期PEG',
					'Delta预期增长率','Delta预期增长率2',
					'Delta预期当年净利润增长率2'
					]


		sorted_columns = value+profit+lev+growth+growth_moment+trade+predicted
		date_list = [parse(str(x)).strftime('%Y-%m-%d') for x in get_tc_td(begt,endt)]
		ic_data = pd.read_excel(datapath+'{0}.xlsx'.format(sector))
		ic = ic_data.T.iloc[0:len(sorted_columns)].T
		ic = gen_df(ic,sorted_columns,date_list)
		rank_ic = ic_data.T.iloc[len(sorted_columns):2*len(sorted_columns)].T
		rank_ic = gen_df(rank_ic,sorted_columns,date_list)
		ls = ic_data.T.iloc[2*len(sorted_columns):3*len(sorted_columns)].T
		ls = gen_df(ls,sorted_columns,date_list)
		#import pdb;pdb.set_trace()
		return ic, rank_ic, ls

	def gen_ic_stats(ic,name):
		ic_mean = pd.DataFrame(ic.mean(),columns=['{0}'.format(name)])
		ic_win = pd.DataFrame((ic>0).sum()/len(ic),columns=['{0}>0'.format(name)])
		ic_t = pd.DataFrame(stats.ttest_1samp(ic, 0)[0],index=ic.columns,columns=['{0}t'.format(name)])
		return pd.concat([ic_mean,ic_t,ic_win],axis=1)
	def gen_ls_stats(ls,year):
		#import pdb;pdb.set_trace()
		ls_all = pd.DataFrame((1+ls/100).cumprod().iloc[-1]*100-100)
		ls_all.columns = ['LS']
		ls_data_year = ls[ls.index>='{0}-12-20'.format(str(year-1))]
		ls_thisyear = pd.DataFrame((1+ls_data_year/100).cumprod().iloc[-1]*100-100)
		ls_thisyear.columns = ['LS{0}'.format(str(year))]
		#import pdb;pdb.set_trace()
		return pd.concat([ls_all,ls_thisyear],axis=1)
	datapath = 'E://factor_comb//data//factorIC//'
	ic,rankic,ls = get_excel_data(datapath,sector,begt,endt)

	ic_stats = gen_ic_stats(ic,'IC')
	rankic_stats = gen_ic_stats(rankic,'RankIC')
	ls_stats = gen_ls_stats(ls,latest_year)
	stats_all = pd.concat([ic_stats,rankic_stats,ls_stats],axis=1).round(3)
	#import pdb;pdb.set_trace()
	return stats_all

def compare_sector_ic(sector,bench,begt,endt):
	sector_stats = analysis_sector_ic(sector,begt,endt)
	bench_stats = analysis_sector_ic(bench,begt,endt)
	compare_stats = 100*(sector_stats-bench_stats)/bench_stats.round(2)
	return sector_stats,compare_stats,bench_stats

def factor_t_stats_analysis(sector_df,bench_df):
	'''
	Inputs: returns from compare_sector_ic
	'''
	
	sector_marked = sector_df.index[sector_df['RankICt']>1.6].tolist()
	bench_marked = bench_df.index[bench_df['RankICt']>1.6].tolist()
	
	share = [x for x in sector_marked if x in bench_marked]
	only_sector = [x for x in sector_marked if x not in bench_marked]
	only_bench = [x for x in bench_marked if x not in sector_marked]
	#import pdb;pdb.set_trace()
	return share, only_sector, only_bench


def generate_sectorIC_report(sector_name,bench,begt,endt):
	'''
	Inputs 
		sector_name, bench: excel name from data/factorIC. 材料行业（消费行业）,800
	'''
	sector_0,compare_0,bench_0 = compare_sector_ic(sector_name+'因子_0',bench+'_0',begt,endt)
	sector_1,compare_1,bench_1 = compare_sector_ic(sector_name+'因子_1',bench+'_1',begt,endt)
	
	bench = pd.concat([bench_0,bench_1],axis=1)
	bench.columns = [	['无市值']*len(bench_0.columns)+['市值回归']*len(bench_1.columns),
						bench_0.columns.tolist()+bench_1.columns.tolist()]

	sector = pd.concat([sector_0,sector_1],axis=1)
	sector.columns = [	['无市值']*len(sector_0.columns)+['市值回归']*len(sector_1.columns),
						sector_0.columns.tolist()+sector_1.columns.tolist()]

	compare = pd.concat([compare_0,compare_1],axis=1)
	compare.columns = [	['无市值']*len(compare_0.columns)+['市值回归']*len(compare_1.columns),
						compare_0.columns.tolist()+compare_1.columns.tolist()]
	
	writer = pd.ExcelWriter('{0}_单因子.xlsx'.format(sector_name))
	sector.to_excel(writer,'行业内')
	bench.to_excel(writer,'800内')
	compare.to_excel(writer,'与800比较')
	writer.save()
	
	share_0,only_sector_0,only_bench_0 = factor_t_stats_analysis(sector_0,bench_0)
	share_1,only_sector_1,only_bench_1 = factor_t_stats_analysis(sector_1,bench_1)
	#import pdb;pdb.set_trace()
	txtName = '{0}_单因子基准比较.txt'.format(sector_name)
	f = open(txtName,'a+')
	new_context =	'****************************'+'\n'+\
					"（一）无市值回归"+'\n'+\
					'1.均显著因子:'+'\n'+\
					str(share_0)+'\n'+\
					'2.在_{0}显著而在800不显著:'.format(sector_name)+'\n'+\
					str(only_sector_0)+'\n'+\
					'3.在800显著而在_{0}不显著：'.format(sector_name)+'\n'+\
					str(only_bench_0)+'\n'+\
					'****************************'+'\n'+\
					'（二）市值回归'+'\n'+\
					'1.均显著因子:'+'\n'+\
					str(share_1)+'\n'+\
					'2.在_{0}显著而在800不显著:'.format(sector_name)+'\n'+\
					str(only_sector_1)+'\n'+\
					'3.在800显著而在_{0}不显著：'.format(sector_name)+'\n'+\
					str(only_bench_1)+'\n'
	f.write(new_context)
	f.close()

	return sector, compare
	#import pdb;pdb.set_trace()

if __name__=='__main__':
	index = 'SH000906'
	BegT = 20081220
	BegT_ = 20090730
	begt = 20121220
	EndT = 20170630
	cycle = '月线'
	KIND = VALUE
	'''
	slt_short = cal_ic(index,begt,EndT,cycle,KIND)
	slt_long = cal_ic(index,BegT,EndT,cycle,KIND,chosen=slt_short.index.values.tolist())
	
	cal_factor_corr(index,BegT,EndT,'ALL')
	import pdb;pdb.set_trace()
	'''

	'''
	sector = '申万金融'
	MODE =''
	cal_ic(index,BegT_,EndT,cycle,hcg+VALUE+PROFIT+GROWTH+GROWTH_2+OTHER ,sector=sector,mode=MODE)
	'''

	# y = analysis_sector_ic('800_1',20081220,20170531)
	# import pdb;pdb.set_trace()
	generate_sectorIC_report('中证工业','800',20090720,20170531)
	cal_factorIC_corr('中证工业')