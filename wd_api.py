import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
from data import (	DATA_PATH,
					load_zzhy_all)
from config import DATA_PATH
from WindPy import *
w.start()
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')




def get_sector_px(index,start,end,freq):
	'''
	index 000932.SH str
	start 2007-01-01 str
	end 2017-07-24 str
	freq: 'M' str
	'''

	data = w.wsd(index, "close", start, end, "Period={0}".format(freq))
	px_df = pd.DataFrame(np.array(data.Data).T,index=data.Times,columns=[data.Codes[0].split('.')[1]+data.Codes[0].split('.')[0]])
	#import pdb;pdb.set_trace()
	return px_df

def get_sectors(start,end,freq):
	sector_list = [	'000932.SH',
					'000931.SH',
					'000934.SH',
					'000930.SH',
					'000937.SH',
					'000929.SH',
					'000933.SH',
					'000935.SH',
					'000928.SH',
					'000936.SH',
					'000300.SH',
					'000842.SH',
					'000905.SH']

	# sector_list = [	'000930.SH',
	# 				'000300.SH',
	# 				'000842.SH',
	# 				'000905.SH',
	# 			]
	df = []
	for idx in sector_list:
		if not len(df):
			df = get_sector_px(idx,start,end,freq)
		else:
			df = pd.concat([df,get_sector_px(idx,start,end,freq)],axis=1)
	#corr_df = df.corr()
	#import pdb;pdb.set_trace()
	df.index.name='date'
	return df

yy = get_sectors('2006-01-01','2017-09-01','Y')
import pdb;pdb.set_trace()
ret = yy.pct_change()
nv = (1+ret.fillna(0)).cumprod()
ret_rank = ret.rank(axis=1)
nv_rank = nv.rank(axis=1)
sym = 'index'
ret.to_csv('ret_{0}.csv'.format(sym))
nv.to_csv('nv_{0}.csv'.format(sym))
ret_rank.to_csv('ret_rank_{0}.csv'.format(sym))
nv_rank.to_csv('nv_rank_{0}.csv'.format(sym))



def get_td(start,end,freq):
	#start,end "2017-03-27"
	data = w.tdays(start, end, "Period={0}".format(freq))
	td = [str(x)[:10] for x in data.Data[0]]
	#import pdb;pdb.set_trace()
	return td

#yy = get_td('2007-01-01','2017-07-24','M')

def get_sectorconst(date,index):
	#date:2017-04-27
	index_dict = {	'中证消费':'000932.SH',
					'中证可选':'000931.SH',
					'中证金融':'000934.SH',
					'中证工业':'000930.SH',
					'中证公用':'000937.SH',
					'中证材料':'000929.SH',
					'中证医药':'000933.SH',
					'中证信息':'000935.SH',
					'中证能源':'000928.SH',
					'中证电信':'000936.SH',
					'申万银行':'801790.SI',
					'申万非银':'801780.SI'}
	data = w.wset("sectorconstituent","date={0};windcode={1}".format(date,index_dict[index]))
	#import pdb;pdb.set_trace()
	return data

def get_zzhy(date,index):
	#print(date,index)
	stks = get_sectorconst(date,index)
	df = pd.DataFrame(stks.Data,index=stks.Fields,columns=stks.Codes)
	df = df.T
	#import pdb;pdb.set_trace()
	df['intdate'] = df['date'].apply(lambda x:int(str(x)[:10].replace('-','')))
	df['ts_code'] = df['wind_code'].apply(lambda x: x.split('.')[1]+x.split('.')[0])
	'''
	writer = pd.ExcelWriter('vg_{0}.xlsx'.format(date))
	df.to_excel(writer,'Sheet1')
	'''
	return df

def dump_sectorconst(start,end,index,freq,path=''):
	'''
	获取指数成分股数据
	'''
	td = get_td(start,end,freq)
	#import pdb;pdb.set_trace()
	all_stks = []
	for i in range(len(td)-1):
		month_df = get_zzhy(td[i],index)
		if not len(all_stks):
			all_stks = month_df
		else:
			all_stks = pd.concat([month_df,all_stks])
			#import pdb;pdb.set_trace()
	writer = pd.ExcelWriter(path+'{0}_stks.xlsx'.format(index))
	all_stks.to_excel(writer,'Sheet1')
	return all_stks


def dump_allsectorconst(start,end,freq,path):
	
	hy_list = [		'中证消费',	'中证可选',	'中证工业',	'中证公用',
					'中证材料',	'中证医药',	'中证信息',	'中证能源',	'中证电信',
				]


	for hy in hy_list:
		print(hy)
		dump_sectorconst(start,end,hy,freq,path)
	
	td = get_td(start,end,freq)
	fin_stks = []
	re_stks = [] 	#real estate
	for i in range(len(td)-1):
		zzjd = get_zzhy(td[i],'中证金融')
		swfin = pd.concat([get_zzhy(td[i],'申万非银'),get_zzhy(td[i],'申万银行')])
		zzre = zzjd[~zzjd['ts_code'].isin(swfin['ts_code'])]

		if not len(fin_stks):
			fin_stks = swfin
			re_stks = zzre
		else:
			fin_stks = pd.concat([swfin,fin_stks])
			re_stks = pd.concat([zzre,re_stks])

	writer = pd.ExcelWriter(path+'{0}_stks.xlsx'.format('申万金融'))
	fin_stks.to_excel(writer,'Sheet1')
	writer = pd.ExcelWriter(path+'{0}_stks.xlsx'.format('中证地产'))
	re_stks.to_excel(writer,'Sheet1')
	return 0

#dump_allsectorconst('2014-07-20','2017-07-07','M')

def get_indexweight(index,start,end,freq,path=''):
	'''
	获取指数权重数据
	start,end "2017-03-27"
	index '000300.SH'
	'''
	date_list = get_td(start,end,freq)

	hyratio_dict = {}
	
	hy_list = ['中证消费',	'中证可选',	'申万金融','中证地产','中证工业',	'中证公用',
					'中证材料',	'中证医药',	'中证信息',	'中证能源',	'中证电信']
	for hy in hy_list:
		hyratio_dict[hy] = ['']*len(date_list)

	
	sector_stks_dict = load_zzhy_all(DATA_PATH)
	index_const = []
	for i,d in enumerate(date_list):
		data = w.wset("indexconstituent","date={0};windcode={1}".format(d,index))
		const_df = pd.DataFrame(data.Data,index=data.Fields,columns=data.Codes).T
		const_df['date'] = const_df['date'].apply(lambda x: str(x)[:10])
		
		if not len(index_const):
			index_const = const_df
		else:
			index_const = pd.concat([index_const,const_df])
			#import pdb;pdb.set_trace()

		for hy in hy_list:
			sector_stks = sector_stks_dict[hy]
			sector_stks['date']=  sector_stks['date'].apply(lambda x :str(x)[:10])
			slt_stks = sector_stks.loc[sector_stks['date']==d,'wind_code']
			hyratio_dict[hy][i] = const_df.loc[const_df.wind_code.isin(slt_stks),'i_weight'].sum()
	#import pdb;pdb.set_trace()
	hyratio_df = pd.DataFrame(hyratio_dict,index=date_list)
	writer = pd.ExcelWriter(path+'{0}_indexweight.xlsx'.format(index.replace('.','')))
	#writer = pd.ExcelWriter('indexweight.xlsx')
	const_df.to_excel(writer,'Sheet1')
	hyratio_df.to_excel(writer,'Sheet2')

	#const_df.to_csv('const.csv')
	#hyratio_df.to_csv('hyratio.csv')

	return const_df,hyratio_df

def dump_all(start,start1,end,freq):
	#dump trading date
	
	# td = get_td(start,end,freq)
	# pd.DataFrame(td).to_csv(DATA_PATH+'trading_calendar//trading_date.csv')

	#dump index price
	index_px = get_sectors(start,end,freq)
	index_px.to_csv(DATA_PATH+'index//index_px.csv')
	
	# dump_allsectorconst(start1,end,freq,path=DATA_PATH+'industry//')
	
	# get_indexweight('000300.SH',start1,end,freq,path=DATA_PATH+'indexweight//')
	
	return 0

dump_all('2008-12-20','2009-07-20','2017-09-05','M')