import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')
from datetime import datetime
from dateutil.parser import parse
from config import DATA_PATH


def get_tc_td(begt,endt):
	data = pd.read_csv(DATA_PATH+'trading_calendar\\trading_date.csv')
	data = data[['0']]
	begin = parse(str(begt)).strftime('%Y-%m-%d')
	end = parse(str(endt)).strftime('%Y-%m-%d')
	#import pdb;pdb.set_trace()
	data = data[data>=begin][data<=end].dropna()
	date = [x[0] for x in data.values.tolist()]
	#import pdb;pdb.set_trace()
	return date


def get_index_px(index,begt,endt):
	bench_px = pd.read_csv(DATA_PATH+'index//index_px.csv')
	bench_px['date'] = bench_px['date'].apply(lambda x: str(x)[:10])
	bench_px = bench_px[['date',index]]
	bench_px = bench_px.set_index('date')
	bench_px = bench_px.ix[begt:endt]
	
	bench_pct = bench_px.pct_change().fillna(0)
	bench_nv = (1+bench_pct).cumprod()
	#import pdb;pdb.set_trace()
	return bench_nv

def get_zzhy(datapath,sector,date):
	sector_const = pd.read_excel(datapath+'{0}_stks.xlsx'.format(sector))
	month_stks = sector_const.loc[sector_const.intdate==date,'ts_code'].values.tolist()
	return month_stks

def load_zzhy_all(datapath):
	hy_list = ['中证消费',	'中证可选',	'申万金融','中证地产','中证工业',	'中证公用',
					'中证材料',	'中证医药',	'中证信息',	'中证能源',	'中证电信']
	sector_stks_dict = {}
	for hy in hy_list:
		sector_stks_dict[hy] = pd.read_excel(DATA_PATH+'industry//'+'{0}_stks.xlsx'.format(hy))
	return sector_stks_dict

def load_indexweight_all(datapath):

	indexweight = pd.read_excel(DATA_PATH+'indexweight//'+'000300SH_indexweight.xlsx',sheetname='Sheet2')
	return indexweight

def CSVtoHDF(start,end,**kargs):
	data_path = kargs.get('data_path',DATA_PATH)
	hdf_fn = kargs.get('hdf_fn','FinFactors')
	td = get_tc_td(start,end)
	for d in td:
		data = pd.read_excel(data_path+'{0}.xlsx'.format(d.replace('-','')))
		col = data.columns
		new_col = [x.replace('/','%') for x in col]
		data.columns = new_col
		#import pdb;pdb.set_trace()
		data.to_hdf('%s.h5'%(hdf_fn),
				'Financials/{0}'.format(d.replace('-','')),
				format='table',append=True,data_columns=True)
	return 0

# CSVtoHDF('2008-12-20','2017-09-01',hdf_fn='FinFactorsAll')
# import pdb;pdb.set_trace()

def get_fin(date,**kargs):
	data_path = kargs.get('data_path',DATA_PATH)
	stk_pool  = kargs.get('stk_pool','zz800')
	sector = kargs.get('sector','')
	sector_stks_dict = kargs.get('sector_stks_dict',{})
	
	
	if stk_pool == 'zz800':
		data = pd.read_hdf('FinFactors.h5','Financials/{0}'.format(date))
		col = data.columns
		new_col = [x.replace('%','/') for x in col]
		data.columns = new_col
	else:
		# data = pd.read_excel(data_path+'{0}.xlsx'.format(date))
		data = pd.read_hdf('FinFactorsAll.h5','Financials/{0}'.format(date))
		col = data.columns
		new_col = [x.replace('%','/') for x in col]
		data.columns = new_col
	
	# insert data adjustment if needed below
	# import pdb;pdb.set_trace()
	# data['NOA'] = -data['NOA']

	
	if sector in ['800ew','300','500']:
		return data
	elif len(sector.split(';'))==1:
		
		sector_const = sector_stks_dict[sector]
		month_stks = sector_const.loc[sector_const.intdate==date,'ts_code'].values.tolist()
		'''
		month_stks = get_zzhy(data_path,sector,date)
		'''
		return data[data['代码'].isin(month_stks)]
	else:
		all_stks = []
		for s in sector.split(';'):
			sector_const = sector_stks_dict[s]
			month_stks = sector_const.loc[sector_const.intdate==date,'ts_code'].values.tolist()
			all_stks += month_stks
		return data[data['代码'].isin(all_stks)]