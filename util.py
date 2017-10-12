import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
plt.style.use('ggplot')
from datetime import datetime
from dateutil.parser import parse
from data import (	DATA_PATH,
					get_fin)
# import matlab.engine
# eng = matlab.engine.start_matlab()



def score_factors(factor_df, gauss_factors, linear_factors,factor_list,size_neutral=0):
	'''
	Inputs:
		factor_df: dataframe contains all factor values
		gauss_factors: list , factors scored by zzfb
		linear_facotrs: list, factors scored by xx
		factor_list: selected factors
	Outputs:
		factor_df: factor scores instead of values
	'''

	# for f in factor_list:
	# 	factor_df = factor_df[factor_df[f].notnull()]
	#import pdb;pdb.set_trace()

	for f in factor_list:
		factor_df = factor_df[factor_df[f].notnull()]

	for f in factor_list:
		if f in gauss_factors:
			thresh = 0.025
			upper = factor_df[f].quantile(1-thresh)
			lower = factor_df[f].quantile(thresh)
			slt = factor_df[[f]][(factor_df[f]>lower) & (factor_df[f]<upper)]
			factor_df.loc[:,f] = (factor_df[[f]] - slt.mean())/slt.std()
			factor_df.loc[factor_df[f]>2.5,f] = 2.5
			factor_df.loc[factor_df[f]<-2.5,f] = -2.5

		elif f in linear_factors:
			#import pdb;pdb.set_trace()
			# thresh = 0.2
			# upper = factor_df[f].quantile(1-thresh)
			# lower = factor_df[f].quantile(thresh)
			# factor_df.loc[factor_df[f]>upper,f] = upper
			# factor_df.loc[factor_df[f]<lower,f] = lower
			# factor_df.loc[:,f] =  -2.5+5*(factor_df[[f]] - lower)/(upper-lower)
			thresh = 0.025
			upper = factor_df[f].quantile(1-thresh)
			lower = factor_df[f].quantile(thresh)
			slt = factor_df[[f]][(factor_df[f]>lower) & (factor_df[f]<upper)]
			factor_df.loc[:,f] = (factor_df[[f]] - slt.mean())/slt.std()
			factor_df.loc[factor_df[f]>2.5,f] = 2.5
			factor_df.loc[factor_df[f]<-2.5,f] = -2.5
		else:
			print('such {0} factor is not supposed to be here'.format(f))

		if size_neutral:
			if f not in ['总市值']:
				X = sm.add_constant(factor_df['总市值'].values)
				y = factor_df[f].values
				res = sm.RLM(y,X).fit()
				factor_df[f] = res.resid



	return factor_df


def cal_score_factors_all(date_list,**kargs):
	stk_pool			= kargs.get('stk_pool','zz800')
	sector 				= kargs.get('sector','')
	factor_data_dict 	= kargs.get('factor_data_dict',{})
	sector_stks_dict 	= kargs.get('sector_stks_dict',{})
	gauss 				= kargs.get('gauss_factors',[])
	linear 				= kargs.get('linear_factors',[])
	factor_list 		= kargs.get('factor_list',[])
	filter_fin 			= kargs.get('filter_fin',1)
	size_neutral 		= kargs.get('size_neutral',1)
	for x in range(0,len(date_list)-1):
		# print(date_list[x])
		INTdate = int(date_list[x].replace('-',''))
		factor = get_fin(INTdate,
						stk_pool=stk_pool,
						sector=sector,
						sector_stks_dict=sector_stks_dict)
		
		if filter_fin:
			factor = factor[factor['行业编码']>3]

		#import pdb;pdb.set_trace()
		#temp = factor[factor_list+['代码']].dropna()
		#temp = temp.set_index('代码')
		#import pdb;pdb.set_trace()
		factor = score_factors(	factor,
								gauss,
								linear,
								factor_list,
								size_neutral=size_neutral)
		#import pdb;pdb.set_trace()
		factor = factor.set_index('代码')
		factor_data_dict[INTdate] = factor
	return factor_data_dict

def factor_filter(factor_df,factor,filter_percent):
	'''
	filter stks by single factor
	'''
	#factor_df[factor+'_score'] = factor_df[factor.split(';')].mean(axis=1)
	#import pdb;pdb.set_trace()
	if len(factor) == 1:
		#import pdb;pdb.set_trace()
		factor_df.loc[:,factor[0]+'_score'] = factor_df[factor[0].split(';')].mean(axis=1)
		factor_df = factor_df.sort_values(by=factor[0]+'_score',ascending=False).iloc[:round(filter_percent*len(factor_df))]
	elif len(factor) > 1:
		factor_df['+'.join(factor)+'_score'] = 0
		for f in factor:
			factor_df.loc[:,f+'_score'] = factor_df[f.split(';')].mean(axis=1)
			
			if f.split(';') == 1:
				pass
			else:				
				thresh = 0.01
				# upper = factor_df[f+'_score'].quantile(1-thresh)
				# lower = factor_df[f+'_score'].quantile(thresh)
				# slt = factor_df[[f+'_score']][(factor_df[f+'_score']>lower) & (factor_df[f+'_score']<upper)]
				# factor_df.loc[:,f+'_score'] = (factor_df[[f+'_score']] - slt.mean())/slt.std()
				# factor_df.loc[factor_df[f+'_score']>2.5,f] = 2.5
				# factor_df.loc[factor_df[f+'_score']<-2.5,f] = -2.5
				factor_df['+'.join(factor)+'_score'] = factor_df['+'.join(factor)+'_score']+factor_df[f+'_score']
			
		factor_df = factor_df.sort_values(by='+'.join(factor)+'_score',ascending=False).iloc[:round(filter_percent*len(factor_df))]
	else:
		print('CHECH FACTOR LIST INPUTS. THERE SHOULD NOT BE BLANK LIST')
	return factor_df

def stk_filter(factor_df,**kargs):
	target_stock_num = kargs.get('stk_num',80)
	date = kargs.get('date',0)
	target_percent = kargs.get('target_percent','')
	factor_list = kargs.get('factor_list',[])
	sector_factor_dict = kargs.get('sector_factor_dict',{})
	sector_stks_dict = kargs.get('sector_stks_dict',{})
	if_save_fin = kargs.get('if_save_fin',0)

	#import pdb;pdb.set_trace()
	if len(factor_list):
		hycomb = factor_df.copy()
		round_num = len(factor_list)
		if target_percent == '':
			filter_percent = (target_stock_num/len(factor_df))**(1/round_num)
		else:
			filter_percent = (target_percent)**(1/round_num)
		for f in factor_list:
			hycomb = factor_filter(hycomb,f,filter_percent)
		mycomb = hycomb.index
		return mycomb,[]
	
	if len(sector_factor_dict):
		print(date)
		mycomb = []
		mycomb_hy = []

		for hy,factor_list in sector_factor_dict.items():
			sector_const = []
			for ind in hy.split(';'):
				if not len(sector_const):
					sector_const = sector_stks_dict[ind]
				else:
					sector_const = pd.concat([sector_const,sector_stks_dict[ind]])
					#import pdb;pdb.set_trace()
			hy_stks=sector_const.loc[sector_const.intdate==date,'ts_code'].values.tolist()
		
			hycomb = factor_df[factor_df.index.isin(hy_stks)]
			#import pdb;pdb.set_trace()

			round_num = len(factor_list)
			filter_percent = (target_percent)**(1/round_num)
			for f in factor_list:
				if hy == '申万金融' and if_save_fin:
					break
				else:
					#import pdb;pdb.set_trace()
					hycomb = factor_filter(hycomb,f,filter_percent)
			mycomb_hy = mycomb_hy + [hy]*len(hycomb)
			if not len(mycomb):
				mycomb = hycomb.index.tolist()
			else:
				mycomb = mycomb+hycomb.index.tolist()

		#import pdb;pdb.set_trace()
		return mycomb,mycomb_hy