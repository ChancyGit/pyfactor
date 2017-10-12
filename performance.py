import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
plt.style.use('ggplot')
from datetime import datetime




def create_annual_return(port_returns,periods=12):
    #import pdb;pdb.set_trace()
    equity_curve=(1+port_returns).cumprod()
    ret=equity_curve.ix[-1]/equity_curve.ix[0]
    annual_ret=(equity_curve.ix[-1]/equity_curve.ix[0])**(12/(len(equity_curve)-1))-1
    #import pdb;pdb.set_trace()
    return ret, annual_ret,equity_curve

def create_sharpe_ratio(port_returns,rf=0.0,periods=12):
    return np.sqrt(periods)*(np.mean(port_returns)-rf/periods)/np.std(port_returns)

def create_information_ratio(port_returns,periods=12):
    return ((1+np.mean(port_returns.values))**(periods)-1)/(np.sqrt(periods)*np.std(port_returns.values))

def create_drawdowns(port_returns):
    cumret = (1+port_returns).cumprod()-1
    highwatermark = np.zeros(len(cumret))
    drawdown = np.zeros(len(cumret))
    drawdownduration = np.zeros(len(cumret))

    for i in range(len(cumret)-1):
        t=i+1
        highwatermark[t] = max(highwatermark[t-1],cumret.values[t])
        drawdown[t] = 1-(1+cumret.values[t])/(1+highwatermark[t])
        if not drawdown[t]:
            drawdownduration[t]=0
        else:
            drawdownduration[t] = drawdownduration[t-1]+1
    maxDD = np.max(drawdown)
    maxDDD=np.max(drawdownduration)
    
    return maxDD,maxDDD

def calculate_performance(comb_cumret,bench_cumret,threshhold=0.0):

    #import pdb;pdb.set_trace()
    alpha_value = comb_cumret.pct_change().values-bench_cumret.pct_change().values+1
    alpha = pd.DataFrame(alpha_value,index=comb_cumret.index,columns=['alpha'])
    alpha = alpha.fillna(1)
    alpha = alpha.cumprod()
    alpha_pct = alpha.pct_change()
    alpha_pct.fillna(0,inplace=True)
   
    total_ret,ann_ret,ev=create_annual_return(alpha_pct)
    #import pdb;pdb.set_trace()
    sharpe=create_sharpe_ratio(alpha_pct)
    #info=create_information_ratio((1+comb_cumret).pct_change().fillna(0),(1+bench_cumret).pct_change().fillna(0))
    info=create_information_ratio(alpha_pct)
    max_dd,max_dd_duration=create_drawdowns(alpha_pct)

    #how many days the comb outperforms the bench
    f = lambda x: len([y for y in x[x>threshhold]])
    outperf_day_num=alpha_pct.apply(f).values.tolist()[0]
    stats=alpha_pct.describe()

    stats.ix['winratio']=outperf_day_num/(stats.ix['count'].tolist()[0]-1)
    stats.ix['total_ret']=total_ret
    stats.ix['ann_ret']=ann_ret
    stats.ix['sharpe_ratio']=sharpe
    stats.ix['info_ratio']=info
    stats.ix['max_dd']=max_dd
    stats.ix['max_dd_duration']=max_dd_duration
    stats.ix['calmar']=ann_ret/max_dd
    stats.ix['start']=alpha_pct.index[0]
    stats.ix['end']=alpha_pct.index[-1]
    return stats

def cal_perf_detail(comb_nv,bench_nv,date_list,**kargs):
	by_year = kargs.get('by_year',0)
	plot 	= kargs.get('plot',0)

	#import pdb;pdb.set_trace()
	year_end_date = [x for x in bench_nv.index if x[5:7]=='12']
	if date_list[0] not in year_end_date:
		year_end_date.insert(0,date_list[0])
	year_end_date.append(date_list[-1])
	if by_year:
		perf_yearly={}
		for x in range(1,len(year_end_date)):
			#import pdb;pdb.set_trace()
			year = year_end_date[x][:4]
			print('******************')
			print(year)
			comb_year = comb_nv.ix[year_end_date[x-1]:year_end_date[x]]
			bench_year = bench_nv.ix[year_end_date[x-1]:year_end_date[x]]
			perf_year = calculate_performance(comb_year,bench_year)
			print(perf_year)
			print('******************')
			perf_yearly[year] = perf_year

	
	if plot:
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()

	b = bench_nv
	c = comb_nv
	df = pd.concat([b,c],axis=1)
	if plot:
		df.plot(ax=ax1)
		#import pdb;pdb.set_trace()
	if np.sum(df.columns == ['bench_nv','comb_nv'])==2:
		alpha_value = df['comb_nv'].pct_change().values-df['bench_nv'].pct_change().values+1
	elif df.columns[1] == 'comb_nv':
		alpha_value = df['comb_nv'].pct_change().values-df[bench_nv.columns[0]].pct_change().values+1
	else:
		alpha_value = df['comb'].pct_change().values-df[bench_nv.columns[0]].pct_change().values+1
		
	df['alpha'] = alpha_value
	df[['alpha']] = df[['alpha']].fillna(1)
	df[['alpha']] = df[['alpha']].cumprod()
	if plot:
		print('******Final CUMRET*******')
		print(df['alpha'].iloc[-1])


	df.to_csv('net_values.csv')		

	latest_year = df.ix[year_end_date[-2]:year_end_date[-1]].copy()
	latest_year = latest_year.pct_change().fillna(0)
	latest_year = (1+latest_year).cumprod()
	if plot:
		print('***Latest Year Performance***')
		print(latest_year)

	perf = calculate_performance(comb_nv,bench_nv)
	if plot:
		print('******************')
		print('All Period Performance')
		print(perf)
		
		df[['alpha']].plot(ax=ax2,color='m')
		plt.show()
		
	return perf.ix['ann_ret'],perf.ix['sharpe_ratio'],perf.ix['max_dd'],perf.ix['calmar']





if __name__=='__main__':
	pass