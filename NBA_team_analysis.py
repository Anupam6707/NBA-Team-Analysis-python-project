import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
class team_analysis:
    def __init__(self,df):
        self.df=df.copy(deep=True)

    def basic_info(self):
        print('first 5 rows of table')
        print(self.df.head())
        total_obs=self.df.shape[0]
        print('a) total no of obs is:', total_obs)
        total_col=self.df.shape[1]
        print('a) total no of columns is:', total_col)
        miss_val=self.df.isnull().sum()
        print('b) missing values are:')
        print(miss_val)

    def req_cleaning(self):
        self.df['year']=self.df['year'].str[:4].astype(int)
        no_team=self.df['TEAM'].unique()
        print('teams in NBA:', no_team)
        self.df['season_type'].replace('Regular%20Season','rs_20', inplace=True)

    def data_permin_vis(self):
        reqd_cols=['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS' ]
        data_permin=self.df.groupby(['PLAYER','PLAYER_ID','year'])[reqd_cols].sum()
        print(data_permin)
        for c in data_permin.columns[4:]:
            data_permin[c]=data_permin[c]/data_permin['MIN']
        
        data_permin['FG%']=data_permin['FGM']/data_permin['FGA']
        data_permin['3PT%']=data_permin['FG3M']/data_permin['FG3A']
        data_permin['FT%']=data_permin['FTM']/data_permin['FTA']
        data_permin['FG3A%']=data_permin['FG3A']/data_permin['FGA']
        data_permin['PTS/FGA%']=data_permin['PTS']/data_permin['FGA']
        data_permin['FG3M/FGM%']=data_permin['FG3M']/data_permin['FGM']
        data_permin['FTA/FGA%']=data_permin['FTA']/data_permin['FGA']
        data_permin['TRU%']=data_permin['PTS']/(data_permin['FGA']+0.475*data_permin['FTA'])
        data_permin['AST_TOV%']=data_permin['AST']/data_permin['TOV']
        above_50=data_permin[data_permin['MIN']>=50]
        print('correlation of attributes')
        fig=px.imshow(data_permin.corr(),title='heatmap of players',labels=dict(color='correlation'))
        fig.update_layout(coloraxis_colorbar_title='correlation coefficient')
        fig.show()
        
    def time_series_gen_vis(self):
        reqd_cols=['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS' ]
        dt=self.df.groupby('year')[reqd_cols].sum().reset_index()
        dt['POSS']=dt['FGA']-dt['OREB']+dt['TOV']+0.44*dt['FTA']
        dt1=dt[list(dt.columns[:2])+['POSS']+list(dt.columns[2:-1])]
        dt1['FG%']=dt1['FGM']/dt1['FGA']
        dt1['3PT%']=dt1['FG3M']/dt1['FG3A']
        dt1['FT%']=dt1['FTM']/dt1['FTA']
        dt1['FG3A%']=dt1['FG3A']/dt1['FGA']
        dt1['PTS/FGA%']=dt1['PTS']/dt1['FGA']
        dt1['FG3M/FGM%']=dt1['FG3M']/dt1['FGM']
        dt1['FTA/FGA%']=dt1['FTA']/dt1['FGA']
        dt1['TRU%']=dt1['PTS']/(dt1['FGA']+0.475*dt1['FTA'])
        dt1['AST_TOV%']=dt1['AST']/dt1['TOV']
        print(dt1)
        dt1_copy=dt1.copy()
        for c in dt1_copy.columns[2:18]:
            dt1_copy[c]=(dt1_copy[c]/dt1_copy['MIN'])*48*5
        dt1_copy.drop(columns='MIN',inplace=True)
        fig=go.Figure()
        for c in dt1_copy.columns[1:]:
            fig.add_trace(go.Scatter(x=dt1_copy['year'],y=dt1_copy[c],name=f'{c}'))
        fig.update_layout(title='Time series analysis of all teams by attributes',xaxis_title='Year',yaxis_title='stat value',legend_title='statistics')
        fig.show()

    def time_series_ssn(self):
        season_copy=self.df[self.df['season_type']=='rs_20']
        fgm1=season_copy[['year','TEAM','FGM']]
        dg=fgm1[fgm1['TEAM']=='CLE']
        dg.head()
        lg=dg.groupby('year')['FGM'].sum().reset_index()
        lg.rename(columns={'FGM':'CLE'},inplace=True)
        teams=['OKC','MIA', 'HOU', 'NYK', 'GSW', 'MIL', 'POR', 'TOR',
       'BKN', 'CHA', 'LAC', 'BOS', 'UTA', 'PHI', 'IND', 'SAS', 'ATL',
       'LAL', 'NOP', 'DET', 'CHI', 'SAC', 'DAL', 'DEN', 'MEM', 'PHX',
       'ORL', 'MIN', 'WAS']
        for t in teams:
            df=fgm1[fgm1['TEAM']==t]
            df1=df.groupby('year')['FGM'].sum().reset_index()
            df1.rename(columns={'FGM':t},inplace=True)
            lg=pd.merge(lg,df1,how='left',on='year')
        print('the time series data of teams for seasons')
        print(lg)
        print('time series visualisation for teams in season')
        fig1=go.Figure()
        for col in lg.columns[1:]:
            fig1.add_trace(go.Scatter(x=lg['year'],y=lg[col],name=f'{col}'))
        
        fig1.show()


    def time_series_pfs(self):
        Playoff_data=self.df[self.df['season_type']=='Playoffs']
        fgm2=Playoff_data[['year','TEAM','FGM']]
        dg=fgm2[fgm2['TEAM']=='CLE']
        lg=dg.groupby('year')['FGM'].sum().reset_index()
        lg.rename(columns={'FGM':'CLE'},inplace=True)
        teams=['OKC','MIA', 'HOU', 'NYK', 'GSW', 'MIL', 'POR', 'TOR',
       'BKN', 'CHA', 'LAC', 'BOS', 'UTA', 'PHI', 'IND', 'SAS', 'ATL',
       'LAL', 'NOP', 'DET', 'CHI', 'SAC', 'DAL', 'DEN', 'MEM', 'PHX',
       'ORL', 'MIN', 'WAS']
        for t in teams:
            df=fgm2[fgm2['TEAM']==t]
            df1=df.groupby('year')['FGM'].sum().reset_index()
            df1.rename(columns={'FGM':t},inplace=True)
            lg=pd.merge(lg,df1,how='left',on='year')
        print('the time series data of teams for the playoff')
        print(lg)
        print('time series visualisation for teams for playoffs')
        fig1=go.Figure()
        for col in lg.columns[1:]:
            fig1.add_trace(go.Scatter(x=lg['year'],y=lg[col],name=f'{col}'))
        
        fig1.show()

    def team_attr_comparision(self):
        Playoff_data = self.df[self.df['season_type'] == 'Playoffs']
        rows = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        gpd = Playoff_data.groupby('TEAM', as_index=False)[rows].sum()
        gpd['FG%'] = gpd['FGM'] / gpd['FGA']
        gpd['3PT%'] = gpd['FG3M'] / gpd['FG3A']
        gpd['FT%'] = gpd['FTM'] / gpd['FTA']
    
        print('Contribution of steals from each team in playoffs')
        plt.figure(figsize=(10, 6))
        plt.pie(gpd['STL'], labels=gpd['TEAM'], autopct='%.2f%%')
        plt.title('Steals Contribution by Team in Playoffs')
        plt.legend(title='Teams', loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.show()


    def eda_on_3_attributes(self):
        Playoff_data=self.df[self.df['season_type']=='Playoffs']
        f = Playoff_data[['TEAM', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV']]
        rows = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV']        
        gpd = f.groupby('TEAM', as_index=False)[rows].sum()  # Keep TEAM as a column        
        gpd['FG%'] = gpd['FGM'] / gpd['FGA']
        gpd['3PT%'] = gpd['FG3M'] / gpd['FG3A']
        gpd['FT%'] = gpd['FTM'] / gpd['FTA'] 
        print(gpd.head())
        print('Comparision of teams based on three atttributes')
        fig,axes=plt.subplots(3,1,figsize=(20,21))
        axes[0].bar(gpd['TEAM'],gpd['FG%'],color='b')
        axes[1].bar(gpd['TEAM'],gpd['3PT%'],color='g')
        axes[2].bar(gpd['TEAM'],gpd['FT%'],color='y')
        plt.show()

    def last_ten_yr_winners(self):
        req_col=['FGM','FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        gp=self.df.groupby(['year','TEAM'])[req_col].sum().reset_index()
        dg=gp[gp['TEAM']=='GSW']
        dg_y=dg[dg['year']==2014]
        lis=["CLE", "GSW", "GSW", "TOR", "LAL", "MIL", "GSW", "DEN", "BOS"]
        year=2014
        for i in lis:
            data=gp[gp['TEAM']==i]
            data_year=data[data['year']==year+1]
            dg_y=pd.concat([dg_y,data_year],axis=0)
            year+=1
        print('time series analysis of the last 10 years winner based on attributes')
        comp=go.Figure()
        for i in dg_y.columns[2:]:
            comp.add_trace(go.Scatter(x=dg_y['year'],y=dg_y[i],name=f'{i}'))
            comp.update_layout(title='Last 10 Years NBA Winners - Attribute Comparison', xaxis_title='Year', yaxis_title='Stat Value', legend_title="Attributes")
            
        comp.show()

    def comp_gsw_team(self):
        req_col=['FGM','FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        gp1=self.df.groupby(['year','TEAM','PLAYER'])[req_col].sum().reset_index()
        gp2=gp1[gp1['year']==2014]
        gp3=gp2[gp2['TEAM']=='GSW']
        gp3['Total goals_pts']=2*(gp3['FGM']-gp3['FG3M'])+3*(gp3['FG3M'])+gp3['FTM']
        gp3['total_goals']=gp3['FGM']+gp3['FTM']
        gp4=gp3[['PLAYER', 'Total goals_pts',
               'total_goals']]
        fig,axes=plt.subplots(4,2,figsize=(20,20))
        axes[0,0].pie(gp4['Total goals_pts'],labels=gp4['PLAYER'],autopct='%.2f%%')
        axes[0,1].pie(gp4['total_goals'],labels=gp4['PLAYER'],autopct='%.2f%%')
        a=gp1[gp1['year']==2016]
        b=a[a['TEAM']=='GSW']
        b['Total goals_pts']=2*(b['FGM']-b['FG3M'])+3*(b['FG3M'])+b['FTM']
        b['total_goals']=b['FGM']+b['FTM']
        c=b[['PLAYER', 'Total goals_pts',
               'total_goals']]
        axes[1,0].pie(c['Total goals_pts'],labels=c['PLAYER'],autopct='%.2f%%')
        axes[1,1].pie(c['total_goals'],labels=c['PLAYER'],autopct='%.2f%%')
        a1=gp1[gp1['year']==2017]
        b1=a1[a1['TEAM']=='GSW']
        b1['Total goals_pts']=2*(b1['FGM']-b1['FG3M'])+3*(b1['FG3M'])+b1['FTM']
        b1['total_goals']=b1['FGM']+b1['FTM']
        c1=b1[['PLAYER', 'Total goals_pts',
               'total_goals']]
        axes[2,0].pie(c1['Total goals_pts'],labels=c1['PLAYER'],autopct='%.2f%%')
        axes[2,1].pie(c1['total_goals'],labels=c1['PLAYER'],autopct='%.2f%%')
        a2=gp1[gp1['year']==2021]
        b2=a2[a2['TEAM']=='GSW']
        b2['Total goals_pts']=2*(b2['FGM']-b2['FG3M'])+3*(b2['FG3M'])+b2['FTM']
        b2['total_goals']=b2['FGM']+b2['FTM']
        c2=b2[['PLAYER', 'Total goals_pts',
               'total_goals']]
        axes[3,0].pie(c2['Total goals_pts'],labels=c2['PLAYER'],autopct='%.2f%%')
        axes[3,1].pie(c2['total_goals'],labels=c2['PLAYER'],autopct='%.2f%%')
        plt.show()
                


    def final_output(self):
        self.basic_info()
        self.req_cleaning()
        self.data_permin_vis()
        self.time_series_gen_vis()
        self.time_series_ssn()
        self.time_series_pfs()
        self.team_attr_comparision()
        self.eda_on_3_attributes()
        self.last_ten_yr_winners()
        self.comp_gsw_team()

