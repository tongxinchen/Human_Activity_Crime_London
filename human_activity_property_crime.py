"""
Human activity (footfall) & Property Crime pattern
Unit of analysis in London: 4835 LSOAs * 24 Months = 116,040
X: static measurements: social disorganisation (sociodemographic) variables (percentage),
                        crime generators (POI) variables (counts),
   dynamic human activity variables (MDAF: Monthly daily average footfall)
y: crime rate = crime counts / resident population

"""

##
import os
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
##
"""
PART 1: X, Y dataset processing
"""

df_xy = pd.read_csv('data/df_xy.csv', index_col=0)
df_des = df_xy.describe().T


## 1. static and dynamic cols
static_x_list = ['unemployed_perc', 'rent_social_perc', 'level4_perc', 'car_3_over_perc', 'young_16_34_per',
                 '102', '1057', '320', '422', '425', '947']  # static x -- 11
print(len(static_x_list))

dynamic_x_list = ['ww_0_6', 'ww_7_10', 'ww_11_15', 'ww_16_19', 'ww_20_23',
                  'we_0_6', 'we_7_10', 'we_11_15', 'we_16_19', 'we_20_23']  # dynamic x -- 10
print(len(dynamic_x_list))

p_crime = ['p_crime', 'Bicycle theft', 'Burglary', 'Criminal damage and arson',
           'Robbery', 'Shoplifting', 'Theft from the person', 'Vehicle crime']

## 2. train and test set split, then z-score standardisation at X

def standardscaler_x(df_xy, x_cnames, y_cnames):
    # only scaling the X

    from sklearn.preprocessing import StandardScaler
    df_xy.index = range(len(df_xy))
    df_y = df_xy[['lsoa', 'month'] + y_cnames]  # select y
    scaler = StandardScaler().fit(df_xy[x_cnames])  # only fit the X
    df_xn = pd.DataFrame(scaler.transform(df_xy[x_cnames]), columns=x_cnames)
    df_xy_n = pd.concat([df_y, df_xn], axis=1)

    return df_xy_n  #  this function do ###

df_xy_train = df_xy[~df_xy.month.isin(['2021-08', '2021-09', '2021-10', '2021-11', '2021-12'])]  # train part: 19 month
df_xy_test = df_xy[df_xy.month.isin(['2021-08', '2021-09', '2021-10', '2021-11', '2021-12'])]  # test part: 5 month

df_xy_train_n = standardscaler_x(df_xy_train, static_x_list+dynamic_x_list, p_crime)
df_xy_test_n = standardscaler_x(df_xy_test, static_x_list+dynamic_x_list, p_crime)


"""
PART 2: LASSO Regressors/model training and testing 
"""
##
def LASSO_MODEL(df_xy_train, df_xy_test, p_crime, static_x_list, dynamic_x_list):
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.linear_model import LassoCV, LassoLarsCV
    from numpy import arange

    lasso_results = []
    for i in p_crime: # 8 types of y
        print(i)
        y_train = df_xy_train[i].to_list()
        # print(np.round(np.var(y) - np.mean(y),2))
        print('y_shape', len(y_train) )
        for j in [static_x_list, dynamic_x_list, static_x_list+dynamic_x_list]: # 3 sets of x
            df_x_train = df_xy_train[j]
            print('X_shape:', df_x_train.shape)

            mod = LassoCV(alphas=arange(0.001, 1, 0.001), cv=10, max_iter=1000) # cv lasso set
            mod.fit(df_x_train, y_train)  # selecting the alpha
            best_mod = Lasso(alpha=mod.alpha_).fit(df_x_train, y_train) # get best alpha for best lasso

            alpha = mod.alpha_
            mod_r_sq = best_mod.score(df_x_train, y_train) # best lasso r2

            y_pre_train = best_mod.predict(df_x_train)
            mse_1 = mean_squared_error(y_pre_train, y_train).round(8)
            mod_rmse = np.sqrt(mse_1)

            df_x_test = df_xy_test[j]
            y_predict = best_mod.predict(df_x_test)
            y_test = df_xy_test[i].to_list()

            mse = mean_squared_error(y_predict, y_test).round(8)
            rmse = np.sqrt(mse)

            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(pd.DataFrame(y_predict), y_test)
            r_sq = lr.score(pd.DataFrame(y_predict), y_test)

            from regressors import stats
            p_values = stats.coef_pval(best_mod, df_x_train, y_train)
            # print(p_values)


            lasso_results.append([i, len(j), mod_r_sq, mod_rmse, alpha,
                                  np.array(best_mod.coef_),
                                  rmse, r_sq,
                                  p_values[1:]])

        df_results = pd.DataFrame({'crime_type': [n[0] for n in lasso_results],
                                   'x_number': [n[1] for n in lasso_results],
                                   'mod_r2': [n[2] for n in lasso_results],
                                   'mod_rmse': [n[3] for n in lasso_results],
                                   'alpha': [n[4] for n in lasso_results],
                                   'coef': [n[5] for n in lasso_results],
                                   'rmse': [n[6] for n in lasso_results],
                                   'r_sq': [n[7] for n in lasso_results],
                                   'p_values': [n[8] for n in lasso_results]})

    return df_results

def get_df_coef(df_lasso_result_n):
    df_result_lasso_s = df_lasso_result_n[df_lasso_result_n.x_number == 21]
    df_result_lasso_ss = df_result_lasso_s[['coef']]
    df_result_lasso_ss.index = df_result_lasso_s.crime_type.to_list()
    for n in p_crime:
        for j, c in enumerate(static_x_list + dynamic_x_list):
            df_result_lasso_ss.loc[n, c] = df_result_lasso_ss.coef[n][j]

    df_result_lasso_ss = df_result_lasso_ss.drop(columns='coef')
    df_result_lasso_ss = df_result_lasso_ss[dynamic_x_list + static_x_list]
    df_result_lasso_ss = df_result_lasso_ss[['ww_0_6', 'ww_7_10', 'ww_11_15', 'ww_16_19', 'ww_20_23',
                                            'we_0_6', 'we_7_10', 'we_11_15', 'we_16_19', 'we_20_23']+static_x_list]
    return df_result_lasso_ss

def get_df_coef_pvalue(df_lasso_result_n):
    df_result_lasso_s = df_lasso_result_n[df_lasso_result_n.x_number == 21]
    df_result_lasso_ss = df_result_lasso_s[['p_values']]
    df_result_lasso_ss.index = df_result_lasso_s.crime_type.to_list()
    for n in p_crime:
        for j, c in enumerate(static_x_list + dynamic_x_list):
            df_result_lasso_ss.loc[n, c] = df_result_lasso_ss.p_values[n][j]

    df_result_lasso_ss = df_result_lasso_ss.drop(columns='p_values')
    df_result_lasso_ss = df_result_lasso_ss[dynamic_x_list + static_x_list]
    df_result_lasso_ss = df_result_lasso_ss[['ww_0_6', 'ww_7_10', 'ww_11_15', 'ww_16_19', 'ww_20_23',
                                         'we_0_6', 'we_7_10', 'we_11_15', 'we_16_19', 'we_20_23']+static_x_list]
    return df_result_lasso_ss

##

df_lasso_result = LASSO_MODEL(df_xy_train_n,  df_xy_test_n, p_crime, static_x_list, dynamic_x_list)
df_lasso_coef = get_df_coef(df_lasso_result)
df_lasso_coef_pvalues = get_df_coef_pvalue(df_lasso_result)

##

df_lasso_result.to_csv('data/df_lasso_result.csv')
df_lasso_coef.to_csv('data/df_lasso_result_coef.csv')
df_lasso_coef_pvalues.to_csv('data/df_lasso_result_coef_pvalues.csv')

##

"""
PART 3: LASSO Regressors (one month data as training set: 2020-01; 2020-04; 2020-11; 2021-01)
"""

## before lockdown, first national lockdown, recovery

one_train_set = [['2020-02'], ['2020-04'], ['2020-08'], ['2020-11'], ['2021-01']]
one_test_set = [['2020-03'], ['2020-05'], ['2020-09'], ['2020-12'], ['2021-02']]

for m in range(5):

    df_xy_tr = df_xy_train_n[df_xy_train_n.month.isin(one_train_set[m])]
    df_xy_te = df_xy_train_n[df_xy_train_n.month.isin(one_test_set[m])]
    df_lasso_result_n = LASSO_MODEL(df_xy_tr, df_xy_te, p_crime, static_x_list, dynamic_x_list)
    print(m, df_lasso_result_n.query('crime_type == "Theft from the person" and x_number == 21').mod_r2.values)
    df_result_lasso_ss = get_df_coef(df_lasso_result_n)
    df_lasso_coef1_pvalues = get_df_coef_pvalue(df_lasso_result_n)
    df_lasso_result_n.to_csv(f'data/df_lasso_result_{m}.csv')
    df_result_lasso_ss.to_csv(f'data/df_lasso_result_coef_{m}.csv')
    df_lasso_coef1_pvalues.to_csv(f'data/df_lasso_result_coef_pvalues_{m}.csv')


"""
PART 4: mGWR (x: the highest coefficient variable and y: theft from person)
"""

##

from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

gdf_london = gpd.read_file('data/gdf_london.shp')
x_list_select = ['ww_11_15', 'ww_20_23', 'ww_11_15', 'we_16_19']
month_select = ['2020-02', '2020-04', '2020-08', '2020-11']

for c in ['Theft from the person']:

    print(c)
    r2_list = []
    bw_list = []
    df_l_list = []
    for i, m in enumerate(month_select):
        print(m)

        df_xy_s = df_xy_train[df_xy_train.month.str.contains(m)] # select month
        g_coords = list(zip(gdf_london.centroid.x, gdf_london.centroid.y))
        df_xy_geo = pd.DataFrame(pd.merge(gdf_london, df_xy_s, left_on='geo_code', right_on='lsoa', how='left'))
        df_xy_geo[c] = [(n - np.mean(df_xy_geo[c].to_list()))/np.std(df_xy_geo[c].to_list()) for n in df_xy_geo[c].to_list()]
        g_y = df_xy_geo[c].values.reshape((-1, 1))
        print('y_shape:', g_y.shape)

        df_xy_geo[x_list_select[i]] = [(n - np.mean(df_xy_geo[x_list_select[i]].to_list())) / np.std(df_xy_geo[x_list_select[i]].to_list()) for n in df_xy_geo[x_list_select[i]].to_list()]

        g_X = df_xy_geo[[x_list_select[i]]].values # selcet X
        print('X_shape:', g_X.shape)

        gwr_selector = Sel_BW(g_coords, g_y, g_X)
        gwr_bw = gwr_selector.search(bw_min=2)
        print('bw', gwr_bw)
        gwr_results = GWR(g_coords, g_y, g_X, gwr_bw).fit()
        print('general R2:', gwr_results.R2)

        # mgwr_selector = Sel_BW(g_coords, g_y, g_X, multi=True)
        # gwr_bw = mgwr_selector.search(multi_bw_min=[2])
        # print('bw', gwr_bw)
        # gwr_results = MGWR(g_coords, g_y, g_X, mgwr_selector).fit()
        # print('general R2:', gwr_results.R2)
        x1_coef = gwr_results.params[:, 1]
        df_coef = pd.DataFrame({m: x1_coef})

        # local_r2 = [r[0] for r in gwr_results.localR2]
        # df_lr2 = pd.DataFrame({m: local_r2})
        # print('max_local_R2', np.max(local_r2))

        r2_list.append(gwr_results.R2)
        # df_l_list.append(df_lr2)
        df_l_list.append(df_coef)
        bw_list.append(gwr_bw)

    df_gwr_result_global = pd.DataFrame({'month': month_select, 'r2': r2_list, 'bw': bw_list})
    df_gwr_result_global.to_csv(f'data/df_gwr_result_global.csv')
    df_gwr_result_local = pd.concat(df_l_list, axis=1)
    df_gwr_result_local['lsoa'] = gdf_london.geo_code.to_list()
    df_gwr_result_local.to_csv(f'data/df_gwr_result_local.csv')

##
"""
PART 5: Plotting results
"""

## 5.1 Fig: local coef for all crime models
def plot_local_coef(df_result_lasso_ss, df_result_lasso_pvalues, month):
    static_names = ['Unemployment', 'Rent social house ', 'Education above level 4', 'Own cars above 3', 'Young residents 16-34',
                    'Eating and drinking', 'Public transport', 'Tourism',  'Gambling', 'Venues, stage and screen',
                    'Food, drink and multi-item retail']
    dynamic_names = ['Early morning (WD)', 'Morning (WD)', 'Midday (WD)', 'Afternoon (WD)', 'Evening (WD)',
                    'Early morning (WE)', 'Morning (WE)', 'Midday (WE)', 'Afternoon (WE)', 'Evening (WE)']
    crime_names = ['All crime', 'Bicycle theft', 'Burglary', 'Criminal damage and arson',
                    'Robbery', 'Shoplifting', 'Theft from the person', 'Vehicle crime']

    sns.set_theme(style="ticks")
    from matplotlib.ticker import FormatStrFormatter

    # Set up the matplotlib figure
    fig,ax = plt.subplots(8, 1, figsize=(8, 12), sharex=True)

    for i,n in enumerate(p_crime):

        barlist = ax[i].bar(x=df_result_lasso_ss.columns.to_list(),
                            height=df_result_lasso_ss.loc[n].values, width=1, label=n)

        for j,m in enumerate(df_result_lasso_pvalues.loc[n].to_list()):
            if m < 0.05:
                barlist[j].set_color('#4C72B0')
            else:
                barlist[j].set_color('#6A6A6A')

        ax[i].hlines(y=0, color="k", xmin=-1, xmax=21, clip_on=False, lw=0.8)
       # ax[i].set_ylabel(n[0:3], fontsize = 18,  rotation = '90', loc = 'center')
        ax[i].set_ylim([np.min(df_result_lasso_ss.loc[n].to_list()) - 0.05, np.max(df_result_lasso_ss.loc[n].to_list())+0.05])
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].tick_params(axis='y', labelsize=13)

        ax[i].tick_params(axis='x',  # changes apply to the x-axis
                          which='both',  # both major and minor ticks are affected
                          bottom=False,  # ticks along the bottom edge are off
                          top=False,  # ticks along the top edge are off
                          labelbottom=False)  # labels along the bottom edge are off

        from matplotlib.lines import Line2D
        ax[i].legend(handles=[Line2D([0], [0], lw=0, color='orange', label=crime_names[i])],
                     fontsize=14, frameon=False, loc='upper right')


    ax[4].set_ylabel('Coefficient value', fontsize=15, loc='center')
    ax[7].tick_params(
          axis='x',  # changes apply to the x-axis
          which='both',  # both major and minor ticks are affected
          bottom=True,  # ticks along the bottom edge are off
          top=False,  # ticks along the top edge are off
          labelbottom=True)  # labels along the bottom edge are off

    ax[7].set_xticks(df_result_lasso_ss.columns.to_list())
    ax[7].set_xticklabels(labels=dynamic_names + static_names, rotation='270', fontsize=13)

    import matplotlib.patches as mpatches
    colors = ['#6A6A6A', '#4C72B0']
    fig.legend(bbox_to_anchor=(0.55, 1.01),  # position
               handles=[mpatches.Patch(color=colors[m],
                                       label=['Not significant', 'p < 0.05'][m]) for m in range(2)],
               loc='upper center', fontsize=13, frameon=False, ncol=2)

    # ax[0].set_title(f'Obsevation period {month}')
    sns.despine(bottom=True, left=False)

    plt.tight_layout(h_pad=0.5)

    # plt.annotate(r"$\}$",fontsize=24,
    #             xy=(0.27, 0.77), xycoords='figure fraction', rotation = '270'
    #             )
    plt.savefig(f'data/cs_1_model_local_coef_crime_all.jpg', dpi=300, bbox_inches='tight')

    plt.show()

df_result_lasso_ss = pd.read_csv(f'data/df_lasso_result_coef.csv', index_col=0)
df_result_lasso_pvalues = pd.read_csv(f'data/df_lasso_result_coef_pvalues.csv', index_col=0)
plot_local_coef(df_result_lasso_ss, df_result_lasso_pvalues, 'all')


## 5.2 Fig: local coef for selected crime model at four observation period

df_list0 = []
df_list1 = []
for m in range(4):
    df_result_lasso_ss = pd.read_csv(f'data/df_lasso_result_coef_{m}.csv', index_col=0)
    df_result_lasso_pp = pd.read_csv(f'data/df_lasso_result_coef_pvalues_{m}.csv', index_col=0)
    df_list0.append(df_result_lasso_ss.loc[['Theft from the person']])
    df_list1.append(df_result_lasso_pp.loc[['Theft from the person']])
df_plot_0 = pd.concat(df_list0)
df_plot_1 = pd.concat(df_list1)

op_names = ['Before lockdown (Feb 2020)', 'First national lockdown (Apr 2020)',
            'Lockdown easing (Aug 2020)', 'Second national lockdown (Nov 2020)']
static_names = ['Unemployment', 'Rent social house ', 'Education above level 4', 'Own cars above 3', 'Young residents 16-34',
                'Eating and drinking', 'Public transport', 'Tourism',  'Gambling', 'Venues, stage and screen',
                'Food, drink and multi-item retail']
dynamic_names = ['Early morning (WD)', 'Morning (WD)', 'Midday (WD)', 'Afternoon (WD)', 'Evening (WD)',
                 'Early morning (WE)', 'Morning (WE)', 'Midday (WE)', 'Afternoon (WE)', 'Evening (WE)']
df_plot_0.index = op_names
df_plot_1.index = op_names

sns.set_theme(style="ticks")
from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
for i, n in enumerate(op_names):

    barlist = ax[i].bar(x=df_plot_0.columns.to_list(), height=df_plot_0.loc[n].values, width=1, label=n)

    ax[i].hlines(y=0, color="k", xmin=-1, xmax=21, clip_on=False, lw=0.8)
    # ax[i].set_ylabel(n[0:3], fontsize = 18,  rotation = '90', loc = 'center')
    ax[i].set_ylim([np.min(df_plot_0.loc[n].to_list()) - 0.05, np.max(df_plot_0.loc[n].to_list()) + 0.05])
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[i].tick_params(axis='y', labelsize=13)

    ax[i].tick_params(axis='x',  # changes apply to the x-axis
                          which='both',  # both major and minor ticks are affected
                          bottom=False,  # ticks along the bottom edge are off
                          top=False,  # ticks along the top edge are off
                          labelbottom=False)  # labels along the bottom edge are off

    for j, m in enumerate(df_plot_1.loc[n].to_list()):
        if m < 0.05:
            barlist[j].set_color('#4C72B0')
        else:
            barlist[j].set_color('#6A6A6A')

    from matplotlib.lines import Line2D
    ax[i].legend(
                 bbox_to_anchor=(0.9, 1.2),
                 handles=[Line2D([0], [0], lw=0, color='orange', label=op_names[i])],
                 fontsize=15, frameon=False, loc='upper right')

ax[2].set_ylabel('Coefficient value', fontsize=15, loc='center')
ax[3].tick_params(
          axis='x',  # changes apply to the x-axis
          which='both',  # both major and minor ticks are affected
          bottom=True,  # ticks along the bottom edge are off
          top=False,  # ticks along the top edge are off
          labelbottom=True)  # labels along the bottom edge are off
ax[3].set_xticks(df_plot_0.columns.to_list())
ax[3].set_xticklabels(labels=dynamic_names + static_names, rotation='270', fontsize=13)
import matplotlib.patches as mpatches
colors = ['#6A6A6A', '#4C72B0']
fig.legend(bbox_to_anchor=(0.55, -.002),  # position
           handles=[mpatches.Patch(color=colors[m],
                                   label=['Not significant', 'p < 0.05'][m]) for m in range(2)],
           loc='lower center', fontsize=12, frameon=False, ncol=4)
sns.despine(bottom=True, left=False)
plt.tight_layout(pad=0.2, h_pad=0.1)
plt.savefig(f'data/cs_1_local_coef_four_op.jpg', dpi=300, bbox_inches='tight')
plt.show()


## 5.3 Fig: map the GWR local coefficient at selected x variables

def cla(x):
    y = 's'
    if -4 <= x < -2:
        y = -3
    elif -2 <= x < -0.5:
        y = -2
    elif -0.5 <= x < 0:
        y = -1
    elif 0 <= x < 0.5:
        y = 0
    elif 0.5 <= x < 2:
        y = 1
    elif 2 <= x < 4.5:
        y = 2
    elif 4.5 <= x < 6:
        y = 3
    elif 6 <= x < 9:
        y = 4
    else:
        print(x)

    return y

my_colors = ['#00006D', '#B3C8D8','#F2EEEE',
             '#E7A3A8', '#D93A46', '#E06E76', '#A1195A', '#781737']

def map_gwr_coef(months):
    op_names = ['Before lockdown (Feb 2020)', 'First national lockdown (Apr 2020)',
                'Lockdown easing (Aug 2020)', 'Second national lockdown (Nov 2020)']
    r_list = ['0.92', '0.36', '0.82', '0.74']
    bw_list = ['14', '80', '23', '34']
    x_list = ['Midday (WD)', 'Evening (WD)', 'Midday (WD)', 'Afternoon (WE)']
    gdf_london = gpd.read_file('data/gdf_london.shp')
    gdf_london.index = range(len(gdf_london))
    df_gwr_result_local = pd.read_csv('data/df_gwr_result_local.csv')
    gdf_london = pd.merge(gdf_london, df_gwr_result_local, left_on='geo_code', right_on='lsoa', how='left')
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.ravel()
    for k in range(4):
        gdf_london['cla'] = gdf_london[months[k]].apply(lambda x: cla(np.round(x, 2)))
        for i in gdf_london['cla'].unique():
            gdf_london.query('cla ==@i').plot(color=my_colors[i+2], edgecolor='black', lw=0.06, ax=ax[k])
        ax[k].set_title(op_names[k]+'\n'
                        +'Selected variable: '+x_list[k]+'\n'
                        +'BW='+bw_list[k]+', '+'$R^2$='+r_list[k], fontsize=11)
        ax[k].axis('off')

    import matplotlib.patches as mpatches
    fig.legend(bbox_to_anchor=(0.5, 0.001),  # position
               handles=[mpatches.Patch(color=my_colors[m],
                                       label=['[-4 , -2)',
                                               '[-2 , -0.5)',
                                              '[-0.5 , 0)',
                                              '[0 , 0.5)',
                                              '[0.5 , 2)',
                                              '[2 , 4.5)',
                                              '[4.5 , 6)',
                                              '[6 , 9)'][m]) for m in range(8)],
               loc='lower center', fontsize=10,
               frameon=False,
               ncol=4,
               title='Local coefficient value',
               title_fontsize=10,
               handlelength=1.5,
               handleheight=0.5)
    plt.tight_layout(pad=0.2, h_pad=0.01, w_pad=0.01)
    plt.savefig(f'data/cs_1_gwr_local.jpg', dpi=300, bbox_inches='tight')

    plt.show()

map_gwr_coef(['2020-02', '2020-04', '2020-08', '2020-11'])

