import string
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, boxplot, xticks
from scipy.linalg import svd
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import source.Data as Da
pio.renderers.default = 'svg'
pio.templates


def plot_attribute_against(data: Da.Data, attribute_x: int, attribute_y: int, plot_title: string):
    f = figure()
    title(plot_title)
    for c in range(data.C):
        class_mask = data.y == c
        plot(data.y2[class_mask,
             attribute_x],
             data.y2[class_mask,
             attribute_y],
             'o',
             alpha=1,
             marker='.',
             markersize=5)
        plot(data.x_tilda[class_mask, attribute_x], data.x_tilda[class_mask, attribute_y], 'o', alpha=1)

    xlabel(data.attributes[attribute_x])
    ylabel(data.attributes[attribute_y])

    show()


def plot_visualized_data(data: Da.Data, plot_title: string):
    
    u, s, vh = svd(data.x_tilda, full_matrices=False)

    rho = (s * s) / (s * s).sum()

    threshold = 0.90

    plt.figure()
    plt.title(plot_title)
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title(plot_title)
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


def plot_visualized_pca(data: Da.Data, first_pc: int, second_pc: int, plot_title: string):
    u, s, vh = svd(data.x_tilda, full_matrices=False)
    v = vh.T
    z = data.x_tilda @ v

    plt.figure()
    plt.title(plot_title)
    for c in range(data.C):
        class_mask = data.y == c
        plot(z[class_mask, first_pc], z[class_mask, second_pc], 'o', alpha=1, marker='.', markersize=5)
    xlabel('PC{0}'.format(first_pc + 1))
    ylabel('PC{0}'.format(second_pc + 1))

    show()


def plot_visualized_coefficients(data: Da.Data, pc_count: int, plot_title: string, legend: bool = True):
    pcs = [i for i in range(0, pc_count)]
    legend_strs = ['PC' + str(e + 1) for e in pcs]

    u, s, vh = svd(data.x_tilda, full_matrices=False)
    v = vh.T

    bw = .2
    r = np.arange(0, data.M)
    print(r)
    print(data.M)
    for i in pcs:
        plt.bar(r + i * bw, v[:, i], width=bw)

    plt.xticks(r + bw, data.attributes)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')

    if legend:
        plt.legend(legend_strs)

    plt.grid()
    plt.title(plot_title)
    plt.show()


def plot_boxplots(data: Da.Data, plot_title: string):
    boxplot(data.y2)
    xticks(range(1, data.M), data.attributes)
    ylabel('y')
    title(plot_title)
    show()


def plot_boxplot(data: Da.Data, attr: int, plot_title: string):
    boxplot(data.x[:, attr])
    xticks(range(1, 2), [data.attributes[attr]])
    ylabel(data.attribute_units[attr])
    title(plot_title)
    show()


def plot_box_plot(data: Da.Data):
    
    # not normalized plot
    # fig = px.box(data.df, color='Fire size', color_discrete_sequence=["red", "blue"]) #, points='all')
    # fig.update_layout(font=dict(size=14))
    # fig.update_yaxes(title='', visible=True, showticklabels=True)
    # fig.update_xaxes(title='', visible=True, showticklabels=True)
    # fig.show()
    
    #normalized plot
    fig = px.box(data.df_tilda, color='Burned area', color_discrete_sequence=["red", "blue"]) #, points='all')
    fig.update_layout(font=dict(size=14))
    fig.update_yaxes(title='', visible=True, showticklabels=True)
    fig.update_xaxes(title='', visible=True, showticklabels=True)
    fig.update_xaxes(tickangle=30)
    fig.show()

    
def plot_correlation_matrix(data: Da.Data):    
    corr = data.df.corr()
    
    figure, ax = plt.subplots(figsize=(14, 10))
    plt.xticks(fontsize=17,weight='bold')
    plt.yticks(fontsize=17 ,weight='bold', rotation=45)
    sns.set(font_scale = 1)
    # cmap=sns.diverging_palette(220, 10, as_cmap=True,center='dark')
    cmap= "YlGnBu_r"
    g = sns.heatmap(corr, annot=True,mask=np.zeros_like(corr, dtype=np.bool), cmap=cmap ,
                square=True,ax=ax);
    # g.set_xticklabels(labels=('X','Y','month','day','FFMC','DMC','DC','ISI',
    #                           'temp','RH','wind','rain','area'),rotation=45);
    g.set_xticklabels(labels=('FFMC','DMC','DC','ISI',
                              'temp','RH','wind','rain','area'),rotation=45);
    g.set_yticklabels(labels=data.df_attributes,rotation=30);
    # g.set_yticklabels(labels=data.df_attributes, rotation=30, horizontalalignment='left')

    
def plot_cum_variance(data: Da.Data):
    
    # not normailized data
    data.df_data = data.df_data - data.df_data.mean(axis=0)
    U,S,VT = np.linalg.svd(data.df_data, full_matrices=False)
    rho = (S*S) / (S*S).sum() 
    threshold = 0.9

    fig1 = plt.figure(figsize=(24,5))
    ax = fig1.add_subplot(122)
    ax.plot(range(1,len(rho)+1),rho,'x-')
    ax.plot([1,len(rho)],[threshold, threshold],'k--')
    ax.plot(range(1,len(rho)+1),np.cumsum(rho),'o-',c='red')
    ax.legend(['Individual','Threshold','Cumulative'],fontsize=17)
    plt.xlabel("Principal Component",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Variance Explained",fontsize=20)
    plt.show()
    print('Cumulative not normalized var: ', np.cumsum(rho))
          
    # normailized data
    
    U,S,VT = np.linalg.svd(data.df_data_tilda, full_matrices=False)
    rho = (S*S) / (S*S).sum() 
    threshold = 0.9

    fig1 = plt.figure(figsize=(24,5))
    ax = fig1.add_subplot(122)
    ax.plot(range(1,len(rho)+1),rho,'x-')
    ax.plot([1,len(rho)],[threshold, threshold],'k--')
    ax.plot(range(1,len(rho)+1),np.cumsum(rho),'o-',c='red')
    ax.legend(['Individual','Threshold','Cumulative'],fontsize=17)
    plt.xlabel("Principal Component",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Variance Explained",fontsize=20)
    plt.show()
    
    print('Cumulative normalized var: ', np.cumsum(rho))


def plot_pca(data: Da.Data):
    
    # not normailized data
    data.df_data = data.df_data - data.df_data.mean(axis=0)
    U,S,VT = np.linalg.svd(data.df_data, full_matrices=False)
    V = VT.T 
    Z = data.df_data @ V

    color_dict = dict({'Small':'red', 'Big':'blue'})
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig = plt.figure(figsize=(18,10))
    sns.scatterplot(x = Z[0], y = Z[1], hue = data.df['Burned area'], s = 50 ,palette=color_dict)
    plt.xlabel("PCA1",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("PCA2",fontsize=20)
    plt.legend(title="Burnead area",fontsize=12, loc ="lower left")
    plt.show()
    
    # normailized data
    
    U,S,VT = np.linalg.svd(data.df_data_tilda, full_matrices=False)
    V = VT.T 
    Z = data.df_data_tilda @ V

    color_dict = dict({'Small':'red', 'Big':'blue'})
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig = plt.figure(figsize=(18,10))
    sns.scatterplot(x = Z[0], y = Z[1], hue = data.df['Burned area'], s = 50 ,palette=color_dict)
    plt.xlabel("PCA1",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("PCA2",fontsize=20)
    plt.legend(title="Burnead area",fontsize=12, loc ="lower left")
    plt.show()


def plot_pca_coeff(data: Da.Data):

    # not normailized data
    data.df_data = data.df_data - data.df_data.mean(axis=0)
    U,S,VT = np.linalg.svd(data.df_data, full_matrices=False)
    test_df = pd.DataFrame(VT, 
                           index = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7',
                                        'PC8','PC9','PC10','PC11','PC12'],
                           columns = data.df_attributes[:-1])
    
    test_df = test_df.transpose()
    coefficents = test_df[['PC1','PC2','PC3','PC4']]
    
    ax = coefficents.plot(kind='bar', figsize=(18, 10), legend=True, fontsize=25)
    ax.set_ylabel("Principal Component Coefficients", fontsize=25)
    plt.xticks(rotation=-30)
    plt.legend(fontsize = 18)
    plt.show()
    
    # normailized data
    
    U,S,VT = np.linalg.svd(data.df_data_tilda, full_matrices=False)
    test_df = pd.DataFrame(VT, 
                           index = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7',
                                        'PC8','PC9','PC10','PC11','PC12'],
                           columns = data.df_attributes[:-1])
    
    test_df = test_df.transpose()
    coefficents = test_df[['PC1','PC2','PC3','PC4']]
    
    ax = coefficents.plot(kind='bar', figsize=(18, 10), legend=True, fontsize=25)
    ax.set_ylabel("Principal Component Coefficients", fontsize=25)
    plt.xticks(rotation=-30)
    plt.legend(fontsize = 18)
    plt.show()
    
    
def plot_distribution(data:Da.Data):

    # using original dataset without the outpute variable

    fig = plt.figure(figsize=(37, 25))
    
    bins_grid = [1,2,3,4,5,6,7,8,9]
    bins_month = [1,2,3,4,5,6,7,8,9,10,11,12]
    bins_day = ['mon','tue','wed','thu','fri','sat','sun']
    
    ax1 = plt.subplot(4, 4, 1)
    ax1.hist(data.df['X'],color='royalblue',bins=9)
    ax1.set_title('X (index)',fontsize=30, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=25)
    ax1.tick_params(labelsize=25)
    
    ax2 = plt.subplot(4, 4, 2)
    ax2.hist(data.df['Y'],color='royalblue',bins=9)
    ax2.set_title('Y (index)',fontsize=30, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=25)
    ax2.tick_params(labelsize=25)
    
    ax3 = plt.subplot(4, 4, 3)
    ax3.hist(data.df['month'],color='royalblue',bins=12)
    ax3.set_title('Month (index)',fontsize=30, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=25)
    ax3.tick_params(labelsize=25)
    
    ax4 = plt.subplot(4, 4, 4)
    ax4.hist(data.df['day'],color='royalblue',bins=7)
    ax4.set_title('Day (index)',fontsize=30, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=25)
    ax4.tick_params(labelsize=25)
    
    ax5 = plt.subplot(4, 4, 5)
    ax5.hist(data.df['FFMC'],color='royalblue',bins=15)
    ax5.set_title('FFMC (code)',fontsize=30, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=25)
    ax5.tick_params(labelsize=25)
    
    ax6 = plt.subplot(4, 4, 6)
    ax6.hist(data.df['DMC'],color='royalblue',bins=15)
    ax6.set_title('DMC (code)',fontsize=30, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=25)
    ax6.tick_params(labelsize=25)
    
    ax7 = plt.subplot(4, 4, 7)
    ax7.hist(data.df['DC'],color='royalblue',bins=15)
    ax7.set_title('DC (code)',fontsize=30, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=25)
    ax7.tick_params(labelsize=25)
    
    ax8 = plt.subplot(4, 4, 8)
    ax8.hist(data.df['ISI'],color='royalblue',bins=15)
    ax8.set_title('ISI (code)',fontsize=30, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=25)
    ax8.tick_params(labelsize=25)
    
    ax9 = plt.subplot(4, 4, 9)
    ax9.hist(data.df['temp'],color='royalblue',bins=15)
    ax9.set_title('Temperature (°C)',fontsize=30, fontweight='bold')
    ax9.set_ylabel('Frequency', fontsize=25)
    ax9.tick_params(labelsize=25)

    ax10 = plt.subplot(4, 4, 10)
    ax10.hist(data.df['RH'],color='royalblue',bins=15)
    ax10.set_title('Relative Humidity (%)',fontsize=30, fontweight='bold')
    ax10.set_ylabel('Frequency', fontsize=25)
    ax10.tick_params(labelsize=25)
    
    ax11 = plt.subplot(4, 4, 11)
    ax11.hist(data.df['wind'],color='royalblue',bins=15)
    ax11.set_title('Wind (km/h)',fontsize=30, fontweight='bold')
    ax11.set_ylabel('Frequency', fontsize=25)
    ax11.tick_params(labelsize=25)
    
    ax12 = plt.subplot(4, 4, 12)
    ax12.hist(data.df['rain'],color='royalblue',bins=15)
    ax12.set_title('Rain (mm/m^2)',fontsize=30, fontweight='bold')
    ax12.set_ylabel('Frequency', fontsize=25)
    ax12.tick_params(labelsize=25)
