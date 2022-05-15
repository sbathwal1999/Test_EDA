import pandas as pd
import numpy as np
import scipy, os, platform, io
from jinja2 import Template
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')



class EDA:

    # Initialisation
    def __init__(self, DataFrame, TargetFeature, TargetType = None, CategoricalFeatures = None,  NumericalFeatures = None, OtherFeatures = None):
        self.df = DataFrame
        self.TargetFeature = TargetFeature
        self.CategoricalFeatures = CategoricalFeatures
        self.NumericalFeatures = NumericalFeatures
        self.OtherFeatures = OtherFeatures
        self.TargetType = TargetType
        self.PreProcessing()

    # Target Specific Analysis
    def TargetSpecificAnalysis(self):
        pth = pd.__path__
        this_dir, filename = os.path.split(pth[0])

        if self.TargetType == 'error':
            return

        elif self.TargetType == 'category':
            filename = 'Template\\Target_Categorical.html'

            if platform.system() == 'Windows':			
                this_dir, this_filename = os.path.split(__file__)
                Template_PATH = os.path.join(this_dir, filename)
            else:
                filename = 'Test_EDA/Template/Target_Categorical.html'
                Template_PATH = os.path.join(this_dir, filename)

        elif self.TargetType == 'numerical':
            filename = 'Template\\Target_Continuous.html'
            if platform.system() =='Linux':
                filename = 'Test_EDA/Template/Target_Continuous.html'
                Template_PATH = os.path.join(this_dir, filename)
            else:
                this_dir, this_filename = os.path.split(__file__)
                Template_PATH = os.path.join(this_dir, filename)

        with open(Template_PATH) as file:
            template = Template(file.read())
            
        if self.TargetFeature in self.NumericalFeatures:
            html = template.render(dataset_statistics = [self.dataset_statistics()],
                                      feature_details = [self.list_of_fields()],
                                      continuous = self.NumericalFeatures,
                                      categorical = self.CategoricalFeatures,
                                      other = self.OtherFeatures,
                                      pie_1 = self.piechart_datatype(),
                                      stats_cat = [self.stats_cat()],
                                      charts_out_cat = [self.charts_out_cat()],
                                      stats_con = [self.stats_con()],
                                      charts_out_con =  [self.charts_out_con()],
                                      target = self.TargetFeature,
                                      stats_target_con = [self.stats_target_con()],
                                      target_chart = self.target_chart_con(),
                                      corr_target = [self.corr_target()],
                                      scatter_chart =  self.scatter_chart(),
                                      chart_correlation = self.chart_correlation(),
                                      anova_target =  [self.anova_target_con()]
                            )
        # elif self.target in self.ContinuousFeatures:
        #     html = template.render(title = self.title)
            
        out_filename = os.path.join(this_dir, 'Template\\result.html')
        if platform.system() == 'Linux':
            out_filename = os.path.join(this_dir, 'Test_EDA/Template/result.html')
        with io.open(out_filename, mode='w', encoding='utf-8') as f:
            f.write(html)
        
        return display(HTML(html))

    # Preprocessing
    def PreProcessing(self):

        # Categorical 
        if self.CategoricalFeatures == None:

            # Treating Object and Category data type as Categorical Feature
            self.CategoricalFeatures = [feature for feature in self.df.columns if self.df[feature].dtype=='O' or self.df[feature].dtype=='category']
        else:
            for feature in self.CategoricalFeatures:
                self.df[feature] = self.df[feature].astype(str)

        # Numerical
        if self.NumericalFeatures == None:

            # Treating Integer and Float data type as Numerical Feature
            self.NumericalFeatures = [feature for feature in self.df.columns if self.df[feature].dtype=='float' or self.df[feature].dtype=='int']
        else:
            for feature in self.NumericalFeatures:
                if self.df[feature].dtype != 'float' or self.df[feature].dtype != 'int':
                    print(f'{feature} is passed as numerical feature but is not an integer or a float type. Ignoring it!!!')
                    self.NumericalFeatures.remove(feature)
        
        # Target
        if self.TargetFeature in self.CategoricalFeatures:
            self.CategoricalFeatures.remove(self.TargetFeature)
            self.TargetType = 'category'

        elif self.TargetFeature in self.NumericalFeatures:
            self.NumericalFeatures.remove(self.TargetFeature)
            self.TargetType = 'numerical'
        
        if self.TargetType == None:
            if self.df[self.TargetFeature].dtype == 'O' or self.df[self.TargetFeature].dtype == 'category':
                self.TargetType = 'category'
            elif self.df[self.TargetFeature].dtype == 'int' or self.df[self.TargetFeature].dtype == 'float':
                self.TargetType = 'numerical'
            else:
                print('Only Numerical and String type allowed as Target Type. Quitting!!!')
                self.TargetType = 'error'

    # Dataset Description
    def dataset_statistics(self):
        data_stats = {}
        data_stats['Number of Variables'] = self.df.shape[1]
        data_stats['Number of Rows'] = self.df.shape[0]
        data_stats['Missing Value Count'] = sum(self.df.isnull().sum())
        data_stats['Missing (%)'] = str(round(sum(self.df.isnull().sum())/(self.df.shape[0]*self.df.shape[1])*100,1))+"%"
        data_stats['Duplicate Count'] = self.df.duplicated().sum()
        data_stats['Duplicate (%)'] = str(round(self.df.duplicated().sum()/(self.df.shape[0]*self.df.shape[1])*100,1))+"%"
        var = {}
        var['Categorical'] = len([feature for feature in self.df.columns if self.df[feature].dtype=='O'])
        var['Continuous'] = len([feature for feature in self.df.columns if self.df[feature].dtype=='float'])
        var['Other'] = len([feature for feature in self.df.columns if self.df[feature].dtype!='O' and self.df[feature].dtype!='float'])
        data_stats['Variable Types'] = var
        return data_stats
             
    # List of Fields
    def list_of_fields(self):
        lof = {}
        for feature in self.df.columns:
            if self.df[feature].nunique ==self.df.shape[0]:
                detail ='Unique'
            elif self.df[feature].isnull().mean()>0.05:
                detail ='Missing'
            elif feature in self.NumericalFeatures:
                if scipy.stats.shapiro(self.df[feature]).pvalue > 0.05:
                    detail = 'Normal'
                elif np.nanmean(self.df[feature])>np.nanmedian(self.df[feature]):
                    detail = 'Right Skewed'
                else:
                    detail = 'Left Skewed'
            elif feature in self.CategoricalFeatures:
                detail = str(self.df[feature].nunique()) + ' distinct values'
            else:
                detail = 'Other'
            lof[feature] = detail
        return lof

    # Categorical Description
    def stats_cat(self):
        stats = {}
        for feature in self.CategoricalFeatures:
            if self.df[feature].nunique()>=0.7*self.df[feature].shape[0]:
                continue
            if  self.df[feature].value_counts().values[:5].sum()<=0.1 * self.df[feature].value_counts().values[5:].sum():
                continue

            stats_cat = {}
            stats_cat['Distinct Value Count'] = self.df[feature].nunique()
            stats_cat['Missing'] = self.df[feature].isnull().sum()
            stats_cat['Missing (%)'] = str(round(self.df[feature].isnull().mean()*100,1))+'%'
            stats_cat['Mode'] = self.df[feature].mode()[0]
            if self.df[feature].nunique()>=3:
                stats_cat['Top 3 Labels based on Count'] = [f for f in self.df[feature].value_counts().index[:3]]
            elif self.df[feature].nunique()==2:
                stats_cat['Top 2 Labels based on Count'] = [f for f in self.df[feature].value_counts().index[:2]]
            else:
                pass
            stats[feature] = stats_cat
            
        return stats

    # Continuous Description
    def stats_con(self):
        stats = {}
        for feature in self.NumericalFeatures:
            stats_1 = {}
            stats_1['Minimum'] = round(self.df[feature].describe()['min'],4)
            stats_1['Q1'] = round(self.df[feature].describe()['25%'],4)
            stats_1['Median'] = round(self.df[feature].describe()['50%'],4)
            stats_1['Q3'] = round(self.df[feature].describe()['75%'],4)
            stats_1['Maximum'] = round(self.df[feature].describe()['max'],4)
            stats_1['IQR'] = round(self.df[feature].describe()['75%'] - self.df[feature].describe()['25%'],4)
            z= (self.df[feature] - self.df[feature].mean())/self.df[feature].std()
            stats_1['Outliers Count'] = (z>=3).sum() + (z<=-3).sum()

            stats_2 = {}
            stats_2['Count'] = round(self.df[feature].notnull().sum(),4)
            stats_2['Mean'] = round(self.df[feature].mean(),4)
            stats_2['Standard Deviation'] = round(self.df[feature].std(),4)
            stats_2['Sum'] = round(self.df[feature].sum(),4)
            stats_2['Skewness'] = round(self.df[feature].skew(),4)
            stats_2['Kurtosis'] = round(self.df[feature].kurtosis(),4)
            stats_2['Missing (%)'] = str(round(self.df[feature].isnull().mean()*100,2)) + '%'

            stats_3 = {}
            stats_3['Descriptive'] = stats_2
            stats_3['Quantile'] = stats_1
            stats[feature] = stats_3

        return stats

    # ANOVA
    def anova_target_con(self):
        anova_dict = {}
        for feature in self.CategoricalFeatures:
            df1 = self.df[[feature, self.TargetFeature]].dropna()
            try:
                f, p = scipy.stats.f_oneway(*[list(df1[df1[feature]==name][self.TargetFeature]) for name in set(df1[feature])])
                anova_dict[feature] = np.round(p,4)
            except:
                pass

        return anova_dict

    # Continuous Target Description
    def stats_target_con(self):

        feature = self.TargetFeature
        stats_1 = {}
        stats_1['Minimum'] = round(self.df[feature].describe()['min'],4)
        stats_1['Q1'] = round(self.df[feature].describe()['25%'],4)
        stats_1['Median'] = round(self.df[feature].describe()['50%'],4)
        stats_1['Q3'] = round(self.df[feature].describe()['75%'],4)
        stats_1['Maximum'] = round(self.df[feature].describe()['max'],4)
        stats_1['IQR'] = round(self.df[feature].describe()['75%'] - self.df[feature].describe()['25%'],4)
        z= (self.df[feature] - self.df[feature].mean())/self.df[feature].std()
        stats_1['Outliers Count'] = (z>=3).sum() + (z<=-3).sum()
        
        stats_2 = {}
        stats_2['Count'] = round(self.df[feature].notnull().sum(),4)
        stats_2['Mean'] = round(self.df[feature].mean(),4)
        stats_2['Standard Deviation'] = round(self.df[feature].std(),4)
        stats_2['Sum'] = round(self.df[feature].sum(),4)
        stats_2['Skewness'] = round(self.df[feature].skew(),4)
        stats_2['Kurtosis'] = round(self.df[feature].kurtosis(),4)
        stats_2['Missing (%)'] = str(round(self.df[feature].isnull().mean()*100,2)) + '%'

        stats_3 = {}
        stats_3['Descriptive'] = stats_2
        stats_3['Quantile'] = stats_1
        
        return stats_3

    # Continuous Target Correlation
    def corr_target(self):
        corr_dict = {}
        for feature in self.NumericalFeatures:
            corr_dict[feature] = round(self.df[feature].corr(self.df[self.TargetFeature]),4)
        return corr_dict
        
    # Charts Preprocessing
    def charts_preprocess_univariate(data, feature):
        data = data[data[feature].notnull()]
        data = data[feature][np.isfinite(data[feature])]
        return data

    def charts_preprocess_bivariate(data, feature1, feature2):
        data = data[[feature1,feature2]].dropna()
        data = data[feature1][np.isfinite(data[feature1])]
        data = data[feature2][np.isfinite(data[feature2])]
        return data
    
    # PieChart DataType BreakDown
    def piechart_datatype(self):
        return [['Type','Count'],['Categorical',len(self.CategoricalFeatures)],['Continuous',len(self.NumericalFeatures)],['Other',len(self.OtherFeatures)]]

    # Charts Categorical Features
    def charts_out_cat(self):
        stats_1 = {}
        for feature in self.CategoricalFeatures:
            data = self.charts_preprocess_univariate(self.df.copy(),feature)
            flag = 0
            l = []
            l.append(['Label','Count'])
            
            for i in range(data[feature].nunique()):
                if i<=4:
                    l.append([data[feature].value_counts().index[i], data[feature].value_counts().values[i]])
                else:
                    flag=1
                    break
                    
            if flag==1:
                l.append(["Others", data[feature].value_counts().values[5:].sum()])
            stats_1[feature] = l
            
        return stats_1

    # Charts Continuous Features
    def charts_out_con(self):
        stats_1 = {}
        for feature in self.NumericalFeatures:
            data = self.charts_preprocess_univariate(self.df.copy(),feature)
            stats_1[feature] = list(data[feature])
        return stats_1

    # Correlation Plot
    def chart_correlation(self):
        cols_to_consider = []

        for feature in self.NumericalFeatures:
            if np.var(self.df[feature])!=0:
                cols_to_consider.append(feature)

        df1 = self.df[cols_to_consider]
        CorrDf = df1.corr()

        corr_dict = dict()
        corr_list = list()

        for i in list(CorrDf.columns):
            corr_list.append(list(CorrDf[i].values))

        corr_dict['values'] = corr_list
        corr_dict['rows'] = list(CorrDf.columns)
        corr_dict['cols'] = list(CorrDf.columns)
        
        return corr_dict

    # Charts Target Continuous 
    def target_chart_con(self):
        data = self.charts_preprocess_univariate(self.df.copy(),self.TargetFeature)
        return list(data[self.TargetFeature])

    # Scatter Plot - Target Continouos 
    def scatter_chart(self):
        scatter_dict = {}
        for feature in self.NumericalFeatures:
            data = self.charts_preprocess_bivariate(self.df.copy(), feature, self.TargetFeature)
            scatter_list = []
            scatter_list.append([feature, self.TargetFeature])
            for i in range(data.shape[0]):
                scatter_list.append([data[feature].iloc[i], data[self.TargetFeature].iloc[i]])
            scatter_dict[feature] = scatter_list
        return scatter_dict




