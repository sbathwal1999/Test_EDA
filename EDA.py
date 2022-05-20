import pandas as pd
import numpy as np
import scipy, os, platform, io
from jinja2 import Template
from IPython.display import display, HTML
import warnings, webbrowser
from scipy.stats import shapiro
from statsmodels.stats.multicomp import MultiComparison
warnings.filterwarnings('ignore')

class EDA:

    # Initialisation
    def __init__(self, DataFrame, TargetFeature, TargetType = None, CategoricalFeatures = None,  NumericalFeatures = None, OtherFeatures = None):
        self.df = self.validate_df(DataFrame)
        self.TargetFeature = self.validate_TargetFeature(TargetFeature)
        self.CategoricalFeatures = self.validate_CategoricalFeatures(CategoricalFeatures)
        self.NumericalFeatures = self.validate_NumericalFeatures(NumericalFeatures)
        self.OtherFeatures = self.validate_OtherFeatures(OtherFeatures)
        self.TargetType = TargetType
        self.PreProcessing()

    # Validation
    def validate_df(self, DataFrame):
        if not isinstance(DataFrame, pd.DataFrame):
            raise ValueError('Only Pandas DataFrame Supported')
        return DataFrame

    def validate_TargetFeature(self, TargetFeature):
        if not isinstance(TargetFeature, str):
            raise ValueError('Target Feature should be a string')
        if TargetFeature not in self.df.columns:
            raise ValueError('Target Feature not an attribute of DataFrame')
        return TargetFeature
    
    def validate_CategoricalFeatures(self, CategoricalFeatures):
        if CategoricalFeatures == None:
            return CategoricalFeatures
        if not isinstance(CategoricalFeatures, list):
            raise ValueError('Categorical Features should be a list')
        for feature in CategoricalFeatures:
            if feature not in self.df.columns:
                raise ValueError(f'{feature} not an attribute of DataFrame')
        return CategoricalFeatures

    def validate_NumericalFeatures(self, NumericalFeatures):
        if NumericalFeatures == None:
            return NumericalFeatures
        if not isinstance(NumericalFeatures, list):
            raise ValueError('Numerical Features should be a list')
        for feature in NumericalFeatures:
            if feature not in self.df.columns:
                raise ValueError(f'{feature} not an attribute of DataFrame')
        return NumericalFeatures

    def validate_OtherFeatures(self, OtherFeatures):
        if OtherFeatures == None:
            return OtherFeatures
        if not isinstance(OtherFeatures, list):
            if isinstance(OtherFeatures, str):
                OtherFeatures = [OtherFeatures]
            else:
                raise ValueError('Data Type for Other Features should be a string or list')
        for feature in OtherFeatures:
            if feature not in self.df.columns:
                raise ValueError(f'{feature} not an attribute of DataFrame')
        return OtherFeatures

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
        
        try:
            import google.colab
            colab = True
        except:
            colab = False

        def isnotebook():
            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    return True   # Jupyter notebook or qtconsole
                elif shell == 'TerminalInteractiveShell':
                    return False  # Terminal running IPython
                else:
                    return False  # Other type (?)
            except NameError:
                return False 

        if self.TargetType == 'numerical':
            if colab:
                Template_PATH = '/content/Test_EDA/Template/Target_Continuous.html'
            elif isnotebook():
                Template_PATH = './Test_EDA/Template/Target_Continuous.html'
            with open(Template_PATH) as file:
                template = Template(file.read())
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
                                      target_chart_con = self.target_chart_con(),
                                      corr_target = [self.corr_target()],
                                      scatter_chart =  self.scatter_chart_con(),
                                      chart_correlation = self.chart_correlation(),
                                      anova_target =  [self.anova_target_con()],
                                      sim_dist = self.similar_distribution_con()
                            )
        elif self.TargetType == 'category':
            if colab:
                Template_PATH = '/content/Test_EDA/Template/Target_Categorical.html'
            elif isnotebook():
                Template_PATH = './Test_EDA/Template/Target_Categorical.html'
            with open(Template_PATH) as file:
                template = Template(file.read())
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
                                      stats_target_cat = [self.stats_target_cat()],
                                      target_chart_cat = self.target_chart_cat(),
                                      chart_correlation = self.chart_correlation(),
                                      chi_target = [self.ChiSquareTest_target_cat()],
                                      target_grouped = self.target_grouped_chart_cat()
                                      )
            
        out_filename = os.path.join(this_dir, 'Template\\result.html')
        if platform.system() == 'Linux':
            out_filename = os.path.join(this_dir, 'Test_EDA/Template/result.html')
        
        if colab:
            out_filename = '/content/Test_EDA/Template/result.html'
            with io.open(out_filename, mode='w', encoding='utf-8') as f:
                f.write(html)
    
            return display(HTML(html))
        elif isnotebook():
            with io.open(out_filename, mode='w', encoding='utf-8') as f:
                f.write(html)
                url = 'file://'+out_filename
                webbrowser.open(url, new=2)
                return out_filename

    # Preprocessing
    def PreProcessing(self):

        # Categorical 
        if self.CategoricalFeatures == None:

            # Treating Object and Category data type as Categorical Feature
            self.CategoricalFeatures = [feature for feature in self.df.columns if (self.df[feature].dtype=='O' or self.df[feature].dtype=='category')]
            for feature in self.CategoricalFeatures:
                if self.OtherFeatures != None:
                    if feature in self.OtherFeatures:
                        self.CategoricalFeatures.remove(feature)
        else:
            for feature in self.CategoricalFeatures:
                if feature not in self.OtherFeatures:
                    self.df[feature] = self.df[feature].astype(str)
                else:
                    self.CategoricalFeatures.remove(feature)

        # Numerical
        if self.NumericalFeatures == None:

            # Treating Integer and Float data type as Numerical Feature
            self.NumericalFeatures = [feature for feature in self.df.columns if (self.df[feature].dtype=='float' or self.df[feature].dtype=='int64' or self.df[feature].dtype=='int32')]
            for feature in self.NumericalFeatures:
                if self.OtherFeatures != None:
                    if feature in self.OtherFeatures:
                        self.NumericalFeatures.remove(feature)
        else:
            for feature in self.NumericalFeatures:
                if self.df[feature].dtype != 'float' and self.df[feature].dtype != 'int64':
                    print(f'{feature} is passed as numerical feature but is not an integer or a float type. Ignoring it!!!')
                    self.NumericalFeatures.remove(feature)
                if feature in self.OtherFeatures:
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
            elif self.df[self.TargetFeature].dtype == 'int64' or self.df[self.TargetFeature].dtype == 'float' or self.df[self.TargetFeature].dtype == 'int32':
                self.TargetType = 'numerical'
            else:
                print('Only Numerical and String type allowed as Target Type. Quitting!!!')
                self.TargetType = 'error'

        # Others
        for feature in self.df.columns:
            if feature not in self.CategoricalFeatures and feature not in self.NumericalFeatures and feature != self.TargetFeature:
                if self.OtherFeatures == None:
                    self.OtherFeatures = []
                if feature not in self.OtherFeatures:
                    self.OtherFeatures.append(feature)

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
        var['Categorical'] = len(self.CategoricalFeatures)
        var['Continuous'] = len(self.NumericalFeatures)
        if self.OtherFeatures != None:
            var['Other'] = len(self.OtherFeatures)
        else:
            var['Other'] = 0
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
              try:
                if shapiro(self.df[feature]).pvalue > 0.05:
                    detail = 'Normal'
                elif np.nanmean(self.df[feature])>np.nanmedian(self.df[feature]):
                    detail = 'Right Skewed'
                else:
                    detail = 'Left Skewed'
              except:
                detail = 'Numerical'
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
    
    # ANOVA Target Continouos
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

    # Categorical Target Description
    def stats_target_cat(self):
        stats = {}
        stats_cat = {}
        stats_cat['Distinct Value Count'] = self.df[self.TargetFeature].nunique()
        stats_cat['Missing'] = self.df[self.TargetFeature].isnull().sum()
        stats_cat['Missing (%)'] = str(round(self.df[self.TargetFeature].isnull().mean()*100,1))+'%'

        stats['Descriptive'] = stats_cat

        x = self.df[self.TargetFeature].value_counts()
        target_lof = {}
        for i in range(len(x)):
            target_lof[x.index[i]]= x.values[i]    
        stats['target_lof'] = target_lof

        return stats

    # Chi Square Target Categorical
    def ChiSquareTest_target_cat(self):
        DependentVar = self.TargetFeature
        chi_dict = {}
        if len(self.CategoricalFeatures) > 1:
            for IndependentVar in self.CategoricalFeatures:
                if IndependentVar != DependentVar:
                    groupsizes = self.df.groupby([DependentVar, IndependentVar]).size()
                    ctsum = groupsizes.unstack(DependentVar)
                    ChiSq,PValue = list(scipy.stats.chi2_contingency(ctsum.fillna(0)))[0:2]
                    chi_dict[IndependentVar] = round(PValue, 4)
        return chi_dict
        
     # Charts Preprocessing
    def charts_preprocess_univariate(self, feature):
        data = self.df.copy()
        data = data[data[feature].notnull()]
        if feature in self.NumericalFeatures or (feature in self.TargetFeature and self.TargetType == 'numerical'):
            data = data[np.isfinite(data[feature])]
        return data

    def charts_preprocess_bivariate(self, feature1, feature2):
        data = self.df.copy()
        data = data[[feature1,feature2]].dropna()
        if feature1 in self.NumericalFeatures or (feature1 in self.TargetFeature and self.TargetType == 'numerical'):
            data = data[np.isfinite(data[feature1])]
        if feature2 in self.NumericalFeatures or (feature2 in self.TargetFeature and self.TargetType == 'numerical'):
            data = data[np.isfinite(data[feature2])]
        return data
    
    # PieChart DataType BreakDown
    def piechart_datatype(self):
        if self.OtherFeatures != None:
            return [['Type','Count'],['Categorical',len(self.CategoricalFeatures)],['Continuous',len(self.NumericalFeatures)],['Other',len(self.OtherFeatures)]]
        else:
            return [['Type','Count'],['Categorical',len(self.CategoricalFeatures)],['Continuous',len(self.NumericalFeatures)]]

    # Charts Categorical Features
    def charts_out_cat(self):
        stats_1 = {}
        for feature in self.CategoricalFeatures:
            data = self.charts_preprocess_univariate(feature)
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
            data = self.charts_preprocess_univariate(feature)
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
        data = self.charts_preprocess_univariate(self.TargetFeature)
        return list(data[self.TargetFeature])

    # Scatter Plot - Target Continouos 
    def scatter_chart_con(self):
        scatter_dict = {}
        for feature in self.NumericalFeatures:
            data = self.charts_preprocess_bivariate( feature, self.TargetFeature)
            scatter_list = []
            scatter_list.append([feature, self.TargetFeature])
            for i in range(data.shape[0]):
                scatter_list.append([data[feature].iloc[i], data[self.TargetFeature].iloc[i]])
            scatter_dict[feature] = scatter_list
        return scatter_dict

    # Charts Target Categorical
    def target_chart_cat(self):
        target_lof = []
        target_lof.append(['Label','Count'])
        data = self.charts_preprocess_univariate(self.TargetFeature) 
        x = data[self.TargetFeature].value_counts()
        for i in range(len(x)):
            target_lof.append([x.index[i], x.values[i]])       
        return target_lof

    # Grouped Chart Categorical Target vs Categorical Feature
    def target_grouped_chart_cat(self):
        chart = {}
        for feature in self.CategoricalFeatures:
            data = self.charts_preprocess_bivariate(feature, self.TargetFeature)
            if data[feature].nunique() <=5:
                p = pd.crosstab(data[feature],data[self.TargetFeature], margins = False)
                dict1 ={}
                for label in data[self.TargetFeature].value_counts().index:
                    dict2 = {}
                    dict2['y'] = list(p[label].values)
                    dict2['x'] = list(p[label].index)
                    dict1[label] = dict2
                chart[feature] = dict1
        return chart

    # Similar Distribution Tuckey DataFrame
    def GroupTukeyHSD(self, continuous, categorical):
        try:
            mc = MultiComparison(continuous, categorical)
            result = mc.tukeyhsd()
            reject = result.reject
            meandiffs = result.meandiffs
            UniqueGroup = mc.groupsunique
            group1 = [UniqueGroup[index] for index in mc.pairindices[0]]
            group2 = [UniqueGroup[index] for index in mc.pairindices[1]]
            reject = result.reject
            meandiffs = [round(float(meandiff),3) for meandiff in result.meandiffs]
            columns = ['Group 1', "Group 2", "Mean Difference", "Reject"]
            TukeyResult = pd.DataFrame(np.column_stack((group1, group2, meandiffs, reject)), columns=columns)
            TukeyResult_false = TukeyResult[TukeyResult['Reject']=='False']
            overall_distribution_list = []
            same_distribution_list = []
            if len(TukeyResult_false) > 0:
                for group1 in TukeyResult_false['Group 1'].unique():
                    if group1 not in overall_distribution_list:
                        temp_list=[]
                        temp_result = TukeyResult_false[TukeyResult_false['Group 1']== group1]
                        overall_distribution_list.append(group1)
                        for entry in list(temp_result['Group 2'].unique()):
                            if entry not in overall_distribution_list:
                                overall_distribution_list.append(entry)
                                temp_list.append(entry)
                        temp_list.append(group1)
                        same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=temp_list, length=len(temp_list)))
                if len(set(categorical.unique())-set(overall_distribution_list)) >0:
                    missing_categories = list(set(categorical.unique())-set(overall_distribution_list))
                    for group1 in missing_categories:
                        same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=[group1], length=1))

            else:
                for group1 in categorical.unique():
                    same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=[group1], length=1))

            g1 = pd.DataFrame(same_distribution_list).sort_values('length', ascending=False)
        except:
            g1 = pd.DataFrame()
        return g1

    # Equal Width Bins
    def get_bins(self, data, feature, n_bins=25):
        gap = (data[feature].max() - data[feature].min())/n_bins
        bins = []
        i=data[feature].min()
        while i<=data[feature].max():
            bins.append(round(i,4))
            i+=gap
        return bins

    # PDF Graph - Tuckey Similarity
    def tukey_pdf(self, t, bins):
        pdf_value, axis_value = np.histogram(t, density=True, bins=bins)
        x_axis = []
        for i in range(len(axis_value)):
            try:
                x_axis.append(axis_value[i]+axis_value[i+1])
            except:
                break
        return list(pdf_value), x_axis 

    # Tuckey Definition
    def tukey(self, gth, categorical_feature, target, df):
        x = []
        i=0
        graph1 = []
        graph1.append(target)
        bins = self.get_bins(df, target)
        for index, row in gth.iterrows():
            
            cat_list = pd.DataFrame({'category':row['lists']})
            cat_name = row['list_name']
            graph1.append(cat_name)
            d = df[[categorical_feature,target]]
            d[categorical_feature] = d[categorical_feature].astype(str)
            d = d.merge(cat_list, left_on=categorical_feature, right_on='category', how='inner')
            a,b = self.tukey_pdf(list(d[target].dropna()),bins)
            while i<1:
                x.append(b)
                i+=1
            x.append(a)
        
        graph = []
        graph.append(graph1)
        for i in range(len(x[0])):
            graph2=[]
            for j in range(len(graph1)):
                graph2.append(x[j][i])
            graph.append(graph2)
        return graph

    #Tuckey Simialr Distribution
    def similar_distribution_con(self):
        dist = {}
        for feature in self.CategoricalFeatures:
            if self.df[feature].nunique()<=30:
                dist_1 = {}
                dist_2 = {}
                gth = self.GroupTukeyHSD(self.df[self.TargetFeature], self.df[feature].astype(str))
                for i in range(gth.shape[0]):
                    dist_1[gth['list_name'].iloc[i]] = gth['lists'].iloc[i]
                dist_2['Values'] = dist_1
                dist_2['Graph'] = self.tukey(gth, feature, self.TargetFeature, self.df)
                dist[feature]=dist_2
        return dist