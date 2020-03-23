import pandas
import numpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class excelProcessor(object):

    def __init__(self, filename):
        self.name = filename
        self.df = None
        if isinstance(filename, list):
            self.df = pandas.DataFrame(filename)
        else:
            self.df = pandas.read_excel(filename)
        self.df_matrix = self.df.values
        self.df_rows, self.df_cols = self.df_matrix.shape

        self.col_start = 0

        for i in range(self.df_cols):
            # print(type(df_matrix[0][i]))
            if isinstance(self.df_matrix[0][i], float):
                self.col_start = i
                break

        self.dfvalue = self.df.iloc[:, self.col_start:]
        self.valuearray = self.dfvalue.values
        self.col_name = [column_name for column_name in self.df][self.col_start:]

    def get_column_names(self):
        return [column_name for column_name in self.df]

    def get_data(self):
        """
        get data from excel by rows
        :return: a list containing each row in dict (column names are keys)
        """
        materials = []

        for row in self.df.iterrows():
            material = {}

            for column_name in self.get_column_names():
                material[column_name.replace('.', '').replace('$', '')] = row[1][column_name]

            materials.append(material)

        return materials

    def has_blank_cell(self):
        if 0 == numpy.where(self.df.isnull())[0].size:
            return False
        return True

    def blank_cells(self):
        blank_matrix = numpy.where(self.df.isnull())
        return list(zip(blank_matrix[0], blank_matrix[1]))

    def statistics_data_check(self):
        geo_mean = []
        # 求每列的几何平均
        for i in range(len(self.col_name)):
            col_i = self.valuearray[:, i]
            prod = 1.0
            for j in col_i:
                prod *= j
            geo_mean.append(pow(prod, 1.0 / self.df_rows))

        desc = self.dfvalue.describe()
        skew = self.dfvalue.skew()
        rangem = desc.loc['max'] - desc.loc['min']

        IQR = desc.loc['75%'] - desc.loc['25%']
        upper = desc.loc['75%'] + 1.5 * IQR
        lower = desc.loc['25%'] - 1.5 * IQR

        desc.loc['range'] = rangem
        desc.loc['IQR'] = IQR
        desc.loc['lower'] = lower
        desc.loc['upper'] = upper
        desc.loc['geomean'] = geo_mean
        desc.loc['skew'] = skew

        return desc

    def eudist_data_check(self):
        df_arg = self.df.iloc[:, self.col_start:-1]
        df_func = self.df.iloc[:, -1]

        arg_mat = df_arg.values  # 非决策属性值矩阵
        func_mat = df_func.values.reshape(-1, 1)  # 决策属性值矩阵

        arg_pair_dist = pairwise_distances(arg_mat)  # 非决策属性距离矩阵
        func_pair_dist = pairwise_distances(func_mat)  # 决策属性距离矩阵

        # turn pair dist matrix to list
        arg_dist_list = [arg_pair_dist[i][j] for i in range(self.df_rows) for j in range(i + 1, self.df_rows)]
        func_dist_list = [func_pair_dist[i][j] for i in range(self.df_rows) for j in range(i + 1, self.df_rows)]

        # 属性上下限，判断样本属性是否相似或差异
        arg_dist_upper = numpy.quantile(arg_dist_list, 0.75)
        arg_dist_lower = numpy.quantile(arg_dist_list, 0.25)
        func_dist_upper = numpy.quantile(func_dist_list, 0.9)
        func_dist_lower = numpy.quantile(func_dist_list, 0.1)

        suspected_samples = []  # 记录嫌疑样本
        suspected_samples_count = {}  # 记录嫌疑样本出现次数

        for i in range(self.df_rows):
            for j in range(i + 1, self.df_rows):
                #  如果样本非决策属性差异大而决策属性相似，记入嫌疑样本
                if arg_pair_dist[i][j] >= arg_dist_upper and func_pair_dist[i][j] <= func_dist_lower:
                    suspected_samples.append(i)
                    suspected_samples.append(j)
                #  如果样本非决策属性相似而决策属性差异大，记入嫌疑样本
                if arg_pair_dist[i][j] <= arg_dist_lower and func_pair_dist[i][j] >= func_dist_upper:
                    suspected_samples.append(i)
                    suspected_samples.append(j)

        for i in suspected_samples:
            # 若嫌疑样本出现2次以上，视为异常样本
            if suspected_samples.count(i) > 2:
                suspected_samples_count[i] = suspected_samples.count(i)

        # {样本编号：出现次数}
        return suspected_samples_count

    def algorithm_data_check(self):
        lof = LocalOutlierFactor(n_neighbors=self.df_rows // 2, contamination=.1)
        llof = lof.fit_predict(self.valuearray)
        llof_idx = [i for i in range(len(llof)) if llof[i] == -1]

        iif = IsolationForest(n_estimators=len(self.col_name) * 2, contamination=.1)
        lif = iif.fit_predict(self.valuearray)
        lif_idx = [i for i in range(len(lif)) if lif[i] == -1]

        desc = {}
        desc['lof'] = llof_idx
        desc['if'] = lif_idx
        return desc
