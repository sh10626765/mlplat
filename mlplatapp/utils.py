import pandas
import numpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class excelProcessor(object):

    def __init__(self, filename):
        self.name = filename
        self.df =pandas.read_excel(filename)

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
