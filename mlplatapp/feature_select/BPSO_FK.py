import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import csv


class BPSO_FK(object):
    def __init__(self, data_info, population_size, best_fitness, max_steps, feature_kernel, verbose=None,
                 index_of_sets=None):
        self.data = data_info
        self.w_max = 0.9  # 权重最大值
        self.w_min = 0.4  # 权重最小值
        self.w = 0.9  # 初始权重
        self.v_max = 6  # 速度的最大值
        self.c1 = self.c2 = 2  # 学习因子
        self.population_size = population_size  # 粒子群数量
        self.dim = self.data.dim - len(feature_kernel)  # 搜索空间的维度
        self.best_fitness = best_fitness  # 最佳适应度阈值
        self.max_steps = max_steps  # 最大迭代次数
        self.x, self.v = self.init_population()  # 当前粒子群的所有x和v信息
        self.feature_kernel = np.array(feature_kernel)  # 特征核
        self.verbose = verbose  # 冗余特征
        self.feature_space = np.delete(range(self.data.dim), feature_kernel)  # 特征空间
        self.index_of_sets = index_of_sets  # 数据集合划分
        self.fitness = self.init_fitness()  # 当前粒子群的所有个体的适应度
        self.p_best = np.array(list(zip(self.x, self.fitness)))  # 局部最优x和fitness，zip打包后局部最优x类型为array()
        self.g_best, self.r2_best = self.cal_g_best()  # 粒子群最优x、fitness和r2

    # 初始化个体的位置和速度
    def init_population(self):
        x = np.random.randint(0, 2, (1, self.dim))  # 初始化位置随机二进制序列
        v = np.zeros((self.population_size, self.dim))  # 初始化速度为0
        for i in range(self.population_size - 1):
            x = np.vstack((x, np.random.randint(0, 2, (1, self.dim))))
        return x, v

    # 初始化各个体的适应度
    def init_fitness(self):
        fitness = np.array([])
        for i in range(self.population_size):
            # 获取被选择的特征（即二进制为1的特征）   ps: 增加and index not in self.verbose，强制选取实现选择的冗余特征，下同
            selected_feature_list = self.feature_space[
                [index for index, values in enumerate(self.x[i]) if values == 1 and index not in self.verbose]]
            # 在selected_feature_list的基础上强制所有特征核被选择
            options = np.append(selected_feature_list, self.feature_kernel).astype('int8')
            data_x = self.data.X[:, options]
            data_y = self.data.Y
            avg_rmse, r2 = self.cal_fitness(data_x, data_y)
            fitness = np.append(fitness, avg_rmse)  # 以平均RMSE为适应度值
        return fitness

    @staticmethod
    # 获取用于预测的机器学习模型
    def get_model(train_x, train_y):
        # model = Ridge(alpha=1e-3)       # 岭回归模型
        # model = RidgeCV(alphas=np.array([0.000092, 0.000093, 0.000094, 0.000095, 0.000091, 0.000090]))
        # model = RandomForestRegressor(n_estimators=10)
        model = LinearRegression()  # 最小二乘法
        model.fit(train_x, train_y)
        return model

    # 计算个体的适应度
    def cal_fitness(self, data_x, data_y):
        # 留一法

        loo = LeaveOneOut()
        sse = np.array([])
        sst = np.array([])
        avg_y = np.mean(data_y)
        for train, test in loo.split(data_x):
            train_x = data_x[train]
            train_y = data_y[train]
            test_x = data_x[test]
            test_y = data_y[test]
            ridge = self.get_model(train_x, train_y)
            predict_y = ridge.predict(test_x)
            sst = np.append(sst, (test_y - avg_y) ** 2)
            sse = np.append(sse, (test_y - predict_y) ** 2)
        avg_rmse = np.sqrt(np.mean(sse))  # 计算平均RMSE
        avg_r2 = 1 - (np.sum(sse) / np.sum(sst))  # 计算拟合优度

        # 交叉验证（50组，5折交叉验证）
        '''
        rmses = []
        r2s = []
        for k in range(len(self.index_of_sets)):
            train_index = np.delete(range(50), self.index_of_sets[k])        # 获取训练集的样本编号
            test_index = self.index_of_sets[k]                               # 获取验证集的样本编号
            train_x = data_x[train_index]
            train_y = data_y[train_index]
            test_x = data_x[test_index]
            test_y = data_y[test_index]
            model = self.get_model(train_x, train_y)
            predict_y = model.predict(test_x)
            rmse = np.sqrt(mean_squared_error(test_y, predict_y))
            r2 = r2_score(test_y, predict_y)
            rmses.append(rmse)
            r2s.append(r2)
        avg_rmse = np.average(rmses)
        avg_r2 = np.average(r2s)
        '''
        return avg_rmse, avg_r2

    # 更新个体的适应度和局部最优的相关信息（位置+适应度）
    def update_fitness(self):
        fitness = np.array([])
        for i in range(self.population_size):
            # 判断个体对应的特征子集是否为空
            if np.any(np.array(self.x[i]) != 0):
                # 获取被选择的特征（即二进制为1的特征）
                selected_feature_list = self.feature_space[
                    [index for index, values in enumerate(self.x[i]) if values == 1 and index not in self.verbose]]
                # 在selected_feature_list的基础上强制所有特征核被选择
                options = np.append(selected_feature_list, self.feature_kernel).astype('int8')
                data_x = self.data.X[:, options]
                data_y = self.data.Y
                avg_rmse, r2 = self.cal_fitness(data_x, data_y)
            else:
                # 如果特征子集为空，则将RMSE置为无穷大
                avg_rmse = np.inf
            fitness = np.append(fitness, avg_rmse)
            # 更新（每个个体的）局部最优
            if self.p_best[i][1] > avg_rmse:
                self.p_best[i][0] = self.x[i]
                self.p_best[i][1] = avg_rmse
        self.fitness = fitness

    # 计算全局最优个体的相关信息（位置+适应度）
    def cal_g_best(self):
        min_index = np.argmin(self.p_best[:, 1])
        g_best = np.array([np.array(self.p_best[min_index, 0]), self.p_best[min_index, 1]])
        # 计算最优R2值
        selected_feature_list = self.feature_space[
            [index for index, values in enumerate(g_best[0]) if values == 1 and index not in self.verbose]]
        options = np.append(selected_feature_list, self.feature_kernel).astype('int8')
        data_x = self.data.X[:, options]
        data_y = self.data.Y
        # 判断最优个体对应的特征子集是否为空
        if np.any(g_best[0] != 0):
            avg_rmse, r2 = self.cal_fitness(data_x, data_y)
        else:
            r2 = 0
        return g_best, r2

    # 进化
    def evolve(self):
        m = 0  # 记录当前迭代次数
        n = 0  # 记录最优个体值连续不变的次数
        history_g_best = [[], np.inf, 0]  # 历史最优个体（位置，适应度，R2）
        history_g_best[0] = self.g_best[0]
        history_g_best[1] = self.g_best[1]
        history_g_best[2] = self.r2_best
        for step in range(self.max_steps):
            # 生成population_size个范围在0-1的随机dim维数组（引入随机因素）
            r1 = np.random.uniform(0, 1, (self.population_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.population_size, self.dim))

            # 动态改变权重
            self.w = self.w_max - (self.w_max - self.w_min) * m / self.max_steps

            # 更新个体的速度
            self.v = self.w * self.v + self.c1 * r1 * (np.array(list(self.p_best[:, 0])) - self.x) + self.c2 * r2 * (
                        self.g_best[0] - self.x)
            self.v[self.v < -self.v_max] = -self.v_max  # 检查速度是否越界，若越界就取靠近的边界值
            self.v[self.v > self.v_max] = self.v_max

            # 更新个体的位置和适应度
            sigmod_x = 1 / (1 + np.exp(- self.v))  # 计算概率值（作为某二进制位是否翻转的阈值）
            self.x = np.array([[1 if np.random.rand() < v else 0 for v in vs] for vs in sigmod_x])
            self.update_fitness()  # 同时更新局部最优

            # 更新全局最优x和fitness
            before_g_best = self.g_best  # 记录上一次粒子群中的最优个体
            self.g_best, self.r2_best = self.cal_g_best()  # 计算当前粒子群中的最优个体
            # 获取最优个体对应的特征子集（不含特征核）
            selected_best_result = self.feature_space[
                [index for index, values in enumerate(self.g_best[0]) if values == 1 and index not in self.verbose]]
            # 获取最优个体对应的特征子集（含特征核）
            best_result_options = list(np.sort(np.append(self.feature_kernel, selected_best_result).astype('int8')))
            print('第%d次迭代， 最优适应度为：%.5f, 最优拟合优度为：%.5f' % (m, self.g_best[1], self.r2_best))
            print('最优粒子个体：', self.g_best[0])
            print('最优的特征子集是：', best_result_options)

            # 记录“历史”全局最优个体
            if self.g_best[1] < history_g_best[1]:
                history_g_best[0] = self.g_best[0]
                history_g_best[1] = self.g_best[1]
                history_g_best[2] = self.r2_best

            m = m + 1
            if before_g_best[1] == self.g_best[1] and (before_g_best[0] == self.g_best[0]).all():
                n += 1
            else:
                n = 0
            if n >= 20:
                # 如果最优个体连续20次迭代均不变，则证明算法收敛，如果此时最优个体的适应度达到阈值则跳出循环
                if self.g_best[1] <= self.best_fitness:
                    break
                # 如果此时最优个体的适应度未达到阈值，则证明算法陷入局部最优，采用CatfishBPSO中的方法跳出局部最优
                else:
                    n = 0
                    # CatfishBPSO，将粒子群按照适应度大小排序，后10%的粒子一部分变为全1（最大搜索空间），一部分变为全0（最小搜索空间）
                    num_of_change = int(self.population_size * 0.1)  # 需要改变的粒子的数量
                    all_fitness = self.fitness  # 当前种群内所有个体的适应度
                    order_x_index = np.argsort(-all_fitness)[:num_of_change]  # 获取需要改变的粒子的坐标
                    for k in range(num_of_change):  # 随机改变上述粒子的位置x
                        if np.random.rand() > 0.5:
                            self.x[order_x_index[k]] = np.ones(self.dim)
                        else:
                            self.x[order_x_index[k]] = np.zeros(self.dim)

                    # 重置全局最优解，对于位置x来说，随机选择一个位置置为1，其他位置置为0
                    mut_index = np.random.randint(0, self.dim)  # 获取随机位置信息
                    # 重置全局最优个体的位置x
                    self.g_best[0] = np.array([1 if i == mut_index else 0 for i in range(self.dim)])
                    selected_feature_list = self.feature_space[[index for index, values in enumerate(self.g_best[0]) if
                                                                values == 1 and index not in self.verbose]]
                    options = np.append(self.feature_kernel, selected_feature_list).astype('int8')
                    data_x = self.data.X[:, options]
                    data_y = self.data.Y
                    avg_rmse, r2 = self.cal_fitness(data_x, data_y)
                    # 重置全局最优个体的适应度
                    self.g_best[1] = avg_rmse

        # 最终模型：
        final_result = self.feature_space[
            [index for index, values in enumerate(history_g_best[0]) if values == 1 and index not in self.verbose]]
        final_options = list(np.sort(np.append(self.feature_kernel, final_result).astype('int8')))
        data_x = self.data.X[:, final_options]
        data_y = self.data.Y
        rr = self.get_model(data_x, data_y)
        print(rr.coef_, rr.intercept_)
        return final_options, history_g_best[1], history_g_best[2]
