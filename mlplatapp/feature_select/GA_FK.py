import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import csv


# 计算二进制数之间的汉明距离
def hamming(a: list, b: list):
    count = 0  # 初始化计数器
    for i in range(max(len(a), len(b))):
        a_bit = a[i] if i < len(a) else 0  # 提取a列表当前位
        b_bit = b[i] if i < len(b) else 0  # 提取b列表当前位
        count = count if a_bit == b_bit else count + 1  # 如果提取出的两个数字不相等，计数器+1
    return count


def ridge_regression(train_x, train_y):
    # model = Ridge(alpha=1e-3)       # 岭回归模型
    # model = RidgeCV(alphas=np.array([0.000092, 0.000093, 0.000094, 0.000095, 0.000091, 0.000090]))
    model = LinearRegression()  # 最小二乘法
    # model = RandomForestRegressor(n_estimators=10)
    model.fit(train_x, train_y)
    return model


class GA(object):
    def __init__(self, data_info, population_size, best_fitness, max_steps):
        self.data = data_info
        self.not_kernel = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        self.feature_kernel = np.array([])  # 特征核
        self.k1, self.k2, self.k3, self.k4 = 1, 1, 1, 0.2
        self.C = 0.4
        self.retain_rate = 0.2
        self.random_select_rate = 0.5
        self.mutation_rate = 0.01
        self.population_size = population_size  # 种群数量
        self.dim = self.data.dim - len(self.feature_kernel)  # 搜索空间的维度
        self.best_fitness = best_fitness  # 最佳适应度阈值
        self.max_steps = max_steps  # 最大迭代次数
        self.population = self.gen_population()   # 初始化种群
        self.fitness = self.cal_all_fitness()  # 种群中的所有个体的适应度

    # 初始化单个个体
    def gen_chromosome(self):
        # 随机生成长度为self.dim的染色体
        temp = np.random.randint(0, 2, self.dim)
        while True:
            if 1 not in temp:
                temp = np.random.randint(0, 2, (1, self.dim))
            else:
                break
        return temp

    # 初始化种群
    def gen_population(self):
        population = np.array([self.gen_chromosome() for i in range(self.population_size)])
        return population

    # 解码染色体，将二进制序列转化为对应特征的编号序列（添加特征核）
    def decode(self, chromosome):
        if len(self.feature_kernel) == 0:
            selected_feature_list = np.array([index for index, values in enumerate(chromosome) if values == 1])
            options = selected_feature_list
        else:
            selected_feature_list = self.not_kernel[[index for index, values in enumerate(chromosome) if values == 1]]
            options = np.append(self.feature_kernel, selected_feature_list)
        return options

    # 计算个体的适应度
    def cal_fitness(self, data_x, data_y):
        loo = LeaveOneOut()
        sse = np.array([])
        sst = np.array([])
        avg_y = np.mean(data_y)
        for train, test in loo.split(data_x):
            train_x = data_x[train]
            train_y = data_y[train]
            test_x = data_x[test]
            test_y = data_y[test]
            ridge = ridge_regression(train_x, train_y)
            predict_y = ridge.predict(test_x)
            sst = np.append(sst, (test_y - avg_y)**2)
            sse = np.append(sse, (test_y - predict_y) ** 2)
        avg_rmse = np.sqrt(np.mean(sse))            # 计算平均RMSE
        avg_r2 = 1 - (np.sum(sse) / np.sum(sst))        # 计算拟合优度


        return avg_rmse, avg_r2

    # 计算种群的适应度
    def cal_all_fitness(self):
        fitness = np.array([])
        for i in range(self.population_size):
            if np.any(np.array(self.population[i]) != 0):  # 避免变异可能导致的特征子集为空的状况
                if len(self.feature_kernel) == 0:
                    selected_feature_list = np.array([index for index, values in enumerate(self.population[i]) if values == 1])
                    options = selected_feature_list
                else:
                    selected_feature_list = self.not_kernel[[index for index, values in enumerate(self.population[i]) if values == 1]]
                    options = np.append(self.feature_kernel, selected_feature_list)
                data_x = self.data.X[:, options]
                data_y = self.data.Y
                avg_rmse, r2 = self.cal_fitness(data_x, data_y)
            else:
                avg_rmse = np.inf
            fitness = np.append(fitness, avg_rmse)
        return fitness

    # 采用轮盘赌法进行选择
    def selection(self):
        # 按照适应度从大到小（取值从小到大），对适应度和个体序列进行排序
        sorted_index = np.argsort(self.fitness)             # 获取排序后的坐标
        self.fitness = np.sort(self.fitness)                # 获取排序后的之适应度
        self.population = self.population[sorted_index]     # 获取排序后的种群
        # 选出适应性强的染色体
        retain_length = int(self.population_size * self.retain_rate)
        parents = self.population[: retain_length]
        # 选出适应性不强，但幸存的个体
        for chromosome in self.population[retain_length:]:
            if np.random.random() < self.random_select_rate:
                parents = np.vstack((parents, chromosome))
        return parents

    # 交叉
    def crossover(self, parents):
        # 求父代的平均适应度和最大适应度(最小值)
        population = list(self.population.tolist())
        f_avg = np.mean([self.fitness[index] for index in [population.index(parents[i].tolist()) for i in range(len(parents))]])
        f_max = np.max([self.fitness[index] for index in [population.index(parents[i].tolist()) for i in range(len(parents))]])
        children = []
        target_count = self.population_size - len(parents)
        while len(children) < target_count:
            male = np.random.randint(0, len(parents) - 1)
            female = np.random.randint(0, len(parents) - 1)
            if male != female:
                male_individual = parents[male]
                female_individual = parents[female]
                # 父亲和母亲中的最大适应度（最小值）
                f_com_max = np.max([self.fitness[population.index(male_individual.tolist())],
                                    self.fitness[population.index(female_individual.tolist())]])
                # 自适应计算交叉概率p
                if f_com_max < f_avg:
                    p = self.k1 * (f_max - f_com_max) / (f_max - f_avg)
                else:
                    p = self.k2
                if np.random.random() < p:
                    # 随机选取交叉点
                    cross_pos = np.random.randint(0, self.dim)
                    children1 = male_individual[0:cross_pos]
                    children1 = np.append(children1, female_individual[cross_pos:])
                    children2 = female_individual[0:cross_pos]
                    children2 = np.append(children2, male_individual[cross_pos:])
                    children.append(children1.tolist())
                    children.append(children2.tolist())
        children = np.array(children)
        return children

    # 变异
    def mutation(self, parents, children, children_fitness):
        # 求种群的平均适应度和最大适应度(值最小)
        f_avg = np.mean(self.fitness)
        f_max = np.max(self.fitness)
        # 求当前种群的汉明距离
        haming_distance = 0
        for i in range(self.population_size):
            for j in range(i+1, self.population_size):
                haming_distance += hamming(self.population[i], self.population[j])
        # 计算f用来定量表示种群的分布特性
        f = (4 * haming_distance) / (self.dim * self.population_size**2)
        for i in range(len(children)):
            # 自适应计算变异率p
            f_current = children_fitness[i]     # 当前个体的适应度
            if f_current < f_avg:
                p = self.C * (1 - f + self.k3 * (f_max - f_current) / (f_max - f_avg))
            else:
                p = self.k4
            if np.random.random() < p:
                j = np.random.randint(0, self.dim - 1)
                children[i][j] = 1 - children[i][j]
        self.population = np.vstack((parents, children))

    # 进化
    def evolve(self):
        parents = self.selection()              # 选择
        children = self.crossover(parents)      # 交叉
        # 计算交叉后的个体（孩子）的适应度
        children_fitness = np.array([])
        for i in range(len(children)):
            if len(self.feature_kernel) == 0:
                selected_feature_list = np.array([index for index, values in enumerate(children[i]) if values == 1])
                options = selected_feature_list
            else:
                selected_feature_list = self.not_kernel[[index for index, values in enumerate(children[i]) if values == 1]]
                options = np.append(self.feature_kernel, selected_feature_list)
            data_x = self.data.X[:, options]
            data_y = self.data.Y
            avg_rmse, r2 = self.cal_fitness(data_x, data_y)
            children_fitness = np.append(children_fitness, avg_rmse)
        self.mutation(parents, children, children_fitness)        # 变异
        self.fitness = self.cal_all_fitness()


# 迭代求解
def GA_FK(data_info, population_size, best_fitness, max_steps):
    final_fitness = np.inf
    final_selected_feature = np.array([])
    final_r2 = 0
    for i in range(1):
        ga = GA(data_info, population_size, best_fitness, max_steps)
        before_best = ga.population[np.argmin(ga.fitness)]
        before_best_fitness = np.min(ga.fitness)
        current_best = ga.population[np.argmin(ga.fitness)]
        selected_feature = ga.decode(current_best)
        data_x = ga.data.X[:, selected_feature]
        data_y = ga.data.Y
        current_best_fitness, current_best_r2 = ga.cal_fitness(data_x, data_y)
        m = 0  # 记录内层迭代次数
        n = 0  # 记录最优个体连续不变的次数
        for j in range(max_steps):
            ga.evolve()
            current_best = ga.population[np.argmin(ga.fitness)]
            selected_feature = ga.decode(current_best)
            data_x = ga.data.X[:, selected_feature]
            data_y = ga.data.Y
            current_best_fitness, current_best_r2 = ga.cal_fitness(data_x, data_y)
            print('第%d次迭代， 最优适应度为：%.5f, 最优拟合优度为：%.5f' % (m, current_best_fitness, current_best_r2))
            print('最优粒子个体：', current_best)
            print('最优的特征子集是：', np.sort(selected_feature))
            m += 1
            if (before_best == current_best).all() and before_best_fitness == current_best_fitness:
                n += 1
            else:
                n = 0
            if n >= 10:
                break
        if final_fitness > current_best_fitness:
            final_fitness = current_best_fitness
            final_selected_feature = selected_feature
            final_r2 = current_best_r2
        if final_fitness < best_fitness:
            break
    # 最终模型：
    data_x = data_info.X[:, final_selected_feature]
    data_y = data_info.Y
    rr = ridge_regression(data_x, data_y)
    print('最优的特征子集是：', final_selected_feature)
    print('最优适应度为：%.5f, 最优拟合优度为：%.5f' % (final_fitness, final_r2))
    print(rr.coef_, rr.intercept_)