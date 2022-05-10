import os
import os.path as osp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time
from matplotlib import colors
import matplotlib.pyplot as plt


def getFileList(root, file_list, ext=None):
    root_new = root
    if os.path.isfile(root):
        if ext is None:
            file_list.append(root)
        else:
            if ext in root[-3:]:
                file_list.append(root)
    elif os.path.isdir(root):
        for s in os.listdir(root):
            root_new = os.path.join(root, s)
            getFileList(root_new, file_list, ext)
    return file_list


def loadImg(path_img):
    mat = cv2.imread(path_img)  # 载入图像
    mat = cv2.resize(mat, IMG_SIZE)  # 统一尺寸
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
    img_mat = cv2.equalizeHist(mat)  # 直方图均衡
    return img_mat


def saveImg(mat, path_img, vcenter=0.0):
    array = np.copy(mat)
    array = array.flatten()
    array = array[abs(array) > 0.0000001]

    # 寻找异常点上下界
    percentile = np.percentile(array, [0, 25, 50, 75, 100])
    IQR = percentile[3] - percentile[1]
    limitUp = percentile[3] + IQR * 1.5
    limitDown = percentile[1] - IQR * 1.5

    # 剔除异常点
    red = np.copy(mat)
    red[red < limitDown] = limitDown
    red[red > limitUp] = limitUp
    # print("Max Value:", red.max())
    # print("Min Value:", red.min())

    # 映射颜色区间，输出伪彩色图
    output = np.copy(red)
    divnorm = colors.TwoSlopeNorm(vcenter=vcenter)
    # plt.imshow(output, cmap=plt.cm.jet, norm=divnorm)
    plt.imshow(output, cmap=plt.get_cmap('Greys'), norm=divnorm)
    plt.colorbar()
    plt.savefig(path_img)
    plt.close()
    # plt.show()


def dataloader(path_data, mode="eigen"):
    # 获取所有的图像列表
    img_list = []
    img_list = getFileList(path_data, img_list, "jpg")
    # 划分数据集
    list_train, list_test = train_test_split(img_list, test_size=0.2, random_state=0)
    # 数据排序
    list_train.sort()
    list_test.sort()
    # 数据处理
    # 对于Eigenface方法
    if mode == "eigen":
        def processEigen(_list):
            _x = np.zeros((IMG_SIZE[0] * IMG_SIZE[1], 1))
            _y = []
            for img_path in _list:
                img_mat = loadImg(img_path)
                img_mat = np.reshape(img_mat, (-1, 1))
                _x = np.column_stack((_x, img_mat))
                _y.append(osp.basename(img_path).split(".")[0])
            _x = _x[:, 1:]
            return _x, _y

        x_train, y_train = processEigen(list_train)
        x_test, y_test = processEigen(list_test)
        return x_train, y_train, x_test, y_test

    # 对于Fisherface方法
    elif mode == "fisher":
        def processFisher(_list):
            _x = []
            _y = []
            # 获取标签列表
            for img_path in _list:
                _y.append(osp.basename(img_path).split(".")[0])
            # 获取数据列表（元素为每一类的矩阵）
            _pre_l = _y[0]
            _pre_m = np.zeros((IMG_SIZE[0] * IMG_SIZE[1], 1))
            for i in range(len(_list)):
                if _y[i] != _pre_l:
                    _pre_l = _y[i]
                    _x.append(_pre_m[:, 1:])
                    _pre_m = np.zeros((IMG_SIZE[0] * IMG_SIZE[1], 1))
                img_mat = loadImg(_list[i])
                img_mat = np.reshape(img_mat, (-1, 1))
                _pre_m = np.column_stack((_pre_m, img_mat))
            return _x, _y

        x_train, y_train = processFisher(list_train)
        x_test, y_test = processFisher(list_test)
        return x_train, y_train, x_test, y_test

    # 对于LBPH方法
    else:
        def processLbp(_list):
            _x = []
            _y = []
            for img_path in _list:
                img_mat = loadImg(img_path)
                _x.append(img_mat)
                _y.append(osp.basename(img_path).split(".")[0])
            return _x, _y

        x_train, y_train = processLbp(list_train)
        x_test, y_test = processLbp(list_test)
        return x_train, y_train, x_test, y_test


class Eigenface:
    x_train, y_train, x_test, y_test = np.zeros(0), [], np.zeros(0), []
    low_mat, eig_vec = np.zeros(0), np.zeros(0)

    def __init__(self, path):
        self.x_train, self.y_train, self.x_test, self.y_test = dataloader(path, mode="eigen")
        self.algorithm_pca(self.x_train)

    def algorithm_pca(self, data_mat):
        # 均值矩阵（平均脸）
        mean_mat = np.mat(np.mean(data_mat, 1)).T
        # 差值矩阵
        diff_mat = data_mat - mean_mat
        # 协方差矩阵
        cov_mat = (diff_mat.T * diff_mat) / float(diff_mat.shape[1])
        # 特征值，特征向量
        eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_mat))
        # 特征值，特征向量 的处理与选取
        eig_vecs = diff_mat * eig_vecs
        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)
        eig_val = (np.argsort(eig_vals)[::-1])[:DIM]
        self.eig_vec = eig_vecs[:, eig_val]
        # 低维矩阵
        self.low_mat = self.eig_vec.T * diff_mat

        # 可视化中间变量
        ## 平均脸
        saveImg(np.reshape(mean_mat / mean_mat.max(), IMG_SIZE), osp.join(PATH_LOG, "eigen", "mean_face_{}x{}.jpg".format(IMG_SIZE[0], IMG_SIZE[1])),
                vcenter=0.5)
        ## 特征脸
        for i in range(0, self.eig_vec.shape[1]):
            pic = np.reshape(self.eig_vec[:, i], IMG_SIZE)
            saveImg(pic.real, osp.join(PATH_LOG, "eigen", "eigen_face_{}x{}_{}.jpg".format(IMG_SIZE[0], IMG_SIZE[1], i)))

    def predict(self):
        results = []
        accs = []
        for test_vec, label in zip(self.x_test.T, self.y_test):
            test_vec = np.reshape(test_vec, (-1, 1))
            test_vec = np.reshape(self.eig_vec.T * test_vec, (1, -1))
            distance = []
            for vec in self.low_mat.T:
                distance.append(np.linalg.norm(test_vec - vec))
            pred_index = np.argsort(distance)[0]
            pred_y = self.y_train[pred_index]
            results.append(pred_y)
            accs.append(pred_y == label)
        acc = np.sum(accs) / len(accs)
        return results, accs, acc


class Fisherface:
    x_train, y_train, x_test, y_test = [], [], [], []
    low_mat_list, eig_vec = [], np.zeros(0)

    def __init__(self, path):
        self.x_train, self.y_train, self.x_test, self.y_test = dataloader(path, mode="fisher")
        self.algorithm_lda(self.x_train)

    def algorithm_lda(self, data_list):
        n = IMG_SIZE[0] * IMG_SIZE[1]  # 数据维度
        Sw = np.zeros((n, n))  # 类内散度
        Sb = np.zeros((n, n))  # 类间散度
        Mu = np.zeros((n, 1))  # 总数据的均值
        N = 0  # 总样本数
        means = []  # 记录每一类的均值矩阵
        sample_nums = []  # 记录每一类的样本数

        # 计算类内散度Sw
        for i, data_mat in enumerate(data_list):
            mean_mat = np.mat(np.mean(data_mat, 1)).T
            means.append(mean_mat)
            sample_nums.append(data_mat.shape[1])
            Mu += data_mat.shape[1] * mean_mat
            N += data_mat.shape[1]

            diff_mat = data_mat - mean_mat
            Sw += diff_mat * data_mat.T
        Mu = Mu / N

        # 计算类间散度Sb
        for num, mean_mat in zip(sample_nums, means):
            Sb += num * (mean_mat - Mu) * (mean_mat - Mu).T

        # 计算并选取特征值
        eig_vals, eig_vecs = np.linalg.eig(np.mat(np.linalg.inv(Sw) * Sb))
        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)
        eig_val = (np.argsort(eig_vals)[::-1])[:DIM]
        self.eig_vec = eig_vecs[:, eig_val]

        # 降维
        self.low_mat_list = []
        for data_mat in data_list:
            self.low_mat_list.append(self.eig_vec.T * data_mat)

        # 可视化中间结果
        ## 平均脸
        saveImg(np.reshape(Mu / Mu.max(), IMG_SIZE), osp.join(PATH_LOG, "fisher", DATASET, "mean_face_{}x{}.jpg".format(IMG_SIZE[0], IMG_SIZE[1])), vcenter=0.5)
        ## 特征脸
        for i in range(0, self.eig_vec.shape[1]):
            pic = np.reshape(self.eig_vec[:, i], IMG_SIZE)
            saveImg(pic.real, osp.join(PATH_LOG, "fisher", DATASET, "fisher_face_{}x{}_{}.jpg".format(IMG_SIZE[0], IMG_SIZE[1], i)))

    def predict(self):
        results = []
        accs = []
        index = 0
        for test_mat in self.x_test:
            for test_vec in test_mat.T:
                test_vec = np.reshape(test_vec, (-1, 1))
                test_vec = np.reshape(self.eig_vec.T * test_vec, (1, -1))
                distance = []
                for low_mat in self.low_mat_list:
                    for low_vec in low_mat.T:
                        distance.append(np.linalg.norm(test_vec - low_vec))
                pred_index = np.argsort(distance)[0]
                pred_y = self.y_train[pred_index]
                results.append(pred_y)
                accs.append(pred_y == self.y_test[index])
                index += 1
        acc = np.sum(accs) / len(accs)

        return results, accs, acc


class Lbp:
    def __init__(self, path):
        self.table = {}
        self.ImgSize = IMG_SIZE
        self.BlockNum = 5
        self.count = 0
        self.path = path

    def get_hop_counter(self, num):
        """
        计算二进制序列是否只变化两次
        :param num: 数字
        :return: 01变化次数
        """
        bin_num = bin(num)
        bin_str = str(bin_num)[2:]
        n = len(bin_str)
        if n < 8:
            bin_str = "0" * (8 - n) + bin_str
        n = len(bin_str)
        counter = 0
        for i in range(n):
            if i != n - 1:
                if bin_str[i + 1] != bin_str[i]:
                    counter += 1
            else:
                if bin_str[0] != bin_str[i]:
                    counter += 1
        return counter

    def get_table(self):
        """
        生成均匀对应字典
        :return: 均匀LBP特征对应字典
        """
        counter = 1
        for i in range(256):
            if self.get_hop_counter(i) <= 2:
                self.table[i] = counter
                counter += 1
            else:
                self.table[i] = 0
        return self.table

    def get_lbp_feature(self, img_mat):
        """
        计算LBP特征
        :param img_mat:图像矩阵
        :return: LBP特征图
        """
        # cv2.imwrite('./data/face_test/lbp_img' + str(self.count) + '.jpg', img_mat)
        m = img_mat.shape[0]
        n = img_mat.shape[1]
        neighbor = [0] * 8
        feature_map = np.mat(np.zeros((m, n)))
        t_map = np.mat(np.zeros((m, n)))
        for y in range(1, m - 1):
            for x in range(1, n - 1):
                neighbor[0] = img_mat[y - 1, x - 1]
                neighbor[1] = img_mat[y - 1, x]
                neighbor[2] = img_mat[y - 1, x + 1]
                neighbor[3] = img_mat[y, x + 1]
                neighbor[4] = img_mat[y + 1, x + 1]
                neighbor[5] = img_mat[y + 1, x]
                neighbor[6] = img_mat[y + 1, x - 1]
                neighbor[7] = img_mat[y, x - 1]
                center = img_mat[y, x]
                temp = 0
                for k in range(8):
                    temp += (neighbor[k] >= center) * (1 << k)
                feature_map[y, x] = self.table[temp]
                t_map[y, x] = temp
        feature_map = feature_map.astype('uint8')  # 数据类型转换为无符号8位型，如不转换则默认为float64位，影响最终效果
        t_map = t_map.astype('uint8')
        # print('t_map', t_map.shape)
        # cv2.imwrite('./data/face_test/lbp_face' + str(self.count) + '.jpg', t_map)
        # print('feature_map', feature_map.shape)
        # cv2.imwrite('./data/face_test/lbp_map' + str(self.count) + '.jpg', feature_map)
        self.count += 1
        return feature_map

    def get_hist(self, roi):
        """
        计算直方图
        :param roi:图像区域
        :return: 直方图矩阵
        """
        hist = cv2.calcHist([roi], [0], None, [59], [0, 256])  # 第四个参数是直方图的横坐标数目，经过均匀化降维后这里一共有59种像素
        return hist

    def compare(self, sampleImg, test_img):
        """
        比较函数，这里使用的是欧氏距离排序，也可以使用KNN，在此处更改
        :param sampleImg: 样本图像矩阵
        :param test_img: 测试图像矩阵
        :return: k2值
        """
        testFeatureMap = self.get_lbp_feature(test_img)
        sampleFeatureMap = self.get_lbp_feature(sampleImg)
        # 计算步长，分割整个图像为小块
        ystep = int(self.ImgSize[0] / self.BlockNum)
        xstep = int(self.ImgSize[1] / self.BlockNum)

        k2 = 0
        for y in range(0, self.ImgSize[0], ystep):
            for x in range(0, self.ImgSize[1], xstep):
                testroi = testFeatureMap[y:y + ystep, x:x + xstep]
                sampleroi = sampleFeatureMap[y:y + ystep, x:x + xstep]
                testHist = self.get_hist(testroi)
                sampleHist = self.get_hist(sampleroi)
                k2 += np.sum((sampleHist - testHist) ** 2) / np.sum((sampleHist + testHist))
        # print('k2的值为', k2)
        return k2

    def predict(self):
        self.table = self.get_table()

        img_list, label, test_img_list, test_labels = dataloader(self.path, mode="lbp")

        results = []
        accs = []
        for i in range(0, len(test_img_list)):
            test_img = test_img_list[i]
            k2_list = []
            for img in img_list:
                k2 = self.compare(img, test_img)
                k2_list.append(k2)
            result = label[np.argsort(k2_list)[0]]
            results.append(result)
            accs.append(result == test_labels[i])

        acc = np.sum(accs) / len(accs)

        return results, accs, acc


# DATASET = "faces94"
# DATASET = "faces95"
# DATASET = "faces96"
# DATASET = "grimace"
IMG_SIZE = (16, 16)
DIM = 20
PATH_LOG = "./data/logs/"


datasets = ["faces94", "faces95", "faces96", "grimace"]
for i in range(4):
    DATASET = datasets[3 - i]
    print(DATASET)

    time_begin = time.time()

    # eigenface = Eigenface('./data/Face_Recognition_Data/{}'.format(DATASET))
    # results, accs, acc = eigenface.predict()

    # fisherface = Fisherface('./data/Face_Recognition_Data/{}'.format(DATASET))
    # results, accs, acc = fisherface.predict()

    lbp = Lbp('./data/Face_Recognition_Data/{}'.format(DATASET))
    results, accs, acc = lbp.predict()

    print("\nAccu:", acc)
    print("\nCost Time:", time.time() - time_begin)
