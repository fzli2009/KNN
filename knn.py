import csv
import math
import numpy as np
import operator
import time
#创建一个类用于读取csv文件
class CsvLoad(object):
    data_load=[]
    def __init__(self,filename):
        self.filename=filename
    def load_csv(self):
        with open(self.filename) as csvfile:
            CsvLoad.data_load = csv.reader(csvfile)
            data_set=list(CsvLoad.data_load)
            for i in range(len(data_set)):#使用列表推导式遍历列表并将其转化为浮点型
                data_set[i]=[float(f) for f in data_set[i]]

            # for i in range(len(data_set)):#普通for循环遍历列表并转化为浮点型
            #     for j in range(len(data_set[0])):
            #         data_set[i][j]=float(data_set[i][j])
        csvfile.close()
        return data_set
#定义knn算法为一个类
class Knn(object):
    distance_list=[]
    classcount={}
    distance=0
    distances=[]
    def __init__(self,dataSet,target_data,label,k):#dataSet已知数据标签的数据集，target_data待预测的数据集，label训练数据集的标签值，k最近邻的距离
        self.dataSet=dataSet
        self.target_data=target_data
        self.label=label
        self.k=k
    def euclideanDistance(self):#计算欧式距离并将距离保存到类属性distance_list中
        for i in range(len(self.dataSet)):
            for j in range(len(self.dataSet[0])):
                Knn.distance += pow((self.target_data[j] - self.dataSet[i][j]), 2)
            Knn.distance = math.sqrt(Knn.distance)
            Knn.distance_list.append(Knn.distance)
    def kNN(self):#将保存的距离排序并运算出k近邻内重复最多的标签值
        distance_list = np.array(Knn.distance_list)#将列表转换为数组
        sort_Index = distance_list.argsort()#对距离列表排序并返回其索引值
        for i in range(self.k):
            label_index = label[sort_Index[i]]
            #字典的形式保存键值，标签值保存为key，将标签出现的次数保存为value。
            Knn.classcount[label_index] = Knn.classcount.get(label_index, 0) + 1#在字典里查询key=label_index对应的value，然后值加1
        #print("标签及其出现次数是%s" % Knn.classcount)
        sortedClassCount = sorted(Knn.classcount.items(), key=operator.itemgetter(1), reverse=True)#dict.items 将字典中的键值转化为元组并保存到列表中
        return sortedClassCount[0][0]

    def __del__(self):
        class_name = self.__class__.__name__
        Knn.distance_list = []
        Knn.classcount = {}
        Knn.distance = 0
        Knn.distances = []
        #print(class_name, "销毁")

####################################主函数###########################################

start=time.time()
csv_load_train=CsvLoad('D:/experiment/train/feature_mfcc.csv')
data_set=csv_load_train.load_csv()
print("训练集一共%s条数据"%len(data_set))
csv_load_pred=CsvLoad('D:/experiment/test/feature_mfcc.csv')
data_pred=csv_load_pred.load_csv()
print("测试集一共%s条数据"%len(data_pred))
label=[1]*700+[2]*700+[3]*700# 1表示汽车碰撞声  2表示汽车鸣笛声  3表示语音声
accu_glass = 0  # 用于保存正确识别的汽车碰撞声的个数
accu_gunshots = 0  # 用于保存正确识别的鸣笛声的个数
accu_screams = 0  # 用于正确识别人语音的个数
label_count_glass=[]
label_count_gunshots=[]
label_count_screams=[]

for z in range(len(data_pred)):
    #print('目前运算到第%s条待预测的数据'%z)
    #key_value_contain = []#用于保存k近邻内各个标签出现的次数
    knn_data = Knn(data_set, data_pred[z], label, 22)#创建k近邻实例
    knn_data.euclideanDistance()#计算待预测数据与训练集的欧式距离
    label_predict=knn_data.kNN()#预测到的标签值
    # key_value_contain.append(Knn.classcount)
    if z<300:
        label_known=1
        label_count_glass.append(Knn.classcount)
        if label_known == label_predict:
            accu_glass=accu_glass+1
    elif z>=300 and z<600:
        label_known=2
        label_count_gunshots.append(Knn.classcount)
        if label_known == label_predict:
            accu_gunshots=accu_gunshots+1
    else:
        label_known=3
        label_count_screams.append(Knn.classcount)
        if label_known == label_predict:
            accu_screams=accu_screams+1
time_speed=time.time()-start
print ('该段程序共耗时%s秒'%time_speed)
print('本次实验选取玻璃破碎声、枪声和人的喊叫声各700条声音作为训练集，各300条声音作为测试集，下面就是此次预测的结果！')
print('短时能量特征+knn分类器能正确识别到玻璃破碎声的个数为%s,其识别率为%s'%(accu_glass,float(accu_glass/300)))
print('短时能量特征+knn分类器能正确识别到枪声的个数为%s,其识别率为%s'%(accu_gunshots,float(accu_gunshots/300)))
print('短时能量特征+knn分类器能正确识别到人的尖叫声的个数为%s,其识别率为%s'%(accu_screams,float(accu_screams/300)))

print(label_count_glass)
print(label_count_gunshots)
print(label_count_screams)

