#!/usr/bin/python
# coding=utf-8

	
def calcShannonEnt(dataSet):
	# 求list的长度，表示计算参与训练的数据量
	numEntries = len(dataSet)
	# 计算分类标签label出现的次数
	labelCounts = {}
	# the the number of unique elements and their occurance
	for featVec in dataSet:
		# 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
		currentLabel = featVec[-1]
#		print 'currentLabel-----'+currentLabel
		# 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	
	# 对于 label 标签的占比，求出 label 标签的香农熵
	shannonEnt = 0.0
	for key in labelCounts:
		# 使用所有类标签的发生频率计算类别出现的概率。
		prob = float(labelCounts[key])/numEntries
		# 计算香农熵，以 2 为底求对数
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

dataSet, labels = createDataSet()
calcShannonEnt(dataSet)
#print calcShannonEnt(createDataSet())