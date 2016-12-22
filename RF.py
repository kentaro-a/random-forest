# -*- coding: utf-8 -*-

import numpy as np
import math
import csv
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets


"""
	ランダムフォレスト実行クラス
	- date:2016-12-22
	- author:kentaro-a
"""
class RF:


	"""
		Initialize
		- input_file:csvファイルパス（先頭列はラベル）
	"""
	def __init__(self, input_file):
		csv_data = csv.reader(open(input_file, "r"))
		self.input_list = np.array([v for v in csv_data])
		

	"""
		トレーニングデータ生成
		- testdata_per:input_fileのなかでテストデータとして仕様する割合(%)
						指定しない場合はテストデータを生成しない。
	"""
	def createTrainingData(self, testdata_per=0):
		if (type(testdata_per) is int) and (testdata_per != 0):
			np.random.shuffle(self.input_list)
			div_index = math.floor(len(self.input_list) * testdata_per / 100)
			vsplit = np.vsplit(self.input_list, [div_index])
			self.test_data = np.hsplit(vsplit[0], [1])[1]
			# 先頭列をラベル、その他をデータとして配列を分割する
			split = np.hsplit(vsplit[1], [1])
			self.tr_label = [v[0] for v in split[0]]
			self.tr_data = split[1]

		else:
			# 先頭列をラベル、その他をデータとして配列を分割する
			split = np.hsplit(self.input_list, [1])
			self.tr_label = [v[0] for v in split[0]]
			self.tr_data = split[1]

		return self


	"""
		テストデータ生成
		- input_file:csvファイルパス（先頭列ラベルなし）
	"""
	def createTestData(self, input_file):
		csv_data = csv.reader(open(input_file, "r"))
		self.test_data = np.array([v for v in csv_data])
		return self


	"""
		モデル学習実行
	"""
	def execute(self):
		self.model = RandomForestClassifier()
		self.model.fit(self.tr_data, self.tr_label)
		return self


	"""
		test_dataを使って予測する
	"""
	def executePredict(self):
		self.predict = self.model.predict(self.test_data)
		return self


	"""
		予測結果取得
	"""
	def getPredict(self):
		return self.predict


	"""
		特徴量の重要度(Importance)取得
	"""
	def getImportance(self):
		return self.model.feature_importances_


	"""
		model取得
	"""
	def getModel(self):
		return self.model




