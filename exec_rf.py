# -*- coding: utf-8 -*-

from pprint import pprint
from RF import RF

rf = RF("csv/bk/testdata.csv", "csv/20170110171713_header_output.csv");
#rf = RF("csv/bk/testdata.csv");

rf.createTrainingData().execute()
pprint(rf.importanceToCsv("csv/importance.csv"))
