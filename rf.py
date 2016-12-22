# -*- coding: utf-8 -*-

from pprint import pprint
from RF import RF

rf = RF("testdata.csv");
rf.createTrainingData().execute()
pprint(rf.getImportance())
