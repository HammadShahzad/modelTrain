from itertools import count
from unicodedata import category
from matplotlib import axes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

import pandas
import pandas as pd
import numpy as np
import regex as re
from collections import Counter


df=pd.read_csv("studentsResponses.csv")


y=df.iloc[:,49]
print("less than 1.7 = ",(y.where(y<1.7)).count())
print("less than y<2  = " ,(y.where(y<2.0)).count())
print("less than y<2.3  = ",(y.where(y<2.3)).count())
print("less than y<2.7  = ",(y.where(y<2.7)).count())

