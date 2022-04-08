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


y=df.iloc[:,54]
print(y)

# y= y.str.strip().str.replace('[^\w\s]',' ',regex=True).str.lower().str.replace(",", " ", regex=True)
# sugges_freqone=Counter(" ".join(y).split()).most_common(900)
# sugges_freqtwo=pd.DataFrame(sugges_freqone,columns=['Word','Frequency'])
# sugges_freqtwo.plot(x='Word',y='Frequency',kind='bar')
# print(sugges_freqtwo)
# sugges_freqtwo.to_csv('final_freq0_1.csv')

