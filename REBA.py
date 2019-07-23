import pandas as pd
import numpy as np


name_list = ['bulb+dryer', 'bulb+es+dryer', 'bulb+es', 'bulb' ,'dryer', 'es+dryer', 'es', 'inactive']


data = pd.DataFrame(columns=['label', 'signal'])

features = list()
label = list()
count = 0
for n in name_list:
    for i in range(1,30+1):

        if i<10:
            fname = n+'0'+str(i)+'.csv'
        else:
            fname = n+str(i)+'.csv'

        df = pd.read_csv(fname)
        df = df['Irms']



        df.replace('', np.nan, inplace=True)
        df = df.fillna(float(0.0))


        df = df.replace(df[df == '\r\n'],float(0.0))
        df = df.iloc[0:15]

        df = pd.to_numeric(df, errors='coerce')



        features.append(df.tolist())

        label.append(n)



import matplotlib.pyplot as plt
from random import randint

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.4)
axs = axs.ravel()

for i in range(10):
    datanum = randint(0, len(label))
    axs[i].plot(features[datanum])
    axs[i].set_title(label[datanum])
plt.show()

from sklearn.model_selection import train_test_split

feat_train, feat_test, label_train, label_test = train_test_split(features,
                                                                  label, test_size=0.2, random_state=42, shuffle=True)




from sklearn.naive_bayes import GaussianNB

models = []
models.append([GaussianNB(), 'GaussianNB'])



for clf, name in models:

    print(name)
    clf = clf.fit(feat_train, label_train)

    from sklearn.metrics import accuracy_score

    # print('Overfitting :', accuracy_score(clf.predict(feat_train), label_train)) # overfitting

    print('Accuracy', accuracy_score(clf.predict(feat_test), label_test))
    print()








