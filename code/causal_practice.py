import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression

lalonde = pd.read_csv('datasets/lalonde.csv')

control_dat = lalonde[lalonde.treat == 0]
trt_dat = lalonde[lalonde.treat == 1]

trt_dat

covs = lalonde.drop('treat', axis=1)



sns.kdeplot(control_dat.re78, label='Control')
sns.kdeplot(trt_dat.re78, label='Treatment')
plt.show()

sns.kdeplot(control_dat.educ, label='Control')
sns.kdeplot(trt_dat.educ, label='Treatment')
plt.show()


propensity = LogisticRegression()
propensity = propensity.fit(covs, lalonde.treat)
pscore = propensity.predict_proba(covs)[:,1] # The predicted propensities by the model
print(pscore[:5])

lalonde['ps'] = pscore

control_dat = lalonde[lalonde.treat == 0]
trt_dat = lalonde[lalonde.treat == 1]


sns.kdeplot(control_dat.ps, label='Control')
sns.kdeplot(trt_dat.ps, label='Treatment');
