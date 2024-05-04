# Pyspectra

# Import basic libraries

import spc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Read .spc file

# Read a single file

from pyspectra.readers.read_spc import read_spc
spc=read_spc('pyspectra/sample_spectra/VIAVI/JDSU_Phar_Rotate_S06_1_20171009_1540.spc')
spc.plot()
plt.xlabel("nm")
plt.ylabel("Abs")
plt.grid(True)
print(spc.head())

# Read multiple .spc files from a directory

from pyspectra.readers.read_spc import read_spc_dir

df_spc, dict_spc=read_spc_dir('pyspectra/sample_spectra/VIAVI')
display(df_spc.transpose())
f, ax =plt.subplots(1, figsize=(18,8))
ax.plot(df_spc.transpose())
plt.xlabel("nm")
plt.ylabel("Abs")
ax.legend(labels= list(df_spc.transpose().columns))
plt.show()

# Read .dx spectral files

# Read a single .dx file

# Single file, single spectra

# Single file with single spectra

from pyspectra.readers.read_dx import read_dx
#Instantiate an object
Foss_single= read_dx()
# Run  read method
df=Foss_single.read(file='pyspectra/sample_spectra/DX multiple files/Example1.dx')
df.transpose().plot()

# Single file, multiple spectra:

Foss_single= read_dx()
# Run  read method
df=Foss_single.read(file='pyspectra/sample_spectra/FOSS/FOSS.dx')
df.transpose().plot(legend=False)

for c in Foss_single.Samples['29179'].keys():
    print(c)

# Spectra preprocessing

from pyspectra.transformers.spectral_correction import msc, detrend ,sav_gol,snv

MSC= msc()
MSC.fit(df)
df_msc=MSC.transform(df)


f, ax= plt.subplots(2,1,figsize=(14,8))
ax[0].plot(df.transpose())
ax[0].set_title("Raw spectra")

ax[1].plot(df_msc.transpose())
ax[1].set_title("MSC spectra")
plt.show()

SNV= snv()
df_snv=SNV.fit_transform(df)

Detr= detrend()
df_detrend=Detr.fit_transform(spc=df_snv,wave=np.array(df_snv.columns))

f, ax= plt.subplots(3,1,figsize=(18,8))
ax[0].plot(df.transpose())
ax[0].set_title("Raw spectra")

ax[1].plot(df_snv.transpose())
ax[1].set_title("SNV spectra")

ax[2].plot(df_detrend.transpose())
ax[2].set_title("SNV+ Detrend spectra")

plt.tight_layout()
plt.show()

# Modelling of spectra

# Decompose using PCA

pca=PCA()
pca.fit(df_msc)
plt.figure(figsize=(18,8))
plt.plot(range(1,len(pca.explained_variance_)+1),100*pca.explained_variance_.cumsum()/pca.explained_variance_.sum())
plt.grid(True)
plt.xlabel("Number of components")
plt.ylabel(" cumulative % of explained variance")

df_pca=pd.DataFrame(pca.transform(df_msc))
plt.figure(figsize=(18,8))
plt.plot(df_pca.loc[:,0:25].transpose())


plt.title("Transformed spectra PCA")
plt.ylabel("Response feature")
plt.xlabel("Principal component")
plt.grid(True)
plt.show()

# Using automl libraries to deploy faster models

import tpot
from tpot import TPOTRegressor
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = TPOTRegressor(generations=10, population_size=50, scoring='neg_mean_absolute_error',
                      cv=cv, verbosity=2, random_state=1, n_jobs=-1)

y=Foss_single.Conc[:,0]
x=df_pca.loc[:,0:25]
model.fit(x,y)

from sklearn.metrics import r2_score
r2=round(r2_score(y,model.predict(x)),2)
plt.scatter(y,model.predict(x),alpha=0.5, color='r')
plt.plot([y.min(),y.max()],[y.min(),y.max()],LineStyle='--',color='black')
plt.xlabel("y actual")
plt.ylabel("y predicted")
plt.title("Spectra model prediction R^2:"+ str(r2))

plt.show()