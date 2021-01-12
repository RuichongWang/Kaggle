# Kaggle Mechanisms of Action Prediction - Solo Bronze Medal Solution
About this Competition: https://www.kaggle.com/c/lish-moa
## Repository contents
## Methods
In this competition, I used A, B and C as base model, then blend the predicted result of all three models as final submission. 

The model structure of multi head is:

<p align="middle">
  <img src="img/.png" width="400"/>
</p>

About TabNet,...

Since the data for this competition has pretty much features, more than 800 original features and 1700+ after feature engineering, TabNet, a ..., seems to be very suitable for this data. The original Paper about TabNet can be found [here](https://arxiv.org/pdf/1908.07442.pdf). The basic structure of TabNet is: 

<p align="middle">
  <img src="img/.png" width="400"/>
  <img src="img/.png" width="400"/>
</p>

## Data loading
1. Original competition data from kaggle
```
  mkdir input
  cd input
  kaggle competitions download -c liverpool-ion-switching
```
2. Data without drift
As discussed in [this passage](https://www.kaggle.com/c/liverpool-ion-switching/discussion/133874), the original data is synthetic data with real life "electrophysiological" noise and synthetic drift added. [Chris Deotte](https://www.kaggle.com/cdeotte) created [this Data Set](https://www.kaggle.com/cdeotte/data-without-drift) which removed the drift and presented a cleaned data.
```
  kaggle datasets download -d cdeotte/data-without-drift
```
3. RAPIDS
In rapids-knn-as-features.ipynb, I used RAPIDS-KNN to create additional features, so RAPIDS is needed before running it.
```
  kaggle datasets download -d cdeotte/rapids
```
4. Ion shifted rfc proba
Thanks to [Sergey Bryansky](https://www.kaggle.com/sggpls/competitions), [this data](https://www.kaggle.com/sggpls/ion-shifted-rfc-proba) is the forcasted probability of train and test data using random forest classifier. This can be used as extra features.
```
  kaggle datasets download -d sggpls/ion-shifted-rfc-proba
```
5. Kalman cleaned data
Thanks to [ragnar](https://www.kaggle.com/ragnar123), [this data](https://www.kaggle.com/ragnar123/clean-kalman) removed the noise in original signal.
```
  kaggle datasets download -d ragnar123/clean-kalman
```
