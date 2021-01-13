# Kaggle Mechanisms of Action Prediction - Solo Bronze Medal Solution
About this Competition: https://www.kaggle.com/c/lish-moa
## Methods
* Highly Unbalanced Data: The data is highly unbalanced, so for each base model, I set 7 different seeds and blend 7 different models to have a robust result.
* Normalization: RankGauss, a method which works usually much better than standard mean/std scaler or min/max. [Michael Jahrer](https://www.kaggle.com/mjahrer) introduced this method [here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629).
* Dimension Reduction: PCA + Variance Threshold
* Models: I used TabNet, multi-heads-ResNet and simple fully connected neural network as base models, then blended the result of all three models as final submission. 

This is a multi-label task, and our labels are protein targets' mechanism of action responses, which is reasonable to have some correlation from each other, so I didn't used LightGBM model which can't explore this relationship.

TabNet outperforms or is on par with other tabular learning models on various datasets for classification and regression problems from different domains. It is a Tree-based learning that uses sequential attention to choose which features to reason from at each decision step.

Since the data for this competition has pretty much features, 870+ original features and 1500+ after feature engineering, TabNet seems to be very suitable for this data. The original Paper about TabNet can be found [here](https://arxiv.org/pdf/1908.07442.pdf). The basic structure of TabNet is: 

<p align="middle">
  <img src="img/tabnet_feature_selection.png" width="400"/>
  <img src="img/tabnet2_architecture.png" width="400"/>
</p>

The model structure of multi-heads-ResNet is:

<p align="middle">
  <img src="img/Multi_head_simple.png" width="200"/>
</p>

<p align="middle">
  <a href="https://github.com/RuichongWang/Kaggle/blob/main/Mechanisms-of-Action-Prediction/img/Multi_head.png">Structure in detail</a>
</p>

## Data loading
1. Original competition data from kaggle
```
  mkdir input
  cd input
  kaggle competitions download -c lish-moa
```
2. Iterative stratification
As introduced [here](https://github.com/trent-b/iterative-stratification), Iterative stratification is cross validator with stratification for multilabel data. 
```
  kaggle datasets download -d yasufuminakama/iterative-stratification
```
3. Pytorch tabnet
This is an offline code competition, [ryati](https://www.kaggle.com/ryati131457) uploaded TabNet's installation package. 
```
  kaggle datasets download -d ryati131457/pytorchtabnet
```
