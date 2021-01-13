import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
! pip install ../input/pytorchtabnet/pytorch_tabnet-2.0.0-py3-none-any.whl

import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.decomposition import PCA
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (QuantileTransformer, RobustScaler)
from tensorflow.keras import (Model, backend, callbacks, layers,
                              losses, metrics, optimizers, regularizers)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import itertools

warnings.filterwarnings('ignore')

seed = 42
num_of_seeds = 5
n_strats = 5

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map(
        {'trt_cp': 0, 'ctl_vehicle': 1})
    df = pd.get_dummies(df, columns=["cp_time", "cp_dose"])
    del df['sig_id']
    return df


def read_data():
    train = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    test = pd.read_csv('../input/lish-moa/test_features.csv')
    sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

    train = preprocess(train)
    test = preprocess(test)

    del train_targets['sig_id']

    train_targets = train_targets.loc[train['cp_type'] == 0].reset_index(
        drop=True)   # when cp_type == 0, all labels equal to 0
    train = train.loc[train['cp_type'] == 0].reset_index(drop=True)
    return train, test, train_targets, sample_submission


def fe(train, test):
    seed_everything(42)

    # Scaling
    features = train.columns[2:]
    scaler = RobustScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis=0))

    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])

    # RankGauss
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    for col in (GENES + CELLS):
        transformer = QuantileTransformer(
            n_quantiles=100, random_state=0, output_distribution="normal")
        vec_len = len(train[col].values)
        vec_len_test = len(test[col].values)
        raw_vec = train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train[col] = transformer.transform(
            raw_vec).reshape(1, vec_len)[0]
        test[col] = transformer.transform(
            test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
    print('data shape, RankGauss', train.shape)

    # PCA
    n_comp_g = 600         # determined by PCA, 90% Variance Explained
    n_comp_c = 50          # determined by PCA, 90% Variance Explained

    # GENES
    Gene_data = pd.concat([pd.DataFrame(train[GENES]),
                           pd.DataFrame(test[GENES])])
    Gene_gene_PCA = (PCA(n_components=n_comp_g,
                         random_state=seed).fit_transform(Gene_data[GENES]))

    train_gene_PCA = Gene_gene_PCA[:train.shape[0]]
    test_gene_PCA = Gene_gene_PCA[-test.shape[0]:]

    train_gene_PCA = pd.DataFrame(
        train_gene_PCA, columns=[f'pca_G-{i}' for i in range(n_comp_g)])
    test_gene_PCA = pd.DataFrame(
        test_gene_PCA, columns=[f'pca_G-{i}' for i in range(n_comp_g)])

    train = pd.concat((train, train_gene_PCA), axis=1)
    test = pd.concat((test, test_gene_PCA), axis=1)

    # CELLS
    Cell_data = pd.concat([pd.DataFrame(train[CELLS]),
                           pd.DataFrame(test[CELLS])])
    data_cell_PCA = (PCA(n_components=n_comp_c,
                         random_state=seed).fit_transform(Cell_data[CELLS]))

    train_cell_PCA = data_cell_PCA[:train.shape[0]]
    test_cell_PCA = data_cell_PCA[-test.shape[0]:]

    train_cell_PCA = pd.DataFrame(
        train_cell_PCA, columns=[f'pca_C-{i}' for i in range(n_comp_c)])
    test_cell_PCA = pd.DataFrame(
        test_cell_PCA, columns=[f'pca_C-{i}' for i in range(n_comp_c)])

    train = pd.concat((train, train_cell_PCA), axis=1)
    test = pd.concat((test, test_cell_PCA), axis=1)
    print('data shape, PCA', train.shape)

    # statistic features
    for df in [train, test]:
        df['g_sum'] = df[GENES].sum(axis=1)
        df['g_mean'] = df[GENES].mean(axis=1)
        df['g_std'] = df[GENES].std(axis=1)
        df['g_kurt'] = df[GENES].kurtosis(axis=1)
        df['g_skew'] = df[GENES].skew(axis=1)

        df['c_sum'] = df[CELLS].sum(axis=1)
        df['c_mean'] = df[CELLS].mean(axis=1)
        df['c_std'] = df[CELLS].std(axis=1)
        df['c_kurt'] = df[CELLS].kurtosis(axis=1)
        df['c_skew'] = df[CELLS].skew(axis=1)

        df['gc_sum'] = df[GENES + CELLS].sum(axis=1)
        df['gc_mean'] = df[GENES + CELLS].mean(axis=1)
        df['gc_std'] = df[GENES + CELLS].std(axis=1)
        df['gc_kurt'] = df[GENES + CELLS].kurtosis(axis=1)
        df['gc_skew'] = df[GENES + CELLS].skew(axis=1)

    # Variance Threshold
    data = pd.concat([pd.DataFrame(train),
                      pd.DataFrame(test)])

    ve_threshold = 0.8
    var_thresh = VarianceThreshold(ve_threshold)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_transformed = data_transformed[: train.shape[0]]
    test_transformed = data_transformed[-test.shape[0]:]

    train = train[['cp_type', 'cp_time_24', 'cp_time_48',
                   'cp_time_72', 'cp_dose_D1', 'cp_dose_D2']]
    test = test[['cp_type', 'cp_time_24', 'cp_time_48',
                 'cp_time_72', 'cp_dose_D1', 'cp_dose_D2']]

    train = pd.concat(
        [train, pd.DataFrame(train_transformed)], axis=1)
    test = pd.concat(
        [test, pd.DataFrame(test_transformed)], axis=1)

    print('data shape, VarianceEncoding', train.shape)
    return train, test

train_org, test_org, train_targets_scored, submission = read_data()
train_fe, test_fe = fe(train_org, test_org)

def tabnet(X_train_org, X_train_fe, y_train, X_val_org, X_val_fe, y_val, test_org, test_fe, seed, fold_nb):
    print('Tabnet model #%s, fold #%s, started.'%(seed+1,fold_nb+1))
    seed_everything(seed)
    tabnet_params = dict(n_d=24, n_a=48, n_steps=1, gamma=1.3,
                         lambda_sparse=0,
                         optimizer_fn=torch.optim.Adam, # tabnet package is based on torch, so tf.keras.optimizers.Adam will lead to an error
                         optimizer_params=dict(lr=2e-2),
                         mask_type='entmax',
                         scheduler_params=dict(mode="min",
                                               patience=5,
                                               min_lr=1e-5,
                                               factor=0.9,),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,   # same, here we can't use tf.keras.callbacks.ReduceLROnPlateau,
                         verbose=0,
                         )

    model = TabNetRegressor(**tabnet_params)
    model.fit(X_train=X_train_fe,
              y_train=y_train,
              eval_set=[(X_val_fe, y_val)],
              eval_name=["val"],
              eval_metric=['logloss'],
              max_epochs=250,
              patience=15, batch_size=256, virtual_batch_size=32,
              num_workers=1, drop_last=False,
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits
              ) 

    score = np.min(model.history["val_logloss"])
    return 'tabnet', model.predict(test_fe.values)  # it will automatically use best weights from best epoch, so we don't need to save & load weights


def fcn(X_train_org, X_train_fe, y_train, X_val_org, X_val_fe, y_val, test_org, test_fe, seed, fold_nb):
    print('FCN model #%s, fold #%s, started.'%(seed+1,fold_nb+1))
    seed_everything(seed)

    inp = layers.Input(shape=(X_train_fe.shape[1],))

    x = layers.BatchNormalization()(inp)
    x = layers.Dense(1500)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.26)(x)

    x = layers.Dense(1500)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.26)(x)

    out = layers.Dense(206)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adam())

    checkpoint = ModelCheckpoint(filepath='/kaggle/output/fcn_%s.h5' %
                                 seed,
                                 monitor=losses.BinaryCrossentropy(),
                                 mode='min',
                                 save_best_only='True')
    model.fit(X_train_fe,
              y_train,
              epochs=25,
              callbacks=[checkpoint],
              batch_size=128,
              verbose=1,
              validation_data=(X_val_fe, y_val))

    return 'fcn', model.predict(test_fe.values)

def multi_head(X_train_org, X_train_fe, y_train, X_val_org, X_val_fe, y_val, test_org, test_fe, seed, fold_nb):
    print('Multi_head model #%s, fold #%s, started.'%(seed+1,fold_nb+1))
    seed_everything(seed)

    # head 1
    inp_1 = layers.Input(shape=(X_train_fe.shape[1],))

    head_1 = layers.BatchNormalization()(inp_1)
    head_1 = layers.Dropout(0.3)(head_1)
    head_1 = layers.Dense(2048)(head_1)
    head_1 = layers.LeakyReLU()(head_1)
    head_1 = layers.BatchNormalization()(head_1)
    head_1 = layers.Dense(1024)(head_1)

    # head 2
    inp_2 = layers.Input(shape=(X_train_org.shape[1],))
    concat_1 = layers.Concatenate()([head_1, inp_2])

    head_2 = layers.BatchNormalization()(concat_1)
    head_2 = layers.Dropout(0.3)(head_2)

    head_2 = layers.Dense(2048)(head_2)
    head_2 = layers.LeakyReLU()(head_2)
    head_2 = layers.BatchNormalization()(head_2)

    head_2 = layers.Dense(2048)(head_2)
    head_2 = layers.LeakyReLU()(head_2)
    head_2 = layers.BatchNormalization()(head_2)

    head_2 = layers.Dense(1024)(head_2)
    head_2 = layers.LeakyReLU()(head_2)
    head_2 = layers.BatchNormalization()(head_2)

    head_2 = layers.Dense(1024)(head_2)
    head_2 = layers.LeakyReLU()(head_2)

    # skip connect & merge
    concat_2 = layers.Average()([head_1, head_2])

    concat_2 = layers.BatchNormalization()(concat_2)
    concat_2 = layers.Dense(1024)(concat_2)
    concat_2 = layers.LeakyReLU()(concat_2)
    concat_2 = layers.BatchNormalization()(concat_2)

    concat_2 = layers.Dense(206)(concat_2)
    concat_2 = layers.ReLU()(concat_2)
    concat_2 = layers.BatchNormalization()(concat_2)

    concat_2 = layers.Dense(206)(concat_2)
    out = layers.Softmax()(concat_2)

    model = Model(inputs=[inp_1, inp_2], outputs=out)
    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adam())

    checkpoint = ModelCheckpoint(filepath='/kaggle/output/multi_head_%s.h5' %
                                 seed,
                                 monitor=losses.BinaryCrossentropy(),
                                 mode='min',
                                 save_best_only='True')

    model.fit([X_train_fe, X_train_org],
              y_train,
              validation_data=([X_val_fe, X_val_org], y_val),
              epochs=35, batch_size=256,
              callbacks=[checkpoint], verbose=1)
    return 'multi_head', model.predict([test_fe.values,test_org.values])

def run_nn(model_func, seed):
    mskf = MultilabelStratifiedKFold(
        n_splits=n_strats, random_state=seed, shuffle=True)
    test_cv_preds = []
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_fe, train_targets_scored)):
        X_train_fe,  y_train = train_fe.values[train_idx,
                                               :], train_targets_scored.values[train_idx, :]
        X_train_org, y_train = train_org.values[train_idx,
                                                :], train_targets_scored.values[train_idx, :]
        X_val_fe,    y_val = train_fe.values[val_idx,
                                             :], train_targets_scored.values[val_idx, :]
        X_val_org,   y_val = train_org.values[val_idx,
                                              :], train_targets_scored.values[val_idx, :]

        model_name, preds_test = model_func(
            X_train_org, X_train_fe, y_train, X_val_org, X_val_fe, y_val, test_org, test_fe, seed, fold_nb)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))


    test_preds_all = np.stack(test_cv_preds)
    all_labels = [col for col in submission.columns if col not in ["sig_id"]]
    submission[all_labels] = test_preds_all.mean(axis=0)
    submission.loc[test_org['cp_type'] == 1, submission.columns[1:]] = 0
    submission.to_csv('%s_%s.csv' % (model_name, seed+1), index=None)


for model_func, model_seed in itertools.product([tabnet, fcn, multi_head], range(num_of_seeds)):
    run_nn(model_func, model_seed)
