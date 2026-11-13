#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation fin-à-fin de la RMSE de prime pure en assurance automobile.

Source des transformations : xgboost_frequence_v2_explique.py du projet original.
La gravité est apprise UNIQUEMENT sur les contrats sinistrés dans chaque fold.
Mise à l'écart par id_client, prétraitements réappris dans chaque fold,
aucun accès au holdout pendant les recherches Optuna.

Candidats : fréquence XGBoost x gravité XGBoost Gamma,
            XGBoost direct Tweedie/RMSE, LightGBM et CatBoost directs,
            combinaison convexe régularisée sélectionnée sur les OOF.
Tous les objectifs d'optimisation visent la RMSE finale du montant_sinistre.
Aucune RMSE ni prétention de supériorité n'est fabriquée sans train.csv réel.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
EPS = 1e-8
FREQ_INITIAL = dict(n_estimators=350, max_depth=3, learning_rate=.045,
                    min_child_weight=5, subsample=.85, colsample_bytree=.85,
                    reg_lambda=5., reg_alpha=0., gamma=0.,
                    min_frequency=20, max_categories=70)
FREQ_V2 = dict(n_estimators=900, max_depth=2, learning_rate=.016422,
               min_child_weight=27, subsample=.86968, colsample_bytree=.86799,
               reg_lambda=19.0732, reg_alpha=3e-6, gamma=1.0143,
               min_frequency=20, max_categories=30)
FREQ_V3 = dict(n_estimators=1400, max_depth=2, learning_rate=.0091573,
               min_child_weight=24, subsample=.85296, colsample_bytree=.8588,
               reg_lambda=8.3545, reg_alpha=.0001, gamma=.94077,
               min_frequency=40, max_categories=30)
FREQ_TPE100 = dict(n_estimators=1800, max_depth=2, learning_rate=.009371,
                   min_child_weight=24, subsample=.85046, colsample_bytree=.95094,
                   reg_lambda=13.92, reg_alpha=0., gamma=2.6356,
                   min_frequency=60, max_categories=15)
SEV_INITIAL = dict(n_estimators=350, max_depth=2, learning_rate=.035,
                   min_child_weight=8, subsample=.85, colsample_bytree=.85,
                   reg_lambda=10., reg_alpha=.1, gamma=0.,
                   min_frequency=20, max_categories=70)
DIRECT_INITIAL = dict(n_estimators=700, max_depth=2, learning_rate=.025,
                      min_child_weight=20, subsample=.85, colsample_bytree=.85,
                      reg_lambda=10., reg_alpha=.1, gamma=1.,
                      min_frequency=30, max_categories=30,
                      objective='reg:tweedie', tweedie_variance_power=1.5)
BUDGETS = {
    'test': dict(gamma=1, freq=1, gamma2=0, direct=1, light=1, cat=1),
    'rapide': dict(gamma=8, freq=8, gamma2=4, direct=8, light=8, cat=6),
    'complet': dict(gamma=35, freq=35, gamma2=20, direct=50, light=40, cat=25),
    'maximum': dict(gamma=100, freq=100, gamma2=60, direct=120, light=100, cat=60),
    'ultra': dict(gamma=280, freq=280, gamma2=150, direct=450, light=300, cat=240),
    'extreme': dict(gamma=500, freq=500, gamma2=250, direct=850, light=500, cat=400),
}


def args_parser():
    a = argparse.ArgumentParser(description='Optimiser la prime pure sur RMSE OOF, fréquence+gravité et coût direct')
    a.add_argument('--train', type=Path, default=Path('train.csv'))
    a.add_argument('--test', type=Path, default=Path('test.csv'))
    a.add_argument('--output', type=Path, default=Path('resultats_rmse'))
    a.add_argument('--budget', choices=BUDGETS, default='complet')
    a.add_argument('--folds', type=int, default=4)
    a.add_argument('--cv-repeats', type=int, default=1, help='Nombre de répartitions CV groupées indépendantes (multiplication du coût de calcul)')
    a.add_argument('--threads', type=int, default=7)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--seeds-ensemble', type=str, default='',
                   help='Graines prédéfinies séparées par virgules, ex. 42,2026,1337,17,73. Vide : seed initiale seulement.')
    a.add_argument('--sans-catboost', action='store_true')
    a.add_argument('--sans-lightgbm', action='store_true')
    a.add_argument('--autoriser-paiement', action='store_true',
                   help='UNIQUEMENT si paiement est prouvé disponible AVANT tout sinistre.')
    return a.parse_args()


def source_frequency():
    path = BASE / 'xgboost_frequence_v2_explique.py'
    spec = importlib.util.spec_from_file_location('frequence_v2_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard_execution(args):
    """Empêche de réutiliser une étude SQLite avec un autre jeu de données/protocole.

    La reprise est autorisée après interruption, mais pas si les objectifs de
    phase ont été changés (budget, colonnes, folds, seed ou code).
    """
    sources = [args.train, args.test, BASE/'xgboost_frequence_v2_explique.py', Path(__file__)]
    checksums = {}
    for p in sources:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                h.update(block)
        checksums[p.name]=h.hexdigest()
    config={'empreintes':checksums,'budget':args.budget,'folds':args.folds,'cv_repeats':args.cv_repeats,
            'seed':args.seed,'autoriser_paiement':args.autoriser_paiement,
            'sans_catboost':args.sans_catboost,
            'sans_lightgbm':args.sans_lightgbm,'seeds_ensemble':args.seeds_ensemble}
    file=args.output/'identite_execution.json'
    if file.exists():
        old=json.loads(file.read_text(encoding='utf-8'))
        if old!=config:
            raise ValueError('Dossier de résultats déjà utilisé avec une autre version ou un '
                             'autre protocole. Choisis un nouveau --output pour éviter de '
                             'mélanger des essais Optuna non comparables.')
    else:
        save_json(file,config)


def load_train_test(a):
    if not a.train.is_file() or not a.test.is_file():
        raise FileNotFoundError(
            'Les vrais fichiers train.csv et test.csv sont obligatoires. '
            'Place-les dans le dossier du script, ou passe --train et --test. '
            'Aucune performance ne peut être calculée sans leurs valeurs réelles.'
        )
    train = pd.read_csv(a.train, dtype={'code_postal':'string'}, low_memory=False)
    test = pd.read_csv(a.test, dtype={'code_postal':'string'}, low_memory=False)
    required = {'nombre_sinistres','montant_sinistre','id_client','index'}
    if required - set(train):
        raise ValueError('Cibles/identifiants manquants dans train.csv : ' + str(required-set(train)))
    if 'index' not in test or test['index'].duplicated().any() or test['index'].isna().any():
        raise ValueError('test.csv doit comporter une colonne index unique et non manquante')
    if train['id_client'].isna().any():
        raise ValueError('id_client manquant : pas de validation groupée possible')
    yfreq = pd.to_numeric(train['nombre_sinistres'], errors='coerce')
    y = pd.to_numeric(train['montant_sinistre'], errors='coerce')
    if not np.isfinite(y).all() or not yfreq.isin([0,1]).all():
        raise ValueError('Cibles non finies ou nombre_sinistres différent de 0/1')
    if (y<0).any() or ((yfreq==0)&(y!=0)).any() or ((yfreq==1)&(y<=0)).any():
        raise ValueError('Fréquence et coût incohérents : valeurs négatives, zéro ou sinistre incohérent.')
    print(f'Train : {len(train):,} contrats ; {int(yfreq.sum()):,} sinistres ; test : {len(test):,} contrats')
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rmse(y,p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('Prédiction NaN ou infinie')
    return float(np.sqrt(mean_squared_error(np.asarray(y), p)))


def save_json(path, obj):
    def cnv(o):
        if isinstance(o, (np.integer,np.floating)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(str(type(o)))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=cnv), encoding='utf-8')


def make_x_sets(freq, train, test, allow_paiement):
    enriched_tr, enriched_te = freq.make_features(train), freq.make_features(test)
    dtr, dte = {}, {}
    for name in ('base','selection','complet','etendu'):
        xtr, xte = freq.x_for(enriched_tr, name), freq.x_for(enriched_te, name)
        if not allow_paiement:
            xtr = xtr.drop(columns=['paiement'], errors='ignore')
            xte = xte.drop(columns=['paiement'], errors='ignore')
        if list(xtr.columns) != list(xte.columns):
            raise ValueError(f'Colonnes différentes entre Train/Test pour {name}')
        dtr[name], dte[name] = xtr, xte
    return dtr, dte


def model_xgb(freq, family, x, params, threads, seed):
    common = {k:v for k,v in params.items() if k not in
              ('min_frequency','max_categories','objective','tweedie_variance_power','target_transform')}
    if family=='freq':
        return freq.make_model(x, params, threads, seed)
    if family=='light':
        from lightgbm import LGBMRegressor
        p={k:v for k,v in params.items() if k not in ('min_frequency','max_categories')}
        obj=p.pop('objective','tweedie')
        if obj != 'tweedie':
            p.pop('tweedie_variance_power',None)
        model=LGBMRegressor(objective=obj, verbosity=-1, n_jobs=threads,
                            random_state=seed, subsample_freq=1, **p)
        pre=freq.preprocessor(x,params['min_frequency'],params['max_categories'])
        return Pipeline([('preprocess',pre),('reg',model)])
    objective = 'reg:gamma' if family=='gamma' else params.get('objective','reg:tweedie')
    extra = {'tweedie_variance_power':params.get('tweedie_variance_power',1.5)} if objective=='reg:tweedie' else {}
    reg = XGBRegressor(objective=objective, eval_metric='rmse', tree_method='hist',
                       random_state=seed, n_jobs=threads, **extra, **common)
    pre = freq.preprocessor(x, params['min_frequency'], params['max_categories'])
    return Pipeline([('preprocess',pre),('reg',reg)])


def train_predict(freq, family, xtr, target, xva, params, threads, seed, save_model=None):
    model = model_xgb(freq, family, xtr, params, threads, seed)
    if family=='gamma' and np.any(np.asarray(target)<=0):
        raise ValueError('Gamma uniquement sur les coûts de sinistre strictement positifs')
    transform = params.get('target_transform','identity') if family=='direct' else 'identity'
    ytrain=np.asarray(target,dtype=float)
    if transform=='sqrt': ytrain=np.sqrt(np.maximum(ytrain,0))
    elif transform=='log1p': ytrain=np.log1p(np.maximum(ytrain,0))
    model.fit(xtr, ytrain)
    if save_model is not None:
        joblib.dump({'pipeline':model,'family':family,'params':params},save_model)
    if family=='freq':
        out = model.predict_proba(xva)[:,1]
        return np.clip(out, EPS, 1-EPS)
    out = np.asarray(model.predict(xva), dtype=float)
    if transform=='sqrt': out=np.maximum(out,0)**2
    elif transform=='log1p': out=np.expm1(np.clip(out,0,22))
    return np.maximum(out, EPS if family=='gamma' else 0.)


def cb_frame(x):
    x = x.copy()
    cats = x.select_dtypes(exclude='number').columns.tolist()
    for c in cats:
        x[c] = x[c].astype('string').fillna('Inconnu').astype(str)
    for c in x.select_dtypes(include='number').columns:
        x[c] = x[c].astype(float).replace([np.inf, -np.inf], np.nan)
    return x, cats


def train_cb_predict(xtrain, ytrain, xval, p, threads, seed, save_model=None):
    from catboost import CatBoostRegressor
    xtrain, cats = cb_frame(xtrain)
    xval, _ = cb_frame(xval)
    kind = p['loss_kind']
    if kind=='tweedie':
        # CatBoost sur CPU déconseille de trop grandes cibles Tweedie.
        scale = max(1., float(np.quantile(np.asarray(ytrain), .995))/800.)
        loss = f'Tweedie:variance_power={p["tweedie_variance_power"]:.5f}'
    else:
        scale = 1.
        loss = 'RMSE'
    model = CatBoostRegressor(
        iterations=int(p['iterations']), depth=int(p['depth']),
        learning_rate=float(p['learning_rate']), l2_leaf_reg=float(p['l2_leaf_reg']),
        random_strength=float(p['random_strength']),
        border_count=int(p.get('border_count',254)),
        bagging_temperature=float(p.get('bagging_temperature',1.0)),
        rsm=float(p.get('rsm',1.0)), bootstrap_type='Bayesian',
        loss_function=loss, random_seed=seed, thread_count=threads,
        verbose=False, allow_writing_files=False,
        boosting_type='Plain', max_ctr_complexity=int(p.get('max_ctr_complexity',1)),
        one_hot_max_size=int(p.get('one_hot_max_size',10)),
        leaf_estimation_iterations=1,
    )
    model.fit(xtrain, np.asarray(ytrain)/scale, cat_features=cats)
    if save_model is not None:
        joblib.dump({'pipeline':model,'family':'cat','params':p,
                     'scale':scale,'cat_columns':cats},save_model)
    pred = np.maximum(np.asarray(model.predict(xval), dtype=float)*scale,0.)
    return pred


def cv_predict(freq, kind, xsets, y, yfreq, splits, p, feature, threads, seed):
    if isinstance(splits[0], list):
        # Chaque repetiton CV fournit une prediction OOF par contrat.
        # L'average diminue la variance de la repartition en folds.
        return np.mean([cv_predict(freq,kind,xsets,y,yfreq,one,p,feature,threads,seed)
                        for one in splits],axis=0)
    x = xsets[feature]
    result = np.full(len(x), np.nan, dtype=float)
    for k,(tr,va) in enumerate(splits):
        if kind=='gamma':
            positive_tr = tr[yfreq[tr]==1]
            if len(positive_tr)<10: raise ValueError('Trop peu de sinistres pour le fold')
            xx = x.iloc[positive_tr]
            yy = y[positive_tr]
        else:
            xx, yy = x.iloc[tr], (yfreq[tr] if kind=='freq' else y[tr])
        if kind=='cat':
            pred = train_cb_predict(xx, yy, x.iloc[va], p, threads, seed+k)
        else:
            pred = train_predict(freq, kind, xx, yy, x.iloc[va], p, threads, seed+k)
        result[va] = pred
    if not np.isfinite(result).all():
        raise ValueError(f'OOF incomplet pour {kind}/{feature}')
    return result


def sample_params(trial, kind):
    if kind in ('freq','gamma','direct'):
        d = {
            'n_estimators':trial.suggest_int('n_estimators',250,3200,step=50),
            'max_depth':trial.suggest_int('max_depth',1,7),
            'learning_rate':trial.suggest_float('learning_rate',.003,.13,log=True),
            'min_child_weight':trial.suggest_int('min_child_weight',1,100),
            'subsample':trial.suggest_float('subsample',.60,1.),
            'colsample_bytree':trial.suggest_float('colsample_bytree',.60,1.),
            'reg_lambda':trial.suggest_float('reg_lambda',.01,150.,log=True),
            'reg_alpha':trial.suggest_categorical('reg_alpha',[0.,1e-6,1e-4,.001,.05,.2,1.,5.]),
            'gamma':trial.suggest_float('gamma',0.,10.),
            'max_bin':trial.suggest_categorical('max_bin',[128,256,512]),
            'max_delta_step':trial.suggest_categorical('max_delta_step',[0,1,3,6]),
            'min_frequency':trial.suggest_categorical('min_frequency',[5,10,20,40,60,100]),
            'max_categories':trial.suggest_categorical('max_categories',[10,15,30,70,110]),
            'features':trial.suggest_categorical('features',['base','selection','complet','etendu']),
        }
        if kind=='direct':
            d['objective'] = trial.suggest_categorical('objective',['reg:tweedie','reg:squarederror'])
            if d['objective']=='reg:tweedie':
                d['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power',1.05,1.95)
            else:
                d['target_transform'] = trial.suggest_categorical('target_transform',['identity','sqrt','log1p'])
        return d
    if kind=='light':
        d={
            'n_estimators':trial.suggest_int('n_estimators',250,1800,step=50),
            'num_leaves':trial.suggest_int('num_leaves',4,40),
            'max_depth':trial.suggest_int('max_depth',2,8),
            'learning_rate':trial.suggest_float('learning_rate',.003,.13,log=True),
            'min_child_samples':trial.suggest_int('min_child_samples',8,250),
            'min_child_weight':trial.suggest_float('min_child_weight',1e-4,20.,log=True),
            'min_split_gain':trial.suggest_float('min_split_gain',0.,4.),
            'path_smooth':trial.suggest_categorical('path_smooth',[0,1,5,20]),
            'subsample':trial.suggest_float('subsample',.60,1.),
            'colsample_bytree':trial.suggest_float('colsample_bytree',.60,1.),
            'reg_lambda':trial.suggest_float('reg_lambda',.01,150.,log=True),
            'reg_alpha':trial.suggest_float('reg_alpha',0.,10.),
            'min_frequency':trial.suggest_categorical('min_frequency',[5,10,20,40,60,100]),
            'max_categories':trial.suggest_categorical('max_categories',[10,15,30,70,110]),
            'features':trial.suggest_categorical('features',['base','selection','complet','etendu']),
            'objective':trial.suggest_categorical('objective',['tweedie','regression']),
        }
        if d['objective']=='tweedie':
            d['tweedie_variance_power']=trial.suggest_float('tweedie_variance_power',1.15,1.85)
        return d
    if kind=='cat':
        d = {
            'iterations':trial.suggest_int('iterations',250,2400,step=50),
            'depth':trial.suggest_int('depth',2,8),
            'learning_rate':trial.suggest_float('learning_rate',.012,.12,log=True),
            'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1,30,log=True),
            'random_strength':trial.suggest_float('random_strength',.001,10.,log=True),
            'bagging_temperature':trial.suggest_float('bagging_temperature',0.,5.),
            'rsm':trial.suggest_float('rsm',.55,1.),
            'border_count':trial.suggest_categorical('border_count',[32,64,128,254]),
            'one_hot_max_size':trial.suggest_categorical('one_hot_max_size',[2,10,30]),
            'max_ctr_complexity':trial.suggest_categorical('max_ctr_complexity',[1,2]),
            'features':trial.suggest_categorical('features',['base','selection','complet','etendu']),
            'loss_kind':trial.suggest_categorical('loss_kind',['rmse','tweedie'])
        }
        if d['loss_kind']=='tweedie':
            d['tweedie_variance_power']=trial.suggest_float('tweedie_variance_power',1.15,1.85)
        return d
    raise ValueError(kind)


def study_optimize(name, kind, n_trials, p_fixed, xsets, y, yfreq,
                   splits, freq, args, db_path):
    if n_trials<=0: return None
    study = optuna.create_study(
        study_name=name, direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=f'sqlite:///{db_path.resolve().as_posix()}', load_if_exists=True,
    )
    already=len(study.trials)
    remaining=max(0,n_trials-already)
    print(f'\n=== {name}: {already} essais présents ; {remaining} nouveaux essais ===',flush=True)
    def obj(t):
        params = sample_params(t,kind)
        feature = params.pop('features')
        pred = cv_predict(freq,kind,xsets,y,yfreq,splits,params,feature,args.threads,args.seed)
        if kind=='freq': pred = pred*p_fixed
        if kind=='gamma': pred = pred*p_fixed
        # Calage scalaire appris uniquement sur les predictions OOF du developpement.
        # Crucial pour comparer équitablement les objectifs Tweedie, RMSE, sqrt et log1p.
        fact=rmse_scalar_adjustment(y,pred)
        result=rmse(y,pred*fact)
        t.set_user_attr('calage_ooof',fact)
        t.set_user_attr('feature',feature)
        return result
    def callback(st,t):
        if t.value is not None:
            print(f'  {name} essai {t.number+1}/{n_trials}: RMSE={t.value:.5f} | record={st.best_value:.5f}',flush=True)
    if remaining:
        study.optimize(obj,n_trials=remaining,n_jobs=1,callbacks=[callback],
                       show_progress_bar=False, gc_after_trial=True)
    return study


def study_candidate(study):
    if study is None or study.best_trial is None: return None
    p=dict(study.best_params)
    features=p.pop('features')
    return dict(params=p,feature=features,score=study.best_value)


def parse_ensemble_seeds(args):
    if not args.seeds_ensemble.strip():
        return [int(args.seed)]
    ss=[int(x.strip()) for x in args.seeds_ensemble.split(',') if x.strip()]
    if not ss or len(ss)>12 or len(ss)!=len(set(ss)):
        raise ValueError('Utilise entre 1 et 12 seeds distinctes, ex. --seeds-ensemble 42,2026,1337')
    if ss[0] != int(args.seed):
        raise ValueError('La premiere seed de --seeds-ensemble doit correspondre à --seed')
    return ss


def cv_seed_average(freq, kind, cand, xsets, y, yfreq, splits, threads, seeds):
    return np.mean([cv_predict(freq,kind,xsets,y,yfreq,splits,cand['params'],
                               cand['feature'],threads,int(s)) for s in seeds],axis=0)


def fitted_seed_average(freq,kind,cand,xsets,y,yfreq,idx_train,idx_pred,threads,seeds,
                        save_prefix=None):
    preds=[]
    for i,s in enumerate(seeds):
        filename = None
        if save_prefix is not None:
            stem=Path(save_prefix)
            filename=stem.with_name(stem.stem+f'_seed{s}'+stem.suffix)
        preds.append(final_fit_predict(freq,kind,cand,xsets,y,yfreq,idx_train,idx_pred,
                                       threads,int(s),filename))
    return np.mean(preds,axis=0)


def evaluate_candidate(freq,kind,cand,xsets,y,yfreq,splits,threads,seed):
    out=cv_predict(freq,kind,xsets,y,yfreq,splits,cand['params'],cand['feature'],threads,seed)
    return out


def final_fit_predict(freq,kind,cand,xsets,y,yfreq,ix_train,ix_val,threads,seed,save_model=None):
    x=xsets[cand['feature']]
    if kind=='gamma':
        indices=ix_train[yfreq[ix_train]==1]
        xx,yy=x.iloc[indices],y[indices]
    else:
        xx,yy=x.iloc[ix_train],yfreq[ix_train] if kind=='freq' else y[ix_train]
    if kind=='cat':
        return train_cb_predict(xx,yy,x.iloc[ix_val],cand['params'],threads,seed,save_model)
    return train_predict(freq,kind,xx,yy,x.iloc[ix_val],cand['params'],threads,seed,save_model)



def calibrated_oof_rmse(y,pred):
    return rmse(y,pred*rmse_scalar_adjustment(y,pred))


def rmse_scalar_adjustment(y, pred):
    """Calage multiplicatif en RMSE sur OOF, gardé hors du holdout.
    Retourne 1.0 si l'ajustement n'améliore pas réellement la RMSE.
    """
    denom=float(np.dot(pred,pred))
    if denom<=1e-15: return 1.0
    factor=float(np.clip(np.dot(y,pred)/denom,.5,1.5))
    if rmse(y,pred*factor)>=rmse(y,pred)*.99999:
        return 1.0
    return factor

def nonnegative_weights(P,y):
    n=P.shape[1]
    scores=[rmse(y,P[:,k]) for k in range(n)]
    best=int(np.argmin(scores))
    start=np.zeros(n);start[best]=1.
    norm=max(float(np.mean((y-np.mean(y))**2)),1.)
    def loss(w):
        mse=np.mean((y-P@w)**2)/norm
        # faible régularisation vers le meilleur modèle seul : réduit le surajustement.
        return mse+.00015*np.sum((w-start)**2)
    result=minimize(loss,start,method='SLSQP',
                    constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1}],
                    bounds=[(0.,1.)]*n,options={'maxiter':500,'ftol':1e-10})
    weights=np.clip(result.x if result.success else start,0,1)
    weights=weights/weights.sum()
    # Ne retenir le mélange que s'il apporte un gain visible sur OOF.
    if rmse(y,P@weights)>= min(scores)*.9995:
        weights=start
    return weights,scores


def save_plots(df, out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    items=df.sort_values('rmse_oof')
    ax.barh(items['modele'],items['rmse_oof'])
    ax.invert_yaxis()
    ax.set_xlabel('RMSE OOF de la prime (plus faible = mieux)')
    ax.set_title('Comparaison réellement mesurée : mêmes folds, même cible')
    for idx, val in enumerate(items['rmse_oof']):
        ax.text(val,idx,f' {val:.4f}',va='center',fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'comparaison_rmse_oof.png',dpi=180)
    plt.close(fig)


def save_optuna_history(db_path,out):
    import matplotlib.pyplot as plt
    names=['gamma_phase1','freq_rmse','gamma_phase2','direct_tweedie_rmse',
           'lightgbm_direct','catboost_direct']
    fig,ax=plt.subplots(figsize=(10,5))
    plotted=False
    for name in names:
        try:
            study=optuna.load_study(study_name=name,storage=f'sqlite:///{db_path.resolve().as_posix()}')
        except (KeyError,ValueError):
            continue
        vals=[t.value for t in study.trials if t.value is not None and np.isfinite(t.value)]
        if vals:
            ax.plot(range(1,len(vals)+1),np.minimum.accumulate(vals),label=name)
            plotted=True
    if plotted:
        ax.set_xlabel('Essais terminés (pour chaque recherche)')
        ax.set_ylabel('Meilleure RMSE OOF de la prime')
        ax.set_title('Convergence des recherches Optuna (objectifs RMSE)')
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out/'convergence_optimisations_rmse.png',dpi=180)
    plt.close(fig)


def run(args):
    if args.folds<2 or args.threads<1 or args.cv_repeats<1 or args.cv_repeats>5:
        raise ValueError('folds>=2, threads>=1, 1 <= cv-repeats <= 5 requis')
    ensemble_seeds=parse_ensemble_seeds(args)
    print("Seeds de l'ensemble final :", ensemble_seeds, flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    guard_execution(args)
    freq=source_frequency()
    train,test=load_train_test(args)
    if not args.autoriser_paiement:
        print('Prudence anti-fuite : colonne paiement exclue par défaut (sémantique temporelle inconnue).')
    xtr, xte = make_x_sets(freq,train,test,args.autoriser_paiement)
    y=np.asarray(train['montant_sinistre'],dtype=float)
    yfreq=np.asarray(train['nombre_sinistres'],dtype=int)
    dev,hold=freq.holdout_indices(train,args.seed)
    dev=np.asarray(dev,dtype=int); hold=np.asarray(hold,dtype=int)
    if args.cv_repeats==1:
        splits=freq.client_folds(pd.Series(yfreq[dev]),pd.Series(train['id_client'].iloc[dev].to_numpy()),args.folds,args.seed+1)
    else:
        splits=[freq.client_folds(pd.Series(yfreq[dev]),pd.Series(train['id_client'].iloc[dev].to_numpy()),
                                  args.folds,args.seed+1+1009*r) for r in range(args.cv_repeats)]
    xs_dev={name:x.iloc[dev].reset_index(drop=True) for name,x in xtr.items()}
    y_dev,yf_dev=y[dev],yfreq[dev]
    print(f'Développement={len(dev)}, holdout={len(hold)}, folds={args.folds}, répétitions CV={args.cv_repeats}, threads={args.threads}.')
    B=BUDGETS[args.budget]
    baseline=np.full(len(y_dev),np.nan)
    sevmean=np.full(len(y_dev),np.nan)
    fold_groups=splits if isinstance(splits[0],list) else [splits]
    bs=[]
    for fold_group in fold_groups:
        baseline_one=np.full(len(y_dev),np.nan)
        for tr,va in fold_group:
            baseline_one[va]=np.mean(y_dev[tr])
        bs.append(baseline_one)
    baseline=np.mean(bs,axis=0)
    initial_score=rmse(y_dev,baseline)
    print(f'Référence constante RMSE OOF={initial_score:.6f}')

    # Fréquences historiques : comparer sur le critère RMSE (non sur Log Loss).
    fcandidates=[
        ('freq_initiale',dict(params=FREQ_INITIAL,feature='selection')),
        ('freq_V2_46',dict(params=FREQ_V2,feature='etendu')),
        ('freq_V3_30',dict(params=FREQ_V3,feature='selection')),
        ('freq_TPE100_30',dict(params=FREQ_TPE100,feature='selection')),
    ]
    pbest=None; freq_sel=None; best_freq_rmse=np.inf
    g_default=dict(params=SEV_INITIAL,feature='selection')
    print('Calcul de la gravité Gamma initiale et des quatre fréquences historiques...',flush=True)
    m_init=evaluate_candidate(freq,'gamma',g_default,xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
    if not np.isfinite(m_init).all(): raise ValueError('Gravité initiale OOF invalide')
    for name,cand in fcandidates:
        p=evaluate_candidate(freq,'freq',cand,xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
        s=rmse(y_dev,p*m_init)
        print(f'  {name:<20} RMSE OOF avec gravité initiale={s:.6f}',flush=True)
        if s<best_freq_rmse:
            pbest,freq_sel,best_freq_rmse=p,(name,cand),s
    print(f'Référence fréquence sélectionnée : {freq_sel[0]}')

    db=args.output/'etudes_optuna.sqlite3'
    # Optimiser d'abord le coût conditionnel dans le critère final p * m.
    sg=study_optimize('gamma_phase1','gamma',B['gamma'],pbest,xs_dev,y_dev,yf_dev,
                      splits,freq,args,db)
    gcandidates=[('gamma_initiale',g_default)]
    if sg is not None: gcandidates.append(('gamma_optuna',study_candidate(sg)))
    gamma_sel=min(gcandidates,
                  key=lambda it:calibrated_oof_rmse(y_dev,pbest*evaluate_candidate(freq,'gamma',it[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)))
    m_best=evaluate_candidate(freq,'gamma',gamma_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)

    # Optimiser ensuite les probabilités en RMSE de prime (pour la gravité fixe OOF).
    sf=study_optimize('freq_rmse','freq',B['freq'],m_best,xs_dev,y_dev,yf_dev,
                      splits,freq,args,db)
    if sf is not None: fcandidates.append(('freq_RMSE_Optuna',study_candidate(sf)))
    freq_sel=min(fcandidates,
                 key=lambda it:calibrated_oof_rmse(y_dev,evaluate_candidate(freq,'freq',it[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)*m_best))
    pbest=evaluate_candidate(freq,'freq',freq_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
    print(f'Meilleure fréquence pour la prime: {freq_sel[0]}')

    # Deuxième passe possible : réoptimiser la gravité avec la nouvelle fréquence.
    sg2=study_optimize('gamma_phase2','gamma',B['gamma2'],pbest,xs_dev,y_dev,yf_dev,
                       splits,freq,args,db)
    if sg2 is not None: gcandidates.append(('gamma_reoptimisee',study_candidate(sg2)))
    gamma_sel=min(gcandidates,
                  key=lambda it:calibrated_oof_rmse(y_dev,pbest*evaluate_candidate(freq,'gamma',it[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)))
    mbest=evaluate_candidate(freq,'gamma',gamma_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
    p2part=np.maximum(pbest*mbest,0.)
    print(f'Modèle fréquence × gravité : RMSE OOF={rmse(y_dev,p2part):.6f}',flush=True)

    # Modèle de coût global entraîné sur tous les contrats, y compris les zéros.
    d_default=dict(params=DIRECT_INITIAL,feature='selection')
    sd=study_optimize('direct_tweedie_rmse','direct',B['direct'],None,xs_dev,y_dev,yf_dev,
                      splits,freq,args,db)
    dcandidates=[('tweedie_initial',d_default)]
    if sd is not None: dcandidates.append(('direct_optuna',study_candidate(sd)))
    dir_sel=min(dcandidates,
                key=lambda it:calibrated_oof_rmse(y_dev,evaluate_candidate(freq,'direct',it[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)))
    pdirect=evaluate_candidate(freq,'direct',dir_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
    print(f'Coût direct XGBoost : RMSE OOF={rmse(y_dev,pdirect):.6f}',flush=True)

    model_oof={'constante':baseline,'frequence_x_gravite':p2part,'cout_direct_xgb':pdirect}
    light_sel=None
    if not args.sans_lightgbm:
        try:
            import lightgbm  # noqa F401
            sl=study_optimize('lightgbm_direct','light',B['light'],None,xs_dev,y_dev,yf_dev,
                              splits,freq,args,db)
            if sl is not None:
                light_sel=('lightgbm_optuna',study_candidate(sl))
                plight=evaluate_candidate(freq,'light',light_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
                model_oof['cout_direct_lightgbm']=plight
                print(f'Coût direct LightGBM : RMSE OOF={rmse(y_dev,plight):.6f}',flush=True)
        except ImportError:
            print('LightGBM indisponible : modèle LightGBM ignoré.')
    cat_sel=None
    if not args.sans_catboost:
        try:
            import catboost  # noqa F401
            sc=study_optimize('catboost_direct','cat',B['cat'],None,xs_dev,y_dev,yf_dev,
                              splits,freq,args,db)
            if sc is not None:
                cat_sel=('catboost_optuna',study_candidate(sc))
                pcat=evaluate_candidate(freq,'cat',cat_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,args.seed)
                model_oof['cout_direct_catboost']=pcat
                print(f'Coût direct CatBoost : RMSE OOF={rmse(y_dev,pcat):.6f}',flush=True)
        except ImportError:
            print('CatBoost indisponible : modèle CatBoost ignoré.')

    # Ensemble multi-seed: la liste est définie AVANT le holdout.
    # Les hyperparamètres ont été sélectionnés sur le développement uniquement.
    if len(ensemble_seeds)>1:
        print('Construction OOF des ensembles multi-seeds (sans toucher au holdout)...',flush=True)
        pbest=cv_seed_average(freq,'freq',freq_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,ensemble_seeds)
        mbest=cv_seed_average(freq,'gamma',gamma_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,ensemble_seeds)
        model_oof['frequence_x_gravite']=np.maximum(pbest*mbest,0.)
        model_oof['cout_direct_xgb']=cv_seed_average(freq,'direct',dir_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,ensemble_seeds)
        if light_sel is not None:
            model_oof['cout_direct_lightgbm']=cv_seed_average(freq,'light',light_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,ensemble_seeds)
        if cat_sel is not None:
            model_oof['cout_direct_catboost']=cv_seed_average(freq,'cat',cat_sel[1],xs_dev,y_dev,yf_dev,splits,args.threads,ensemble_seeds)
        print('RMSE OOF 2-part après seeds:',rmse(y_dev,model_oof['frequence_x_gravite']))

    # Calibrage multiplicatif de la prime sur les seules prédictions OOF.
    scales={}
    for name in model_oof:
        factor=1. if name=='constante' else rmse_scalar_adjustment(y_dev,model_oof[name])
        scales[name]=factor
        model_oof[name]=model_oof[name]*factor
        if factor!=1.:
            print(f'  Calage OOF du modèle {name}: ×{factor:.5f}')

    names=list(model_oof)
    P=np.column_stack([model_oof[n] for n in names])
    w,individual=nonnegative_weights(P,y_dev)
    pred_oof=P@w
    pred_oof=np.maximum(pred_oof,0.)
    print('\nPoids de mélange OOF : ' + ', '.join(f'{name}: {weight:.4f}' for name,weight in zip(names,w)))
    print(f'RMSE OOF ensemble={rmse(y_dev,pred_oof):.6f} | constante={initial_score:.6f}')
    pd.DataFrame({'INDEX':train['index'].iloc[dev].to_numpy(),'reel':y_dev,
                  'prediction_prime_oof':pred_oof,'prediction_frequence_oof':pbest,
                  'prediction_gravite_oof':mbest,**model_oof}).to_csv(args.output/'predictions_developpement_oof.csv',index=False)

    # Holdout : jamais consulté avant ce point pour guider le choix des hyperparamètres.
    xfull=xtr
    fhold=fitted_seed_average(freq,'freq',freq_sel[1],xfull,y,yfreq,dev,hold,args.threads,ensemble_seeds)
    mhold=fitted_seed_average(freq,'gamma',gamma_sel[1],xfull,y,yfreq,dev,hold,args.threads,ensemble_seeds)
    dhold=fitted_seed_average(freq,'direct',dir_sel[1],xfull,y,yfreq,dev,hold,args.threads,ensemble_seeds)
    hold_preds={'constante':np.full(len(hold),np.mean(y[dev])),
                'frequence_x_gravite':fhold*mhold,'cout_direct_xgb':dhold}
    if light_sel is not None:
        hold_preds['cout_direct_lightgbm']=fitted_seed_average(freq,'light',light_sel[1],xfull,y,yfreq,dev,hold,args.threads,ensemble_seeds)
    if cat_sel is not None:
        hold_preds['cout_direct_catboost']=fitted_seed_average(freq,'cat',cat_sel[1],xfull,y,yfreq,dev,hold,args.threads,ensemble_seeds)
    hold_preds={name:pred*scales[name] for name,pred in hold_preds.items()}
    hold_matrix=np.column_stack([hold_preds[n] for n in names])
    hold_ensemble=np.maximum(hold_matrix@w,0.)
    hold_report=pd.DataFrame([{'modele':name,'rmse_holdout':rmse(y[hold],hold_preds[name])} for name in names] +
                              [{'modele':'ensemble_choisi_avant_holdout','rmse_holdout':rmse(y[hold],hold_ensemble)}])
    hold_report.to_csv(args.output/'evaluations_holdout.csv',index=False)
    pd.DataFrame({'INDEX':train['index'].iloc[hold].to_numpy(), 'reel':y[hold],
                  'prime_predite':hold_ensemble,'frequence_predite':fhold,
                  'gravite_predite':mhold}).to_csv(args.output/'predictions_holdout.csv',index=False)
    print('\nHOLDOUT INDÉPENDANT (aucun réglage après lecture):\n'+hold_report.to_string(index=False))

    # Final : réentraîner sur tous les contrats étiquetés, prédire le test externe.
    all_idx=np.arange(len(train),dtype=int)
    external_x={name:pd.concat([xfull[name],xte[name]],ignore_index=True) for name in xfull}
    ext_idx=np.arange(len(train),len(train)+len(test),dtype=int)
    ftest=fitted_seed_average(freq,'freq',freq_sel[1],external_x,y,yfreq,all_idx,ext_idx,args.threads,ensemble_seeds,args.output/'modele_frequence.joblib')
    mtest=fitted_seed_average(freq,'gamma',gamma_sel[1],external_x,y,yfreq,all_idx,ext_idx,args.threads,ensemble_seeds,args.output/'modele_gravite.joblib')
    dtest=fitted_seed_average(freq,'direct',dir_sel[1],external_x,y,yfreq,all_idx,ext_idx,args.threads,ensemble_seeds,args.output/'modele_cout_direct_xgb.joblib')
    test_preds={'constante':np.full(len(test),np.mean(y)),
                'frequence_x_gravite':ftest*mtest, 'cout_direct_xgb':dtest}
    if light_sel is not None:
        test_preds['cout_direct_lightgbm']=fitted_seed_average(freq,'light',light_sel[1],external_x,y,yfreq,all_idx,ext_idx,args.threads,ensemble_seeds,args.output/'modele_cout_direct_lightgbm.joblib')
    if cat_sel is not None:
        test_preds['cout_direct_catboost']=fitted_seed_average(freq,'cat',cat_sel[1],external_x,y,yfreq,all_idx,ext_idx,args.threads,ensemble_seeds,args.output/'modele_cout_direct_catboost.joblib')
    test_preds={name:pred*scales[name] for name,pred in test_preds.items()}
    final=np.maximum(np.column_stack([test_preds[n] for n in names])@w,0.)
    if not np.isfinite(final).all() or len(final)!=len(test):
        raise ValueError('Soumission non finie ou taille incorrecte')
    pd.DataFrame({'INDEX':test['index'].to_numpy(),'pred':final}).to_csv(args.output/'soumission_prime_pure.csv',index=False)
    pd.DataFrame({'INDEX':test['index'].to_numpy(),'pred':ftest}).to_csv(args.output/'soumission_frequence.csv',index=False)
    pd.DataFrame({'INDEX':test['index'].to_numpy(),'pred':mtest}).to_csv(args.output/'soumission_gravite.csv',index=False)
    results=pd.DataFrame([{'modele':n,'rmse_oof':v} for n,v in zip(names,individual)] +
                         [{'modele':'ensemble','rmse_oof':rmse(y_dev,pred_oof)}])
    results.to_csv(args.output/'comparaison_rmse_oof.csv',index=False)
    save_plots(results,args.output)
    save_optuna_history(db,args.output)
    settings={
        'criterium':'RMSE du montant_sinistre sur TOUS les contrats (zéros inclus)',
        'holdout_client_grouped':True,'folds':args.folds,'seed':args.seed,
        'seeds_ensemble':ensemble_seeds,'budget_optimisation':args.budget,'cv_repeats':args.cv_repeats,
        'developpement_n':len(dev),'holdout_n':len(hold),
        'paiement_conserve':args.autoriser_paiement,
        'freq':{'nom':freq_sel[0],**freq_sel[1]},
        'gamma':{'nom':gamma_sel[0],**gamma_sel[1]},
        'direct':{'nom':dir_sel[0],**dir_sel[1]},
        'catboost':None if cat_sel is None else {'nom':cat_sel[0],**cat_sel[1]},
        'lightgbm':None if light_sel is None else {'nom':light_sel[0],**light_sel[1]},
        'poids_melange':dict(zip(names,w.tolist())),
        'calages_multiplicatifs':scales,
        'rmse_oof_ensemble':rmse(y_dev,pred_oof),
        'rmse_holdout_ensemble':rmse(y[hold],hold_ensemble),
        'fichiers_modeles_par_seed':len(ensemble_seeds),
        'limitations':['Optimisation guidée par le RMSE CV, holdout consulté une fois à la fin',
                       'Un meilleur RMSE CV ne garantit pas une meilleure RMSE Kaggle',
                       'La catégorie paiement est exclue sauf validation de sa disponibilité antérieure au sinistre',
                       'Une seule séparation holdout : variance d\'évaluation possible'],
    }
    save_json(args.output/'rapport_final_rmse.json',settings)
    print(f'\nSoumission créée : {args.output/"soumission_prime_pure.csv"}')
    print('À transmettre : comparaison_rmse_oof.csv, evaluations_holdout.csv et rapport_final_rmse.json.')


if __name__=='__main__':
    try:
        run(args_parser())
    except KeyboardInterrupt:
        print('\nInterrompu. Études Optuna enregistrées en SQLite ; relancer la même commande.',file=sys.stderr)
        raise SystemExit(130)
