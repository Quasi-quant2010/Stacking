# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'

import os, re
import time
import cPickle as cp
import itertools
from model_wrappers import SklearnModel, XgboostModel
from utils.config_utils import Config
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def write_cv_res_csv(cv_out, cv_csv_out):
    cv_out = os.path.join(Config.get_string('data.path'), 'output', cv_out) if cv_out else None

    param_keys, param_vals, scores = cp.load(open(cv_out, 'rb'))
    assert len(param_vals) == len(scores), 'Error: param value list length do not match score list length!'
    assert len(param_keys) == len(param_vals[0]), 'Error: param key count and value count do not match!'
    if isinstance(param_vals[0], dict):
        param_keys = param_vals[0].keys()
        param_vals = [param.values() for param in param_vals]
    f = open(cv_csv_out, 'w')
    for key in param_keys:
        f.write('{0},'.format(key))
    for i in xrange(len(scores[0])):
        f.write('score_{0},'.format(i))
    f.write('score_mean,score_std\n')
    for i, params in enumerate(param_vals):
        for p in params:
            f.write('{0},'.format(p))
        for s in scores[i]:
            f.write('{0},'.format(s))
        f.write('{0},{1}\n'.format(scores[i].mean(), scores[i].std()))
    f.close()



class GridSearch:
    def __init__(self, wrapper_class, experiment, model_param_keys, model_param_vals,
                 cv_folder=None,
                 cv_out=None, cv_pred_out=None, refit_pred_out=None):
        """
        Constructor of grid search.
        Support search on a set of model parameters, and record the cv result of each param configuration.

        :param wrapper_class: model wrapper type string like 'XgboostModel' or 'SklearnModel'
        :param experiment: experiment object of ExperimentL1 at 1-Level or ExperimentL2 at 2-Level
        :param model_param_keys: list of model param keys. eg. ['paramA', 'paramB', 'paramC']
        :param model_param_vals: list of model param values (iterable). eg. [['valAa', 'valAb'], [0.1, 0.2], (1, 2, 3)]
        :param cv_out: Output pickle file name of cross validation score results.
        :param cv_pred_out: prediction of cross validation each fold.
        :param refit_pred_out: refit on full train set and predict on test set.
        :return: (best parameters, best score)
        """

        self.wrapper_class = wrapper_class
        self.experiment = experiment
        self.model_param_keys = model_param_keys
        self.model_param_vals = model_param_vals
        self.str_match = re.compile(r'loss')

        if wrapper_class == SklearnModel:
            self.model_name = model_param_vals[0]
        else:
            self.model_name = 'xgb'
        self.cv_out = os.path.join(Config.get_string('data.path'), cv_folder, cv_out) if cv_out else None
        self.cv_pred_out = os.path.join(Config.get_string('data.path'), cv_folder, cv_pred_out) if cv_pred_out else None
        self.refit_pred_out = os.path.join(Config.get_string('data.path'), cv_folder, refit_pred_out) if refit_pred_out else None


    def search_by_cv(self, validation_metrics = None):
        """
        Search by cross validation.

        :return: best parameter dict
        """

        if validation_metrics is None:
            validation_metrics = 'auc'

        # create dataframe of results
        scores_list = []
        preds_list = []
        param_vals_list = []
        best_param = None
        best_score_mean = None
        best_score_std= None
        for v in itertools.product(*(self.model_param_vals[::-1])):
            param_dic = {}
            for i in xrange(len(self.model_param_keys)):
                param_dic[self.model_param_keys[-(i+1)]] = v[i]
            model = self.wrapper_class(param_dic)
            scores, preds = self.experiment.cross_validation(model)
            scores_list.append(scores[validation_metrics])
            preds_list.append(preds)
            param_vals_list.append(v)
            
            #print "score", scores[validation_metrics].mean()
            #print "param", param_dic

            if self.cv_pred_out:
                cp.dump(preds_list, open(self.cv_pred_out, 'wb'), protocol=2)
            cp.dump((self.model_param_keys, param_vals_list, scores_list), open(self.cv_out, 'wb'), protocol=2)
            if self.str_match.search(validation_metrics):
                # hinge_loss, log_loss
                if best_score_mean is None or best_score_mean > scores[validation_metrics].mean():
                    best_param = param_dic
                    best_score_mean = scores[validation_metrics].mean()
            else:
                # auc, accuracy, precision, recall, f_value
                if best_score_mean is None or best_score_mean < scores[validation_metrics].mean():
                    best_param = param_dic
                    best_score_mean = scores[validation_metrics].mean()
        #print "best_score", best_score_mean
        #print "best_param", best_param
        # do not need
        #if self.refit_pred_out:
        #    self.fit_full_set_and_predict(self.refit_pred_out)
        return best_param, best_score_mean

    def fit_full_set_and_predict(self, refit_pred_out):
        preds_list = []
        for v in itertools.product(*self.model_param_vals[::-1]):
            param_dic = {}
            for i in xrange(len(self.model_param_keys)):
                param_dic[self.model_param_keys[-(i+1)]] = v[i]
            model = self.wrapper_class(param_dic)
            preds = self.experiment.fit_fullset_and_predict(model)
            preds_list.append(preds)
        cp.dump(preds_list, open(refit_pred_out, 'wb'), protocol=2)


    def to_string(self):
        return self.model_name+ '_cv_'


class BayesSearch:
    def __init__(self, wrapper_class, experiment, model_param_keys, model_param_space,
                 cv_out=None, cv_pred_out=None, refit_pred_out=None, dump_round=10, use_lower=0,n_folds=5):
        """
        Constructor of bayes search.
        Support search on a set of model parameters, and record the cv result of each param configuration.

        :param wrapper_class: model wrapper type string like 'XgboostModel' or 'SklearnModel'
        :param experiment: experiment object of ExperimentL1 or ExperimentL2
        :param model_param_keys: list of model param keys. eg. ['paramA', 'paramB', 'paramC']
        :param model_param_space: list of model param space
        :param cv_out: Output pickle file name of cross validation score results.
        :param cv_pred_out: prediction of cross validation each fold.
        :param refit_pred_out: refit on full train set and predict on test set.
        :return: None
        """

        self.wrapper_class = wrapper_class
        self.experiment = experiment
        self.model_param_keys = model_param_keys
        self.model_param_space = model_param_space
        self.integer_params = set()
        self.n_folds = n_folds
        for k, v in model_param_space.iteritems():
            vstr = str(v)
            if vstr.find('quniform') >= 0 \
                    or vstr.find('qloguniform') >= 0\
                    or vstr.find('qnormal') >= 0\
                    or vstr.find('qnormal') >= 0:
            #if v == hp.quniform or v == hp.qlognormal or v == hp.qnormal:
                self.integer_params.add(k)
            pass
        self.param_vals_list = []
        self.preds_list = []
        self.scores_list = []
        self.refit_preds_list = []
        self.model_name = self.wrapper_class.__name__

        self.cv_out = os.path.join(Config.get_string('data.path'), 'output', cv_out) if cv_out else None
        self.cv_pred_out = os.path.join(Config.get_string('data.path'), 'output', cv_pred_out) if cv_pred_out else None
        self.refit_pred_out = os.path.join(Config.get_string('data.path'), 'output', refit_pred_out) if refit_pred_out else None

        self.eval_round = 0
        self.dump_round = dump_round
        self.trials = Trials()
        self.use_lower=use_lower
        pass

    def objective(self, param_dic):
        if self.eval_round > 0 and self.eval_round % self.dump_round == 0:
            self.dump_result()
        self.eval_round += 1
        for k in param_dic:
            if k in self.integer_params:
                param_dic[k] = int(param_dic[k])
        print param_dic

        model = self.wrapper_class(param_dic)
        scores, preds = self.experiment.cross_validation(model,n_folds=self.n_folds)
        if self.refit_pred_out:
            model = self.wrapper_class(param_dic)
            refit_pred = self.experiment.fit_fullset_and_predict(model)
            self.refit_preds_list.append(refit_pred)
        self.param_vals_list.append(param_dic)
        self.scores_list.append(scores)
        self.preds_list.append(preds)
        loss = -scores.mean()
        if self.use_lower:
            loss += scores.std()
        return {
            'loss': loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            # -- attachments are handled differently
        }

    def dump_result(self):
        if self.cv_pred_out:
            cp.dump(self.preds_list, open(self.cv_pred_out, 'wb'), protocol=2)
        if self.cv_out:
            cp.dump((self.model_param_keys, self.param_vals_list, self.scores_list), open(self.cv_out, 'wb'), protocol=2)
        if self.refit_pred_out:
            cp.dump(self.refit_preds_list, open(self.refit_pred_out, 'wb'), protocol=2)
           #self.fit_full_set_and_predict(self.refit_pred_out)
        pass

    def search_by_cv(self, max_evals=201):
        best = fmin(self.objective, space=self.model_param_space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials)
        print 'Best Param:'
        print best
        return best

    def fit_full_set_and_predict(self, refit_pred_out, topK=None):
        trials = sorted(self.trials.trials[:-1], key = lambda x: x['result']['loss'])
        if topK:
            trials = trials[:topK]
        idxs = [t['tid'] for t in trials]
        refit_preds_list = []
        # fit on whole training set and predict
        for i in idxs:
            model = self.wrapper_class(self.param_vals_list[i])
            preds = self.experiment.fit_fullset_and_predict(model)
            refit_preds_list.append(preds)
        if topK:
            cp.dump((self.model_param_keys, self.preds_list[idxs], refit_preds_list),
                    open(refit_pred_out, 'wb'), protocol=2)
        else:
            cp.dump(refit_preds_list, open(refit_pred_out, 'wb'), protocol=2)
        pass
