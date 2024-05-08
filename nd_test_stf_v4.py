from __future__ import print_function, division

if __name__ == '__main__':

    import os
    import sys
    import zipfile
    import time
    import random
    import itertools
    import math
    import pandas as pd
    import numpy as np
    from scipy import stats, linalg
    from scipy.stats import randint as sp_randint
    import scipy.sparse
    import pickle
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pylab as pl
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer, \
        OneHotEncoder,PolynomialFeatures,Normalizer
    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression,RidgeClassifier,\
        Perceptron,PassiveAggressiveClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM
    from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.ensemble import GradientBoostingClassifier, RandomTreesEmbedding,\
        RandomForestClassifier, VotingClassifier, IsolationForest,\
        AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
    from sklearn.neural_network import BernoulliRBM, MLPClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF,DotProduct,\
        Matern,RationalQuadratic,ExpSineSquared,ConstantKernel as CK,\
        CompoundKernel,PairwiseKernel,WhiteKernel,Product,Exponentiation,Sum
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,\
        LinearDiscriminantAnalysis
    from sklearn.kernel_approximation import RBFSampler,AdditiveChi2Sampler,\
        Nystroem,SkewedChi2Sampler
    from sklearn import mixture
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import KMeans,MiniBatchKMeans,FeatureAgglomeration,\
        DBSCAN,MeanShift,estimate_bandwidth,AgglomerativeClustering

    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold,\
        train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
    from sklearn.metrics.pairwise import chi2_kernel,laplacian_kernel,additive_chi2_kernel
    from sklearn import metrics
    from sklearn.metrics import classification_report,confusion_matrix,\
        precision_score, f1_score, accuracy_score, recall_score,\
        roc_auc_score,roc_curve, auc, consensus_score, brier_score_loss, log_loss
    from sklearn.decomposition import PCA, KernelPCA, SparsePCA, NMF, FactorAnalysis, \
        FastICA, MiniBatchDictionaryLearning, TruncatedSVD, LatentDirichletAllocation
    from sklearn.cross_decomposition import PLSCanonical,PLSSVD,PLSRegression,CCA
    from sklearn import random_projection, manifold
    from sklearn.manifold import TSNE
    from sklearn.covariance import ShrunkCovariance, LedoitWolf,\
        EmpiricalCovariance, MinCovDet, EllipticEnvelope
    from sklearn.feature_selection import VarianceThreshold, SelectPercentile, \
        f_classif, RFECV, RFE, SelectKBest, chi2, mutual_info_classif, SelectFromModel, \
        SelectFwe
    from sklearn.feature_extraction.image import grid_to_graph
    from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE, \
        BorderlineSMOTE, SVMSMOTE, SMOTENC
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion, make_union
#    from sklearn.datasets import dump_svmlight_file
    from sklearn.calibration import CalibratedClassifierCV
    import scipy.signal


    import tensorflow as tf
    import keras
    from keras import backend as K, metrics as krmetrics, regularizers, objectives
    import keras.backend.tensorflow_backend as ktf
    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Activation, Dropout, ZeroPadding2D, \
        advanced_activations as AC, Reshape, Flatten, Embedding, \
        Conv2D, Conv1D, GlobalMaxPooling2D, GlobalMaxPooling1D, \
        MaxPooling2D, MaxPooling1D, AveragePooling2D, GlobalAveragePooling1D,\
        LocallyConnected1D, AveragePooling1D, UpSampling2D,\
        BatchNormalization, Lambda, Layer, Conv2DTranspose, \
        LSTM, GRU, TimeDistributed, SimpleRNN, ConvLSTM2D, \
        merge, Permute, RepeatVector, Cropping1D, Cropping2D, Add, \
        SeparableConv2D, LocallyConnected2D, Multiply, Concatenate, \
        SeparableConv1D, CuDNNLSTM, CuDNNGRU, GlobalAveragePooling2D, \
        ZeroPadding1D
    from keras.layers.wrappers import Bidirectional
    from keras.preprocessing.image import ImageDataGenerator,\
        img_to_array,array_to_img,load_img
    from keras.regularizers import l2,l1
    from keras import initializers
    from keras.layers.noise import GaussianNoise, GaussianDropout
    from keras.optimizers import RMSprop, SGD, Adam
    from keras.utils.np_utils import to_categorical
    from keras.utils import multi_gpu_model
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.preprocessing import sequence
    from keras.callbacks import TensorBoard,ModelCheckpoint,\
        LearningRateScheduler,ReduceLROnPlateau,EarlyStopping
    from keras.utils import plot_model,np_utils
    from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
    from keras.layers.advanced_activations import LeakyReLU
    from keras import utils
    from keras import activations
    import keras.utils.np_utils as kutils
    import json
    import argparse
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss
    from keras.models import load_model,save_model,clone_model
    import skimage.transform
    from keras_contrib.callbacks import CyclicLR


    def get_session(devices="0",gpu_fraction=0.25):
        np.random.seed(random_state_tf)
#        set_random_seed(random_state_tf)
        tf.set_random_seed(random_state_tf)

#        server = tf.train.Server.create_local_server()
        #"", "0,1", "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
#        intra_threads = os.environ.get('OMP_NUM_THREADS')
#        intra_threads = 1
#        inter_threads = 1

        sess = tf.Session(
#                            server.target,
                            config=tf.ConfigProto(
                                    gpu_options=gpu_options,
#                                    intra_op_parallelism_threads=intra_threads,
#                                    inter_op_parallelism_threads=inter_threads,
#                                    allow_soft_placement=True,
#                                    log_device_placement=True
                            )
                        )
#        sess = tf.Session()
        return sess

    class CustomModelCheckpoint(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            if in_cv:
                self.ts_type = "cv"+str(n_cv)
            else:
                self.ts_type = "blind%s"%conf_permtest
            self.rocp = []
            self.f1 = []
            self.val_acc = []
            self.rfa = []
            self.df_hist = pd.DataFrame(
                    columns=["epoch","loss","acc","val_loss","val_acc",
                             "rocp","f1","rfa"])
            self.idx = 0
            self.best_epoch = 1
#            if not ('rocp' in self.params['metrics']):
#                self.params['metrics'].append('rocp')
#            if not ('f1' in self.params['metrics']):
#                self.params['metrics'].append('f1')
#            if not ('rfa' in self.params['metrics']):
#                self.params['metrics'].append('rfa')
        def on_train_end(self, logs={}):
            print("\n======================BEST:======================")
#            best_idx = self.df_hist[metric_save].argmax()
##            print(best_idx)
#            print(self.df_hist.iloc[best_idx:best_idx+1,:])
            best_idx = self.best_epoch-1
#            print(best_idx)
            print(self.df_hist.iloc[best_idx:best_idx+1,:])
            if verbose_callback:
                print(self.df_hist)
                print(self.df_hist.describe())
            print("")

        def on_epoch_end(self, epoch, logs={}):
            if in_cv:
                x_val = x_test_dl
                y_val = y_test_dl[:,watch_cls]
            else:
                x_val = x_blind_dl
                y_val = y_blind_dl[:,watch_cls]
            label_out_pred_prob_d = model.predict(x_val)
            label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
            label_out_pred_m = label_out_pred_d
            label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
            label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]
            rocp = roc_auc_score(y_val, label_out_pred_prob)
            f1 = f1_score(y_val, label_out_pred)
            val_acc = logs.get("val_accuracy") #acc accuracy
            acc = logs.get("accuracy") #acc accuracy
            #5/3/2
            rfa = rocp*w_rocp+f1*w_f1+val_acc*w_acc
            self.df_hist.loc[self.idx]=[epoch+1,logs.get("loss"),acc,
                            logs.get("val_loss"),val_acc,rocp,f1,rfa]
            self.idx+=1
            if verbose_callback:
                print("======================EPOCH %06d:======================"%(epoch+1))
                print(self.df_hist.iloc[self.idx-1:self.idx,:])
#            logs["rocp"] = rocp
#            logs["f1"] = f1
#            logs["rfa"] = rfa
            need_add = False
            if len(self.rfa)==0:
                need_add = True
            elif rfa>np.max(self.rfa):
                need_add = True
            if need_add:
                self.rfa.append(rfa)
                if not verbose_callback:
                    print("======================EPOCH %06d:======================"%(epoch+1))
                    print(self.df_hist.iloc[self.idx-1:self.idx,:])
                print("RFA :", rfa)
                if save_histweights:
                    model.save_weights(filepath_custom % (
                            conf_tags_dl,self.ts_type,batch_size_cv,
                            epoch,logs.get("loss"),acc,
                            logs.get("val_loss"),val_acc,
                            rocp,f1,rfa #roc,rocp,f1
                            ))
                if metric_save=="rfa":
                    self.best_epoch = epoch+1
                    model.save_weights(save_dir+"/"+conf_tags_dl+"_%s.h5"%self.ts_type)
#            need_add = False
#            if len(self.rocp)==0:
#                need_add = True
#            elif rocp>np.max(self.rocp):
#                need_add = True
#            if need_add:
#                self.rocp.append(rocp)
#                if not verbose_callback:
#                    print("======================EPOCH %06d:======================"%(epoch+1))
#                    print(self.df_hist.iloc[self.idx-1:self.idx,:])
#                print("ROCP:", rocp)
#                if save_histweights:
#                    model.save_weights(filepath_custom % (
#                            conf_tags_dl,self.ts_type,batch_size_cv,
#                            epoch,logs.get("loss"),acc,
#                            logs.get("val_loss"),val_acc,
#                            rocp,f1,rfa #roc,rocp,f1
#                            ))
#                if metric_save=="rocp":
#                    self.best_epoch = epoch+1
#                    model.save_weights(save_dir+"/"+conf_tags_dl+"_%s.h5"%self.ts_type)
#            need_add = False
#            if len(self.f1)==0:
#                need_add = True
#            elif f1>np.max(self.f1):
#                need_add = True
#            if need_add:
#                self.f1.append(f1)
#                if not verbose_callback:
#                    print("======================EPOCH %06d:======================"%(epoch+1))
#                    print(self.df_hist.iloc[self.idx-1:self.idx,:])
#                print("F1  :", f1)
#                if save_histweights:
#                    model.save_weights(filepath_custom % (
#                            conf_tags_dl,self.ts_type,batch_size_cv,
#                            epoch,logs.get("loss"),acc,
#                            logs.get("val_loss"),val_acc,
#                            rocp,f1,rfa #roc,rocp,f1
#                            ))
#                if metric_save=="f1":
#                    self.best_epoch = epoch+1
#                    model.save_weights(save_dir+"/"+conf_tags_dl+"_%s.h5"%self.ts_type)
#            need_add = False
#            if len(self.val_acc)==0:
#                need_add = True
#            elif val_acc>np.max(self.val_acc):
#                need_add = True
#            if need_add:
#                self.val_acc.append(val_acc)
#                if not verbose_callback:
#                    print("======================EPOCH %06d:======================"%(epoch+1))
#                    print(self.df_hist.iloc[self.idx-1:self.idx,:])
#                print("VACC:", val_acc)
#                if save_histweights:
#                    model.save_weights(filepath_custom % (
#                            conf_tags_dl,self.ts_type,batch_size_cv,
#                            epoch,logs.get("loss"),acc,
#                            logs.get("val_loss"),val_acc,
#                            rocp,f1,rfa #roc,rocp,f1
#                            ))
#                if metric_save=="val_acc":
#                    self.best_epoch = epoch+1
#                    model.save_weights(save_dir+"/"+conf_tags_dl+"_%s.h5"%self.ts_type)
            print("")

    def lr_log(epoch):
        lr = model.optimizer.get_config()["learning_rate"] #lr learning_ate
        print('Learning rate: %0.8f'%lr)
        return lr

    def lr_schedule(epoch):
        if epoch < int(epochs*lr_factors[0]):
            lr = lr_init
            return lr
        if epoch < int(epochs*lr_factors[1]):
            lr = lr_init*0.1
            return lr
        if epoch < int(epochs*lr_factors[2]):
            lr = lr_init*0.01
            return lr
        if epoch < int(epochs*lr_factors[3]):
            lr = lr_init*0.001
            return lr
        lr = lr_init*0.0001
        return lr


    def clf_sigmoid(x):
        raw = 1.0/(1+np.exp(-x*1.0))
        norm = preprocessing.normalize(raw,norm="l1")
        return norm


##=============================================================
    pd.set_option('display.width', 1600)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.max_columns', 20)

    #===================================
    random_state_tf=2100

    tf_sess = get_session(devices="0",gpu_fraction=0.9)
    ktf.set_session(tf_sess)

    #======================================================================
    indata_dir = "./CS800"
    indata_path_train = indata_dir+"/df_train.csv"
    indata_path_test = indata_dir+"/df_test.csv"

    num_classes = 2
    watch_cls=1 #0 1

    img_rows, img_cols = 32, 32
    channels = 1

    fs_core=881 #881
    fs_core_idxs=np.arange(fs_core).tolist()
    fs_ext=24 #24
    fs_ext_idxs=(np.arange(fs_ext)+fs_core).tolist()
    fs_num=fs_core+fs_ext #905

    ##fs test
    df_train = pd.read_csv(indata_path_train)
    cols_core=np.array(df_train.columns[:fs_core],dtype=str)
    # df_fs_all=pd.DataFrame()
    # df_fs_all["fs_names"]=cols_core
    # df_fs_sel=pd.read_csv(indata_dir+"/fs20.csv")
    # fs_names_sel=np.array(df_fs_sel.fs_name.values,dtype=np.str)
    # df_fs_sel=df_fs_all[df_fs_all.fs_names.isin(fs_names_sel)]
    # fs_core_idxs_sel=np.array(df_fs_sel.index.values,dtype=np.int64).tolist()

    #======================================================================
    ## b51s2100v10s2100-stf_dn0mm10_osbfv2b
    conf_ds = "b51s2100v10s2100-stf_dn0mm10"
    conf_tags=conf_ds

    save_ds = False #True False

    #===================================
    cv_as_all=False
    n_cv_splits=10
    cv_seed=2100
    cv_partial=False #False True
    n_cv_partial=int(190 *n_cv_splits/(n_cv_splits-1))
    cv_partial_seed=1000

    # fs_core_idxs=fs_core_idxs_sel
    fs_idxs=fs_core_idxs+fs_ext_idxs
    # fs_idxs=fs_core_idxs
    # fs_idxs=fs_ext_idxs

    #===================================
    minmax_scale=True
    minmax_scale_factor=10.0 #1.0 2.0* 4** 10***

    #===================================
    conf_osbfb = "_osbfv2b" #_osbfv2b*
    conf_tags += conf_osbfb

    train_ros = True #True* False
    batch_fix = True #False True*
    batch_fix_v2 = True #False?+ True v2ba- v2+? v1ba+? v1=-?
    batch_fix_balance = True #False? True+
    osbfb_seed = 2100

    #======================================================================
    ## mlite-d10_b32e200clr2lr02-85frpt stf st24 st(d25)
    m_dl = "mlite"
    conf_dl = "_"+m_dl+"-d25_b32e200clr2lr01-85frpt" #mlcnew/new2/3** mglr5/6/3* -dl/dlnew/sk/x/xk/dxk
    conf_dl += "-t01" #-v190s10 -tn
    conf_tags_dl=conf_tags+conf_dl

    enable_dl = True #True False
    need_training_cv = True #True False
    cv_using_selflast = False #True False
    cv_using_selfbest = True #True False
    need_training_bl = False #True False
    bl_using_selflast = False #True False
    bl_using_selfbest = False #True False
    bl_using_cvbests = True #True False
    tb_enabled = False #True False

    if enable_dl:
        conf_tags+=conf_dl

    #===================================
    batch_size_cv = 32 #32*001e800 16*0005e400
    epochs_cv = 200 #200*clr1*-2 400*clr2*-4
    batch_size = batch_size_cv
    epochs = epochs_cv

    metric_save = "rfa" #val_acc rocp f1 rfa
    w_rocp = 0.3 #0.315 0.3**
    w_f1 = 0.35 #0.33 0.35**
    w_acc = 0.35 #0.355 0.35**
    w_merge_exp_factor = 0.5 #25 1 5 10 0.5**

    opt_name = "adam" #0.0002/0.5 0.001/0.75/0.9 0.0005
    lr_init = 0.001 #0.001*/0005adam 0.0002sgd
    beta_1=0.75 #0.75

    use_clr = True #True* False
    if not use_clr:
        lr_factors = [0.4, 0.6, 0.8, 0.9] #lrv1*
#        lr_factors = [0.7, 0.8, 0.9, 0.95] #lrv2

#        metric_lr_reduce = metric_save #val_acc rocp f1 rfa val_loss
#        lr_reduce_patience = 12 #100/2/4
    else:
        step_mode='triangular' #triangular triangular2 exp_range
        epochs_clr_cv = epochs_cv/2 *1/2 #/2* 1**? 4-
        epochs_clr = epochs/2 *1/2 #/2* 1**? 4-
        base_lr=1e-7 #0.0000001
#        step_lr = 0.0001
#        max_lr_cv=step_lr*epochs_clr_cv #0004*noaug 0005*aug 00033* 0003
#        max_lr=step_lr*epochs_clr #0004*noaug 0005*aug 00033* 0003
        max_lr_cv=0.01 #0.001*c1? 02**c1? 0005c2
        max_lr=max_lr_cv #0.001*c1? 02**c1? 0005c2

    data_augmentation = True #False* True
    mix_enable=False #False* True?
    mix_alpha=0.2 #0.2? 0.5?
    mix_datagen=False #False* True?

    show_model = True #True False
    verbose_dl = 2 #0 2 1

    #======================================================================
    conf_xk = "_l1" #_l1 _svc _rf
    if enable_dl:
        conf_xk+="d50"

    enable_sk = False #True False
    w_dl = 0.50 #0.33 0.5*+? 0.75* 0.9*?

    if enable_sk:
        conf_tags+=conf_xk

    #======================================================================
    conf_extra = "" #_test
    conf_tags+=conf_extra

    print("\n===========================================================================\n")
    print(conf_tags)
    print("\n===========================================================================\n")

    #================================================
    permtest_seed=None #for permtest only
    conf_permtest="" #for permtest only

    #======================================================================
    #os.chdir(b"C:\HDC\ND") / os.getcwd()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'saved_models_st/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if save_ds:
        outdata_dir = os.path.join(save_dir, conf_ds)
        if not os.path.isdir(outdata_dir):
            os.makedirs(outdata_dir)

    log_dir = os.path.join(save_dir, conf_tags)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    #===================================
    callbacks_dl = []

    if not use_clr:
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks_dl.append(lr_scheduler)

#        callbacks_dl.append(
#            ReduceLROnPlateau(monitor=metric_lr_reduce,factor=0.5, #0.5/0.55 val_loss val_acc
#                              patience=lr_reduce_patience / 2,
#                              cooldown=lr_reduce_patience / 4, verbose=1))
    else:
        callbacks_dl.append(None)

    verbose_callback = False #True False
    save_histweights = False #True False
    model_name_custom = '%s_%s-b%03d.e%03d-l%0.3fd-a%0.3f-tl%0.3f-ta%0.3f-trp%0.3f-tf%0.3f-tw%0.3f.h5'
    filepath_custom = os.path.join(save_dir, model_name_custom)
    custom_checkpoint = CustomModelCheckpoint()
    callbacks_dl.append(custom_checkpoint)

#    metric_lr_es = metric_save #val_acc rocp f1 rfa val_loss
#    early_stopping_patience = 12 #100/2/4
#    callbacks_dl.append(
#        EarlyStopping(monitor=metric_lr_es,minlr=1e-6,
#                      patience=early_stopping_patience, verbose=1))

    lr_logger = LearningRateScheduler(lr_log)
    callbacks_dl += [lr_logger]


    #===================================
    if tb_enabled:
        tb_dir = os.path.join(save_dir, conf_tags+"_tb")
        if not os.path.isdir(tb_dir):
            os.mkdir(tb_dir)
        tb_callback = keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 write_graph=False, #模型结构图 True
                                                 batch_size=batch_size, #直方图计算 batch_size 50
                                                 histogram_freq=epochs//10, #参数和激活直方图，要有测试集 0 2
                                                 write_grads=True, #False True?
                                                 write_images=False, #模型参数做为图片 False
                                                 update_freq='epoch', #epoch,batch,样本数
                                                 embeddings_freq=0, #0 5
                                                 embeddings_layer_names=None, #None ["xxx"]
                                                 embeddings_data=None, #None xxx
                                                 )
        callbacks_dl+=[tb_callback]
        if sys.platform.startswith("win"):
            os.system("start tensorboard --logdir %s"%tb_dir)
        else:
            os.system("setsid tensorboard --logdir %s >%s/stnew-tblog.txt 2>&1 &"%(tb_dir,save_dir))
##===================================================================


##===================================================================
    act_cnn = "AC.LeakyReLU()" #**0.3/0.2
#    act_cnn = "Activation(\"elu\")" #***
#    act_cnn = "Activation(\"relu\")" #*
#    act_cnn_fn = "elu"
#    act_cnn_fn = "relu"

    act_ds = "AC.LeakyReLU()" #**0.3/0.2
#    act_ds = "Activation(\"elu\")" #***
#    act_ds = "Activation(\"relu\")" #*
#    act_ds_fn = "elu"
#    act_ds_fn = "relu"

#    k_init = 'glorot_normal' #*
#    k_init = 'glorot_uniform'
#    k_init = 'he_normal'
    k_init = 'he_uniform' #**

    k_regu = None
    a_regu = None

    #=======================================
    global model_showed
    model_showed = False

    def create_dlmodel(m_name=None):
        np.random.seed(random_state_tf)
#        set_random_seed(random_state_tf)
        tf.set_random_seed(random_state_tf)

        #==========================================
        if m_name is None:
            m_name = m_dl

        if opt_name=="adam":
            optimizer = Adam(lr=lr_init, beta_1=beta_1) #0.0002/0.5 0.001/0.75/0.9
        else:
            optimizer = keras.optimizers.Adadelta()

        losses_used="categorical"
#        losses_used="sparse"
        metrics_show = ['accuracy'] #accuracy
        #==========================================

        #==========================================
        if m_name=="mlite":
            n_filters_nl = 96 #noad*+ a* 96**
            drop_rate = 0.25 #0.10**all/ext(25??) 0.25**core 10*/12*/fs16/25+40/49* 15fs25,40 16/fs49
##            dense_drop_rate = 0.25 #--

            x_in = Input(shape=(img_rows,img_cols,1))

            x = Reshape((-1,1))(x_in)
            x = LocallyConnected1D(18, 32, padding="valid", strides=32)(x)
            x = Reshape((-1,1))(x)
            x = LocallyConnected1D(9, 18, padding="valid", strides=18)(x)
            x = Reshape((-1,1))(x)
            x = LocallyConnected1D(5, 9, padding="valid", strides=9)(x)
            x = Reshape((-1,1))(x)
            x = LocallyConnected1D(3, 5, padding="valid", strides=5)(x)
            x = Flatten()(x)

            x0 = x
            x0d = Dropout(drop_rate)(x0)
            x0t = Dense(n_filters_nl,activation="tanh")(x0)
            x0s = Dense(n_filters_nl,activation="sigmoid")(x0)
            x0l = Dense(n_filters_nl,activation="linear")(x0)
            xm = Multiply()([x0,x0l])
            xm = Dropout(drop_rate)(xm)
            xts = Multiply()([x0t,x0s])
            xts = Dropout(drop_rate)(xts)
            x = Add()([x0d,xm,xts])

#            x = Activation('relu')(x) #++
##            x = Dropout(dense_drop_rate)(x)

            prediction = Dense(num_classes,
                       kernel_initializer=k_init,
                       activation="softmax"
                      )(x)
            model = Model(x_in, prediction)
        #==========================================

        global model_showed
        if show_model and not model_showed:
            model.summary()
            plot_model(model, to_file=save_dir+"/model-%s.png"%conf_tags_dl,
                       show_layer_names=True,show_shapes=True)
            model_showed = True

        if losses_used=="sparse":
            model.compile(optimizer=optimizer,
                          loss=keras.losses.sparse_categorical_crossentropy,
                          metrics=metrics_show)
        if losses_used=="categorical":
            model.compile(optimizer=optimizer,
#                          loss=xcustom_loss_batch,
                          loss='categorical_crossentropy',
                          metrics=metrics_show)
        return model

#    model = create_dlmodel()
    model = None


##===================================================================
    crop_prob = -0.01 #-0.01* 1.01 0.33 0.8 0.5
    crop_factor = 0.80 #0.5 0.8 0.2 0.7/6
    crop_factor_ub = 1.0 #1 0.9
    crop_fill_mode = "constant" #constant reflect**
    crop_strict=False #False True
    crop_preserve_range=False #False True
    crop_anti_aliasing=False #False True

    from random_eraser import get_random_eraser
    random_eraser = get_random_eraser(p=0.85, #0.5* 0.75** 0.85***
                                      #minimum / maximum proportion of erased area
                                      s_l=0.02, s_h=0.4, #s_l=0.02, s_h=0.4*
                                      #minimum / maximum aspect ratio of erased area
                                      r_1=0.3, r_2=1/0.3, #r_1=0.3, r_2=1/0.3*
                                      #minimum / maximum value for erased area
                                      v_l=-1.0*minmax_scale_factor/1.0, #-same? dr* fr**
                                      v_h= 1.0*minmax_scale_factor/1.0,
                                      pixel_level=True) #False* True**

    zoom_fill_mode = "constant"  #nearest* reflect** constant** wrap
    zoom_fill_val = -1*minmax_scale_factor if minmax_scale else 0

    def random_crop_image(image):
        skip_crop = np.random.random()
        if skip_crop<=(1-crop_prob):
            image_crop = image
        else:
            random_array = np.random.random(size=4)
            height,width,channels = image.shape
            if not crop_strict:
                w = int(width*(crop_factor+random_array[0]*(crop_factor_ub-crop_factor)))
    #            h = int(height*(crop_factor+random_array[1]*(crop_factor_ub-crop_factor)))
            else:
                w = int((width*crop_factor)*(1+random_array[0]*(1-crop_factor)))
    #            h = int((height*crop_factor)*(1+random_array[1]*(1-crop_factor)))
            h = w
            x = int(random_array[2]*(width-w))
            y = int(random_array[3]*(height-h))
    #        print(x,w,y,h)
            image_crop = image[y:h+y,x:w+x,0:channels]
            image_crop = skimage.transform.resize(image_crop,image.shape,
                                                  preserve_range=crop_preserve_range,
                                                  anti_aliasing=crop_anti_aliasing,
                                                  mode=crop_fill_mode,
                                                  )
    #        image_crop = image_crop.astype('float32')
        image_crop=random_eraser(image_crop)
        return image_crop

    datagen = ImageDataGenerator(
        featurewise_center=False,  #True False* subtract_pixel_mean set input mean to 0 over the dataset
        samplewise_center=False,  #True False* set each sample mean to 0
        featurewise_std_normalization=False,  #True False* subtract_pixel_mean divide inputs by std of the dataset
        samplewise_std_normalization=False,  #True False* divide each input by its std
        zca_whitening=False,  #True False* apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #    rotation_range=15,  #10*/15**/30/45 0++ randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
    #    width_shift_range=1/32, #5/32** 2++/3/32*,0.1*
        # randomly shift images vertically (fraction of total height)
    #    height_shift_range=1/32, #5/32** 2++/3/32*,0.1*
    #    shear_range=15,  #0** 0.1 set range for random shear
    #    zoom_range=[0.8,1.67],  #0** 0.1 [0.8,1.67]/[0.25/33,2.5/2.2]/[0.67,2.0] set range for random zoom
    #    horizontal_flip=True,  #True** False++ randomly flip images
    #    vertical_flip=True,  #False** randomly flip images
        # set mode for filling points outside the input boundaries
        fill_mode=zoom_fill_mode, #nearest reflect** constant wrap
        cval=zoom_fill_val,  #-1.0 0/std value used for fill_mode = "constant"
        brightness_range=None, #[0,25]
        channel_shift_range=0.0,  # set range for random channel shifts
        # set function that will be applied on each input
        preprocessing_function=random_crop_image, #random_crop_image None
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # image data format, either "channels_first" or "channels_last"
        data_format='channels_last',
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
        dtype=np.float32)
##===================================================================


##=============================================================

    random_state = 2100

    l1lg=linear_model.LogisticRegression(dual=False,C=1.0,penalty='l1',n_jobs=1,#tol=1e-5,
                                         max_iter=5000,solver="liblinear",random_state=random_state)

    rf = RandomForestClassifier(n_estimators=150, criterion='gini',
                           max_depth=5, min_samples_split=5, oob_score=False,
                           random_state=random_state)

    svc=SVC(kernel="rbf",gamma=0.002,C=15.0,probability=True,random_state=random_state)


    vc=VotingClassifier(estimators=[
                                        ('l1lg', l1lg),
                                        # ('svc', svc),
                                        # ('rf',  rf),
                                   ],
                            voting='soft') #soft hard
    ml=vc

##=============================================================


##=============================================================

    def load_hdxdata(fs_idxs, #
                     minmax_scale=True, #True False
                     ):
        df_train = pd.read_csv(indata_path_train)
        df_test = pd.read_csv(indata_path_test)

        cols_org=np.array(df_train.columns[:fs_num],dtype=str)
        cols_zero=np.array(["NULL%05d"%(i+1)\
            for i in range(img_rows*img_cols-fs_num)],dtype=str)
        cols_new=np.concatenate([cols_org,cols_zero])

        df_train=df_train.reset_index(drop=True)
        df_test=df_test.reset_index(drop=True)
        df_train_org=df_train.copy()
        df_test_org=df_test.copy()

        fs_zero=list(set(np.arange(fs_num).tolist())-set(fs_idxs))
        fs_zero.sort()

        x_train_org=np.array(df_train.iloc[:,:fs_num].copy().values,dtype=np.float32)
        y_train=np.array(df_train["label"].values,dtype=np.int32)
        data_zeros=np.zeros([df_train.shape[0],
                             img_rows*img_cols-fs_num],dtype=np.float32)
        x_train=np.concatenate([x_train_org,data_zeros],axis=1)
        x_train[:,fs_zero]=0.0

        x_test_org=np.array(df_test.iloc[:,:fs_num].copy().values,dtype=np.float32)
        y_test=np.array(df_test["label"].values,dtype=np.int32)
        data_zeros=np.zeros([df_test.shape[0],
                             img_rows*img_cols-fs_num],dtype=np.float32)
        x_test=np.concatenate([x_test_org,data_zeros],axis=1)
        x_test[:,fs_zero]=0.0

        data_all = np.concatenate([x_train,x_test])
        if minmax_scale:
            data_all=preprocessing.minmax_scale(data_all,(-1,1))
            data_all*=minmax_scale_factor
        else:
            data_all=data_all-data_all.mean(axis=0)
        x_train=data_all[:df_train.shape[0],:]
        x_test=data_all[df_train.shape[0]:,:]

        df_train_std=pd.DataFrame(np.copy(x_train),columns=cols_new)
        df_train_std["label"]=y_train
        df_train_std["sno"]=df_train.sno.values
        df_test_std=pd.DataFrame(np.copy(x_test),columns=cols_new)
        df_test_std["label"]=y_test
        df_test_std["sno"]=df_test.sno.values

        return (x_train, y_train),(x_test, y_test),cols_new,\
                (df_train_org,df_test_org),(df_train_std,df_test_std)

    def make_cv_splits(data_cv,label_cv,n_cv_splits,cv_seed):
        kf_cv = StratifiedKFold(
                n_splits=n_cv_splits,shuffle=True,random_state=cv_seed)
        splits_cv = []
        for tridx_cv,tsidx_cv in kf_cv.split(data_cv,label_cv):
            x_train_org = data_cv[tridx_cv,:]
            y_train_org = label_cv[tridx_cv]
            x_test_org = data_cv[tsidx_cv,:]
            y_test_org = label_cv[tsidx_cv]
            splits_cv.append([x_train_org,y_train_org,
                              x_test_org,y_test_org])
#            splits_cv.append([tridx_cv,tsidx_cv])
        return splits_cv

    def fix_batch(batch_size,seed,x_train,y_train_flat,
                  train_ros=True,batch_fix=True):
        if train_ros:
            x_train_flat = x_train.reshape((x_train.shape[0],-1))
            ros=RandomOverSampler(random_state=seed)
#            ros=SMOTE(random_state=seed)
#            ros=ADASYN(random_state=seed)
#            ros=BorderlineSMOTE(random_state=seed, kind='borderline-1')
#            ros=BorderlineSMOTE(random_state=seed, kind='borderline-2')
#            ros=SVMSMOTE(random_state=seed)
#            ros=SMOTENC(categorical_features=[0,1], random_state=seed) #[0,1,2]
            x_train_ros,y_train_ros=ros.fit_sample(x_train_flat,y_train_flat)
            print('Over sampling:',x_train.shape[0],"->",x_train_ros.shape[0])
            x_train_new=x_train_ros.reshape(
                    (x_train_ros.shape[0],img_rows,img_cols,channels))
            y_train_new = y_train_ros
        else:
            x_train_new = x_train
            y_train_new = y_train_flat

        if batch_fix:
            if x_train_new.shape[0]%batch_size!=0:
                n_add=batch_size-x_train_new.shape[0]%batch_size
                rnd=np.random.RandomState(seed)
                if batch_fix_v2:
                    x_train_lb=x_train
                    y_train_lb=y_train_flat
                else:
                    x_train_lb=x_train_new
                    y_train_lb=y_train_new
                if not batch_fix_balance:
                    idxs_add=rnd.choice(x_train_lb.shape[0],n_add,replace=False)
                    x_tr_add=x_train_lb[idxs_add]
                    y_tr_add=y_train_lb[idxs_add]
                    x_train_new = np.concatenate([x_train_new,x_tr_add])
                    y_train_new = np.concatenate([y_train_new,y_tr_add])
                else:
                    for i in range(2):
                        lb_add=i
                        if lb_add==0:
                            n_add_lbc=n_add//2
                        else:
                            n_add_lbc=n_add-n_add_lbc
                        ids_lbc=np.where(y_train_lb==lb_add)[0]
                        x_train_lbc=x_train_lb[ids_lbc]
                        y_train_lbc=y_train_lb[ids_lbc]
                        idxs_add=rnd.choice(x_train_lbc.shape[0],n_add_lbc,replace=False)
                        x_tr_add=x_train_lbc[idxs_add]
                        y_tr_add=y_train_lbc[idxs_add]
                        x_train_new = np.concatenate([x_train_new,x_tr_add])
                        y_train_new = np.concatenate([y_train_new,y_tr_add])
                print('Add train samples:',n_add)

        return x_train_new, y_train_new


    ##prepare data
    (data_cv,label_cv),(data_out,label_out),cols_new,\
    (df_train,df_test),(df_train_std,df_test_std) = load_hdxdata(fs_idxs,
                                                                 minmax_scale=minmax_scale)

    ##cv_partial
    if cv_partial:
        n_cv_partial=min(n_cv_partial,df_train.shape[0])
        rnd_cv_partial=np.random.RandomState(cv_partial_seed)
        cv_sel=rnd_cv_partial.choice(df_train.shape[0],n_cv_partial,replace=False)
        data_cv=data_cv[cv_sel]
        label_cv=label_cv[cv_sel]
        df_train=df_train.iloc[cv_sel,:]
        df_train_std=df_train_std.iloc[cv_sel,:]

    ##save and tracking
    if save_ds:
        df_train_std.to_csv(outdata_dir+"/df_train_std.csv",index=False)
        df_test_std.to_csv(outdata_dir+"/df_test_std.csv",index=False)
        ids_cv=np.array(df_train["sno"].values)
        ids_out=np.array(df_test["sno"].values)
#        print(np.unique(ids_cv).shape)
#        print(np.unique(ids_out).shape)
        data_cv[:,-1]=ids_cv
        data_out[:,-1]=ids_out

    if cv_as_all:
        data_cv=np.concatenate((data_cv,data_out),axis=0)
        label_cv=np.concatenate((label_cv,label_out),axis=0)

    splits_cv=make_cv_splits(data_cv,label_cv,n_cv_splits,cv_seed)

    ##save splits and recover tracking to original data
    if save_ds:
        def save_cv_ds(i, cv_type):
            cv_dir=outdata_dir+"/cv%03d"%(i+1)
            if not os.path.isdir(cv_dir):
                os.makedirs(cv_dir)

            if cv_type=="train":
                ds_pos=0
            else:
                ds_pos=2

            data_cvtt=splits_cv[i][ds_pos]
            snos=np.copy(data_cvtt[:,-1])
            data_cvtt[:,-1]=data_cvtt[:,-2]

            df_cvtt_org=df_train.loc[df_train.sno.isin(snos)]
            df_cvtt_org=df_cvtt_org.reset_index(drop=True)
            df_cvtt_org.to_csv(cv_dir+"/%s.csv"%cv_type,index=False)

            df_cvtt=pd.DataFrame(np.copy(data_cvtt),columns=cols_new)
            df_cvtt["label"]=splits_cv[i][ds_pos+1]
            df_cvtt["sno"]=snos
            df_cvtt.to_csv(cv_dir+"/%s_std.csv"%cv_type,index=False)
        for i in range(n_cv_splits):
            save_cv_ds(i, "train")
            save_cv_ds(i, "val")
        data_cv[:,-1]=data_cv[:,-2]
        data_out[:,-1]=data_out[:,-2]
##=============================================================


##===============================================================================
    start=time.time()

##=============================================================
    df_hist = pd.DataFrame()
    hist_acc = []
    hist_prec = []
    hist_recall = []
    hist_f1 = []
    hist_rocp = []
    y_allcv = np.array([],dtype=int)
    rocp_allcv = np.array([],dtype=float)

    n_cv=0
    for split_cv in splits_cv: #tr_idx,ts_idx
        n_cv+=1
        if tb_enabled or conf_permtest!="":
            break
        print("\n=====================CV[%s]========================\n"%n_cv)
        in_cv = True

        # the data, split between train and test sets
        (x_train_org,y_train_org,x_test_org,y_test_org)=split_cv
#        x_train_org = data_cv[tr_idx,:]
#        y_train_org = label_cv[tr_idx]
#        x_test_org = data_cv[ts_idx,:]
#        y_test_org = label_cv[ts_idx]

        # convert class vectors to binary class matrices
        y_train_cat = keras.utils.to_categorical(y_train_org, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test_org, num_classes)
        y_test_watch=y_test_cat[:,watch_cls]

        print(x_train_org.shape[0], 'train samples')
        print(x_test_org.shape[0], 'test samples\n')

        ##=============================================================
        x_train_dl = np.reshape(x_train_org,[-1,img_rows,img_cols,channels])
        x_test_dl = np.reshape(x_test_org,[-1,img_rows,img_cols,channels])

        y_train_dl = y_train_cat
        y_test_dl = y_test_cat

        x_train_dl,y_train_bf=fix_batch(
                batch_size_cv,osbfb_seed,x_train_dl,y_train_org,
                train_ros,batch_fix)
        y_train_dl = keras.utils.to_categorical(y_train_bf, num_classes)
        x_train_org = np.reshape(x_train_dl,[x_train_dl.shape[0],-1])
        y_train_org = y_train_bf
        y_train_cat = y_train_dl
        print(x_train_org.shape[0], 'new train samples\n')
        ##=============================================================

        ##=============================================================
        if enable_dl:
            del(model)
            model = create_dlmodel()

            if need_training_cv:
                callbacks_cv = callbacks_dl.copy()

                csv_logger = keras.callbacks.CSVLogger(log_dir+"/hist_cv%d.csv"%n_cv)
                callbacks_cv.append(csv_logger)

                if use_clr:
                    step_size=math.ceil(
                            x_train_dl.shape[0]/batch_size_cv)*epochs_clr_cv #half 2-8
                    clr = CyclicLR(base_lr=base_lr,max_lr=max_lr_cv,
                                   step_size=step_size,mode=step_mode)
                    callbacks_cv[0]=clr

                if data_augmentation:
                    if mix_enable:
                        from mixup_generator import MixupGenerator
                        generator_cv = MixupGenerator(x_train_dl,y_train_dl,
                                                    batch_size=batch_size_cv,
                                                    alpha=mix_alpha, #0.5
                                                    datagen=datagen if mix_datagen else None,
                                                    )()
                    else:
                        generator_cv = datagen.flow(x_train_dl,y_train_dl,
                                                     batch_size=batch_size_cv,
                                                     shuffle=True,seed=2100)

                if not data_augmentation:
                    model.fit(x_train_dl,y_train_dl, #y_train_cat y_train_org
                              batch_size=batch_size_cv,
                              epochs=epochs_cv,
                              verbose=verbose_dl,
                              shuffle=True,
                              callbacks=callbacks_cv,
                              validation_data=(x_test_dl,y_test_dl), #y_test_cat y_test_org
                              )
                else:
                    model.fit_generator(
                                generator=generator_cv,workers=1,
                                epochs=epochs_cv,
                                validation_data=(x_test_dl,y_test_dl),
                                steps_per_epoch=x_train_dl.shape[0] // batch_size_cv,
                                callbacks=callbacks_cv,
                                use_multiprocessing=False, #True False
                                verbose=verbose_dl)
                model.save_weights(save_dir+"/"+conf_tags_dl+"_cv%d_last.h5"%n_cv)

            if cv_using_selflast:
                model.load_weights(save_dir+"/"+conf_tags_dl+"_cv%d_last.h5"%n_cv)
                label_out_pred_prob_d = model.predict(x_test_dl)

                if cv_using_selfbest:
                    label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
                    label_out_pred_m = label_out_pred_d
                    label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
                    label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]

                    print(classification_report(y_test_org, label_out_pred_m))
                    print(confusion_matrix(y_test_org, label_out_pred_m))
                    print("ACCURACY:", accuracy_score(y_test_org, label_out_pred_m))
                    print("PRECISION:", precision_score(y_test_watch, label_out_pred))
                    print("RECALL:", recall_score(y_test_watch, label_out_pred))
                    print("F1:", f1_score(y_test_watch, label_out_pred))
                    print("ROC_AUC(Pr.):", roc_auc_score(y_test_watch, label_out_pred_prob))

            if cv_using_selfbest:
                model.load_weights(save_dir+"/"+conf_tags_dl+"_cv%d.h5"%n_cv)
                label_out_pred_prob_d = model.predict(x_test_dl)

            label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
            label_out_pred_m = label_out_pred_d
            label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
            label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]

            print(classification_report(y_test_org, label_out_pred_m))
            print(confusion_matrix(y_test_org, label_out_pred_m))
            print("ACCURACY:", accuracy_score(y_test_org, label_out_pred_m))
            print("PRECISION:", precision_score(y_test_watch, label_out_pred))
            print("RECALL:", recall_score(y_test_watch, label_out_pred))
            print("F1:", f1_score(y_test_watch, label_out_pred))
            print("ROC_AUC(Pr.):", roc_auc_score(y_test_watch, label_out_pred_prob))
        ##=============================================================

        ##=============================================================
        if enable_sk:
            print('\n')
            ml.fit(x_train_org, y_train_org)

            label_out_pred_prob_sk = ml.predict_proba(x_test_org)
            label_out_pred_sk = np.argmax(label_out_pred_prob_sk,axis=1)
            label_out_pred_m = label_out_pred_sk
            label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
            label_out_pred_prob = label_out_pred_prob_sk[:,watch_cls]

            print(classification_report(y_test_org, label_out_pred_m))
            print(confusion_matrix(y_test_org, label_out_pred_m))
            print("ACCURACY:", accuracy_score(y_test_org, label_out_pred_m))
            print("PRECISION:", precision_score(y_test_watch, label_out_pred))
            print("RECALL:", recall_score(y_test_watch, label_out_pred))
            print("F1:", f1_score(y_test_watch, label_out_pred))
            print("ROC_AUC(Pr.):", roc_auc_score(y_test_watch, label_out_pred_prob))
        ##=============================================================

        ##=============================================================
        if enable_sk:
            label_out_pred_prob_hyb = label_out_pred_prob_sk
        if enable_dl:
            if not enable_sk:
                label_out_pred_prob_hyb = label_out_pred_prob_d
            else:
                label_out_pred_prob_hyb = label_out_pred_prob_hyb*(1-w_dl) + label_out_pred_prob_d*w_dl
        label_out_pred_hyb = np.argmax(label_out_pred_prob_hyb,axis=1)
        label_out_pred_m = label_out_pred_hyb
        label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
        label_out_pred_prob = label_out_pred_prob_hyb[:,watch_cls]
        ##=============================================================

        print('\n')
        print(classification_report(y_test_org, label_out_pred_m))
        print(confusion_matrix(y_test_org, label_out_pred_m))
        print("ACCURACY:", accuracy_score(y_test_org, label_out_pred_m))
        print("PRECISION:", precision_score(y_test_watch, label_out_pred))
        print("RECALL:", recall_score(y_test_watch, label_out_pred))
        print("F1:", f1_score(y_test_watch, label_out_pred))
        print("ROC_AUC(Pr.):", roc_auc_score(y_test_watch, label_out_pred_prob))

        hist_acc.append(accuracy_score(y_test_org, label_out_pred_m))
        hist_prec.append(precision_score(y_test_watch, label_out_pred))
        hist_recall.append(recall_score(y_test_watch, label_out_pred))
        hist_f1.append(f1_score(y_test_watch, label_out_pred))
        hist_rocp.append(roc_auc_score(y_test_watch, label_out_pred_prob))

        y_allcv = np.concatenate([y_allcv,y_test_watch])
        rocp_allcv = np.concatenate([rocp_allcv,label_out_pred_prob])

        fpr_oc, tpr_oc, _ = roc_curve(
            y_test_watch,label_out_pred_prob,pos_label=1)
        plt.figure(6661,figsize=(10,8))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_oc, tpr_oc, ':',
                 label='CV%s(AUC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'
                 % (str(n_cv),hist_rocp[-1],hist_f1[-1],
                    hist_prec[-1],hist_recall[-1],hist_acc[-1])
                 )
        plt.xlabel('False positive rate(1-Specificity)')
        plt.ylabel('True positive rate(Sensitivity)')
        plt.title('ROC curve' )
        plt.legend(loc='best')
#        plt.show()

    df_hist["acc"] = np.array(hist_acc,dtype=float)
    df_hist["prec"] = np.array(hist_prec,dtype=float)
    df_hist["recall"] = np.array(hist_recall,dtype=float)
    df_hist["f1"] = np.array(hist_f1,dtype=float)
    df_hist["rocp"] = np.array(hist_rocp,dtype=float)


    print("\n========================Blind Test=========================\n")
    in_cv = False

    x_cv_org = data_cv
    y_cv_org = label_cv

    if conf_permtest!="":
        rnd_pt=np.random.RandomState(permtest_seed)
        y_cv_org=rnd_pt.randint(0,num_classes,size=y_cv_org.shape)

    x_blind_org = data_out
    y_blind_org = label_out

    # convert class vectors to binary class matrices
    y_cv_cat = keras.utils.to_categorical(y_cv_org, num_classes)
    y_blind_cat = keras.utils.to_categorical(y_blind_org, num_classes)
    y_blind_watch=y_blind_cat[:,watch_cls]

    print(x_cv_org.shape[0], 'train samples for CV')
    print(x_blind_org.shape[0], 'test samples for blind-test\n')

    ##=============================================================
    x_cv_dl = np.reshape(x_cv_org,[-1,img_rows,img_cols,channels])
    x_blind_dl = np.reshape(x_blind_org,[-1,img_rows,img_cols,channels])

    y_cv_dl = y_cv_cat
    y_blind_dl = y_blind_cat

    x_cv_dl,y_cv_bf=fix_batch(batch_size,osbfb_seed,
                              x_cv_dl,y_cv_org,
                              train_ros,batch_fix)
    y_cv_dl = keras.utils.to_categorical(y_cv_bf, num_classes)
    x_cv_org = np.reshape(x_cv_dl,[x_cv_dl.shape[0],-1])
    y_cv_org = y_cv_bf
    y_cv_cat = y_cv_dl
    print(x_cv_org.shape[0], 'new train samples for CV\n')
    ##=============================================================

    ##=============================================================
    if enable_dl:
        del(model)
        model = create_dlmodel()

        if need_training_bl:
            callbacks_bl = callbacks_dl.copy()

            if conf_permtest=="":
                csv_logger = keras.callbacks.CSVLogger(log_dir+"/hist_blind.csv")
                callbacks_bl.append(csv_logger)

            if use_clr:
                step_size=math.ceil(x_cv_dl.shape[0]/batch_size)*epochs_clr #half 2-8
                clr = CyclicLR(base_lr=base_lr,max_lr=max_lr,
                               step_size=step_size,mode=step_mode)
                callbacks_bl[0]=clr

            if data_augmentation:
                if mix_enable:
                    from mixup_generator import MixupGenerator
                    generator = MixupGenerator(x_cv_dl, y_cv_dl,
                                            batch_size=batch_size,
                                            alpha=mix_alpha, #0.5
                                            datagen=datagen if mix_datagen else None,
                                            )()
                else:
                    generator = datagen.flow(x_cv_dl,y_cv_dl,
                                             batch_size=batch_size,
                                             shuffle=True,seed=2100)

            if not data_augmentation:
                model.fit(x_cv_dl, y_cv_dl, #y_cv_cat y_cv_org
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose_dl,
                          shuffle=True,
                          callbacks=callbacks_bl,
                          validation_data=(x_blind_dl, y_blind_dl) #y_blind_cat y_blind_org
                          )
            else:
                model.fit_generator(
                            generator,workers=1,
                            epochs=epochs,
                            validation_data=(x_blind_dl,y_blind_dl),
                            steps_per_epoch=x_cv_dl.shape[0] // batch_size,
                            callbacks=callbacks_bl,
                            use_multiprocessing=False, #True False
                            verbose=verbose_dl)
            if conf_permtest=="":
                model.save_weights(save_dir+"/"+conf_tags_dl+"_blind_last.h5")

        if bl_using_selflast:
            model.load_weights(save_dir+"/"+conf_tags_dl+"_blind_last.h5")
            label_out_pred_prob_d = model.predict(x_blind_dl)

            if bl_using_selfbest or bl_using_cvbests:
                label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
                label_out_pred_m = label_out_pred_d
                label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
                label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]

                print(classification_report(y_blind_org, label_out_pred_m))
                print(confusion_matrix(y_blind_org, label_out_pred_m))
                print("ACCURACY:", accuracy_score(y_blind_org, label_out_pred_m))
                print("PRECISION:", precision_score(y_blind_watch, label_out_pred))
                print("RECALL:", recall_score(y_blind_watch, label_out_pred))
                print("F1:", f1_score(y_blind_watch, label_out_pred))
                print("ROC_AUC(Pr.):", roc_auc_score(y_blind_watch, label_out_pred_prob))

        if bl_using_selfbest:
            model.load_weights(save_dir+"/"+conf_tags_dl+"_blind%s.h5"%conf_permtest)
            label_out_pred_prob_d = model.predict(x_blind_dl)

            if bl_using_cvbests:
                label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
                label_out_pred_m = label_out_pred_d
                label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
                label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]

                print(classification_report(y_blind_org, label_out_pred_m))
                print(confusion_matrix(y_blind_org, label_out_pred_m))
                print("ACCURACY:", accuracy_score(y_blind_org, label_out_pred_m))
                print("PRECISION:", precision_score(y_blind_watch, label_out_pred))
                print("RECALL:", recall_score(y_blind_watch, label_out_pred))
                print("F1:", f1_score(y_blind_watch, label_out_pred))
                print("ROC_AUC(Pr.):", roc_auc_score(y_blind_watch, label_out_pred_prob))

        if bl_using_cvbests:
        #    w_merge = np.repeat(1/len(splits_cv),len(splits_cv))
            w_merge_org = w_rocp*df_hist.rocp+w_f1*df_hist.f1+w_acc*df_hist.acc
        #    w_merge = w_merge_org/w_merge_org.sum()
            w_merge = np.exp(w_merge_org*w_merge_exp_factor)/ \
                        np.exp(w_merge_org*w_merge_exp_factor).sum()
            label_out_pred_prob_d = np.zeros_like(y_blind_dl,dtype=float)
        #    label_out_pred_probs_d = []
            for i in range(1,len(splits_cv)+1):
                model.load_weights(save_dir+"/"+conf_tags_dl+"_cv%d.h5"%i)
                score = model.evaluate(x_blind_dl, y_blind_dl, verbose=0)
                print('\n')
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                print('\n')
                label_out_pred_prob_d += np.asarray(model.predict(x_blind_dl)) * w_merge[i-1]
        #        label_out_pred_probs_d.append(np.asarray(model.predict(x_blind_dl)))
        #    label_out_pred_probs_d = np.asarray(label_out_pred_probs_d)
        ##    label_out_pred_prob_dv = np.zeros_like(label_out_pred_prob_d)
        ##    voting_from = 0
        ##    voters = len(splits_cv)//2+1
        ##    for i in range(label_out_pred_prob_dv.shape[0]):
        ##        for c in range(num_classes):
        ##            preds = label_out_pred_probs_d[:,i,c]
        ###            pred_cur = preds.max() \
        ###                    if (1-preds.max())<preds.min() else preds.min()
        ##            preds_sort = np.sort(preds)
        ##            pred_cur = preds_sort[-voters-voting_from:-voting_from].mean() \
        ##                    if (1-preds_sort[-voters-voting_from:-voting_from].mean())<\
        ##                        preds_sort[voting_from:voting_from+voters].mean() \
        ##                    else preds_sort[voting_from:voters+voting_from].mean()
        ###            pred_cur = preds_sort[-voters-voting_from:-voting_from].max() \
        ###                    if (1-preds_sort[-voters-voting_from:-voting_from].mean())<\
        ###                        preds_sort[voting_from:voting_from+voters].mean() \
        ###                    else preds_sort[voting_from:voters+voting_from].max()
        ##            label_out_pred_prob_dv[i,c] = pred_cur
        ##        label_out_pred_prob_dv[i,:] = \
        ##            label_out_pred_prob_dv[i,:]/label_out_pred_prob_dv[i].sum()
        ##    label_out_pred_prob_d = label_out_pred_prob_dv

        label_out_pred_d = np.argmax(label_out_pred_prob_d,axis=1)
        label_out_pred_m = label_out_pred_d
        label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
        label_out_pred_prob = label_out_pred_prob_d[:,watch_cls]

        print(classification_report(y_blind_org, label_out_pred_m))
        print(confusion_matrix(y_blind_org, label_out_pred_m))
        print("ACCURACY:", accuracy_score(y_blind_org, label_out_pred_m))
        print("PRECISION:", precision_score(y_blind_watch, label_out_pred))
        print("RECALL:", recall_score(y_blind_watch, label_out_pred))
        print("F1:", f1_score(y_blind_watch, label_out_pred))
        print("ROC_AUC(Pr.):", roc_auc_score(y_blind_watch, label_out_pred_prob))
    ##=============================================================

    ##=============================================================
    if enable_sk:
        print('\n')
        ml.fit(x_cv_org, y_cv_org)

        label_out_pred_prob_sk = ml.predict_proba(x_blind_org)
        label_out_pred_sk = np.argmax(label_out_pred_prob_sk,axis=1)
        label_out_pred_m = label_out_pred_sk
        label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
        label_out_pred_prob = label_out_pred_prob_sk[:,watch_cls]

        print(classification_report(y_blind_org, label_out_pred_m))
        print(confusion_matrix(y_blind_org, label_out_pred_m))
        print("ACCURACY:", accuracy_score(y_blind_org, label_out_pred_m))
        print("PRECISION:", precision_score(y_blind_watch, label_out_pred))
        print("RECALL:", recall_score(y_blind_watch, label_out_pred))
        print("F1:", f1_score(y_blind_watch, label_out_pred))
        print("ROC_AUC(Pr.):", roc_auc_score(y_blind_watch, label_out_pred_prob))
    ##=============================================================

    ##=============================================================
    if enable_sk:
        label_out_pred_prob_hyb = label_out_pred_prob_sk
    if enable_dl:
        if not enable_sk:
            label_out_pred_prob_hyb = label_out_pred_prob_d
        else:
            label_out_pred_prob_hyb = label_out_pred_prob_hyb*(1-w_dl) + label_out_pred_prob_d*w_dl
    label_out_pred_hyb = np.argmax(label_out_pred_prob_hyb,axis=1)
    label_out_pred_m = label_out_pred_hyb
    label_out_pred=np.asarray(label_out_pred_m==watch_cls,dtype=np.int32)
    label_out_pred_prob = label_out_pred_prob_hyb[:,watch_cls]
    ##=============================================================

    print('\n')
    print(classification_report(y_blind_org, label_out_pred_m))
    print(confusion_matrix(y_blind_org, label_out_pred_m))
    print("ACCURACY:", accuracy_score(y_blind_org, label_out_pred_m))
    print("PRECISION:", precision_score(y_blind_watch, label_out_pred))
    print("RECALL:", recall_score(y_blind_watch, label_out_pred))
    print("F1:", f1_score(y_blind_watch, label_out_pred))
    print("ROC_AUC(Pr.):", roc_auc_score(y_blind_watch, label_out_pred_prob))

    if conf_permtest=="":
        fpr_oc, tpr_oc, _ = roc_curve(
            y_blind_watch,label_out_pred_prob,pos_label=1)
        plt.figure(8881,figsize=(10,8))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_oc, tpr_oc, 'r-',
                 label='BlindTest(AUC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'
                 % (roc_auc_score(y_blind_watch, label_out_pred_prob),
                   f1_score(y_blind_watch, label_out_pred),
                   precision_score(y_blind_watch, label_out_pred),
                   recall_score(y_blind_watch, label_out_pred),
                   accuracy_score(y_blind_org, label_out_pred_m)))
        plt.xlabel('False positive rate(1-Specificity)')
        plt.ylabel('True positive rate(Sensitivity)')
        plt.title('ROC curve' )
        plt.legend(loc='best')
        plt.show()
        plt.savefig(log_dir+"/roc-blind.png")

    df_blind=pd.DataFrame()
    df_blind["pred_prob"]=label_out_pred_prob
    df_blind["real_label"]=y_blind_watch
#    df_blind.plot()
    if conf_permtest=="":
        df_blind_file=log_dir+"/df-blind-pred.csv"
    else:
        df_blind_file=log_dir+"/df-blind-pred%s_%d.csv"%(conf_permtest,
                                                         int(round(roc_auc_score(y_blind_watch, label_out_pred_prob)*10000,0)))
    df_blind.to_csv(df_blind_file,index=False)


    if conf_permtest=="" and not tb_enabled:
        print("\n========================CV Total=========================\n")
        print(df_hist)
        print(df_hist.describe())

        rocp_avg = roc_auc_score(y_allcv, rocp_allcv)
        pred_allcv = np.asarray(rocp_allcv>0.5,dtype=np.int32)
        fpr_oc, tpr_oc, _ = roc_curve(
            y_allcv,rocp_allcv,pos_label=1)
        plt.figure(6661,figsize=(10,8))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_oc, tpr_oc, 'r-',
                 label='AVG(AUC:%0.4f|F1:%0.4f|PR:%0.4f|RC:%0.4f|ACC:%0.4f)'
                 % (
#                    rocp_avg,
#                    f1_score(y_allcv, pred_allcv),
#                    precision_score(y_allcv, pred_allcv),
#                    recall_score(y_allcv, pred_allcv),
#                    accuracy_score(y_allcv, pred_allcv),
                    df_hist.rocp.mean(),
                    df_hist.f1.mean(),
                    df_hist.prec.mean(),
                    df_hist.recall.mean(),
                    df_hist.acc.mean()
                    ))
        plt.xlabel('False positive rate(1-Specificity)')
        plt.ylabel('True positive rate(Sensitivity)')
        plt.title('ROC curve' )
        plt.legend(loc='best')
        plt.show()
        plt.savefig(log_dir+"/roc-cvs.png")

        df_cvs=pd.DataFrame()
        df_cvs["pred_prob"]=rocp_allcv
        df_cvs["real_label"]=y_allcv
#        df_cvs.plot()
        df_cvs.to_csv(log_dir+"/df-cvs-pred.csv",index=False)


#    ktf.clear_session()


##==============================================================================
#import os
#import pandas as pd
#import matplotlib
#import matplotlib.pylab as plt
#
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', 10000)
#pd.set_option('display.max_columns', 100)
#
#print(matplotlib.style.available)
#
##matplotlib.style. use('ggplot')
#matplotlib.style. use('seaborn-whitegrid')
##matplotlib.style. use('seaborn-darkgrid')
##matplotlib.style. use('seaborn-paper')
#matplotlib.style. use('seaborn-poster')
##%matplotlib inline
#
#
#df_log=pd.read_csv("saved_models_st/hist.csv")
#
#df_log.loc[:,["accuracy","val_accuracy"]].plot(figsize=(18,9), #20,9
#                                      title="Training Accuray vs Validation Accuracy")
#df_log.loc[:,["loss","val_loss"]].plot(figsize=(18,9), #20,9
#                                      title="Training Loss vs Validation Loss")
##==============================================================================


##===================================================================
#from vis.visualization import visualize_activation,visualize_saliency
#from vis.utils import utils as vis_utils
#
#layer_idx = vis_utils.find_layer_idx(model, model.layers[-1].name)
## Swap softmax with linear
#model.layers[layer_idx].activation = activations.linear
#model = vis_utils.apply_modifications(model)
#
#img10_df=pd.DataFrame()
#img10_df["cols"]=cols_new
#
#filter_idx = 1
#img1 = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(-1., 1.))
#img1 = (img1+1)/2
#plt.figure("img1")
#plt.imshow(img1[..., 0])
#plt.savefig("./saved_models_st/img1-%s.png" % conf_tags)
#
#filter_idx = 0
#img0 = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(-1., 1.))
#img0 = (img0+1)/2
#plt.figure("img0")
#plt.imshow(img0[..., 0])
#plt.savefig("./saved_models_st/img0-%s.png" % conf_tags)
#
#img1_1d=img1.ravel()
#img0_1d=img0.ravel()
#img_diff=np.abs(img1_1d - img0_1d)
#
#img10_df["score1"]=img1_1d
#img10_df["score0"]=img0_1d
#img10_df["score_diff"]=img_diff
#
#
##train1=np.where(y_train_org==1)[0]
##train0=np.where(y_train_org==0)[0]
##
##filter_idx = 1 #1 0 [1,0]
##imga = visualize_saliency(model, layer_idx, filter_indices=filter_idx,
##                           seed_input=x_train[train1]) #tr/ts:0->1 1->0
##plt.imshow(imga, cmap='jet')
##imga_1d=imga.ravel()
##img10_df["score_att1_train"]=imga_1d
##
##filter_idx = 0 #1 0 [1,0]
##imga = visualize_saliency(model, layer_idx, filter_indices=filter_idx,
##                           seed_input=x_train[train0]) #tr/ts:0->1 1->0
##plt.imshow(imga, cmap='jet')
##imga_1d=imga.ravel()
##img10_df["score_att0_train"]=imga_1d
##
#
#trainok=np.where(label_out_pred_dtr==y_train_org)[0]
#x_trainok=x_train[trainok]
#y_trainok=y_train_org[trainok]
#trainok1=np.where(y_trainok==1)[0]
#trainok0=np.where(y_trainok==0)[0]
#
#filter_idx = 1 #1 0 [1,0]
#imga = visualize_saliency(model, layer_idx, filter_indices=filter_idx,
#                           seed_input=x_trainok[trainok1]) #tr/ts:0->1 1->0
#plt.figure("att1_trainok")
#plt.imshow(imga, cmap='jet')
#plt.savefig("./saved_models_st/att1_trainok-%s.png" % conf_tags)
#imga_1d=imga.ravel()
#img10_df["score_att1_trainok"]=imga_1d
#
#filter_idx = 0 #1 0 [1,0]
#imga = visualize_saliency(model, layer_idx, filter_indices=filter_idx,
#                           seed_input=x_trainok[trainok0]) #tr/ts:0->1 1->0
#plt.figure("att0_trainok")
#plt.imshow(imga, cmap='jet')
#plt.savefig("./saved_models_st/att0_trainok-%s.png" % conf_tags)
#imga_1d=imga.ravel()
#img10_df["score_att0_trainok"]=imga_1d
#
#
#img10_df.to_csv("./saved_models_st/dl_fs_atts-%s.csv"%conf_tags,
#                index=False)
#
##img10_df=img10_df.sort_index()
##img10_df=img10_df.sort_values(["score_att1_trainok"],ascending=False)
#
##score_att1_trainok=np.array(img10_df.score_att1_trainok,dtype=np.float32)
##score_att1_trainok_ft=scipy.signal.medfilt(score_att1_trainok,3)
##===================================================================


##===============================================================================
    print("")
    print("\n===========================================================================\n")
    print(conf_tags)
    print("\nRun time:", time.time()-start)
    print("\n===========================================================================\n")



















