import tensorflow as tf
import numpy as np
import uproot
import awkward as ak
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics
import yaml
import tf2onnx
import onnx
import onnxruntime as ort
import shutil
import copy
import psutil
import gc


import threading
from FLAF.RunKit.kinit import cond as kInit_cond, update_kinit_thread


# Need to get train_features and train_labels
class DataWrapper:
    def __init__(self):
        print("Init data wrapper")

        self.feature_names = None
        self.listfeature_names = None
        self.highlevelfeatures_names = None
        self.label_name = None
        self.mbb_name = None

        self.features_no_param = None
        self.features = None
        self.listfeatures = None
        self.hlv = None
        self.param_values = None
        self.labels = None
        self.labels_binary = None
        self.mbb = None
        self.mbb_region = None

        self.class_weight = None
        self.adv_weight = None
        self.class_target = None
        self.adv_target = None

        self.param_list = [
            250,
            260,
            270,
            280,
            300,
            350,
            450,
            550,
            600,
            650,
            700,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2500,
            3000,
            4000,
            5000,
        ]
        self.use_parametric = False

        self.features_paramSet = None

        self.hmefriendfeatures_names = None
        self.hme = None

        self.X_mass = None

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

    def SetParamList(self, param_list):
        self.param_list = param_list

    def SetPredictParamValue(self, param_value):
        # During predict, we want to use a truly random param value even for signal!
        if param_value not in self.param_list:
            print(f"This param value {param_value} is not an option!")
        new_params = np.array([[param_value for x in self.features]]).transpose()

        self.features_paramSet = np.append(self.features_no_param, new_params, axis=1)

    def AddInputFeatures(self, features):
        if self.feature_names == None:
            self.feature_names = features
        else:
            self.feature_names = self.feature_names + features

        print(f"Added features {features}")
        print(f"New feature list {self.feature_names}")

    def AddInputFeaturesList(self, features, index):
        if self.listfeature_names == None:
            self.listfeature_names = [[feature, index] for feature in features]
        else:
            self.listfeature_names = self.listfeature_names + [
                [feature, index] for feature in features
            ]

        print(f"Added features {features} with index {index}")
        print(f"New feature list {self.listfeature_names}")

    def AddHighLevelFeatures(self, features):
        if self.highlevelfeatures_names == None:
            self.highlevelfeatures_names = features
        else:
            self.highlevelfeatures_names = self.highlevelfeatures_names + features

        print(f"Added high level features {features}")
        print(f"New feature list {self.highlevelfeatures_names}")

    def AddHMEFeatures(self, features):
        if self.hmefriendfeatures_names == None:
            self.hmefriendfeatures_names = features
        else:
            self.hmefriendfeatures_names = self.hmefriendfeatures_names + features

    def AddInputLabel(self, labels_name):
        if self.label_name != None:
            print("What are you doing? You already defined the input label branch")
        self.label_name = labels_name

    def SetMbbName(self, mbb_name):
        if self.mbb_name != None:
            print("What are you doing? You already defined the mbb branch")
        self.mbb_name = mbb_name

    def ReadFile(
        self, file_name, entry_start=None, entry_stop=None, hme_friend_file=None
    ):
        if self.feature_names == None:
            print("Uknown branches to read! DefineInputFeatures first!")
            return

        print(f"Reading file {file_name}")

        features_to_load = []
        features_to_load = features_to_load + self.feature_names
        if self.listfeature_names != None:
            for listfeature in self.listfeature_names:
                if listfeature[0] not in features_to_load:
                    features_to_load.append(listfeature[0])
        if self.highlevelfeatures_names != None:
            features_to_load = features_to_load + self.highlevelfeatures_names

        features_to_load.append(self.mbb_name)
        features_to_load.append("X_mass")

        print(f"Only loading these features {features_to_load}")

        print(
            f"Going to open file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

        # file = uproot.open(file_name)
        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                features_to_load, entry_start=entry_start, entry_stop=entry_stop
            )

            print(
                f"Loaded branches. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            self.features = np.array(
                [getattr(branches, feature_name) for feature_name in self.feature_names]
            ).transpose()
            print("Got features, but its a np array")

            default_value = 0.0
            if self.listfeature_names != None:
                self.listfeatures = np.array(
                    [
                        ak.fill_none(
                            ak.pad_none(getattr(branches, feature_name), index + 1),
                            default_value,
                        )[:, index]
                        for [feature_name, index] in self.listfeature_names
                    ],
                    dtype="float32",
                ).transpose()
            print("Got the list features")

            # Need to append the value features and the listfeatures together
            if self.listfeature_names != None:
                print("We have list features!")
                self.features = np.append(self.features, self.listfeatures, axis=1)

            if self.highlevelfeatures_names != None:
                self.hlv = np.array(
                    [
                        getattr(branches, feature_name)
                        for feature_name in self.highlevelfeatures_names
                    ],
                    dtype="float32",
                ).transpose()
                self.features = np.append(self.features, self.hlv, axis=1)

            print(
                f"Set Features. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            self.mbb = np.array(getattr(branches, self.mbb_name), dtype="float32")

            # Add parametric variable
            # self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
            self.X_mass = getattr(branches, "X_mass")
            self.param_values = np.array(
                [getattr(branches, "X_mass")], dtype="float32"
            ).transpose()  # Init wrong parametric, later we will fill with random sample
            print("Got the param values")

        if hme_friend_file:
            print(f"Using HME Friend {hme_friend_file}")
            with uproot.open(hme_friend_file) as file:
                tree = file["Events"]
                branches_hme = tree.arrays(
                    [
                        "SingleLep_DeepHME_mass",
                        "SingleLep_DeepHME_mass_error",
                        "DoubleLep_DeepHME_mass",
                        "DoubleLep_DeepHME_mass_error",
                    ],
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                )
                if self.hmefriendfeatures_names != None:
                    self.hme = np.array(
                        [
                            getattr(branches_hme, feature_name)
                            for feature_name in self.hmefriendfeatures_names
                        ],
                        dtype="float32",
                    ).transpose()
                    self.features = np.append(self.features, self.hme, axis=1)

        self.features_no_param = self.features
        if self.use_parametric:
            self.features = np.append(self.features, self.param_values, axis=1)

        print(
            f"End read. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        # file = uproot.open(weight_name)
        with uproot.open(weight_name) as file:
            tree = file["weight_tree"]
            branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
            self.class_weight = np.array(
                getattr(branches, "class_weight"), dtype="float32"
            )
            self.adv_weight = np.array(getattr(branches, "adv_weight"), dtype="float32")
            self.class_target = np.array(
                getattr(branches, "class_target"), dtype="float32"
            )
            self.adv_target = np.array(getattr(branches, "adv_target"), dtype="float32")
            file.close()


@tf.function
def binary_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)


@tf.function
def binary_focal_crossentropy(target, output, y_class, y_pred_class):
    gamma = 2.0  # Default from keras
    # gamma = 0.0

    # Use signal from multiclass for focal check
    if y_class is not None:
        y_class = y_class[:, 0]
        y_pred_class = y_pred_class[:, 0]

    # Un-nest the output (currently in shape [ [1], [2], [3], ...] and we want in shape [1, 2, 3])
    y_true = target
    y_pred = output[:, 0]

    bce = binary_entropy(y_true, y_pred)

    # return bce

    # Calculate focal factor
    # p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    # focal_factor = tf.keras.ops.power(1.0 - p_t, gamma)

    # focal_bce = focal_factor * bce

    # We want to add more weight to BACKGROUNDS (1 - y_class in binary) when they predict to be SIGNAL
    # class_weight_factor = tf.keras.ops.power((1 - y_class) * y_pred_class, gamma)
    # We want when it is true background (y_class == 1) and when it is accurate (y_pred_class ~ 1) the weight is low

    # Need to normalize the y_pred_class to use whole range [-1, 1]
    # Model might learn to only use a small portion (0.5, 1.0) which can cause problems with gamma factor
    y_pred_class_mean = tf.math.reduce_mean(y_pred_class)
    y_pred_class_std = tf.math.reduce_std(y_pred_class)

    y_pred_class_norm = (y_pred_class - y_pred_class_mean) / (
        2 * y_pred_class_std
    ) + 0.5
    y_pred_class_norm_clipped = tf.clip_by_value(
        y_pred_class_norm, clip_value_min=0, clip_value_max=1
    )

    # 1 - y_class gives only background (when it is not signal)
    # y_pred_class_norm_clipped gives when the background is expected to be signal (incorrect)
    class_weight_factor = tf.keras.ops.power(
        (1 - y_class) * (y_pred_class_norm_clipped), gamma
    )
    # class_weight_factor = tf.expand_dims(class_weight_factor, axis=-1)
    # I lost my understanding of why this isn't (1 - y_pred_class)

    focal_bce = class_weight_factor * bce

    norm_factor = tf.math.reduce_sum(1 - y_class) / tf.math.reduce_sum(
        class_weight_factor
    )

    focal_bce = focal_bce * norm_factor

    # tf.print("Checking the pieces, y_class, y_pred_class_norm_clipped, y_true, y_pred, bce, class_weight, norm_factor")
    # tf.print(y_class)
    # tf.print(y_pred_class_norm_clipped)
    # tf.print(y_true)
    # tf.print(y_pred)
    # tf.print(bce)
    # tf.print(class_weight_factor)
    # tf.print(norm_factor)
    # tf.print(focal_bce)

    return focal_bce


@tf.function
def accuracy(target, output):
    target = tf.expand_dims(target, axis=-1)
    return tf.cast(tf.equal(target, tf.round(output)), tf.float32)


@tf.function
def categorical_crossentropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    output = output / tf.reduce_sum(output, axis=-1, keepdims=True)
    output = tf.clip_by_value(output, epsilon, 1.0 - epsilon)
    log_prob = tf.math.log(output)
    return -tf.reduce_sum(target * log_prob, axis=-1)


@tf.function
def categorical_accuracy(target, output):
    y_true = target
    y_pred = output

    y_true = tf.argmax(y_true, axis=-1)

    reshape_matches = False
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_true.dtype)

    y_true_org_shape = tf.shape(y_true)
    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
    ):
        y_true = tf.squeeze(y_true, -1)
        reshape_matches = True
    y_pred = tf.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, dtype=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    if reshape_matches:
        matches = tf.reshape(matches, y_true_org_shape)
    return matches


@tf.function
def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    if tf.reduce_any(tf.math.is_nan(var_2)):
        tf.print("WARNING: var_2 has NaN entries! Where is it?")
        tf.print(var_2)
        # Print index where it is NaN
        nan_indices = tf.where(tf.math.is_nan(var_2))
        tf.print("NaN indices:", nan_indices)

    if tf.reduce_any(tf.math.is_nan(var_1)):
        tf.print("WARNING: var_1 has NaN entries! Where is it?")
        tf.print(var_1)
        # Print index where it is NaN
        nan_indices = tf.where(tf.math.is_nan(var_1))
        tf.print("NaN indices:", nan_indices)

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])

    yy = tf.transpose(xx)
    amat = tf.math.abs(xx - yy)

    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])

    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx - yy)

    amatavg = tf.reduce_mean(amat * normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat * normedweight, axis=1)

    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat - minuend_1 - minuend_2 + tf.reduce_mean(amatavg * normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat - minuend_1 - minuend_2 + tf.reduce_mean(bmatavg * normedweight)

    ABavg = tf.abs(tf.reduce_mean(Amat * Bmat * normedweight, axis=1))
    AAavg = tf.reduce_mean(Amat * Amat * normedweight, axis=1)
    BBavg = tf.reduce_mean(Bmat * Bmat * normedweight, axis=1)

    # if tf.reduce_any(tf.math.is_nan(ABavg)):
    #     tf.print("WARNING: ABavg has NaN entries! Where is it?")
    #     tf.print(ABavg)
    #     # Print index where it is NaN
    #     nan_indices = tf.where(tf.math.is_nan(ABavg))
    #     tf.print("NaN indices:", nan_indices)

    # if tf.reduce_any(tf.math.is_nan(AAavg)):
    #     tf.print("WARNING: AAavg has NaN entries! Where is it?")
    #     tf.print(AAavg)
    #     # Print index where it is NaN
    #     nan_indices = tf.where(tf.math.is_nan(AAavg))
    #     tf.print("NaN indices:", nan_indices)

    # if tf.reduce_any(tf.math.is_nan(BBavg)):
    #     tf.print("WARNING: BBavg has NaN entries! Where is it?")
    #     tf.print(BBavg)
    #     # Print index where it is NaN
    #     nan_indices = tf.where(tf.math.is_nan(BBavg))
    #     tf.print("NaN indices:", nan_indices)

    # if tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)) == 0:
    #     tf.print("WARNING: Division by zero in distance correlation! Setting dCorr to 0")
    #     tf.print(tf.reduce_mean(AAavg*normedweight))
    #     tf.print(var_1)
    #     tf.print(tf.reduce_mean(BBavg*normedweight))
    #     tf.print(var_2)

    if power == 1:
        dCorr = tf.reduce_mean(ABavg * normedweight) / tf.math.sqrt(
            tf.abs(
                tf.reduce_mean(AAavg * normedweight)
                * tf.reduce_mean(BBavg * normedweight)
            )
            + 1e-12
        )
    elif power == 2:
        dCorr = (tf.reduce_mean(ABavg * normedweight)) ** 2 / (
            tf.abs(
                tf.reduce_mean(AAavg * normedweight)
                * tf.reduce_mean(BBavg * normedweight)
            )
            + 1e-12
        )
    else:
        dCorr = (
            tf.reduce_mean(ABavg * normedweight)
            / tf.math.sqrt(
                tf.reduce_mean(AAavg * normedweight)
                * tf.reduce_mean(BBavg * normedweight)
            )
            + 1e-12
        ) ** power

    if tf.math.is_nan(dCorr):
        tf.print("WARNING: dCorr is NaN! Setting to 0")
        tf.print("How did we get here?")
        tf.print(var_1)
        tf.print(var_2)
        tf.print(Amat)
        tf.print(Bmat)
        tf.print(tf.reduce_mean(ABavg * normedweight))
        tf.print(
            tf.reduce_mean(AAavg * normedweight) * tf.reduce_mean(BBavg * normedweight)
        )
        tf.print(
            tf.math.sqrt(
                tf.reduce_mean(AAavg * normedweight)
                * tf.reduce_mean(BBavg * normedweight)
            )
        )
        tf.print(dCorr)
        dCorr = tf.constant(
            0.0, dtype=tf.float32
        )  # Can't use tf.where, maybe because it is only a single constant? or some reason

    if dCorr < 0:
        tf.print("WARNING: dCorr is negative! Setting to 0")
        tf.print("How did we get here?")
        tf.print(var_1)
        tf.print(var_2)
        tf.print(Amat)
        tf.print(Bmat)
        tf.print(dCorr)
        dCorr = tf.constant(0.0, dtype=tf.float32)

    return dCorr


def ks_test(x, y):
    # x and y are nested, unnest them
    x_sorted = tf.sort(x[:, 0])
    y_sorted = tf.sort(y[:, 0])
    combined = tf.concat([x[:, 0], y[:, 0]], axis=0)
    sorted_combined = tf.sort(combined)

    n_x = tf.shape(x)[0]
    n_y = tf.shape(y)[0]

    cdf_x = tf.cast(
        tf.searchsorted(x_sorted, sorted_combined, side="right"), tf.float32
    ) / tf.cast(n_x, tf.float32)
    cdf_y = tf.cast(
        tf.searchsorted(y_sorted, sorted_combined, side="right"), tf.float32
    ) / tf.cast(n_y, tf.float32)

    delta = tf.abs(cdf_x - cdf_y)
    return tf.reduce_max(delta)


class EpochCounterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.epoch_counter.assign_add(1.0)
        return


class AdvOnlyCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        train_dataset,
        nSteps=100,
        TrackerWindowSize=10,
        on_batch=True,
        on_epoch=False,
        continue_training=False,
        quiet=False,
    ):
        self.train_dataset = train_dataset.repeat()
        self.trackerWindowSize = TrackerWindowSize
        self.nSteps = nSteps
        self.generator = self.looper()
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.continue_training = continue_training  # self.setup['continue_training'] When we continue, there is no point to skipping first epoch
        self.quiet = quiet

    def looper(self):
        yield
        n_window = 0
        nStep = 0
        for data in self.train_dataset:
            self.model._step_adv_only(data, True)
            n_window += 1
            if n_window == self.trackerWindowSize:
                if not self.quiet:
                    print(
                        f"\nSubmodule loss {self.model.adv_loss_tracker_submodule.result()} and accuracy {self.model.adv_accuracy_tracker_submodule.result()} after {nStep+1} nSteps"
                    )
                self.model.adv_loss_tracker_submodule.reset_state()
                self.model.adv_accuracy_tracker_submodule.reset_state()
                n_window = 0
            nStep += 1
            if nStep == self.nSteps:
                nStep = 0  # This is only a counter, so its fine to reset
                yield

    def on_batch_end(self, batch, logs=None):
        if self.nSteps <= 0:
            return
        if self.model.epoch_counter == 0.0 and not self.continue_training:
            return
        if self.on_batch:
            next(self.generator)

    def on_epoch_end(self, epoch, logs=None):
        if self.nSteps <= 0:
            return
        if self.model.epoch_counter == 0.0 and not self.continue_training:
            return
        if self.on_epoch:
            next(self.generator)


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        mode="min",
        min_delta=None,
        min_rel_delta=None,
        save_callback=None,
        patience=None,
        predicate=None,
        input_signature=None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.epochs_since_last_save = 0
        self.msg = None
        self.save_callback = save_callback
        self.patience = patience
        self.predicate = predicate
        self.input_signature = input_signature

        if os.path.exists(filepath):
            shutil.rmtree(filepath)

        self.best = None
        self.monitor_op = self._make_monitor_op(mode, min_delta, min_rel_delta)

    def _make_monitor_op(self, mode, min_delta, min_rel_delta):
        if mode == "min":
            if min_delta is None and min_rel_delta is None:
                return lambda current, best: best is None or best - current > 0
            if min_delta is None:
                return (
                    lambda current, best: best is None
                    or (best - current) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or best - current > min_delta
            return (
                lambda current, best: best is None
                or (best - current) > min_rel_delta * best
                or best - current > min_delta
            )
        elif mode == "max":
            if min_delta is None and min_rel_delta is None:
                return lambda current, best: best is None or current - best > 0
            if min_delta is None:
                return (
                    lambda current, best: best is None
                    or (current - best) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or current - best > min_delta
            return (
                lambda current, best: best is None
                or (current - best) > min_rel_delta * best
                or current - best > min_delta
            )
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

    def _print_msg(self):
        if self.msg is not None:
            print(self.msg)
            self.msg = None

    def on_epoch_begin(self, epoch, logs=None):
        self._print_msg()

    def on_train_end(self, logs=None):
        self._print_msg()
        if self.best == None:
            print("Never found a best model, just save the last one")

            dir_name = f"epoch_final.keras"
            onnx_dir_name = f"epoch_final.onnx"
            os.makedirs(self.filepath, exist_ok=True)
            path = os.path.join(self.filepath, f"{dir_name}")
            # self.model.save(path) # Don't save all keras, its a waste of space
            if self.input_signature is not None:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    self.model, self.input_signature, opset=13
                )
                onnx.save(onnx_model, os.path.join(self.filepath, f"{onnx_dir_name}"))

            path_best = os.path.join(self.filepath, "best.onnx")
            path_best_keras = os.path.join(self.filepath, "best.keras")
            if os.path.exists(path_best):
                os.remove(path_best)
                os.remove(path_best_keras)

            os.symlink(onnx_dir_name, path_best)
            os.symlink(dir_name, path_best_keras)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        current = logs.get(self.monitor)
        # if self.monitor_op(current, self.best) and (self.predicate is None or self.predicate(self.model, logs)):
        if self.predicate is None or self.predicate(
            self.model, logs
        ):  # Save all for now
            dir_name = f"epoch_{epoch+1}.keras"
            onnx_dir_name = f"epoch_{epoch+1}.onnx"
            os.makedirs(self.filepath, exist_ok=True)
            path = os.path.join(self.filepath, f"{dir_name}")
            if self.save_callback is None:
                # self.model.save(path) # Don't save all keras, its a waste of space
                if self.input_signature is not None:
                    onnx_model, _ = tf2onnx.convert.from_keras(
                        self.model, self.input_signature, opset=13
                    )
                    onnx.save(
                        onnx_model, os.path.join(self.filepath, f"{onnx_dir_name}")
                    )

            else:
                self.save_callback(self.model, path)
            path_best = os.path.join(self.filepath, "best.onnx")
            path_best_keras = os.path.join(self.filepath, "best.keras")
            if os.path.exists(path_best):
                os.remove(path_best)
                if os.path.exists(path_best_keras):
                    os.remove(path_best_keras)

            os.symlink(onnx_dir_name, path_best)
            # os.symlink(dir_name, path_best_keras)

            if self.verbose > 0:
                self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
                if self.best is None:
                    self.msg += f"= {current:.5f}."
                else:
                    self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
                self.msg += f" Saving model to {path}\n"
            self.best = current
            self.epochs_since_last_save = 0
        if self.patience is not None and self.epochs_since_last_save >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                if self.msg is None:
                    self.msg = "\n"
                self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."


class AdversarialModel(tf.keras.Model):
    """Goal: discriminate class0 vs class1 vs class2 without learning features that can guess class_adv"""

    def __init__(self, setup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup

        self.nClasses = setup["nClasses"]

        self.epoch_counter = tf.Variable(0.0)

        self.adv_optimizer = tf.keras.optimizers.Nadam(
            learning_rate=setup["adv_learning_rate"],
            weight_decay=setup["adv_weight_decay"],
        )

        self.apply_common_gradients = setup["apply_common_gradients"]

        self.class_grad_factor = setup["class_grad_factor"]

        self.class_loss = categorical_crossentropy
        self.class_accuracy = categorical_accuracy
        self.disco_loss = distance_corr
        self.lambda_disco = 1000

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.disco_loss_tracker = tf.keras.metrics.Mean(name="disco_loss")

        self.adv_grad_factor = setup["adv_grad_factor"]

        self.adv_loss = binary_focal_crossentropy
        self.adv_accuracy = accuracy

        self.adv_loss_tracker = tf.keras.metrics.Mean(name="adv_loss")
        self.adv_accuracy_tracker = tf.keras.metrics.Mean(name="adv_accuracy")

        self.adv_loss_tracker_submodule = tf.keras.metrics.Mean(name="adv_loss")
        self.adv_accuracy_tracker_submodule = tf.keras.metrics.Mean(name="adv_accuracy")

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

        self.adv_min_tracker = tf.keras.metrics.Mean(name="adv_min")
        self.adv_max_tracker = tf.keras.metrics.Mean(name="adv_max")

        self.other_class_min_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_min{n}")
            for n in range(self.nClasses)
            if n != 0
        ]
        self.other_class_max_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_max{n}")
            for n in range(self.nClasses)
            if n != 0
        ]

        self.common_layers = []

        def add_layer(layer_list, n_units, activation, name):
            layer = tf.keras.layers.Dense(
                n_units,
                activation=activation,
                name=name,
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
            )
            layer_list.append(layer)
            if setup["dropout"] > 0:
                dropout = tf.keras.layers.Dropout(
                    setup["dropout"], name=name + "_dropout"
                )
                layer_list.append(dropout)
            if setup["use_batch_norm"]:
                batch_norm = tf.keras.layers.BatchNormalization(
                    name=name + "_batch_norm"
                )
                layer_list.append(batch_norm)

        for n in range(setup["n_common_layers"]):
            add_layer(
                self.common_layers,
                setup["n_common_units"],
                setup["common_activation"],
                f"common_{n}",
            )

        self.class_layers = []
        self.adv_layers = []
        for n in range(setup["n_class_layers"]):
            add_layer(
                self.class_layers,
                setup["n_class_units"],
                setup["class_activation"],
                f"class_{n}",
            )
        for n in range(setup["n_adv_layers"]):
            add_layer(
                self.adv_layers,
                setup["n_adv_units"],
                setup["adv_activation"],
                f"adv_{n}",
            )

        self.class_output = tf.keras.layers.Dense(
            setup["nClasses"], activation="softmax", name="class_output"
        )

        self.adv_output = tf.keras.layers.Dense(
            1, activation="sigmoid", name="adv_output"
        )

        self.output_names = ["class_output", "adv_output"]

    def call(self, x):
        x_common = self.call_common(x)
        class_output = self.call_class(x_common)
        adv_output = self.call_adv(x_common)
        return class_output, adv_output

    def call_common(self, x):
        for layer in self.common_layers:
            x = layer(x)
        return x

    def call_class(self, x_common):
        x = x_common
        for layer in self.class_layers:
            x = layer(x)
        class_output = self.class_output(x)
        return class_output

    def call_adv(self, x_common):
        x = x_common
        for layer in self.adv_layers:
            x = layer(x)
        adv_output = self.adv_output(x)
        return adv_output

    def _step(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)
        y_adv = tf.cast(y[1], dtype=tf.float32)

        class_weight = tf.cast(y[2], dtype=tf.float32)
        adv_weight = tf.cast(y[3], dtype=tf.float32)
        # disco_weight only on background events
        disco_weight = tf.expand_dims(
            tf.cast(y[0][:, 0] == 0, dtype=tf.float32), axis=-1
        )

        def compute_losses():
            y_pred_class, y_pred_adv = self(x, training=training)

            class_loss_vec = self.class_loss(y_class, y_pred_class)
            disco_loss = (
                self.lambda_disco
                * disco_weight
                * self.disco_loss(y_pred_class[:, 0], y_adv, adv_weight)
            )

            class_loss = tf.reduce_mean(class_loss_vec * class_weight) + disco_loss

            adv_loss_vec = self.adv_loss(
                y_adv, y_pred_adv, y_class, y_pred_class
            )  # Focal loss
            # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
            # We want to apply some weights onto the adv loss vector
            # This is to have the SignalRegion and ControlRegion have equal weights

            adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)
            # tf.print("Final adv loss")
            # tf.print(adv_loss)

            # Experimental ks test loss
            # Combine both class and adv loss into one 'loss' and put into only one optimizer
            # new_loss = class_lost + k * ks_test

            # y_adv_SR_mask = (y_adv == 0) & (adv_weight != 0)
            # y_adv_CR_mask = (y_adv == 1) & (adv_weight != 0)
            # k = 0.0

            # new_loss = class_loss + k * ks_test(y_pred_class[y_adv_SR_mask], y_pred_class[y_adv_CR_mask])

            return (
                y_pred_class,
                class_loss_vec,
                class_loss,
                y_pred_adv,
                adv_loss_vec,
                adv_loss,
                disco_loss,
            )

        if training:
            with tf.GradientTape() as class_tape, tf.GradientTape() as adv_tape:
                (
                    y_pred_class,
                    class_loss_vec,
                    class_loss,
                    y_pred_adv,
                    adv_loss_vec,
                    adv_loss,
                    disco_loss,
                ) = compute_losses()
        else:
            (
                y_pred_class,
                class_loss_vec,
                class_loss,
                y_pred_adv,
                adv_loss_vec,
                adv_loss,
                disco_loss,
            ) = compute_losses()

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

        self.adv_min_tracker.update_state(tf.reduce_min(y_pred_adv))
        self.adv_max_tracker.update_state(tf.reduce_max(y_pred_adv))

        for n in range(self.nClasses):
            if n == 0:
                continue
            self.other_class_min_tracker[n - 1].update_state(
                tf.reduce_min(y_pred_class[:, n])
            )
            self.other_class_max_tracker[n - 1].update_state(
                tf.reduce_max(y_pred_class[:, n])
            )

        class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

        self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
        self.class_accuracy_tracker.update_state(
            class_accuracy_vec, sample_weight=class_weight
        )

        self.disco_loss_tracker.update_state(disco_loss)

        adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)

        self.adv_loss_tracker.update_state(adv_loss_vec, sample_weight=adv_weight)
        self.adv_accuracy_tracker.update_state(
            adv_accuracy_vec, sample_weight=adv_weight
        )

        if training:
            common_vars = [
                var for var in self.trainable_variables if "/common" in var.path
            ]
            class_vars = [
                var for var in self.trainable_variables if "/class" in var.path
            ]
            adv_vars = [var for var in self.trainable_variables if "/adv" in var.path]
            n_common_vars = len(common_vars)

            grad_class = class_tape.gradient(class_loss, common_vars + class_vars)
            grad_class_excl = grad_class[n_common_vars:]

            grad_adv = adv_tape.gradient(adv_loss, common_vars + adv_vars)
            grad_adv_excl = grad_adv[n_common_vars:]

            # tf.print("Hi artem, here is grad_class")
            # tf.print(len(grad_class))
            # for g_class, g_adv in zip(grad_class, grad_adv):
            #   if 'common' not in g_class.name: continue
            #   if 'MatMul' not in g_class.name: continue
            #   tf.print(f"Looking at norms for layers {g_class.name} and {g_adv.name}")
            #   tf.print(tf.norm(g_class))
            #   tf.print(tf.norm(g_adv))

            grad_common_growing_adv = [
                self.class_grad_factor * grad_class[i]
                - (self.adv_grad_factor + (0.1 * self.epoch_counter)) * grad_adv[i]
                for i in range(len(common_vars))
            ]

            grad_common = [
                self.class_grad_factor * grad_class[i]
                - self.adv_grad_factor * grad_adv[i]
                for i in range(len(common_vars))
            ]

            grad_common_no_adv = [grad_class[i] for i in range(len(common_vars))]

            grad_common_only_adv = [grad_adv[i] for i in range(len(common_vars))]

            grad_common_with_adv = [
                self.class_grad_factor * grad_class[i]
                + self.adv_grad_factor * grad_adv[i]
                for i in range(len(common_vars))
            ]

            grad_common_only_neg_adv = [
                (1.0) * grad_adv[i] for i in range(len(common_vars))
            ]

            @tf.function
            def cond_true_fn():
                if self.apply_common_gradients:
                    tf.cond(
                        self.epoch_counter <= 0.0
                        and not self.setup["continue_training"],
                        true_fn=apply_common_no_adv,
                        false_fn=apply_common,
                    )
                return

            @tf.function
            def apply_common_no_adv():
                self.optimizer.apply_gradients(
                    zip(grad_common_no_adv + grad_class_excl, common_vars + class_vars)
                )
                return

            def apply_common_only_neg_adv():
                self.optimizer.apply_gradients(
                    zip(grad_common_only_neg_adv, common_vars)
                )
                return

            @tf.function
            def apply_common():
                self.optimizer.apply_gradients(
                    zip(grad_common + grad_class_excl, common_vars + class_vars)
                )
                return

            @tf.function
            def flipflop_class_adv():
                tf.cond(
                    self.epoch_counter % 2 == 0,
                    true_fn=apply_common_no_adv,
                    false_fn=apply_common_only_neg_adv,
                )
                return

            @tf.function
            def cond_false_fn():
                return

            cond_true_fn()
            # flipflop_class_adv()
            # self.optimizer.apply_gradients(zip(grad_common_no_adv + grad_class_excl, common_vars + class_vars))
            # self.adv_optimizer.apply_gradients(zip(grad_common_with_adv + grad_adv_excl, common_vars + adv_vars))
            self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))

        return {m.name: m.result() for m in self.metrics}

    def _step_adv_only(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)
        y_adv = tf.cast(y[1], dtype=tf.float32)

        class_weight = tf.cast(y[2], dtype=tf.float32)
        adv_weight = tf.cast(y[3], dtype=tf.float32)

        def compute_losses(x_common):
            y_pred_class, y_pred_adv = self(x, training=training)
            # y_pred_adv = self.call_adv(x_common)

            # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv, None, None) # Focal loss
            adv_loss_vec = self.adv_loss(
                y_adv, y_pred_adv, y_class, y_pred_class
            )  # Focal loss
            # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
            # We want to apply some weights onto the adv loss vector
            # This is to have the SignalRegion and ControlRegion have equal weights

            adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)

            return y_pred_adv, adv_loss_vec, adv_loss

        if training:
            x_common = self.call_common(x)
            with tf.GradientTape() as adv_tape:
                y_pred_adv, adv_loss_vec, adv_loss = compute_losses(x_common)
        else:
            y_pred_adv, adv_loss_vec, adv_loss = compute_losses()

        adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)

        self.adv_loss_tracker_submodule.update_state(
            adv_loss_vec, sample_weight=adv_weight
        )
        self.adv_accuracy_tracker_submodule.update_state(
            adv_accuracy_vec, sample_weight=adv_weight
        )

        if training:
            adv_vars = [var for var in self.trainable_variables if "/adv" in var.path]

            grad_adv = adv_tape.gradient(adv_loss, adv_vars)

            self.adv_optimizer.apply_gradients(zip(grad_adv, adv_vars))

        return

    def train_step(self, data):
        # self.batch_counter = self.batch_counter.assign_add(1)
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)

    @property
    def metrics(self):
        metric_list = [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.adv_loss_tracker,
            self.adv_accuracy_tracker,
            self.class_min_tracker,
            self.class_max_tracker,
            self.adv_min_tracker,
            self.adv_max_tracker,
            self.disco_loss_tracker,
        ]
        metric_list = (
            metric_list
            + [x for x in self.other_class_min_tracker]
            + [x for x in self.other_class_max_tracker]
        )

        return metric_list


class DiscoModel(tf.keras.Model):
    """Goal: by using distance correlation, make the Classifier not learn the bb-mass information"""

    def __init__(self, setup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup

        self.nClasses = setup["nClasses"]

        self.epoch_counter = tf.Variable(0.0)

        self.class_loss = categorical_crossentropy
        self.class_accuracy = categorical_accuracy
        self.disco_loss = distance_corr
        self.lambda_disco = setup["disco_lambda_factor"]

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.disco_loss_tracker = tf.keras.metrics.Mean(name="disco_loss")

        self.combined_loss_tracker = tf.keras.metrics.Mean(name="combined_loss")

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

        self.other_class_min_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_min{n}")
            for n in range(self.nClasses)
            if n != 0
        ]
        self.other_class_max_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_max{n}")
            for n in range(self.nClasses)
            if n != 0
        ]

        self.disco_layers = []

        def add_layer(layer_list, n_units, activation, name):
            if setup["use_batch_norm"]:
                batch_norm = tf.keras.layers.BatchNormalization(
                    name=name + "_batch_norm"
                )
                layer_list.append(batch_norm)

            layer = tf.keras.layers.Dense(
                n_units,
                activation=activation,
                name=name,
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
                kernel_regularizer=tf.keras.regularizers.l1(0.1),
            )
            layer_list.append(layer)

            if setup["dropout"] > 0:
                dropout = tf.keras.layers.Dropout(
                    setup["dropout"], name=name + "_dropout"
                )
                layer_list.append(dropout)

        for n in range(setup.get("n_disco_layers", 0)):
            add_layer(
                self.disco_layers,
                setup["n_disco_units"],
                setup["disco_activation"],
                f"disco_{n}",
            )

        self.class_output = tf.keras.layers.Dense(
            setup["nClasses"], activation="softmax", name="class_output"
        )

        self.output_names = ["class_output"]

    def call(self, x):
        for layer in self.disco_layers:
            x = layer(x)
        class_output = self.class_output(x)
        return class_output

    def _step(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)
        mbb = tf.cast(y[1], dtype=tf.float32)

        class_weight = tf.cast(y[2], dtype=tf.float32)
        adv_weight = tf.cast(y[3], dtype=tf.float32)
        # disco_weight only on background events
        disco_weight = tf.cast(y[0][:, 0] == 0, dtype=tf.float32)

        def compute_losses():
            y_pred_class = self(x, training=training)
            # tf.print("Computing losses, take example 1")
            # tf.print(x[0], summarize=-1)
            # tf.print(y_class[0])
            # tf.print(y_pred_class[0])
            # tf.print("Finished safety check")

            class_loss_vec = self.class_loss(y_class, y_pred_class)
            disco_loss = (
                self.lambda_disco
                * disco_weight
                * self.disco_loss(mbb, y_pred_class[:, 0], adv_weight, 2)
            )

            class_loss = tf.reduce_mean(class_loss_vec * class_weight)

            combined_loss = tf.reduce_mean(class_loss_vec * class_weight + disco_loss)

            # tf.print("Returning class loss, disco loss, combined loss")
            # tf.print(class_loss)
            # tf.print(disco_loss)
            # tf.print(combined_loss)

            return y_pred_class, class_loss_vec, class_loss, disco_loss, combined_loss

        if training:
            with tf.GradientTape() as class_tape:
                y_pred_class, class_loss_vec, class_loss, disco_loss, combined_loss = (
                    compute_losses()
                )
        else:
            y_pred_class, class_loss_vec, class_loss, disco_loss, combined_loss = (
                compute_losses()
            )

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

        for n in range(self.nClasses):
            if n == 0:
                continue
            self.other_class_min_tracker[n - 1].update_state(
                tf.reduce_min(y_pred_class[:, n])
            )
            self.other_class_max_tracker[n - 1].update_state(
                tf.reduce_max(y_pred_class[:, n])
            )

        class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

        self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
        self.class_accuracy_tracker.update_state(
            class_accuracy_vec, sample_weight=class_weight
        )

        self.disco_loss_tracker.update_state(disco_loss)

        self.combined_loss_tracker.update_state(combined_loss)

        if training:
            grad = class_tape.gradient(combined_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        # self.batch_counter = self.batch_counter.assign_add(1)
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)

    @property
    def metrics(self):
        metric_list = [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.class_min_tracker,
            self.class_max_tracker,
            self.disco_loss_tracker,
            self.combined_loss_tracker,
        ]
        metric_list = (
            metric_list
            + [x for x in self.other_class_min_tracker]
            + [x for x in self.other_class_max_tracker]
        )

        return metric_list


def train_dnn(
    setup,
    training_file,
    weight_file,
    test_training_file,
    test_weight_file,
    output_folder,
    hme_friend_file=None,
    test_hme_friend_file=None,
):
    batch_size = setup["batch_size"]
    test_batch_size = setup["batch_size"]

    output_dnn_name = output_folder

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])
    if setup.get("listfeatures") != None:
        for list_feature in setup.get("listfeatures"):
            dw.AddInputFeaturesList(*list_feature)
    if setup.get("highlevelfeatures") != None:
        dw.AddHighLevelFeatures(setup.get("highlevelfeatures"))
    if setup.get("hmefeatures") != None:
        dw.AddHMEFeatures(setup.get("hmefeatures"))

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    dw.SetMbbName("bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino")

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
        hme_friend_file=hme_friend_file,
    )
    dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)
    # print(config_dict)
    # dw.DefineTrainTestSet(batch_size, 0.0)

    test_dw.ReadFile(
        test_training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
        hme_friend_file=test_hme_friend_file,
    )
    test_dw.ReadWeightFile(
        test_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )
    # dw_val.DefineTrainTestSet(val_batch_size, 0.0)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    if getattr(setup, "adv_model", False):
        # Adversarial Dataset
        batch_size = setup["batch_compression_factor"] * batch_size
        nClasses = setup["nClasses"]
        train_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dw.features,
                (
                    tf.one_hot(dw.class_target, nClasses),
                    dw.adv_target,
                    dw.class_weight,
                    dw.adv_weight,
                ),
            )
        ).batch(batch_size, drop_remainder=True)
        train_tf_dataset = train_tf_dataset.shuffle(
            len(train_tf_dataset), reshuffle_each_iteration=True
        )

        test_batch_size = setup["batch_compression_factor"] * test_batch_size
        test_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                test_dw.features,
                (
                    tf.one_hot(test_dw.class_target, nClasses),
                    test_dw.adv_target,
                    test_dw.class_weight,
                    test_dw.adv_weight,
                ),
            )
        ).batch(test_batch_size, drop_remainder=True)
        test_tf_dataset = test_tf_dataset.shuffle(
            len(test_tf_dataset), reshuffle_each_iteration=True
        )

    else:
        # Disco Dataset
        nClasses = setup["nClasses"]
        train_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dw.features,
                (
                    tf.one_hot(dw.class_target, nClasses),
                    dw.mbb,
                    dw.class_weight,
                    dw.adv_weight,
                ),
            )
        ).batch(batch_size, drop_remainder=True)
        train_tf_dataset = train_tf_dataset.shuffle(
            len(train_tf_dataset), reshuffle_each_iteration=True
        )

        test_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                test_dw.features,
                (
                    tf.one_hot(test_dw.class_target, nClasses),
                    test_dw.mbb,
                    test_dw.class_weight,
                    test_dw.adv_weight,
                ),
            )
        ).batch(test_batch_size, drop_remainder=True)
        test_tf_dataset = test_tf_dataset.shuffle(
            len(test_tf_dataset), reshuffle_each_iteration=True
        )

    @tf.function
    def new_param_map(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass
        parametric_mass_probability = (
            np.ones(len(dw.param_list)) * 1.0 / len(dw.param_list)
        )
        random_param_mass = tf.random.categorical(
            tf.math.log([list(parametric_mass_probability)]),
            tf.shape(features)[0],
            dtype=tf.int64,
        )

        mass_values = tf.constant(dw.param_list)
        mass_keys = tf.constant(np.arange(len(dw.param_list)))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
            default_value=-1,
        )

        actual_new_mass = table.lookup(random_param_mass)
        actual_new_mass = tf.cast(actual_new_mass, tf.float32)

        # Lastly we need to keep the signal events the correct mass
        class_targets = dataset[1][0]
        old_mass_mask = tf.cast(class_targets[:, 0], tf.float32)
        new_mass_mask = tf.cast((class_targets[:, 0] == 0), tf.float32)

        actual_mass = old_mass_mask * features[:, -1] + new_mass_mask * actual_new_mass
        actual_mass = tf.transpose(actual_mass)

        features = tf.concat([features[:, :-1], actual_mass], axis=-1)
        new_dataset = (features, dataset[1])
        return new_dataset

    if setup["UseParametric"]:
        train_tf_dataset = train_tf_dataset.map(new_param_map)
        test_tf_dataset = test_tf_dataset.map(new_param_map)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]

    if getattr(setup, "adv_model", False):
        # AdversarialModel
        model = AdversarialModel(setup)
        model.compile(
            loss=None,
            # optimizer=tf.keras.optimizers.AdamW(learning_rate=setup['learning_rate'],
            #                                     weight_decay=setup['weight_decay']))
            optimizer=tf.keras.optimizers.Nadam(
                learning_rate=setup["learning_rate"], weight_decay=setup["weight_decay"]
            ),
        )
        model(dw.features)
        model.summary()

        def save_predicate(model, logs):
            return (
                abs(logs["val_adv_accuracy"] - 0.5) < 0.001
            )  # How do we stop the model from always guessing 0.49 or 0.51?

        callbacks = [
            ModelCheckpoint(
                output_dnn_name,
                verbose=1,
                monitor="val_class_loss",
                mode="min",
                min_rel_delta=1e-3,
                # patience=setup['patience'], save_callback=None, predicate=save_predicate, input_signature=input_signature),
                patience=setup["patience"],
                save_callback=None,
                input_signature=input_signature,
            ),
            tf.keras.callbacks.CSVLogger(
                f"{output_dnn_name}_training_log.csv", append=True
            ),
            EpochCounterCallback(),
            AdvOnlyCallback(
                train_tf_dataset,
                nSteps=setup["adv_submodule_steps"],
                TrackerWindowSize=setup["adv_submodule_tracker"],
                on_batch=True,
                on_epoch=False,
                continue_training=setup["continue_training"],
                quiet=False,
            ),
            # AdvOnlyCallback(train_tf_dataset, nSteps=5000, TrackerWindowSize=100, on_batch=False, on_epoch=True, skip_epoch0=False, quiet=False),
        ]

    else:
        # DiscoModel
        model = DiscoModel(setup)
        model.compile(
            loss=None,
            optimizer=tf.keras.optimizers.Nadam(
                learning_rate=setup["learning_rate"], weight_decay=setup["weight_decay"]
            ),
        )
        model(dw.features)
        model.summary()

        callbacks = [
            ModelCheckpoint(
                output_dnn_name,
                verbose=1,
                monitor="val_class_loss",
                mode="min",
                min_rel_delta=1e-3,
                patience=100,
                save_callback=None,
                input_signature=input_signature,
            ),
        ]

    verbose = setup["verbose"] if "verbose" in setup else 0
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=False,
        callbacks=callbacks,
    )

    def PlotMetric(history, metric, output_folder):
        if metric not in history.history:
            print(f"Metric {metric} not found in history")
            return
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{metric}.pdf"), bbox_inches="tight")
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)

    PlotMetric(history, "adv_loss", output_folder)

    PlotMetric(history, "class_min", output_folder)

    PlotMetric(history, "class_max", output_folder)

    PlotMetric(history, "adv_min", output_folder)

    PlotMetric(history, "adv_max", output_folder)

    PlotMetric(history, "disco_loss", output_folder)

    PlotMetric(history, "combined_loss", output_folder)

    # model.save(f"{output_dnn_name}.keras") # Don't save all keras, its a waste of space

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, f"{output_dnn_name}.onnx")

    modelname_parity = [output_dnn_name, setup.get("iterate_cut", None)]
    features_config = {
        "features": dw.feature_names,
        "listfeatures": dw.listfeature_names,
        "highlevelfeatures": dw.highlevelfeatures_names,
        "hmefeatures": dw.hmefriendfeatures_names,
        "use_parametric": dw.use_parametric,
        "modelname_parity": modelname_parity,
        "parametric_list": dw.param_list,
        "model_setup": setup,
        "nClasses": setup["nClasses"],
        "nParity": 4,
    }

    with open(os.path.join(output_folder, "dnn_config.yaml"), "w") as file:
        yaml.dump(features_config, file)

    return


def validate_dnn(
    setup,
    validation_file,
    validation_weight_file,
    config_dict,
    output_file,
    model_name,
    model_config,
    hme_friend_file=None,
):
    import ROOT

    print(f"Model load {model_name}")
    sess = ort.InferenceSession(model_name)

    dnnConfig = {}
    with open(model_config, "r") as file:
        dnnConfig = yaml.safe_load(file)

    parametric_list = dnnConfig["parametric_list"]
    use_parametric = dnnConfig["UseParametric"]

    dw = DataWrapper()
    dw.AddInputFeatures(dnnConfig["features"])
    if dnnConfig["listfeatures"] != None:
        for list_feature in dnnConfig["listfeatures"]:
            dw.AddInputFeaturesList(*list_feature)
    if dnnConfig["highlevelfeatures"] != None:
        dw.AddHighLevelFeatures(dnnConfig["highlevelfeatures"])
    if dnnConfig["hmefeatures"] != None:
        dw.AddHMEFeatures(dnnConfig["hmefeatures"])

    dw.UseParametric(use_parametric)
    dw.SetParamList(parametric_list)

    dw.AddInputLabel("sample_type")

    dw.SetMbbName("bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino")

    dw.ReadFile(validation_file, hme_friend_file=hme_friend_file)
    dw.ReadWeightFile(validation_weight_file)
    # dw.validate_output(sess, model_name, output_file)

    # True mbb value
    mbb_value = dw.mbb
    print("mbb_value is ", mbb_value)

    para_masspoint_list = [300, 450, 550, 700, 800, 1000, 3000, 5000]  # [300, 450, 800]
    canvases = []
    for para_masspoint in para_masspoint_list:
        if dw.use_parametric:
            dw.SetPredictParamValue(para_masspoint)
        features = dw.features_paramSet if dw.use_parametric else dw.features_no_param

        pred = sess.run(None, {"x": features})
        pred_class = pred[0]
        pred_signal = pred_class[:, 0]
        pred_adv = pred[1][:, 0]

        adv_weight = dw.adv_weight
        class_weight = dw.class_weight

        nClasses = setup["nClasses"]
        adv_loss_vec = binary_focal_crossentropy(
            tf.cast(dw.adv_target, dtype=tf.float32),
            tf.cast(pred[1], dtype=tf.float32),
            tf.cast(tf.one_hot(dw.class_target, nClasses), dtype=tf.float32),
            tf.cast(pred_class, dtype=tf.float32),
        )
        adv_loss = round(np.average(adv_loss_vec, weights=adv_weight), 3)
        adv_accuracy_vec = accuracy(
            tf.cast(dw.adv_target, dtype=tf.float32), tf.cast(pred[1], dtype=tf.float32)
        )[:, 0]
        adv_accuracy = round(np.average(adv_accuracy_vec, weights=adv_weight), 3)

        class_loss_vec = categorical_crossentropy(
            tf.cast(tf.one_hot(dw.class_target, nClasses), dtype=tf.float32),
            tf.cast(pred_class, dtype=tf.float32),
        )
        class_loss = round(np.average(class_loss_vec, weights=class_weight), 3)
        class_accuracy_vec = categorical_accuracy(
            tf.cast(tf.one_hot(dw.class_target, nClasses), dtype=tf.float32),
            tf.cast(pred_class, dtype=tf.float32),
        )
        class_accuracy = round(np.average(class_accuracy_vec, weights=class_weight), 3)

        # Class Plots
        # Lets build Masks
        Sig_This_Mass = dw.X_mass == para_masspoint
        Sig_SR_mask = (Sig_This_Mass) & (dw.class_target == 0) & (dw.adv_target == 0)
        Sig_CR_high_mask = (dw.class_target == 0) & (dw.adv_target == 1)

        TT_SR_mask = (dw.class_target == 1) & (dw.adv_target == 0)
        TT_CR_high_mask = (dw.class_target == 1) & (dw.adv_target == 1)

        DY_SR_mask = (dw.class_target == 2) & (dw.adv_target == 0)
        DY_CR_high_mask = (dw.class_target == 2) & (dw.adv_target == 1)

        ST_SR_mask = (dw.class_target == 3) & (dw.adv_target == 0)
        ST_CR_high_mask = (dw.class_target == 3) & (dw.adv_target == 1)

        W_SR_mask = (dw.class_target == 4) & (dw.adv_target == 0)
        W_CR_high_mask = (dw.class_target == 4) & (dw.adv_target == 1)

        # Set class quantiles based on signal
        nQuantBins = 10
        quant_binning_class = np.zeros(
            nQuantBins + 1
        )  # Need +1 because 10 bins actually have 11 edges
        quant_binning_class[1:nQuantBins] = np.quantile(
            pred_signal[Sig_SR_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )  # Change list to something dynamic with nQuantBins
        quant_binning_class[-1] = 1.0
        print("We found quant binning class")
        print(quant_binning_class)
        print("From the signal prediction")
        print(pred_signal[Sig_SR_mask])

        quant_binning_adv = np.zeros(
            nQuantBins + 1
        )  # Need +1 because 10 bins actually have 11 edges
        quant_binning_adv[1:nQuantBins] = np.quantile(
            pred_adv[TT_SR_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )  # Change list to something dynamic with nQuantBins
        quant_binning_adv[-1] = 1.0
        print("We found quant binning adv")
        print(quant_binning_adv)

        mask_dict = {
            "Signal": {
                "SR": Sig_SR_mask,
                "CR_high": Sig_CR_high_mask,
            },
            "TT": {
                "SR": TT_SR_mask,
                "CR_high": TT_CR_high_mask,
            },
            "DY": {  # DY weight is turned off
                "SR": DY_SR_mask,
                "CR_high": DY_CR_high_mask,
            },
            "ST": {
                "SR": ST_SR_mask,
                "CR_high": ST_CR_high_mask,
            },
            "W": {
                "SR": W_SR_mask,
                "CR_high": W_CR_high_mask,
            },
        }

        canvases.append(ROOT.TCanvas("c1", "c1", 1200, 600 * len(mask_dict.keys())))
        canvas = canvases[-1]
        canvas.Divide(2, len(mask_dict.keys()))
        pads_list = []
        Class_SR_list = []
        Class_CR_list = []
        Adv_SR_list = []
        Adv_CR_list = []
        legend_list = []
        ratio_list = []
        pavetext_list = []
        for i, process_name in enumerate(mask_dict.keys()):
            canvas.cd(2 * i + 1)
            SR_mask = mask_dict[process_name]["SR"]
            CR_high_mask = mask_dict[process_name]["CR_high"]

            tmp_mbb_value_SR = mbb_value[SR_mask]
            tmp_classOutput_SR = pred_signal[SR_mask]
            tmp_advOutput_SR = pred_adv[SR_mask]
            tmp_SRdict = {
                "mbb": tmp_mbb_value_SR,
                "classOutput": tmp_classOutput_SR,
                "advOutput": tmp_advOutput_SR,
            }

            tmp_mbb_value_CR = mbb_value[CR_high_mask]
            tmp_classOutput_CR = pred_signal[CR_high_mask]
            tmp_advOutput_CR = pred_adv[CR_high_mask]
            tmp_CRdict = {
                "mbb": tmp_mbb_value_CR,
                "classOutput": tmp_classOutput_CR,
                "advOutput": tmp_advOutput_CR,
            }

            class_out_hist_SR, bins = np.histogram(
                pred_signal[SR_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[SR_mask],
            )
            class_out_hist_SR_w2, bins = np.histogram(
                pred_signal[SR_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[SR_mask] ** 2,
            )
            class_out_hist_CR_high, bins = np.histogram(
                pred_signal[CR_high_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[CR_high_mask],
            )
            class_out_hist_CR_high_w2, bins = np.histogram(
                pred_signal[CR_high_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[CR_high_mask] ** 2,
            )

            # Don't use weights for class
            # class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=quant_binning_class, range=(0.0, 1.0))
            # class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=quant_binning_class, range=(0.0, 1.0))

            # ROOT_ClassOutput_SR = ROOT.TH1D(f"ClassOutput_{process_name}_SR", f"ClassOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
            # ROOT_ClassOutput_CR_high = ROOT.TH1D(f"ClassOutput_{process_name}_CR_high", f"ClassOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)
            Class_SR_list.append(
                ROOT.TH1D(
                    f"ClassOutput_{process_name}_SR",
                    f"ClassOutput_{process_name}_SR",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            Class_CR_list.append(
                ROOT.TH1D(
                    f"ClassOutput_{process_name}_CR_high",
                    f"ClassOutput_{process_name}_CR_high",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            ROOT_ClassOutput_SR = Class_SR_list[-1]
            ROOT_ClassOutput_CR_high = Class_CR_list[-1]

            for binnum in range(nQuantBins):
                ROOT_ClassOutput_SR.SetBinContent(binnum + 1, class_out_hist_SR[binnum])
                ROOT_ClassOutput_SR.SetBinError(
                    binnum + 1, class_out_hist_SR_w2[binnum] ** (0.5)
                )

                ROOT_ClassOutput_CR_high.SetBinContent(
                    binnum + 1, class_out_hist_CR_high[binnum]
                )
                ROOT_ClassOutput_CR_high.SetBinError(
                    binnum + 1, class_out_hist_CR_high_w2[binnum] ** (0.5)
                )

            if ROOT_ClassOutput_SR.Integral() == 0:
                print(
                    f"Process {process_name} has no class entries, maybe the background doesn't exist?"
                )
                continue

            ROOT_ClassOutput_SR.Scale(1.0 / ROOT_ClassOutput_SR.Integral())
            ROOT_ClassOutput_CR_high.Scale(1.0 / ROOT_ClassOutput_CR_high.Integral())

            # p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
            pads_list.append(ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0))
            p1 = pads_list[-1]
            p1.SetTopMargin(0)
            p1.Draw()

            # p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
            pads_list.append(ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0))
            p2 = pads_list[-1]
            p2.SetTopMargin(0)
            p2.SetBottomMargin(0)
            p2.Draw()

            p1.cd()

            plotlabel = f"Class Output for {process_name} ParaMass {para_masspoint} GeV"
            ROOT_ClassOutput_SR.Draw()
            ROOT_ClassOutput_SR.SetTitle(plotlabel)
            ROOT_ClassOutput_SR.SetStats(0)
            min_val = max(
                0.0001,
                min(
                    ROOT_ClassOutput_SR.GetMinimum(),
                    ROOT_ClassOutput_CR_high.GetMinimum(),
                ),
            )
            max_val = max(
                ROOT_ClassOutput_SR.GetMaximum(), ROOT_ClassOutput_CR_high.GetMaximum()
            )
            # ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.1*min_val, 20) # 1000*max_val)
            ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.0001, 20)  # 1000*max_val)

            ROOT_ClassOutput_CR_high.SetLineColor(ROOT.kRed)
            ROOT_ClassOutput_CR_high.Draw("same")

            # legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
            legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
            legend = legend_list[-1]
            legend.AddEntry(ROOT_ClassOutput_SR, f"{process_name} m_bb SR")
            legend.AddEntry(ROOT_ClassOutput_CR_high, f"{process_name} m_bb CR High")
            legend.Draw()

            chi2_value = ROOT_ClassOutput_SR.Chi2Test(
                ROOT_ClassOutput_CR_high, option="WW"
            )

            # pt = ROOT.TPaveText(0.1,0.8,0.4,0.9, "NDC")
            pavetext_list.append(ROOT.TPaveText(0.1, 0.8, 0.4, 0.9, "NDC"))
            pt = pavetext_list[-1]
            pt.AddText(f"Loss {class_loss}")
            pt.AddText(f"Accuracy {class_accuracy}")
            pt.AddText(f"Chi2 {chi2_value}")
            pt.Draw()

            print(f"Setting canvas to log scale with range {min_val}, {max_val}")
            p1.SetLogy()
            p1.SetGrid()

            p2.cd()

            # ROOT_ClassOutput_Ratio = ROOT_ClassOutput_SR.Clone()
            ratio_list.append(ROOT_ClassOutput_SR.Clone())
            ROOT_ClassOutput_Ratio = ratio_list[-1]
            ROOT_ClassOutput_Ratio.Divide(ROOT_ClassOutput_CR_high)
            ROOT_ClassOutput_Ratio.SetTitle("Ratio (SR/CR)")
            ROOT_ClassOutput_Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
            ROOT_ClassOutput_Ratio.GetYaxis().SetNdivisions(5)
            ROOT_ClassOutput_Ratio.Draw()

            p2.SetGrid()

            canvas.cd(2 * i + 2)

            # Adv Plots
            adv_out_hist_SR, bins = np.histogram(
                pred_adv[SR_mask],
                bins=quant_binning_adv,
                range=(0.0, 1.0),
                weights=adv_weight[SR_mask],
            )
            adv_out_hist_SR_w2, bins = np.histogram(
                pred_adv[SR_mask],
                bins=quant_binning_adv,
                range=(0.0, 1.0),
                weights=adv_weight[SR_mask] ** 2,
            )
            adv_out_hist_CR_high, bins = np.histogram(
                pred_adv[CR_high_mask],
                bins=quant_binning_adv,
                range=(0.0, 1.0),
                weights=adv_weight[CR_high_mask],
            )
            adv_out_hist_CR_high_w2, bins = np.histogram(
                pred_adv[CR_high_mask],
                bins=quant_binning_adv,
                range=(0.0, 1.0),
                weights=adv_weight[CR_high_mask] ** 2,
            )

            ROOT_AdvOutput_SR = ROOT.TH1D(
                f"AdvOutput_{process_name}_SR",
                f"AdvOutput_{process_name}_SR",
                nQuantBins,
                0.0,
                1.0,
            )
            ROOT_AdvOutput_CR_high = ROOT.TH1D(
                f"AdvOutput_{process_name}_CR_high",
                f"AdvOutput_{process_name}_CR_high",
                nQuantBins,
                0.0,
                1.0,
            )
            Adv_SR_list.append(
                ROOT.TH1D(
                    f"AdvOutput_{process_name}_SR",
                    f"AdvOutput_{process_name}_SR",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            Adv_CR_list.append(
                ROOT.TH1D(
                    f"AdvOutput_{process_name}_CR_high",
                    f"AdvOutput_{process_name}_CR_high",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            ROOT_AdvOutput_SR = Adv_SR_list[-1]
            ROOT_AdvOutput_CR_high = Adv_CR_list[-1]

            for binnum in range(nQuantBins):
                ROOT_AdvOutput_SR.SetBinContent(binnum + 1, adv_out_hist_SR[binnum])
                ROOT_AdvOutput_SR.SetBinError(
                    binnum + 1, adv_out_hist_SR_w2[binnum] ** (0.5)
                )

                ROOT_AdvOutput_CR_high.SetBinContent(
                    binnum + 1, adv_out_hist_CR_high[binnum]
                )
                ROOT_AdvOutput_CR_high.SetBinError(
                    binnum + 1, adv_out_hist_CR_high_w2[binnum] ** (0.5)
                )

            if ROOT_AdvOutput_SR.Integral() == 0:
                print(
                    f"Process {process_name} has no adv entries, maybe the weights are all 0 for adv?"
                )
                continue

            ROOT_AdvOutput_SR.Scale(1.0 / ROOT_AdvOutput_SR.Integral())
            ROOT_AdvOutput_CR_high.Scale(1.0 / ROOT_AdvOutput_CR_high.Integral())

            # p3 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
            pads_list.append(ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0))
            p3 = pads_list[-1]
            p3.SetTopMargin(0)
            p3.Draw()

            # p4 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
            pads_list.append(ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0))
            p4 = pads_list[-1]
            p4.SetTopMargin(0)
            p4.SetBottomMargin(0)
            p4.Draw()

            p3.cd()

            plotlabel = f"Adv Output for {process_name}"
            ROOT_AdvOutput_SR.Draw()
            ROOT_AdvOutput_SR.SetTitle(plotlabel)
            ROOT_AdvOutput_SR.SetStats(0)
            min_val = max(
                0.0001,
                min(
                    ROOT_AdvOutput_SR.GetMinimum(), ROOT_AdvOutput_CR_high.GetMinimum()
                ),
            )
            max_val = max(
                ROOT_AdvOutput_SR.GetMaximum(), ROOT_AdvOutput_CR_high.GetMaximum()
            )
            # ROOT_AdvOutput_SR.GetYaxis().SetRangeUser(0.8*min_val, 20) # 1.5*max_val)
            ROOT_AdvOutput_SR.GetYaxis().SetRangeUser(0.000001, 20)  # 1.5*max_val)

            ROOT_AdvOutput_CR_high.SetLineColor(ROOT.kRed)
            ROOT_AdvOutput_CR_high.Draw("same")

            # legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
            legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
            legend = legend_list[-1]
            legend.AddEntry(ROOT_AdvOutput_SR, f"{process_name} m_bb SR")
            legend.AddEntry(ROOT_AdvOutput_CR_high, f"{process_name} m_bb CR High")
            legend.Draw()

            chi2_value = ROOT_AdvOutput_SR.Chi2Test(ROOT_AdvOutput_CR_high, option="WW")

            # pt = ROOT.TPaveText(0.1,0.7,0.4,0.9, "NDC")
            pavetext_list.append(ROOT.TPaveText(0.1, 0.7, 0.4, 0.9, "NDC"))
            pt = pavetext_list[-1]
            pt.AddText(f"Loss {adv_loss}")
            pt.AddText(f"Accuracy {adv_accuracy}")
            pt.AddText(f"Chi2 {chi2_value}")
            pt.Draw()

            print(f"Setting canvas to log scale with range {min_val}, {max_val}")
            p3.SetLogy()
            p3.SetGrid()

            p4.cd()

            # ROOT_AdvOutput_Ratio = ROOT_AdvOutput_SR.Clone()
            ratio_list.append(ROOT_AdvOutput_SR.Clone())
            ROOT_AdvOutput_Ratio = ratio_list[-1]
            ROOT_AdvOutput_Ratio.Divide(ROOT_AdvOutput_CR_high)
            ROOT_AdvOutput_Ratio.SetTitle("Ratio (SR/CR)")
            ROOT_AdvOutput_Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
            ROOT_AdvOutput_Ratio.GetYaxis().SetNdivisions(5)
            ROOT_AdvOutput_Ratio.Draw()

            p4.SetGrid()

        # canvas.SaveAs(output_file)

        if para_masspoint == para_masspoint_list[0]:
            canvas.Print(f"{output_file}(", f"Title:Mass {para_masspoint} GeV")
            print("Saved [")
        elif para_masspoint == para_masspoint_list[-1]:
            canvas.Print(f"{output_file})", f"Title:Mass {para_masspoint} GeV")
            print("Saved ]")
        else:
            canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} GeV")
        print(f"Saved mass {para_masspoint}")

        # pdf.NewPage()
        # pdf.SetTitle(f"Mass {para_masspoint} GeV")
        # canvas.Paint()

        canvas.Close()


def validate_disco_dnn(
    setup,
    validation_file,
    validation_weight_file,
    config_dict,
    output_file,
    model_name,
    model_config,
    hme_friend_file=None,
):
    print(f"Model load {model_name}")
    sess = ort.InferenceSession(model_name)

    dnnConfig = {}
    with open(model_config, "r") as file:
        dnnConfig = yaml.safe_load(file)

    parametric_list = dnnConfig["parametric_list"]
    use_parametric = dnnConfig["UseParametric"]

    dw = DataWrapper()
    dw.AddInputFeatures(dnnConfig["features"])
    if dnnConfig["listfeatures"] != None:
        for list_feature in dnnConfig["listfeatures"]:
            dw.AddInputFeaturesList(*list_feature)
    if dnnConfig["highlevelfeatures"] != None:
        dw.AddHighLevelFeatures(dnnConfig["highlevelfeatures"])
    if dnnConfig["hmefeatures"] != None:
        dw.AddHMEFeatures(dnnConfig["hmefeatures"])

    dw.UseParametric(use_parametric)
    dw.SetParamList(parametric_list)

    dw.AddInputLabel("sample_type")

    dw.SetMbbName("bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino")

    dw.ReadFile(validation_file, hme_friend_file=hme_friend_file)
    dw.ReadWeightFile(validation_weight_file)
    # dw.validate_output(sess, model_name, output_file)

    # True mbb value
    mbb_value = dw.mbb
    print("mbb_value is ", mbb_value)

    para_masspoint_list = [300, 450, 550, 700, 800, 1000, 3000, 5000]  # [300, 450, 800]
    canvases = []
    for para_masspoint in para_masspoint_list:
        if dw.use_parametric:
            dw.SetPredictParamValue(para_masspoint)
        features = dw.features_paramSet if dw.use_parametric else dw.features_no_param

        pred = sess.run(None, {"x": features})
        pred_class = pred[0]
        pred_signal = pred_class[:, 0]

        class_weight = dw.class_weight

        nClasses = setup["nClasses"]
        class_loss_vec = categorical_crossentropy(
            tf.cast(tf.one_hot(dw.class_target, nClasses), dtype=tf.float32),
            tf.cast(pred_class, dtype=tf.float32),
        )
        class_loss = round(np.average(class_loss_vec, weights=class_weight), 3)
        class_accuracy_vec = categorical_accuracy(
            tf.cast(tf.one_hot(dw.class_target, nClasses), dtype=tf.float32),
            tf.cast(pred_class, dtype=tf.float32),
        )
        class_accuracy = round(np.average(class_accuracy_vec, weights=class_weight), 3)

        # Class Plots
        # Lets build Masks
        Sig_This_Mass = dw.X_mass == para_masspoint
        Sig_SR_mask = (Sig_This_Mass) & (dw.class_target == 0) & (dw.adv_target == 0)
        Sig_CR_high_mask = (dw.class_target == 0) & (dw.adv_target == 1)

        TT_SR_mask = (dw.class_target == 1) & (dw.adv_target == 0)
        TT_CR_high_mask = (dw.class_target == 1) & (dw.adv_target == 1)

        DY_SR_mask = (dw.class_target == 2) & (dw.adv_target == 0)
        DY_CR_high_mask = (dw.class_target == 2) & (dw.adv_target == 1)

        ST_SR_mask = (dw.class_target == 3) & (dw.adv_target == 0)
        ST_CR_high_mask = (dw.class_target == 3) & (dw.adv_target == 1)

        W_SR_mask = (dw.class_target == 4) & (dw.adv_target == 0)
        W_CR_high_mask = (dw.class_target == 4) & (dw.adv_target == 1)

        # Set class quantiles based on signal
        nQuantBins = 10
        quant_binning_class = np.zeros(
            nQuantBins + 1
        )  # Need +1 because 10 bins actually have 11 edges
        quant_binning_class[1:nQuantBins] = np.quantile(
            pred_signal[Sig_SR_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )  # Change list to something dynamic with nQuantBins
        quant_binning_class[-1] = 1.0
        print("We found quant binning class")
        print(quant_binning_class)
        print("From the signal prediction")
        print(pred_signal[Sig_SR_mask])

        mask_dict = {
            "Signal": {
                "SR": Sig_SR_mask,
                "CR_high": Sig_CR_high_mask,
            },
            "TT": {
                "SR": TT_SR_mask,
                "CR_high": TT_CR_high_mask,
            },
            "DY": {  # DY weight is turned off
                "SR": DY_SR_mask,
                "CR_high": DY_CR_high_mask,
            },
            "ST": {
                "SR": ST_SR_mask,
                "CR_high": ST_CR_high_mask,
            },
            "W": {
                "SR": W_SR_mask,
                "CR_high": W_CR_high_mask,
            },
        }

        canvases.append(ROOT.TCanvas("c1", "c1", 1200, 600 * len(mask_dict.keys())))
        canvas = canvases[-1]
        canvas.Divide(1, len(mask_dict.keys()))
        pads_list = []
        Class_SR_list = []
        Class_CR_list = []
        legend_list = []
        ratio_list = []
        pavetext_list = []
        for i, process_name in enumerate(mask_dict.keys()):
            canvas.cd(i + 1)
            SR_mask = mask_dict[process_name]["SR"]
            CR_high_mask = mask_dict[process_name]["CR_high"]

            tmp_mbb_value_SR = mbb_value[SR_mask]
            tmp_classOutput_SR = pred_signal[SR_mask]
            tmp_SRdict = {"mbb": tmp_mbb_value_SR, "classOutput": tmp_classOutput_SR}

            tmp_mbb_value_CR = mbb_value[CR_high_mask]
            tmp_classOutput_CR = pred_signal[CR_high_mask]
            tmp_CRdict = {"mbb": tmp_mbb_value_CR, "classOutput": tmp_classOutput_CR}

            class_out_hist_SR, bins = np.histogram(
                pred_signal[SR_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[SR_mask],
            )
            class_out_hist_SR_w2, bins = np.histogram(
                pred_signal[SR_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[SR_mask] ** 2,
            )
            class_out_hist_CR_high, bins = np.histogram(
                pred_signal[CR_high_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[CR_high_mask],
            )
            class_out_hist_CR_high_w2, bins = np.histogram(
                pred_signal[CR_high_mask],
                bins=quant_binning_class,
                range=(0.0, 1.0),
                weights=class_weight[CR_high_mask] ** 2,
            )

            # Don't use weights for class
            # class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=quant_binning_class, range=(0.0, 1.0))
            # class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=quant_binning_class, range=(0.0, 1.0))

            # ROOT_ClassOutput_SR = ROOT.TH1D(f"ClassOutput_{process_name}_SR", f"ClassOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
            # ROOT_ClassOutput_CR_high = ROOT.TH1D(f"ClassOutput_{process_name}_CR_high", f"ClassOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)
            Class_SR_list.append(
                ROOT.TH1D(
                    f"ClassOutput_{process_name}_SR",
                    f"ClassOutput_{process_name}_SR",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            Class_CR_list.append(
                ROOT.TH1D(
                    f"ClassOutput_{process_name}_CR_high",
                    f"ClassOutput_{process_name}_CR_high",
                    nQuantBins,
                    0.0,
                    1.0,
                )
            )
            ROOT_ClassOutput_SR = Class_SR_list[-1]
            ROOT_ClassOutput_CR_high = Class_CR_list[-1]

            for binnum in range(nQuantBins):
                ROOT_ClassOutput_SR.SetBinContent(binnum + 1, class_out_hist_SR[binnum])
                ROOT_ClassOutput_SR.SetBinError(
                    binnum + 1, class_out_hist_SR_w2[binnum] ** (0.5)
                )

                ROOT_ClassOutput_CR_high.SetBinContent(
                    binnum + 1, class_out_hist_CR_high[binnum]
                )
                ROOT_ClassOutput_CR_high.SetBinError(
                    binnum + 1, class_out_hist_CR_high_w2[binnum] ** (0.5)
                )

            if ROOT_ClassOutput_SR.Integral() == 0:
                print(
                    f"Process {process_name} has no class entries, maybe the background doesn't exist?"
                )
                continue

            ROOT_ClassOutput_SR.Scale(1.0 / ROOT_ClassOutput_SR.Integral())
            ROOT_ClassOutput_CR_high.Scale(1.0 / ROOT_ClassOutput_CR_high.Integral())

            # p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
            pads_list.append(ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0))
            p1 = pads_list[-1]
            p1.SetTopMargin(0)
            p1.Draw()

            # p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
            pads_list.append(ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0))
            p2 = pads_list[-1]
            p2.SetTopMargin(0)
            p2.SetBottomMargin(0)
            p2.Draw()

            p1.cd()

            plotlabel = f"Class Output for {process_name} ParaMass {para_masspoint} GeV"
            ROOT_ClassOutput_SR.Draw()
            ROOT_ClassOutput_SR.SetTitle(plotlabel)
            ROOT_ClassOutput_SR.SetStats(0)
            min_val = max(
                0.0001,
                min(
                    ROOT_ClassOutput_SR.GetMinimum(),
                    ROOT_ClassOutput_CR_high.GetMinimum(),
                ),
            )
            max_val = max(
                ROOT_ClassOutput_SR.GetMaximum(), ROOT_ClassOutput_CR_high.GetMaximum()
            )
            # ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.1*min_val, 20) # 1000*max_val)
            ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.0001, 20)  # 1000*max_val)

            ROOT_ClassOutput_CR_high.SetLineColor(ROOT.kRed)
            ROOT_ClassOutput_CR_high.Draw("same")

            # legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
            legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
            legend = legend_list[-1]
            legend.AddEntry(ROOT_ClassOutput_SR, f"{process_name} m_bb SR")
            legend.AddEntry(ROOT_ClassOutput_CR_high, f"{process_name} m_bb CR High")
            legend.Draw()

            chi2_value = ROOT_ClassOutput_SR.Chi2Test(
                ROOT_ClassOutput_CR_high, option="WW"
            )

            # pt = ROOT.TPaveText(0.1,0.8,0.4,0.9, "NDC")
            pavetext_list.append(ROOT.TPaveText(0.1, 0.8, 0.4, 0.9, "NDC"))
            pt = pavetext_list[-1]
            pt.AddText(f"Loss {class_loss}")
            pt.AddText(f"Accuracy {class_accuracy}")
            pt.AddText(f"Chi2 {chi2_value}")
            pt.Draw()

            print(f"Setting canvas to log scale with range {min_val}, {max_val}")
            p1.SetLogy()
            p1.SetGrid()

            p2.cd()

            # ROOT_ClassOutput_Ratio = ROOT_ClassOutput_SR.Clone()
            ratio_list.append(ROOT_ClassOutput_SR.Clone())
            ROOT_ClassOutput_Ratio = ratio_list[-1]
            ROOT_ClassOutput_Ratio.Divide(ROOT_ClassOutput_CR_high)
            ROOT_ClassOutput_Ratio.SetTitle("Ratio (SR/CR)")
            ROOT_ClassOutput_Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
            ROOT_ClassOutput_Ratio.GetYaxis().SetNdivisions(5)
            ROOT_ClassOutput_Ratio.Draw()

            p2.SetGrid()

        # canvas.SaveAs(output_file)

        if para_masspoint == para_masspoint_list[0]:
            canvas.Print(f"{output_file}(", f"Title:Mass {para_masspoint} GeV")
            print("Saved [")
        elif para_masspoint == para_masspoint_list[-1]:
            canvas.Print(f"{output_file})", f"Title:Mass {para_masspoint} GeV")
            print("Saved ]")
        else:
            canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} GeV")
        print(f"Saved mass {para_masspoint}")

        # pdf.NewPage()
        # pdf.SetTitle(f"Mass {para_masspoint} GeV")
        # canvas.Paint()

        canvas.Close()


def train_step2(
    setup,
    training_file,
    weight_file,
    config_dict,
    test_training_file,
    test_weight_file,
    test_config_dict,
    output_folder,
    hme_friend_file=None,
    test_hme_friend_file=None,
):
    batch_size = config_dict["meta_data"]["batch_dict"]["batch_size"]
    test_batch_size = test_config_dict["meta_data"]["batch_dict"]["batch_size"]

    output_dnn_name = output_folder

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])
    if setup.get("listfeatures") != None:
        for list_feature in setup.get("listfeatures"):
            dw.AddInputFeaturesList(*list_feature)
    if setup.get("highlevelfeatures") != None:
        dw.AddHighLevelFeatures(setup.get("highlevelfeatures"))
    if setup.get("hmefeatures") != None:
        dw.AddHMEFeatures(setup.get("hmefeatures"))

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    dw.SetMbbName("bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino")

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
        hme_friend_file=hme_friend_file,
    )
    dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)
    print(config_dict)
    # dw.DefineTrainTestSet(batch_size, 0.0)

    test_dw.ReadFile(
        test_training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
        hme_friend_file=test_hme_friend_file,
    )
    test_dw.ReadWeightFile(
        test_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )
    # dw_val.DefineTrainTestSet(val_batch_size, 0.0)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    # Disco Dataset
    nClasses = setup["nClasses"]
    # For step 2, force a binary class target, replace anything >= 1 with 1
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dw.features,
            (
                tf.one_hot(tf.where(dw.class_target >= 1, 1, 0), nClasses),
                dw.mbb,
                dw.class_weight,
            ),
        )
    ).batch(batch_size, drop_remainder=True)
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )

    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_dw.features,
            (
                tf.one_hot(tf.where(test_dw.class_target >= 1, 1, 0), nClasses),
                test_dw.mbb,
                test_dw.class_weight,
            ),
        )
    ).batch(test_batch_size, drop_remainder=True)
    test_tf_dataset = test_tf_dataset.shuffle(
        len(test_tf_dataset), reshuffle_each_iteration=True
    )

    @tf.function
    def new_param_map(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass
        parametric_mass_probability = (
            np.ones(len(dw.param_list)) * 1.0 / len(dw.param_list)
        )
        random_param_mass = tf.random.categorical(
            tf.math.log([list(parametric_mass_probability)]),
            tf.shape(features)[0],
            dtype=tf.int64,
        )

        mass_values = tf.constant(dw.param_list)
        mass_keys = tf.constant(np.arange(len(dw.param_list)))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
            default_value=-1,
        )

        actual_new_mass = table.lookup(random_param_mass)
        actual_new_mass = tf.cast(actual_new_mass, tf.float32)

        # Lastly we need to keep the signal events the correct mass
        class_targets = dataset[1][0]
        old_mass_mask = tf.cast(class_targets[:, 0], tf.float32)
        new_mass_mask = tf.cast((class_targets[:, 0] == 0), tf.float32)

        actual_mass = old_mass_mask * features[:, -1] + new_mass_mask * actual_new_mass
        actual_mass = tf.transpose(actual_mass)

        features = tf.concat([features[:, :-1], actual_mass], axis=-1)
        new_dataset = (features, dataset[1])
        return new_dataset

    if setup["UseParametric"]:
        train_tf_dataset = train_tf_dataset.map(new_param_map)
        test_tf_dataset = test_tf_dataset.map(new_param_map)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]

    # DiscoModel
    model = DiscoModel(setup)
    model.compile(
        loss=None,
        optimizer=tf.keras.optimizers.Nadam(
            learning_rate=setup["learning_rate"], weight_decay=setup["weight_decay"]
        ),
    )
    model(dw.features)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            output_dnn_name,
            verbose=1,
            monitor="val_class_loss",
            mode="min",
            min_rel_delta=1e-3,
            patience=100,
            save_callback=None,
            input_signature=input_signature,
        ),
    ]

    verbose = setup["verbose"] if "verbose" in setup else 0
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=False,
        callbacks=callbacks,
    )

    def PlotMetric(history, metric, output_folder):
        if metric not in history.history:
            print(f"Metric {metric} not found in history")
            return
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{metric}.pdf"), bbox_inches="tight")
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)

    PlotMetric(history, "class_min", output_folder)

    PlotMetric(history, "class_max", output_folder)

    PlotMetric(history, "disco_loss", output_folder)

    PlotMetric(history, "combined_loss", output_folder)

    # model.save(f"{output_dnn_name}.keras") # Don't save all keras, its a waste of space

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, f"{output_dnn_name}.onnx")

    modelname_parity = [output_dnn_name, config_dict["meta_data"]["iterate_cut"]]
    features_config = {
        "features": dw.feature_names,
        "listfeatures": dw.listfeature_names,
        "highlevelfeatures": dw.highlevelfeatures_names,
        "hmefeatures": dw.hmefriendfeatures_names,
        "use_parametric": dw.use_parametric,
        "modelname_parity": modelname_parity,
        "parametric_list": dw.param_list,
        "model_setup": setup,
        "nClasses": setup["nClasses"],
        "nParity": 4,
    }

    with open(os.path.join(output_folder, "dnn_config.yaml"), "w") as file:
        yaml.dump(features_config, file)

    return
