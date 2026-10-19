import tensorflow as tf
import numpy as np
import uproot
import os
import shutil
import yaml
import tf2onnx
import onnx
import copy
import psutil
import matplotlib.pyplot as plt
import onnxruntime as ort
import ROOT
import sklearn.metrics

from Studies.DNN.DNN_Class import distance_corr


class DataWrapper:
    def __init__(self):
        print("Init data wrapper")

        self.feature_names = None

        self.features_no_param = None
        self.features = None

        self.param_values = None

        self.labels = None

        self.class_weight = None
        self.class_target = None
        self.multiclass_weight = None

        self.res2b = None
        self.recovery = None
        self.boosted = None

        self.param_list = [
            300,
            400,
            500,
            600,
            700,
            800,
            1000,
        ]
        self.use_parametric = False

        self.features_paramSet = None

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

    def ReadFile(self, file_name, entry_start=None, entry_stop=None):
        if self.feature_names == None:
            print("Unknown branches to read! DefineInputFeatures first!")
            return

        print(f"Reading file {file_name}")

        features_to_load = self.feature_names.copy()

        features_to_load.append("X_mass")
        features_to_load.append("weight_Central")
        features_to_load.append("class_value")
        features_to_load.append("res2b")
        features_to_load.append("recovery")
        features_to_load.append("boosted")

        print(f"Only loading these features {features_to_load}")

        print(
            f"Going to open file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                features_to_load, entry_start=entry_start, entry_stop=entry_stop
            )

            print(
                f"Loaded branches. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            self.features = np.array(
                [
                    getattr(branches, feature_name)
                    for feature_name in self.feature_names
                ],
                dtype="float32",
            ).transpose()

            print(
                f"Set Features. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            # Add parametric variable
            # self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
            self.X_mass = getattr(branches, "X_mass")
            self.physics_weight = getattr(branches, "weight_Central")
            self.class_value = getattr(branches, "class_value")
            self.res2b = getattr(branches, "res2b")
            self.recovery = getattr(branches, "recovery")
            self.boosted = getattr(branches, "boosted")
            self.param_values = np.array(
                [getattr(branches, "X_mass")], dtype="float32"
            ).transpose()  # Init wrong parametric, later we will fill with random sample
            print("Got the param values")

        self.features_no_param = self.features
        if self.use_parametric:
            self.features = np.append(self.features, self.param_values, axis=1)

        print(
            f"End read. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        with uproot.open(weight_name) as file:
            tree = file["weight_tree"]
            branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
            self.class_weight = np.array(
                getattr(branches, "class_weight"), dtype="float32"
            )
            self.class_target = np.array(
                getattr(branches, "class_target"), dtype="float32"
            )
            self.multiclass_weight = np.array(
                getattr(branches, "multiclass_weight", None), dtype="float32"
            )
            file.close()

    def GetHME(self, file_name, entry_start=None, entry_stop=None):
        print(f"Reading HME mass from file {file_name}")
        hme_mass = None
        with uproot.open(file_name) as file:
            tree = file["Events"]
            # Branch was renamed from DoubleLep_DeepHME_mass to DeepHME_mass in v2605a
            hme_name = (
                "DeepHME_mass"
                if "DeepHME_mass" in tree.keys()
                else "DoubleLep_DeepHME_mass"
            )
            branches = tree.arrays(
                [hme_name],
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
            hme_mass = np.array(getattr(branches, hme_name), dtype="float32")
        return hme_mass

    def GetFatJetBTag(self, file_name, entry_start=None, entry_stop=None):
        print(f"Reading FatJet BTag from file {file_name}")
        fatjet_btag = None
        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                ["fatbjet_particleNetWithMass_HbbvsQCD"],
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
            fatjet_btag = np.array(
                getattr(branches, "fatbjet_particleNetWithMass_HbbvsQCD"),
                dtype="float32",
            )
        return fatjet_btag


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

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best) and (
            self.predicate is None or self.predicate(self.model, logs)
        ):
            # if self.predicate is None or self.predicate(self.model, logs):
            os.makedirs(self.filepath, exist_ok=True)
            path_best = os.path.join(self.filepath, f"{self.monitor}.onnx")
            if os.path.exists(path_best):
                os.remove(path_best)
            if self.input_signature is not None:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    self.model, self.input_signature, opset=13
                )
                onnx.save(onnx_model, path_best)

            if self.verbose > 0:
                self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
                if self.best is None:
                    self.msg += f"= {current:.5f}."
                else:
                    self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
                self.msg += f" Saving model to {path_best}\n"
            self.best = current
            self.epochs_since_last_save = 0
        if self.patience is not None and self.epochs_since_last_save >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                if self.msg is None:
                    self.msg = "\n"
                self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."


class WeightedBackgroundAtSignalYield(tf.keras.metrics.Metric):
    def __init__(
        self,
        threshold_yield,
        max_events=100000,
        nParity=4,
        name="weighted_bkg_at_sig_yield",
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.threshold_yield = tf.constant(threshold_yield, dtype=dtype)
        self.max_events = max_events
        self.nParity = nParity

        self.scores = self.add_weight(
            name="scores", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.labels = self.add_weight(
            name="labels", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.weights_var = self.add_weight(
            name="weights", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", shape=(), dtype=tf.int32, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        else:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        batch_size = tf.shape(y_true)[0]
        start = self.count
        end = tf.minimum(start + batch_size, self.max_events)
        actual_batch = end - start

        y_true = y_true[:actual_batch]
        y_pred = y_pred[:actual_batch]
        sample_weight = sample_weight[:actual_batch]

        indices = tf.range(start, end)
        indices_2d = tf.expand_dims(indices, 1)

        self.scores.assign(tf.tensor_scatter_nd_update(self.scores, indices_2d, y_pred))
        self.labels.assign(tf.tensor_scatter_nd_update(self.labels, indices_2d, y_true))
        self.weights_var.assign(
            tf.tensor_scatter_nd_update(self.weights_var, indices_2d, sample_weight)
        )

        self.count.assign(end)

    def result(self):
        n = tf.cast(self.count, tf.int32)  # ensure int32 for indexing

        def no_samples():
            return (
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
            )

        def compute():
            # Safe slicing useing tf.gather instead of tf.slice
            idx = tf.range(n)
            scores = tf.gather(self.scores, idx)
            labels = tf.gather(self.labels, idx)
            weights = tf.gather(self.weights_var, idx)

            sorted_idx = tf.argsort(scores, direction="DESCENDING")
            sorted_labels = tf.gather(labels, sorted_idx)
            sorted_weights = tf.gather(weights, sorted_idx)

            is_signal = tf.cast(tf.equal(sorted_labels, 1.0), tf.float32)
            is_bkg = 1.0 - is_signal

            sig_weights = sorted_weights * is_signal
            bkg_weights = sorted_weights * is_bkg

            cum_sig = tf.cumsum(sig_weights)
            cum_bkg = tf.cumsum(bkg_weights)
            cum_bkg_w2 = tf.cumsum(bkg_weights * bkg_weights)

            # Scale by nParity to 'fake' realistic values
            cum_sig = (
                self.nParity * 0.0264215349425664 * cum_sig
            )  # Scale signal to BR too
            cum_bkg = self.nParity * cum_bkg
            cum_bkg_w2 = self.nParity * self.nParity * cum_bkg_w2

            total_sig = cum_sig[-1]
            threshold = tf.minimum(self.threshold_yield, total_sig)

            def no_signal():
                return (
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                )

            def with_signal():
                mask = cum_sig >= threshold
                idx = tf.argmax(tf.cast(mask, tf.int32))
                return (
                    tf.cast(cum_bkg[idx], self.dtype),
                    tf.keras.ops.power(tf.cast(cum_bkg_w2[idx], self.dtype), 0.5),
                    tf.cast(scores[idx], self.dtype),
                )

            return tf.cond(
                tf.logical_or(tf.equal(total_sig, 0.0), tf.equal(threshold, 0.0)),
                no_signal,
                with_signal,
            )

        value, error, score = tf.cond(tf.equal(n, 0), no_samples, compute)

        return value, error, score

    def reset_state(self):
        self.scores.assign(tf.zeros_like(self.scores))
        self.labels.assign(tf.zeros_like(self.labels))
        self.weights_var.assign(tf.zeros_like(self.weights_var))
        self.count.assign(0)

    def _compute(self):
        return self.result()


class WeightedBackgroundAtSignalYieldValue(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_value", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        # raise RuntimeError("Dummy but second one")
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        value, _, _ = self.parent._compute()
        return tf.cast(value, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldError(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_error", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, error, _ = self.parent._compute()
        return tf.cast(error, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldScore(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_score", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, _, score = self.parent._compute()
        return tf.cast(score, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


@tf.function
def binary_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)


@tf.function
def categorical_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -tf.reduce_sum(target * tf.math.log(x), -1)


@tf.function
def binary_focal_crossentropy(target, output, gamma1=2, gamma2=0.5):
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Un-nest the output (currently in shape [ [1], [2], [3], ...] and we want in shape [1, 2, 3])
    y_true = target[:, 0]
    y_pred = output[:, 0]

    bce = binary_entropy(y_true, y_pred)

    # Custom focal
    gamma_signal = 0  # DO NOT TOUCH
    gamma_bkg = gamma1
    # Target y_pred -> 1
    p_t = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    # Split into a 'signal' gamma and a 'bkg' gamma
    gamma = y_true * gamma_signal + (1 - y_true) * gamma_bkg
    # Calc factor of pT^gamma
    focal_factor = tf.keras.ops.power(p_t, gamma)

    # focal_factor = 1.0
    # focal_bce = focal_factor * bce

    gamma2_signal = gamma2
    gamma2_bkg = 1  # DO NOT TOUCH

    gamma2 = y_true * gamma2_signal + (1 - y_true) * gamma2_bkg
    # tf.print("ff")
    # tf.print(focal_factor)
    # tf.print("bce")
    # tf.print(bce)

    focal_bce = focal_factor * tf.keras.ops.power(bce, gamma2)

    # tf.print("Min?")
    # tf.print(tf.reduce_min(focal_bce))

    return focal_bce


class Model(tf.keras.Model):
    def __init__(self, setup, max_events, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup
        self.gamma1 = setup["gamma1"]
        self.gamma2 = setup["gamma2"]
        self.loss_scale = setup["loss_scale"]

        self.nClasses = setup["nClasses"]

        # self.class_loss = tf.keras.losses.CategoricalCrossentropy(reduction=None)
        # self.class_loss = tf.keras.losses.CategoricalFocalCrossentropy(reduction=None)
        self.class_loss = binary_focal_crossentropy
        self.multiclass_loss = categorical_entropy

        self.class_accuracy = tf.keras.metrics.categorical_accuracy

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.signal_class_loss_tracker = tf.keras.metrics.Mean(name="signal_class_loss")
        self.multiclass_loss_tracker = tf.keras.metrics.Mean(name="multiclass_loss")

        self.l2_loss_tracker = tf.keras.metrics.Mean(name="l2_loss")

        # self.bkgAtSignal = WeightedBackgroundAtSignalYield(
        #     threshold_yield=5.0, max_events=max_events
        # )
        # self.bkgAtSignal_value = WeightedBackgroundAtSignalYieldValue(self.bkgAtSignal)
        # self.bkgAtSignal_error = WeightedBackgroundAtSignalYieldError(self.bkgAtSignal)
        # self.bkgAtSignal_score = WeightedBackgroundAtSignalYieldScore(self.bkgAtSignal)

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

        self.lr_tracker = tf.keras.metrics.Mean(name="learning_rate")

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

        # Input standardization: (x - mean) / std for the physics features, using
        # training-set stats (setup["feature_mean"/"feature_std"], set in train_dnn).
        # Without this the first tanh saturates on raw detector-scale inputs and the
        # classifier never learns. Done with plain tensor constants rather than a
        # keras Normalization layer (which mis-builds inside a subclassed model's
        # call()); these become Sub/Div constant nodes in the exported ONNX.
        if "feature_mean" in setup and "feature_std" in setup:
            self.feature_mean = tf.constant(
                np.array(setup["feature_mean"], dtype=np.float32)
            )
            self.feature_std = tf.constant(
                np.array(setup["feature_std"], dtype=np.float32)
            )
        else:
            self.feature_mean = None
            self.feature_std = None

        self.blocks = []

        n_units = setup["n_units"]
        activation = setup["activation"]

        for n in range(setup["n_layers"]):
            name = f"layer_{n}"
            block = {}
            block["dense"] = tf.keras.layers.Dense(
                n_units,
                activation=activation,
                name=name,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.l2(setup["l2_rate"]),
            )

            if setup["use_batch_norm"]:
                block["bn"] = tf.keras.layers.BatchNormalization(
                    name=name + "_batch_norm"
                )

            if setup["dropout"] > 0:
                block["dropout"] = tf.keras.layers.Dropout(
                    setup["dropout"], name=name + "_dropout"
                )

            self.blocks.append(block)

        self.class_output = tf.keras.layers.Dense(
            setup["nClasses"], activation="softmax", name="class_output"
        )

        self.output_names = ["class_output"]

    def call(self, x):
        if self.feature_mean is not None:
            if self.setup["UseParametric"]:
                # Last column is the appended parametric mass; leave it untouched
                # (feature_mean/std were computed on the non-mass features only).
                feats = (x[:, :-1] - self.feature_mean) / self.feature_std
                h = tf.concat([feats, x[:, -1:]], axis=1)
            else:
                h = (x - self.feature_mean) / self.feature_std
        else:
            h = x
        for i, block in enumerate(self.blocks):
            h = block["dense"](h)

            if self.setup["use_batch_norm"]:
                h = block["bn"](h)

            if self.setup["dropout"] > 0:
                h = block["dropout"](h)

        class_output = self.class_output(h)
        return class_output

    def _step(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)

        class_weight = tf.cast(y[1], dtype=tf.float32)

        physics_weight = tf.cast(y[2], dtype=tf.float32)

        def compute_losses():
            y_pred_class = self(x, training=training)

            signal_class_loss_vec = self.class_loss(
                y_class, y_pred_class, self.gamma1, self.gamma2
            )

            # signal_class_loss_vec = self.class_loss(
            #     y_class, y_pred_class
            # )

            multiclass_loss_vec = self.multiclass_loss(y_class, y_pred_class)

            class_loss_vec = (
                signal_class_loss_vec
                + self.loss_scale * multiclass_loss_vec
                # signal_class_loss_vec
            )

            class_loss = tf.reduce_mean(class_loss_vec * class_weight)

            l2_loss = tf.add_n(self.losses)

            combined_loss = class_loss + l2_loss

            return (
                y_pred_class,
                class_loss_vec,
                class_loss,
                l2_loss,
                combined_loss,
                signal_class_loss_vec,
                multiclass_loss_vec,
            )

        if training:
            with tf.GradientTape() as class_tape:
                (
                    y_pred_class,
                    class_loss_vec,
                    class_loss,
                    l2_loss,
                    combined_loss,
                    signal_class_loss_vec,
                    multiclass_loss_vec,
                ) = compute_losses()
        else:
            (
                y_pred_class,
                class_loss_vec,
                class_loss,
                l2_loss,
                combined_loss,
                signal_class_loss_vec,
                multiclass_loss_vec,
            ) = compute_losses()

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

        # self.bkgAtSignal.update_state(
        #     y_class[:, 0], y_pred_class[:, 0], sample_weight=physics_weight
        # )

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

        self.signal_class_loss_tracker.update_state(
            signal_class_loss_vec, sample_weight=class_weight
        )
        self.multiclass_loss_tracker.update_state(
            multiclass_loss_vec, sample_weight=class_weight
        )

        self.lr_tracker.update_state(self.optimizer.learning_rate)

        self.l2_loss_tracker.update_state(l2_loss)

        if training:
            grad = class_tape.gradient(combined_loss, self.trainable_variables)
            # grad = class_tape.gradient(class_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)

    @property
    def metrics(self):
        metric_list = [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.signal_class_loss_tracker,
            self.multiclass_loss_tracker,
            self.l2_loss_tracker,
            # self.bkgAtSignal_value,
            # self.bkgAtSignal_error,
            # self.bkgAtSignal_score,
            self.class_min_tracker,
            self.class_max_tracker,
            self.lr_tracker,
        ]
        metric_list = (
            metric_list
            + [x for x in self.other_class_min_tracker]
            + [x for x in self.other_class_max_tracker]
        )

        return metric_list


class DiscoModel(Model):
    """Binary sigmoid classifier with single DisCo (arXiv:2007.14400).
    Loss = BCE(1 = signal, 0 = background) + lambda * dCorr(score, HME),
    with the distance correlation computed on background events only."""

    def __init__(self, setup, max_events, *args, **kwargs):
        super().__init__(setup, max_events, *args, **kwargs)

        self.lambda_disco = setup["disco_lambda_factor"]

        # Replace the softmax head with a single sigmoid node (score = P(signal))
        self.class_output = tf.keras.layers.Dense(
            1, activation="sigmoid", name="class_output"
        )

        self.disco_loss_tracker = tf.keras.metrics.Mean(name="disco_loss")
        self.combined_loss_tracker = tf.keras.metrics.Mean(name="combined_loss")

    def _step(self, data, training):
        x, y = data

        y_class = tf.reshape(tf.cast(y[0], dtype=tf.float32), [-1])
        class_weight = tf.reshape(tf.cast(y[1], dtype=tf.float32), [-1])
        physics_weight = tf.cast(y[2], dtype=tf.float32)
        hme = tf.reshape(tf.cast(y[3], dtype=tf.float32), [-1])

        def compute_losses():
            y_pred_class = self(x, training=training)
            score = y_pred_class[:, 0]

            class_loss_vec = binary_entropy(y_class, score)
            class_loss = tf.reduce_mean(class_loss_vec * class_weight)
            # Weighted mean (normalize by sum of weights) instead of reduce_mean over
            # the batch
            # class_loss = tf.reduce_sum(class_loss_vec * class_weight) / (
            #     tf.reduce_sum(class_weight) + 1e-12
            # )

            # DisCo only on background events with a valid HME
            disco_mask = tf.logical_and(
                tf.equal(y_class, 0.0),
                tf.logical_and(hme > 0.0, tf.math.is_finite(hme)),
            )
            score_bkg = tf.boolean_mask(score, disco_mask)
            hme_bkg = tf.boolean_mask(hme, disco_mask)
            weight_bkg = tf.boolean_mask(class_weight, disco_mask)

            n_bkg = tf.cast(tf.size(score_bkg), tf.float32)

            def calc_disco():
                # distance_corr needs weights that sum to the number of entries
                weight_norm = weight_bkg * n_bkg / (tf.reduce_sum(weight_bkg) + 1e-12)
                return distance_corr(hme_bkg, score_bkg, weight_norm)

            disco_loss = tf.cond(
                n_bkg > 1.0, calc_disco, lambda: tf.constant(0.0, dtype=tf.float32)
            )

            l2_loss = tf.add_n(self.losses)

            combined_loss = class_loss + self.lambda_disco * disco_loss + l2_loss

            return (
                y_pred_class,
                class_loss_vec,
                class_loss,
                disco_loss,
                l2_loss,
                combined_loss,
            )

        if training:
            with tf.GradientTape() as class_tape:
                (
                    y_pred_class,
                    class_loss_vec,
                    class_loss,
                    disco_loss,
                    l2_loss,
                    combined_loss,
                ) = compute_losses()
        else:
            (
                y_pred_class,
                class_loss_vec,
                class_loss,
                disco_loss,
                l2_loss,
                combined_loss,
            ) = compute_losses()

        score = y_pred_class[:, 0]

        self.class_min_tracker.update_state(tf.reduce_min(score))
        self.class_max_tracker.update_state(tf.reduce_max(score))

        class_accuracy_vec = tf.cast(
            tf.equal(y_class, tf.cast(score > 0.5, tf.float32)), tf.float32
        )

        self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
        self.class_accuracy_tracker.update_state(
            class_accuracy_vec, sample_weight=class_weight
        )

        self.disco_loss_tracker.update_state(disco_loss)
        self.combined_loss_tracker.update_state(combined_loss)
        self.l2_loss_tracker.update_state(l2_loss)

        self.lr_tracker.update_state(self.optimizer.learning_rate)

        if training:
            grad = class_tape.gradient(combined_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.disco_loss_tracker,
            self.combined_loss_tracker,
            self.l2_loss_tracker,
            self.class_min_tracker,
            self.class_max_tracker,
            self.lr_tracker,
        ]


def train_dnn(
    setup,
    training_file,
    weight_file,
    test_training_file,
    test_weight_file,
    output_folder,
):
    output_dnn_name = os.path.join(output_folder, f"best.onnx")

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = setup.get("entry_stop", None)

    dw.ReadFile(
        training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)

    test_dw.ReadFile(
        test_training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    test_dw.ReadWeightFile(
        test_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )

    use_disco = "disco_lambda_factor" in setup

    if use_disco:
        dw.hme = dw.GetHME(
            training_file, entry_start=entry_start, entry_stop=entry_stop
        )
        test_dw.hme = test_dw.GetHME(
            test_training_file, entry_start=entry_start, entry_stop=entry_stop
        )

    dw.physics_weight = np.where(
        (dw.class_target == 0) & (dw.X_mass != 600), 0.0, dw.physics_weight
    )
    test_dw.physics_weight = np.where(
        (test_dw.class_target == 0) & (test_dw.X_mass != 600),
        0.0,
        test_dw.physics_weight,
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]
    batch_size = setup["batch_size"]
    if use_disco:
        # Binary labels: 1 = signal, 0 = background
        # class_weight balances total signal vs total background 50/50
        train_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dw.features,
                (
                    (dw.class_target == 0).astype("float32"),
                    dw.class_weight,
                    dw.physics_weight,
                    dw.hme,
                ),
            )
        )
    else:
        train_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                dw.features,
                (
                    tf.one_hot(dw.class_target, nClasses),
                    # dw.class_weight,
                    dw.multiclass_weight,
                    dw.physics_weight,
                ),
            )
        )
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )
    # train_tf_dataset = train_tf_dataset.shuffle(
    #     batch_size*100, reshuffle_each_iteration=True
    # )
    batch_size_train = min(batch_size, train_tf_dataset.cardinality().numpy())
    train_tf_dataset = train_tf_dataset.batch(batch_size_train, drop_remainder=True)
    train_cache_name = "train_cache_disco.tfdata" if use_disco else "train_cache.tfdata"
    train_tf_dataset = train_tf_dataset.cache(train_cache_name)

    if use_disco:
        test_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                test_dw.features,
                (
                    (test_dw.class_target == 0).astype("float32"),
                    test_dw.class_weight,
                    test_dw.physics_weight,
                    test_dw.hme,
                ),
            )
        )
    else:
        test_tf_dataset = tf.data.Dataset.from_tensor_slices(
            (
                test_dw.features,
                (
                    tf.one_hot(test_dw.class_target, nClasses),
                    # test_dw.class_weight,
                    test_dw.multiclass_weight,
                    test_dw.physics_weight,
                ),
            )
        )
    test_tf_dataset = test_tf_dataset.shuffle(
        len(test_tf_dataset), reshuffle_each_iteration=True
    )
    # test_tf_dataset = test_tf_dataset.shuffle(
    #     batch_size*100, reshuffle_each_iteration=True
    # )
    batch_size_test = min(batch_size, test_tf_dataset.cardinality().numpy())
    test_tf_dataset = test_tf_dataset.batch(batch_size_test, drop_remainder=True)
    test_cache_name = "test_cache_disco.tfdata" if use_disco else "test_cache.tfdata"
    test_tf_dataset = test_tf_dataset.cache(test_cache_name)

    parametric_mass_probability = np.ones(len(dw.param_list)) * 1.0 / len(dw.param_list)
    log_probs = tf.math.log([parametric_mass_probability])

    mass_values = tf.constant(dw.param_list)
    mass_keys = tf.constant(np.arange(len(dw.param_list)))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
        default_value=-1,
    )

    def new_param_map(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass

        random_param_mass = tf.random.categorical(
            log_probs,
            tf.shape(features)[0],
            dtype=tf.int64,
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

    def new_param_map_binary(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass

        random_param_mass = tf.random.categorical(
            log_probs,
            tf.shape(features)[0],
            dtype=tf.int64,
        )

        actual_new_mass = table.lookup(random_param_mass)
        actual_new_mass = tf.cast(actual_new_mass, tf.float32)

        # Signal (label 1) keeps its true mass, background gets a random one
        labels = dataset[1][0]
        old_mass_mask = tf.cast(labels == 1, tf.float32)
        new_mass_mask = tf.cast(labels == 0, tf.float32)

        actual_mass = old_mass_mask * features[:, -1] + new_mass_mask * actual_new_mass
        actual_mass = tf.transpose(actual_mass)

        features = tf.concat([features[:, :-1], actual_mass], axis=-1)
        new_dataset = (features, dataset[1])
        return new_dataset

    if setup["UseParametric"]:
        param_map = new_param_map_binary if use_disco else new_param_map
        train_tf_dataset = train_tf_dataset.map(param_map)
        train_tf_dataset = train_tf_dataset.prefetch(tf.data.AUTOTUNE)
        test_tf_dataset = test_tf_dataset.map(param_map)
        test_tf_dataset = test_tf_dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="x")]

    nBatches = max(
        train_tf_dataset.cardinality().numpy(), test_tf_dataset.cardinality().numpy()
    )
    max_events = nBatches * max(batch_size_train, batch_size_test)

    if setup["UseParametric"]:
        features_no_mass = dw.features[:, :-1]
    else:
        features_no_mass = dw.features
    #The following was commented out, I uncommented it
    mean = np.mean(features_no_mass, axis=0)
    std = np.std(features_no_mass, axis=0) + 1e-6
    setup["feature_mean"] = mean.tolist()
    setup["feature_std"] = std.tolist()
    if use_disco:
        model = DiscoModel(setup, max_events)
    else:
        model = Model(setup, max_events)
    model.compile(
        loss=None,
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=setup["learning_rate"],
            weight_decay=setup["weight_decay"],
            clipnorm=1.0,
        ),
    )
    model(dw.features)
    model.summary()

    monitor_name = "val_combined_loss" if use_disco else "val_class_loss"

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_name,
        factor=setup["lr_decay"],
        patience=setup["lr_patience"],
        min_lr=0.0000000001,
    )

    callbacks = [
        ModelCheckpoint(
            output_folder,
            verbose=1,
            monitor=monitor_name,
            mode="min",
            min_rel_delta=1e-3,
            patience=setup["patience"],
            save_callback=None,
            input_signature=input_signature,
        ),
        # ModelCheckpoint(
        #     output_folder,
        #     verbose=1,
        #     monitor="val_weighted_bkg_at_sig_yield_value",
        #     mode="min",
        #     min_rel_delta=1e-3,
        #     patience=None,
        #     save_callback=None,
        #     input_signature=input_signature,
        # ),
        reduce_lr,
    ]

    verbose = setup["verbose"] if "verbose" in setup else 0
    # verbose = 1
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=True,
        callbacks=callbacks,
    )

    os.makedirs(output_folder, exist_ok=True)

    def PlotMetric(history, metric, output_folder):
        if metric not in history.history:
            print(f"Metric {metric} not found in history")
            return
        os.makedirs(os.path.join(output_folder, "metrics"), exist_ok=True)
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.yscale("log")
        plt.ylim(bottom=0.0001, top=10.0)
        plt.savefig(
            os.path.join(output_folder, "metrics", f"{metric}.pdf"), bbox_inches="tight"
        )
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)
    PlotMetric(history, "learning_rate", output_folder)
    PlotMetric(history, "l2_loss", output_folder)
    PlotMetric(history, "class_min", output_folder)
    PlotMetric(history, "class_max", output_folder)

    if use_disco:
        PlotMetric(history, "disco_loss", output_folder)
        PlotMetric(history, "combined_loss", output_folder)

    for i in range(1, nClasses):
        PlotMetric(history, f"other_class_min{i}", output_folder)
        PlotMetric(history, f"other_class_max{i}", output_folder)

    PlotMetric(history, "weighted_bkg_at_sig_yield_value", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_error", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_score", output_folder)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="x")]

    # Convert model to ONNX (reuse what you already do later)
    onnx_model_stage1, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

    stage1_model_path = os.path.join(output_folder, "stage1.onnx")
    onnx.save(onnx_model_stage1, stage1_model_path)

    features_config = {
        "features": dw.feature_names,
        "use_parametric": dw.use_parametric,
        "parametric_list": dw.param_list,
        "model_setup": setup,
        "nClasses": setup["nClasses"],
        "nParity": 4,
    }

    if use_disco:
        features_config["disco_lambda_factor"] = setup["disco_lambda_factor"]

    with open(os.path.join(output_folder, "dnn_config.yaml"), "w") as file:
        yaml.dump(features_config, file)

    # Experimental 2-stage DNN
    # Binary stage 2 runs signal vs background using stage 1 score as additional input
    if getattr(setup, "train_stage_2", False):
        print("Running Stage 1 inference to build Stage 2 dataset")

        sess_stage1 = ort.InferenceSession(stage1_model_path)

        # Run inference on FULL training dataset
        features_stage1 = dw.features
        preds_stage1 = sess_stage1.run(None, {"x": features_stage1})[0]

        signal_scores = preds_stage1[:, 0]

        # -------------------------
        # Stage 2 selection
        # -------------------------

        # Filter training data
        preds_stage1_col = preds_stage1

        # Option to use stage 1 scores as input to stage 2
        X2 = np.concatenate([dw.features, preds_stage1_col], axis=1)

        class_weight2 = dw.class_weight
        physics_weight2 = dw.physics_weight

        # Binary labels: signal vs background
        y2 = (dw.class_target != 0).astype(int)

        # Stage 2 test dataset
        features_test = test_dw.features
        preds_stage1_test = sess_stage1.run(None, {"x": features_test})[0]

        preds_stage1_col_test = preds_stage1_test

        # Option to use stage 1 scores as input to stage 2
        X2_test = np.concatenate([test_dw.features, preds_stage1_col_test], axis=1)

        class_weight2_test = test_dw.class_weight
        physics_weight2_test = test_dw.physics_weight

        y2_test = (test_dw.class_target != 0).astype(int)

        nClasses_stage2 = 2
        train_tf_dataset_stage2 = (
            tf.data.Dataset.from_tensor_slices(
                (
                    X2,
                    (
                        tf.one_hot(y2, nClasses_stage2),
                        class_weight2,
                        physics_weight2,
                    ),
                )
            )
            .shuffle(len(X2))
            .batch(batch_size_train, drop_remainder=True)
            .cache("train_cache_stage2.tfdata")
        )

        test_tf_dataset_stage2 = (
            tf.data.Dataset.from_tensor_slices(
                (
                    X2_test,
                    (
                        tf.one_hot(y2_test, nClasses_stage2),
                        class_weight2_test,
                        physics_weight2_test,
                    ),
                )
            )
            .shuffle(len(X2_test))
            .batch(batch_size_test, drop_remainder=True)
            .cache("test_cache_stage2.tfdata")
        )

        print("Training Stage 2 binary model")

        setup_stage2 = copy.deepcopy(setup)
        setup_stage2["nClasses"] = 2
        setup_stage2["loss_scale"] = 0

        model_stage2 = Model(setup_stage2, max_events)

        model_stage2.compile(
            loss=None,
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=setup_stage2["learning_rate"],
                weight_decay=setup_stage2["weight_decay"],
                clipnorm=1.0,
            ),
        )

        model_stage2(X2)
        model_stage2.summary()

        input_shape_stage2 = [None, X2.shape[1]]
        input_signature_stage2 = [
            tf.TensorSpec(input_shape_stage2, tf.float32, name="x")
        ]

        history_stage2 = model_stage2.fit(
            train_tf_dataset_stage2,
            validation_data=test_tf_dataset_stage2,
            verbose=verbose,
            epochs=setup_stage2["n_epochs"],
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(output_folder, "stage2"),
                    verbose=1,
                    monitor="val_class_loss",
                    mode="min",
                    min_rel_delta=1e-3,
                    patience=setup_stage2["patience"],
                    input_signature=input_signature_stage2,
                )
            ],
        )

        output_folder_stage_2 = os.path.join(output_folder, "stage2")
        PlotMetric(history_stage2, "class_loss", output_folder_stage_2)
        PlotMetric(history_stage2, "learning_rate", output_folder_stage_2)
        PlotMetric(history_stage2, "l2_loss", output_folder_stage_2)
        PlotMetric(history_stage2, "class_min", output_folder_stage_2)
        PlotMetric(history_stage2, "class_max", output_folder_stage_2)

        onnx_model_stage2, _ = tf2onnx.convert.from_keras(
            model_stage2, input_signature_stage2, opset=13
        )

        stage2_model_path = os.path.join(output_folder, "stage2.onnx")
        onnx.save(onnx_model_stage2, stage2_model_path)

    return


def validate_dnn(
    setup,
    validation_file,
    validation_weight_file,
    output_folder,
    model_name,
    model_config,
):
    print(f"Model load {model_name}")
    sess = ort.InferenceSession(model_name)

    dnnConfig = {}
    with open(model_config, "r") as file:
        dnnConfig = yaml.safe_load(file)

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        validation_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    dw.ReadWeightFile(
        validation_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )

    hme_values = dw.GetHME(file_name=validation_file)
    fatjet_btag_values = dw.GetFatJetBTag(file_name=validation_file)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]

    os.makedirs(output_folder, exist_ok=True)

    feature_dict = {
        "lep1_pt": [0.0, 500.0],
        "lep1_legType": [-1.0, 3.0],
        "lep2_pt": [0.0, 500.0],
        "lep2_legType": [-1.0, 3.0],
        "PuppiMET_pt": [0.0, 500.0],
        "PuppiMET_phi": [-4.0, 4.0],
        "HT": [0.0, 1000.0],
        "dR_dilep": [0.0, 5.0],
        "dR_dibjet": [0.0, 5.0],
        "dR_dilep_dibjet": [0.0, 5.0],
        "dPhi_lep1_lep2": [-4.0, 4.0],
        "dPhi_MET_dilep": [-4.0, 4.0],
        "dPhi_MET_dibjet": [-4.0, 4.0],
        "MT": [0.0, 300.0],
        "MT2_ll": [0.0, 300.0],
        "MT2_bb": [0.0, 500.0],
        "MT2_blbl": [0.0, 600.0],
        "MT2_blbl2": [0.0, 600.0],
        "ll_mass": [0.0, 100.0],
        "CosTheta_bb": [-1.5, 1.5],
        "bjet1_pt": [0.0, 500.0],
        "bjet1_mass": [0.0, 50.0],
        "bjet1_btagPNetB": [0.0, 1.0],
        "bjet2_pt": [0.0, 500.0],
        "bjet2_mass": [0.0, 50.0],
        "bjet2_btagPNetB": [0.0, 1.0],
    }

    for cat in ["res2b", "boosted", "res1b"]:
        ROOTOut = ROOT.TFile(
            os.path.join(output_folder, f"validation_{cat}.root"), "RECREATE"
        )

        # para_masspoint_list = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 2000, 3000]
        para_masspoint_list = dnnConfig["parametric_list"]
        canvases = []
        for para_masspoint in para_masspoint_list:
            print(f"Validating mass {para_masspoint}")
            fig, ax = plt.subplots()
            if dw.use_parametric:
                dw.SetPredictParamValue(para_masspoint)
            features = (
                dw.features_paramSet if dw.use_parametric else dw.features_no_param
            )

            print("Predicting")
            print("Using features")
            print(features)
            print(type(features))
            print(features.dtype)
            # features = features.astype('double')
            pred = sess.run(None, {"x": features})

            print("What is pred?")
            print(pred)
            for nClass in range(nClasses):
                print(f"Trying class {nClass}")
                output_file = os.path.join(
                    output_folder, f"validation_{cat}_class{nClass}.pdf"
                )

                pred_class = pred[0]
                pred_signal = pred_class[:, nClass]

                class_weight = dw.class_weight
                physics_weight = dw.physics_weight

                # Scale signal to BR
                physics_weight = np.where(
                    dw.class_target == 0,
                    dw.physics_weight * 0.0264215349425664,
                    dw.physics_weight,
                )

                if cat == "res2b":
                    physics_weight = np.where(dw.res2b == 1, physics_weight, 0.0)
                if cat == "boosted":
                    physics_weight = np.where(dw.boosted == 1, physics_weight, 0.0)
                if cat == "res1b":
                    physics_weight = np.where(dw.recovery == 1, physics_weight, 0.0)

                # This class is top score mask
                score_mask = np.argmax(pred_signal) == nClass
                print("Checking score mask")
                print(pred_signal)
                print(nClass)
                print(score_mask)

                # Class Plots
                # Lets build Masks
                Sig_This_Mass = dw.X_mass == para_masspoint
                Sig_mask = (Sig_This_Mass) & (dw.class_target == 0)

                Background_mask = dw.class_target == 1

                TT_mask = dw.class_value == 1

                DY_mask = dw.class_value == 2

                Other_mask = dw.class_value == 3

                nClass_mask = [Sig_mask, TT_mask, DY_mask, Other_mask]

                # Set class quantiles based on signal
                nQuantBins = 50
                quant_binning_class = np.zeros(
                    nQuantBins + 1
                )  # Need +1 because 10 bins actually have 11 edges
                if len(pred_signal[nClass_mask[nClass]]) == 0:
                    print(
                        f"No class {nClass} events in this mass point! Fake Quant Bins!"
                    )
                    quant_binning_class = np.linspace(0, 1, nQuantBins + 1)
                else:
                    quant_binning_class = np.quantile(
                        pred_signal[nClass_mask[nClass]],
                        np.linspace(0, 1, nQuantBins + 1),
                    )
                quant_binning_class[0] = 0.0
                quant_binning_class[-1] = 1.0
                print("We found quant binning class")
                print(quant_binning_class)
                # print("From the signal prediction")
                # print(pred_signal[Sig_mask])

                # Find HME min/max for signal (mean +/- 2 std)
                hme_mean = np.mean(hme_values[Sig_mask])
                hme_std = np.std(hme_values[Sig_mask])
                hme_low = hme_mean - (1 * hme_std)
                hme_high = hme_mean + (1 * hme_std)
                # fatjet_btag_mask = fatjet_btag_values > 0.99
                fatjet_btag_mask = True

                # Do a quick significance scan for HME bounds
                # Take the mean of signal's HME, then scan an asymmetric window around it to maximize significance s/sqrt(s+b)
                best_significance = 0.0
                best_hme_low = hme_low
                best_hme_high = hme_high
                for hme_low_scan in np.linspace(hme_mean - 5 * hme_std, hme_mean, 10):
                    for hme_high_scan in np.linspace(
                        hme_mean, hme_mean + 5 * hme_std, 10
                    ):
                        hme_mask_scan = (
                            (hme_values > hme_low_scan)
                            & (hme_values < hme_high_scan)
                            & (fatjet_btag_mask)
                        )
                        s = np.sum(physics_weight[Sig_mask & hme_mask_scan])
                        b = np.sum(physics_weight[Background_mask & hme_mask_scan])
                        significance = s / np.sqrt(s + b + 1e-6)
                        if significance > best_significance:
                            best_significance = significance
                            best_hme_low = hme_low_scan
                            best_hme_high = hme_high_scan

                print(
                    f"For mass {para_masspoint} we have HME bounds [{best_hme_low}, {best_hme_high}]"
                )
                print(f"This had a significance of {best_significance}")
                hme_mask = (
                    (hme_values > best_hme_low)
                    & (hme_values < best_hme_high)
                    & (fatjet_btag_mask)
                )
                # hme_mask = True  # TEMPORARY, REMOVE THIS TO ENABLE HME CUT

                mask_dict = {
                    "Signal": Sig_mask & hme_mask,
                    "TT": TT_mask & hme_mask,
                    "DY": DY_mask & hme_mask,
                    "Other": Other_mask & hme_mask,
                }

                canvases.append(
                    ROOT.TCanvas("c1", "c1", 1200, 600 * len(mask_dict.keys()))
                )
                canvas = canvases[-1]
                canvas.Divide(1, len(mask_dict.keys()))
                Class_list = []
                legend_list = []
                pads_list = []
                DNN_vs_HME_list = []
                DNN_vs_FatJetBTag_list = []
                DNN_vs_input = []
                for i, process_name in enumerate(mask_dict.keys()):
                    canvas.cd(i + 1)
                    mask = mask_dict[process_name]

                    class_out_hist, bins = np.histogram(
                        pred_signal[mask],
                        bins=quant_binning_class,
                        range=(0.0, 1.0),
                        weights=physics_weight[mask],
                    )
                    class_out_hist_w2, bins = np.histogram(
                        pred_signal[mask],
                        bins=quant_binning_class,
                        range=(0.0, 1.0),
                        weights=physics_weight[mask] ** 2,
                    )

                    Class_list.append(
                        ROOT.TH1D(
                            f"Class{nClass}Output_{process_name}",
                            f"Class{nClass}Output_{process_name}",
                            nQuantBins,
                            0.0,
                            1.0,
                        )
                    )

                    ROOT_ClassOutput = Class_list[-1]

                    for binnum in range(nQuantBins):
                        ROOT_ClassOutput.SetBinContent(
                            binnum + 1, class_out_hist[binnum]
                        )
                        ROOT_ClassOutput.SetBinError(
                            binnum + 1, class_out_hist_w2[binnum] ** (0.5)
                        )

                    ROOTOut.WriteObject(
                        ROOT_ClassOutput,
                        f"m{para_masspoint}_{process_name}_class{nClass}",
                    )

                    # Make a ROOT.TH2D of the DNN prediction vs dw.GetHME(file_name)
                    DNN_vs_HME_list.append(
                        ROOT.TH2D(
                            f"DNN_vs_HME_m{para_masspoint}_{process_name}_class{nClass}",
                            f"DNN_vs_HME_m{para_masspoint}_{process_name}_class{nClass}",
                            100,
                            0.0,
                            1.0,
                            250,
                            0.0,
                            2500.0,
                            # 250,
                            # 5.0,
                            # 8.0,
                        )
                    )
                    DNN_vs_HME = DNN_vs_HME_list[-1]
                    this_dnn_values = pred_signal[mask]
                    this_hme_values = hme_values[mask]
                    for dnn_val, hme_val in zip(this_dnn_values, this_hme_values):
                        # DNN_vs_HME.Fill(dnn_val, np.log(hme_val))
                        DNN_vs_HME.Fill(dnn_val, hme_val)

                    ROOTOut.WriteObject(
                        DNN_vs_HME,
                        f"DNN_vs_HME_m{para_masspoint}_{process_name}_class{nClass}",
                    )

                    # Make a ROOT.TH2D of the DNN prediction vs dw.GetFatJetBTag(file_name)
                    DNN_vs_FatJetBTag_list.append(
                        ROOT.TH2D(
                            f"DNN_vs_FatJetBTag_m{para_masspoint}_{process_name}_class{nClass}",
                            f"DNN_vs_FatJetBTag_m{para_masspoint}_{process_name}_class{nClass}",
                            100,
                            0.0,
                            1.0,
                            100,
                            0.0,
                            1.0,
                        )
                    )
                    DNN_vs_FatJetBTag = DNN_vs_FatJetBTag_list[-1]
                    this_fatjet_btag_values = fatjet_btag_values[mask]
                    for dnn_val, fatjet_btag_val in zip(
                        this_dnn_values, this_fatjet_btag_values
                    ):
                        DNN_vs_FatJetBTag.Fill(dnn_val, fatjet_btag_val)

                    ROOTOut.WriteObject(
                        DNN_vs_FatJetBTag,
                        f"DNN_vs_FatJetBTag_m{para_masspoint}_{process_name}_class{nClass}",
                    )

                    print(f"Input features are {setup['features']}")

                    # For each input feature, do the 2D DNNvsFeature plot
                    plot_features = set(setup["features"])
                    # plot_features.add("fatbjet_particleNetWithMass_HbbvsQCD") # Add the fatjet tagger manually
                    for feature in plot_features:
                        feature_mean = np.mean(
                            dw.features[:, setup["features"].index(feature)][mask]
                        )
                        feature_std = np.std(
                            dw.features[:, setup["features"].index(feature)][mask]
                        )
                        feature_low = feature_mean - (2 * feature_std)
                        feature_high = feature_mean + (2 * feature_std)

                        if feature in feature_dict:
                            feature_low, feature_high = feature_dict[feature]

                        DNN_vs_input.append(
                            ROOT.TH2D(
                                f"DNN_vs_{feature}_m{para_masspoint}_{process_name}_class{nClass}",
                                f"DNN_vs_{feature}_m{para_masspoint}_{process_name}_class{nClass}",
                                100,
                                0.0,
                                1.0,
                                100,
                                # Min and max the features as 2 sigma from mean
                                feature_low,
                                feature_high,
                                # np.min(dw.features[:, setup["features"].index(feature)]),
                                # np.max(dw.features[:, setup["features"].index(feature)])*1.1,  # Add 10% padding on max for better visualization
                            )
                        )
                        DNN_vs_Feature = DNN_vs_input[-1]
                        this_feature_values = dw.features[
                            :, setup["features"].index(feature)
                        ][mask]
                        for dnn_val, feature_val in zip(
                            this_dnn_values, this_feature_values
                        ):
                            DNN_vs_Feature.Fill(dnn_val, feature_val)

                        ROOTOut.WriteObject(
                            DNN_vs_Feature,
                            f"DNN_vs_{feature}_m{para_masspoint}_{process_name}_class{nClass}",
                        )

                    if ROOT_ClassOutput.Integral() == 0:
                        print(
                            f"Process {process_name} has no class entries, maybe the background doesn't exist?"
                        )
                        continue

                    # ROOT_ClassOutput.Scale(1.0 / ROOT_ClassOutput.Integral())

                    pads_list.append(ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0))
                    p1 = pads_list[-1]
                    p1.SetTopMargin(0)
                    p1.Draw()

                    p1.cd()

                    plotlabel = f"Class {nClass} Output for {process_name} ParaMass {para_masspoint} GeV {cat}"
                    ROOT_ClassOutput.Draw()
                    ROOT_ClassOutput.SetTitle(plotlabel)
                    ROOT_ClassOutput.SetStats(0)
                    min_val = max(
                        0.0001,
                        ROOT_ClassOutput.GetMinimum(),
                    )
                    max_val = ROOT_ClassOutput.GetMaximum()

                    ROOT_ClassOutput.GetYaxis().SetRangeUser(
                        0.001 * min_val, 1000 * max_val
                    )

                    legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
                    legend = legend_list[-1]
                    legend.AddEntry(ROOT_ClassOutput, f"{process_name}")
                    legend.Draw()

                    print(
                        f"Setting canvas to log scale with range {min_val}, {max_val}"
                    )
                    p1.SetLogy()
                    p1.SetGrid()

                # if para_masspoint == para_masspoint_list[0]:
                #     canvas.Print(f"{output_file}(", f"Title:Mass {para_masspoint} GeV")
                #     print("Saved [")
                # elif para_masspoint == para_masspoint_list[-1]:
                #     canvas.Print(f"{output_file})", f"Title:Mass {para_masspoint} GeV")
                #     print("Saved ]")
                # else:
                #     canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} GeV")
                # print(f"Saved mass {para_masspoint}")

                canvas.Close()

                canvas = ROOT.TCanvas("c1", "c1", 800, 600)
                legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
                pad_class = ROOT.TPad("class", "class", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
                pad_soverb = ROOT.TPad("soverb", "soverb", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
                pad_class.SetTopMargin(0)
                pad_class.Draw()

                pad_soverb.SetTopMargin(0)
                pad_soverb.SetBottomMargin(0)
                pad_soverb.Draw()

                pad_class.cd()

                soverb = Class_list[0].Clone()
                background = Class_list[1].Clone()

                color_list = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta]
                marker_list = [105, 107, 108, 109]
                for i in range(len(Class_list)):
                    if i == 0:
                        Class_list[i].Draw()
                    else:
                        Class_list[i].Draw("same")
                    if i > 1:
                        background.Add(Class_list[i].Clone())
                    Class_list[i].SetLineColor(color_list[i])
                    Class_list[i].SetMarkerStyle(marker_list[i])
                    Class_list[i].SetMarkerColor(color_list[i])
                    Class_list[i].SetMarkerSize(0.1)
                    legend.AddEntry(Class_list[i], Class_list[i].GetTitle())

                Class_list[0].GetYaxis().SetRangeUser(0.0001, 10000)

                legend.Draw()
                # canvas.SetLogy()
                # canvas.SetGrid()

                pad_class.SetLogy()
                pad_class.SetGrid()

                pad_soverb.cd()

                soverb.SetTitle("S over B")
                soverb.Divide(background)
                soverb.Draw()

                pad_soverb.SetLogy()
                pad_soverb.SetGrid()

                if para_masspoint == para_masspoint_list[0]:
                    canvas.Print(
                        f"{output_file}(",
                        f"Title:Mass {para_masspoint} {cat} class{nClass} GeV",
                    )
                elif para_masspoint == para_masspoint_list[-1]:
                    canvas.Print(
                        f"{output_file})",
                        f"Title:Mass {para_masspoint} {cat} class{nClass} GeV",
                    )
                else:
                    canvas.Print(
                        f"{output_file}",
                        f"Title:Mass {para_masspoint} {cat} class{nClass} GeV",
                    )
                canvas.Close()

                if np.sum(physics_weight[dw.class_target == 1]) > 0:
                    print("sum weights:", np.sum(physics_weight))
                    print("min weight:", np.min(physics_weight))
                    print("max weight:", np.max(physics_weight))
                    display = sklearn.metrics.RocCurveDisplay.from_predictions(
                        dw.class_target == 0,
                        pred_signal,
                        sample_weight=np.clip(physics_weight, 0, None),
                        ax=ax,
                        name=f"Signal vs rest m{para_masspoint} class{nClass}",
                    )
                    _ = display.ax_.set(
                        xlabel="False Positive Rate",
                        ylabel="True Positive Rate",
                        title="Signal-vs-Background ROC curves",
                    )

            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curves {cat} m{para_masspoint} GeV")
            ax.grid()
            plt.savefig(os.path.join(output_folder, f"ROC_{cat}_m{para_masspoint}.pdf"))

            # =========================
            # Feature Importance Study
            # =========================

            print(
                f"Running permutation feature importance for mass {para_masspoint}..."
            )

            # Use correct feature set
            features = (
                dw.features_paramSet if dw.use_parametric else dw.features_no_param
            )

            # Define signal vs background
            y_binary = (dw.class_target == 0).astype(int)

            # Use physics weights
            weights = dw.physics_weight

            # Select only features in this category
            if cat == "res2b":
                category_mask = dw.res2b == 1
            elif cat == "boosted":
                category_mask = dw.boosted == 1
            elif cat == "res1b":
                category_mask = dw.recovery == 1
            else:
                category_mask = np.ones(len(dw.class_target), dtype=bool)

            if np.sum(category_mask) == 0:
                print(
                    f"No events in category {cat} for mass {para_masspoint}, skipping feature importance."
                )
                continue

            features = features[category_mask]
            y_binary = y_binary[category_mask]
            weights = weights[category_mask]

            # Optional: subsample for speed
            max_events = 1000000
            if len(features) > max_events:
                idx = np.random.choice(len(features), max_events, replace=False)
                features_sample = features[idx]
                y_sample = y_binary[idx]
                w_sample = weights[idx]
            else:
                features_sample = features
                y_sample = y_binary
                w_sample = weights

            baseline_auc, importances = permutation_importance_onnx(
                sess,
                features_sample,
                y_sample,
                w_sample,
                n_repeats=3,
            )

            print(f"Baseline AUC: {baseline_auc:.4f}")

            # Sort importance
            sorted_idx = np.argsort(importances)[::-1]

            print("\nFeature importance ranking:")
            for rank, i in enumerate(sorted_idx):
                if i > len(setup["features"]):
                    print(f"i {i} out of range, must be the stage-1 score")
                    feature_name = "stage1_score"
                    print(
                        f"{rank+1:2d}. {feature_name:30s}  ΔAUC = {importances[i]:.6f}"
                    )
                else:
                    print(
                        f"{rank+1:2d}. {setup['features'][i]:30s}  ΔAUC = {importances[i]:.6f}"
                    )

            os.makedirs(
                os.path.join(output_folder, "feature_importance"), exist_ok=True
            )

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[sorted_idx])

            plt.xticks(
                range(len(importances)),
                [setup["features"][i] for i in sorted_idx],
                rotation=90,
            )

            plt.ylabel("AUC drop (importance)")
            plt.title(f"Permutation Feature Importance {cat} m{para_masspoint} GeV")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_folder,
                    "feature_importance",
                    f"importance_{cat}_m{para_masspoint}.pdf",
                )
            )
            plt.close()

        data_obs = ROOT.TH1D(
            f"data_obs",
            f"data_obs",
            nQuantBins,
            0.0,
            1.0,
        )
        ROOTOut.WriteObject(data_obs, f"data_obs")


# =========================================================
# 1. LOAD MODEL + CONFIG
# =========================================================
def load_model_and_config(model_name, model_config):
    print(f"Loading model: {model_name}")
    sess = ort.InferenceSession(model_name)

    with open(model_config, "r") as f:
        config = yaml.safe_load(f)

    return sess, config


# =========================================================
# 2. DATA PREPARATION
# =========================================================
def prepare_datawrapper(setup, validation_file, validation_weight_file):
    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])
    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    dw.ReadFile(validation_file)
    dw.ReadWeightFile(validation_weight_file)

    return dw


# =========================================================
# 3. INFERENCE
# =========================================================
def run_inference(sess, features):
    return sess.run(None, {"x": features})


def get_scores(sess, X, stage="multi"):
    preds = run_inference(sess, X)[0]

    # multi-class
    if preds.ndim == 2:
        return preds

    # binary
    return preds[:, 0]


# =========================================================
# 4. EVENT MASKING
# =========================================================
def build_event_masks(dw, hme_values, cat, para_masspoint, hme_cut=False):

    signal_mask = (dw.X_mass == para_masspoint) & (dw.class_target == 0)
    TT_mask = dw.class_value == 1
    DY_mask = dw.class_value == 2
    Other_mask = dw.class_value == 3
    Background_mask = dw.class_target == 1

    physics_weight = np.copy(dw.physics_weight)
    physics_weight = np.where(
        dw.class_target == 0,
        physics_weight * 0.0264215349425664,
        physics_weight,
    )

    cat_name = "recovery" if cat == "res1b" else cat
    cat_mask = getattr(dw, cat_name) == 1
    physics_weight = np.where(cat_mask, physics_weight, 0)

    if hme_cut:
        hme_mean = np.mean(hme_values[signal_mask])
        hme_std = np.std(hme_values[signal_mask])
        hme_mask = (hme_values > hme_mean - hme_std) & (hme_values < hme_mean + hme_std)

        # Do a quick significance scan for HME bounds
        # Take the mean of signal's HME, then scan an asymmetric window around it to maximize significance s/sqrt(s+b)
        best_significance = 0.0
        best_hme_low = 0
        best_hme_high = 0
        for hme_low_scan in np.linspace(hme_mean - 5 * hme_std, hme_mean, 10):
            for hme_high_scan in np.linspace(hme_mean, hme_mean + 5 * hme_std, 10):
                hme_mask_scan = (hme_values > hme_low_scan) & (
                    hme_values < hme_high_scan
                )
                s = np.sum(physics_weight[signal_mask & hme_mask_scan])
                b = np.sum(physics_weight[Background_mask & hme_mask_scan])
                significance = s / np.sqrt(s + b + 1e-6)
                if significance > best_significance:
                    best_significance = significance
                    best_hme_low = hme_low_scan
                    best_hme_high = hme_high_scan

        print(
            f"For mass {para_masspoint} we have HME bounds [{best_hme_low}, {best_hme_high}]"
        )
        print(f"This had a significance of {best_significance}")
        print(
            f"Original mean, low, high was: {hme_mean}, {hme_mean - hme_std}, {hme_mean + hme_std}"
        )
        hme_mask = (hme_values > best_hme_low) & (hme_values < best_hme_high)

    else:
        hme_mask = True  # Skip hme filtering for now

    mask_dict = {
        "Signal": signal_mask & hme_mask,
        "TT": TT_mask & hme_mask,
        "DY": DY_mask & hme_mask,
        "Other": Other_mask & hme_mask,
    }

    return mask_dict, physics_weight


# =========================================================
# 5. CLASS HISTOGRAMS
# =========================================================
def make_hist(values, mask, weights, bins):
    h, _ = np.histogram(values[mask], bins=bins, weights=weights[mask])
    h2, _ = np.histogram(values[mask], bins=bins, weights=weights[mask] ** 2)
    return h, np.sqrt(h2)


# =========================================================
# 6. ROOT WRITING
# =========================================================
def get_quantile_bins(values, mask, n_bins=50):
    """
    Build quantile bins from signal-only distribution
    """
    v = values[mask]

    if len(v) < 100:
        # fallback to uniform if statistics are too low
        return np.linspace(0, 1, n_bins + 1)

    return np.quantile(v, np.linspace(0, 1, n_bins + 1))


def write_root_outputs(
    ROOTOut,
    pred,
    dw,
    mask_dict,
    physics_weight,
    feature_values,
    setup,
    para_masspoint,
    process_tag,
    class_idx=0,
    bin_format="quantile",
):

    # =========================================================
    # QUANTILE BINNING FROM SIGNAL (stage-2 target)
    # =========================================================

    class_names = ["Signal", "TT", "DY", "Other"]
    class_idx_mask = mask_dict[class_names[class_idx]]

    bins = np.linspace(0, 1, 51)  # Default to uniform bins
    if bin_format == "quantile":
        nBins = 50
        bin_low = 0
        bin_high = 1
        pred_plot = pred
        bins = get_quantile_bins(pred, class_idx_mask, n_bins=nBins)
        bins[0] = 0.0
        bins[-1] = 1.0
    elif bin_format == "raw":
        nBins = 50
        bin_low = 0
        bin_high = 1
        pred_plot = pred
        bins = np.linspace(0, 1, nBins + 1)
    elif bin_format == "logit":
        nBins = 150
        bin_low = -15
        bin_high = 15
        pred_plot = np.clip(pred, 1e-7, 1 - 1e-7)  # Prevent log(0) or log(1)
        pred_plot = np.log(pred_plot / (1 - pred_plot))  # Convert to logit space
        bins = np.linspace(
            bin_low, bin_high, nBins + 1
        )  # Adjust range as needed for logits

    print(f"Starting root output for class {class_idx} on process {process_tag}")
    print(f"Using binning {bins}")

    ROOTOut.cd()
    if not ROOTOut.GetDirectory("2D_plots"):
        ROOTOut.mkdir("2D_plots")  # Always try, returns None or pointer

    dir2d = ROOTOut.GetDirectory("2D_plots")
    if not dir2d:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError("Failed to create or access ROOT directory '2D_plots'.")

    if not ROOTOut.GetDirectory("HME_plots"):
        ROOTOut.mkdir("HME_plots")  # Always try, returns None or pointer

    dirHME = ROOTOut.GetDirectory("HME_plots")
    if not dirHME:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError("Failed to create or access ROOT directory 'HME_plots'.")

    for pname, mask in mask_dict.items():

        # -----------------------
        # 1D DNN histogram
        # -----------------------
        h = ROOT.TH1D(
            f"DNN_{pname}_m{para_masspoint}_c{class_idx}", "", nBins, bin_low, bin_high
        )

        hist, err = make_hist(pred_plot, mask, physics_weight, bins)

        for i in range(nBins):
            h.SetBinContent(i + 1, hist[i])
            h.SetBinError(i + 1, err[i])

        ROOTOut.WriteObject(h, f"m{para_masspoint}_{pname}_class{class_idx}")

        # -----------------------
        # 1D HME histogram
        # -----------------------
        hme_bins = np.linspace(0.0, 2500.0, 251)
        dirHME.cd()
        h = ROOT.TH1D(f"HME_{pname}_m{para_masspoint}", "", 250, 0.0, 2500.0)

        hist, err = make_hist(feature_values["hme"], mask, physics_weight, hme_bins)

        for i in range(nBins):
            h.SetBinContent(i + 1, hist[i])
            h.SetBinError(i + 1, err[i])

        dirHME.WriteObject(h, f"HME_m{para_masspoint}_{pname}")

        # -----------------------
        # 2D DNN vs HME
        # -----------------------
        dir2d.cd()
        h2 = ROOT.TH2D(
            f"DNN_vs_HME_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            100,
            0,
            1,
            250,
            0,
            2500,
        )

        for dnn, hme in zip(pred[mask], feature_values["hme"][mask]):
            h2.Fill(dnn, hme)

        dir2d.WriteObject(
            h2,
            f"DNN_vs_HME_m{para_masspoint}_{pname}_class{class_idx}",
        )

        # -----------------------
        # 2D DNN vs features
        # -----------------------
        for i, feat in enumerate(setup["features"]):

            x = feature_values["features"][:, i][mask]

            h2f = ROOT.TH2D(
                f"DNN_vs_{feat}_{pname}_m{para_masspoint}_c{class_idx}",
                "",
                100,
                0,
                1,
                100,
                np.min(x),
                np.max(x),
            )

            for dnn, val in zip(pred[mask], x):
                h2f.Fill(dnn, val)

            dir2d.WriteObject(
                h2f,
                f"DNN_vs_{feat}_m{para_masspoint}_{pname}_class{class_idx}",
            )

        # -----------------------
        # 2D DNN vs Stage1 Score
        # -----------------------
        if "stage1_score" in feature_values.keys():
            h2 = ROOT.TH2D(
                f"DNN_vs_Stage1_Score_{pname}_m{para_masspoint}_c{class_idx}",
                "",
                100,
                0,
                1,
                100,
                0,
                1,
            )

            for dnn, stage1 in zip(pred[mask], feature_values["stage1_score"][mask]):
                # If stage1 is an array, take first entry
                if isinstance(stage1, np.ndarray):
                    stage1 = stage1[0]
                h2.Fill(dnn, stage1)

            dir2d.WriteObject(
                h2,
                f"DNN_vs_Stage1_Score_{pname}_m{para_masspoint}_class{class_idx}",
            )

        ROOTOut.cd()


# =========================================================
# 7. ROC
# =========================================================
def plot_roc(ax, y_true, pred, weights, label):
    display = sklearn.metrics.RocCurveDisplay.from_predictions(
        y_true,
        pred,
        sample_weight=weights,
        ax=ax,
        name=label,
    )
    return display


# =========================================================
# 8. OPTIONAL FEATURE IMPORTANCE (NOT USED IN FLOW)
# =========================================================
def permutation_importance_onnx(session, X, y, w, n_repeats=3):
    base = get_scores(session, X)[:, 0]
    base_auc = sklearn.metrics.roc_auc_score(y, base, sample_weight=w)

    imps = []

    for i in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            np.random.shuffle(Xp[:, i])
            p = get_scores(session, Xp)[:, 0]
            auc = sklearn.metrics.roc_auc_score(y, p, sample_weight=w)
            drops.append(base_auc - auc)

        imps.append(np.mean(drops))

    return base_auc, np.array(imps)


def feature_importance_scan(
    sess, X, y_true, w, cat, mass, feature_names, output_folder
):
    weights = np.clip(w, 0, None)

    X_feat = X
    y_feat = y_true
    w_feat = weights

    # subsample for speed (important for ROOT-scale datasets)
    max_events = 200_000_000
    if len(X_feat) > max_events:
        idx = np.random.choice(len(X_feat), max_events, replace=False)
        X_feat = X_feat[idx]
        y_feat = y_feat[idx]
        w_feat = w_feat[idx]

    baseline_auc, importances = permutation_importance_onnx(
        sess, X_feat, y_feat, w_feat, n_repeats=3  # stage-1 model
    )

    print(f"[FEATURE IMPORTANCE] baseline AUC = {baseline_auc:.5f}")

    # sort features
    sorted_idx = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    for rank, i in enumerate(sorted_idx):
        print(f"{rank+1:2d}. {feature_names[i]:30s} " f"ΔAUC = {importances[i]:.6f}")

    # =========================================================
    # SAVE PLOT
    # =========================================================

    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(importances)), importances[sorted_idx])

    plt.xticks(
        range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90
    )

    plt.ylabel("ΔAUC (importance)")
    plt.title(f"Feature Importance {cat} m{mass}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"importance_{cat}_m{mass}.pdf"))
    plt.close()


# =========================================================
# 9. MAIN PIPELINE (WITH 2-STAGE SUPPORT)
# =========================================================
def validate_dnn_pipeline(
    setup,
    validation_file,
    validation_weight_file,
    output_folder,
    model_stage1,
    model_stage2=None,
    model_config=None,
):

    os.makedirs(output_folder, exist_ok=True)

    sess1, config = load_model_and_config(model_stage1, model_config)

    sess2 = None
    if model_stage2 is not None:
        print("Loading Stage-2 model")
        sess2 = ort.InferenceSession(model_stage2)

    dw = prepare_datawrapper(setup, validation_file, validation_weight_file)

    hme_values = dw.GetHME(file_name=validation_file)
    nClasses = setup["nClasses"]

    for cat in ["res2b", "boosted", "res1b"]:

        for hme_cut in [True, False]:
            hme_cut_string = "_hme_cut" if hme_cut else ""

            ROOTOut = ROOT.TFile(
                os.path.join(output_folder, f"validation_{cat}{hme_cut_string}.root"),
                "RECREATE",
            )

            ROOTOut2 = (
                ROOT.TFile(
                    os.path.join(output_folder, f"validation_{cat}_stage2.root"),
                    "RECREATE",
                )
                if sess2
                else None
            )

            ROOTOut_raw = ROOT.TFile(
                os.path.join(
                    output_folder, f"validation_raw_{cat}{hme_cut_string}.root"
                ),
                "RECREATE",
            )

            ROOTOut2_raw = (
                ROOT.TFile(
                    os.path.join(output_folder, f"validation_raw_{cat}_stage2.root"),
                    "RECREATE",
                )
                if sess2
                else None
            )

            ROOTOut_logit = ROOT.TFile(
                os.path.join(
                    output_folder, f"validation_logit_{cat}{hme_cut_string}.root"
                ),
                "RECREATE",
            )

            ROOTOut2_logit = (
                ROOT.TFile(
                    os.path.join(output_folder, f"validation_logit_{cat}_stage2.root"),
                    "RECREATE",
                )
                if sess2
                else None
            )

            for mass in config["parametric_list"]:

                print(f"{cat} mass {mass}")

                if dw.use_parametric:
                    dw.SetPredictParamValue(mass)

                X = dw.features_paramSet if dw.use_parametric else dw.features_no_param

                preds1 = get_scores(sess1, X)
                mask_dict, w = build_event_masks(
                    dw, hme_values, cat, mass, hme_cut=hme_cut
                )

                if np.sum(w) == 0:
                    print(f"No events in cat {cat}, continue.")
                    continue

                feature_values = {
                    "hme": hme_values,
                    "features": dw.features,
                }

                # =================================================
                # STAGE 1 OUTPUTS (multi-class)
                # =================================================
                for c in range(nClasses):
                    write_root_outputs(
                        ROOTOut,
                        preds1[:, c],
                        dw,
                        mask_dict,
                        w,
                        feature_values,
                        setup,
                        mass,
                        cat,
                        class_idx=c,
                        bin_format="quantile",
                    )

                    write_root_outputs(
                        ROOTOut_raw,
                        preds1[:, c],
                        dw,
                        mask_dict,
                        w,
                        feature_values,
                        setup,
                        mass,
                        cat,
                        class_idx=c,
                        bin_format="raw",
                    )

                    write_root_outputs(
                        ROOTOut_logit,
                        preds1[:, c],
                        dw,
                        mask_dict,
                        w,
                        feature_values,
                        setup,
                        mass,
                        cat,
                        class_idx=c,
                        bin_format="logit",
                    )

                # =================================================
                # ROC (stage 1)
                # =================================================
                y = (dw.class_target == 0).astype(int)

                fig, ax = plt.subplots()
                plot_roc(ax, y, preds1[:, 0], np.clip(w, 0, None), f"{cat} m{mass}")

                ax.plot([0, 1], [0, 1], "--")
                plt.savefig(os.path.join(output_folder, f"ROC_{cat}_{mass}.pdf"))
                plt.close()

                # =========================================================
                # FEATURE IMPORTANCE (STAGE-1 BASELINE)
                # =========================================================

                print(
                    f"[INFO] Running stage-1 feature importance for {cat} mass {mass}"
                )

                y_true = (dw.class_target == 0).astype(int)

                feature_names = (setup["features"]).copy()
                feature_importance_folder = os.path.join(
                    output_folder, "feature_importance"
                )
                # feature_importance_scan(
                #     sess1, X, y_true, w, cat, mass, feature_names, output_folder
                # )

                if getattr(setup, "train_stage_2", False):
                    # =================================================
                    # STAGE 2 (binary refinement on signal-like events)
                    # =================================================
                    if sess2 is not None:

                        signal_like = np.argmax(preds1, axis=1) == 0

                        # preds1_nested = preds1[:,0].reshape(-1, 1)
                        preds1_nested = preds1
                        X2 = X[signal_like]
                        # X2 = np.concatenate([X[signal_like], preds1_nested[signal_like]], axis=1)

                        preds2 = get_scores(sess2, X2)
                        preds2 = preds2[:, 0]  # Still a 2class technically

                        mask_dict2 = {k: v[signal_like] for k, v in mask_dict.items()}

                        feature_values2 = {
                            "hme": hme_values[signal_like],
                            "features": dw.features[signal_like],
                            "stage1_score": preds1_nested[signal_like],
                        }

                        write_root_outputs(
                            ROOTOut2,
                            preds2,
                            dw,
                            mask_dict2,
                            w[signal_like],
                            feature_values2,
                            setup,
                            mass,
                            cat,
                            class_idx=0,
                        )

                        # =========================================================
                        # FEATURE IMPORTANCE (STAGE-2)
                        # =========================================================

                        print(
                            f"[INFO] Running stage-2 feature importance for {cat} mass {mass}"
                        )

                        y_true = (dw.class_target == 0).astype(int)

                        y_true_stage2 = y_true[signal_like]
                        w_stage2 = w[signal_like]
                        # feature_names.append("stage1_score_signal")
                        # feature_names.append("stage1_score_TT")
                        # feature_names.append("stage1_score_DY")
                        # feature_names.append("stage1_score_Other")
                        feature_importance_folder = os.path.join(
                            output_folder, "feature_importance_stage2"
                        )
                        feature_importance_scan(
                            sess2,
                            X2,
                            y_true_stage2,
                            w_stage2,
                            cat,
                            mass,
                            feature_names,
                            feature_importance_folder,
                        )

    print("Done.")
