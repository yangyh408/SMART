import os
import pickle
import dataclasses

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  
import tensorflow_probability as tfp

from typing import List
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon

@dataclasses.dataclass(frozen=True)
class MetricFeatures:
    linear_accel: tf.Tensor
    linear_speed: tf.Tensor
    yaw_speed: tf.Tensor
    yaw_accel: tf.Tensor
    relative_distance: tf.Tensor
    relative_speed: tf.Tensor
    relative_accel: tf.Tensor
    ttc: tf.Tensor

class RealMetrics:
    def __init__(self):
        self.log_features: MetricFeatures = None
        self.sim_features: List[MetricFeatures] = []
        self.config = {
            'linear_speed': {
                'min_val': 0.0,
                'max_val': 25.0,
                'num_bins': 20,
                'bar_width': 0.5,
            },
            'linear_accel': {
                'min_val': -6.0,
                'max_val': 6.0,
                'num_bins': 20,
                'bar_width': 0.25,
            },
            'yaw_speed': {
                'min_val': -0.314,
                'max_val': 0.314,
                'num_bins': 20,
                'bar_width': 0.01,
            },
            'yaw_accel': {
                'min_val': -1.57,
                'max_val': 1.57,
                'num_bins': 20,
                'bar_width': 0.05,
            },
            'relative_distance': {
                'min_val': 0.0,
                'max_val': 45.0,
                'num_bins': 20,
                'bar_width': 1,
            },
            'relative_speed': {
                'min_val': -18.0,
                'max_val': 18.0,
                'num_bins': 20,
                'bar_width': 0.8,
            },
            'relative_accel': {
                'min_val': -6.0,
                'max_val': 6.0,
                'num_bins': 20,
                'bar_width': 0.25,
            },
            'ttc': {
                'min_val': 0.0,
                'max_val': 5.0,
                'num_bins': 20,
                'bar_width': 0.1,
            },
        }
        self.js_divergence = {field.name: None for field in dataclasses.fields(MetricFeatures)}  
    
    def add_log_features(self, metric_feature):
        self.log_features = metric_feature

    def add_sim_features(self, metric_feature):
        self.sim_features.append(metric_feature)

    def save(self, filename):  
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as file:  
            pickle.dump(self, file)  
  
    @staticmethod  
    def load(filename):  
        with open(filename, 'rb') as file:  
            obj = pickle.load(file)  
            if isinstance(obj, RealMetrics):  
                return obj  
            else:  
                print("WARNING: Loaded object is not an instance of RealMetrics!")  
            
    def cal_JS_with_histogram(self, field, log_feature, sim_feature, *, plot=False):
        edges = tf.cast(tf.linspace(
            self.config[field]['min_val'], self.config[field]['max_val'], self.config[field]['num_bins']+1), tf.float32)

        # log_feature = tf.clip_by_value(
        #     log_feature, self.config[field]['min_val'], self.config[field]['max_val'])
        # sim_feature = tf.clip_by_value(
        #     sim_feature, self.config[field]['min_val'], self.config[field]['max_val'])
        log_feature = log_feature[
            (log_feature >= self.config[field]['min_val']) & 
            (log_feature <= self.config[field]['max_val'])
        ]
        sim_feature = sim_feature[
            (sim_feature >= self.config[field]['min_val']) & 
            (sim_feature <= self.config[field]['max_val'])
        ]

        sim_counts = tfp.stats.histogram(sim_feature, edges)
        sim_counts /= tf.reduce_sum(sim_counts)
        log_counts = tfp.stats.histogram(log_feature, edges)
        log_counts /= tf.reduce_sum(log_counts)

        js_divergence = jensenshannon(sim_counts, log_counts)

        if plot:
            print(f"JS({field}) = {js_divergence}") 
            plt.bar(edges[:-1]+0.5*self.config[field]['bar_width'], sim_counts, align='center', width=self.config[field]['bar_width'], alpha=0.5, label="sim_info")
            plt.bar(edges[:-1]-0.5*self.config[field]['bar_width'], log_counts, align='center', width=self.config[field]['bar_width'], alpha=0.5, label="log_info")
            
            plt.title(f"{field} Distribution Probabilities")
            plt.legend()
            plt.show()
            print("-"*50 + '\n') 
        
        return js_divergence ** 2       # cal js_divergence with **2

    def _KDE_density(self, feature, min_val, max_val):
        eps = 1e-6
        h_bandwidth = 1.06 * np.std(feature) * feature.shape[0] ** (-1/5)
        kde = KernelDensity(
            kernel="gaussian", 
            bandwidth=h_bandwidth, 
            algorithm="auto"
        ).fit(feature[:, tf.newaxis] + eps)
        x_ax = np.linspace(min_val, max_val, 100).reshape(-1, 1)
        log_density = kde.score_samples(x_ax).reshape(-1, 1)
        density_results = np.exp(log_density).reshape(-1, 1)
        return x_ax, density_results

    def cal_JS_with_kernal(self, field, log_feature, sim_feature, *, plot=False):
        # log_feature = tf.clip_by_value(
        #     log_feature, self.config[field]['min_val'], self.config[field]['max_val'])
        # sim_feature = tf.clip_by_value(
        #     sim_feature, self.config[field]['min_val'], self.config[field]['max_val'])
        log_feature = log_feature[
            (log_feature >= self.config[field]['min_val']) & 
            (log_feature <= self.config[field]['max_val'])
        ]
        sim_feature = sim_feature[
            (sim_feature >= self.config[field]['min_val']) & 
            (sim_feature <= self.config[field]['max_val'])
        ]
        
        x_a_preds, ds_preds = self._KDE_density(sim_feature, self.config[field]['min_val'], self.config[field]['max_val'])
        x_a_log, ds_log = self._KDE_density(log_feature, self.config[field]['min_val'], self.config[field]['max_val'])

        m = (ds_preds + ds_log) / 2.0
        
        KL_preds_m = entropy(ds_preds, m)
        KL_log_m = entropy(ds_log, m)

        js_divergence = ((KL_log_m + KL_preds_m) / 2.0)[0]

        if plot:
            print(f"JS({field}) = {js_divergence}") 
            plt.plot(x_a_preds.ravel(), ds_preds.ravel(), label="sim_info")  
            plt.fill_between(x_a_preds.ravel(), ds_preds.ravel(), alpha=0.5)  
            plt.plot(x_a_log.ravel(), ds_log.ravel(), label="log_info")  
            plt.fill_between(x_a_log.ravel(), ds_log.ravel(), alpha=0.5)  

            plt.title(f"{field} Kernel Density Estimate")
            plt.legend()
            plt.show()
            print("-"*50 + '\n') 
        
        return js_divergence
    
    def compute_js_divergence(self, *, method: str="histogram", plot: bool=False):
        for field in self.js_divergence.keys():
            log_field_feature = getattr(self.log_features, field).numpy().reshape(-1)
            log_field_feature = log_field_feature[~np.isnan(log_field_feature)]

            sim_field_feature = np.concatenate(
                [getattr(sim_feature, field).numpy() for sim_feature in self.sim_features]
            ).reshape(-1)
            sim_field_feature = sim_field_feature[~np.isnan(sim_field_feature)]

            if method == "histogram":
                self.js_divergence[field] = self.cal_JS_with_histogram(field, log_field_feature, sim_field_feature, plot=plot)
            elif method == "kernel":
                self.js_divergence[field] = self.cal_JS_with_kernal(field, log_field_feature, sim_field_feature, plot=plot)
            else:
                raise RuntimeError("Unkown method for compute_js_divergence, please use 'histogram' or 'kernel'!")
  