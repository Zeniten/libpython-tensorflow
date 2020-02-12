(ns tf-test.core
  (:require [libpython-clj.python :as py]
            [libpython-clj.jna.base]))

;; Depending on your Python version and virtualenv setup, change accordingly
;; Kan man greie seg uten uttrykket rett nedenfor?
(alter-var-root #'libpython-clj.jna.base/*python-library* (constantly "python3.7m"))
(py/initialize! :python-executable (str (System/getenv "HOME") "/miniconda3/envs/ml/bin/python"))

(require '[libpython-clj.require :refer [require-python]])
(require-python '[tensorflow :as tf]
                '[tensorflow.keras.models :as models]
                '[tensorflow.keras.layers :as layers]
                '[tensorflow.keras.datasets.mnist :as mnist]
                '[numpy :as np]
                '[builtins :as python])

;; load data
(defonce mnist-data (mnist/load_data))

;; prepare data for training and evaluation
(let [[[x-train y-train] [x-test y-test]] mnist-data]
  (def x-train (np/divide x-train 255))
  (def y-train y-train)
  (def x-test (np/divide x-test 255))
  (def y-test y-test))

;; define dense layers
(def relu-dense (layers/Dense 128 :activation "relu"))
(def softmax-dense (layers/Dense 10 :activation "softmax"))

;; specify model architecture
(defonce model (models/Sequential [(layers/Flatten :input_shape [28 28])
                                   relu-dense
                                   (layers/Dropout 0.2)
                                   softmax-dense
                                   ]))

;; ??
(py/py. model compile
        :optimizer "adam"
        :loss "sparse_categorical_crossentropy"
        :metrics (python/list ["accuracy"]))

;; function for training model
(defn train-model
  [model epochs]
  (py/py. model fit x-train y-train :epochs epochs))

;; function for evaluating model
(defn evaluate-model
  [model]
  (py/py. model evaluate x-test y-test :verbose 2))
