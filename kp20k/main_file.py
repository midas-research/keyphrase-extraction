import os
os.mkdir('for_validation/');

import kp20k_process
import sgrank_parameter_select
import singlerank_parameter_select
import Textrank_parameter_select
import precision_recall_keyword_hyperparameter

import shutil
shutil.rmtree('for_validation/');