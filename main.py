from evaluation.mathematics import math
from inspect_ai import eval
import evaluation.diffucoder_model_api

eval(math(), model="diffucoder/7b-instruct")