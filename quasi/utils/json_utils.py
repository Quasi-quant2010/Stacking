# -*- coding: utf-8 -*-


import os
import sys
import json

from utils.config_utils import Config

def dump_stacking_setting(stack_setting_):

    text = json.dumps(stack_setting_, sort_keys=True, ensure_ascii=False, indent=4)

    data_folder = stack_setting_['setting']['folder']
    fname = stack_setting_['setting']['name']
    fname = os.path.join(Config.get_string('data.path'), data_folder, fname)
    with open(fname, 'w') as fh:
        fh.write(text.encode('utf-8'))

    return True
