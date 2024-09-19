# encoding: utf-8
# @Time   : 2023/6/14
# @Author : Spike
# @Descr   :
import copy
import os
import json
import re

from common.func_box import Shell, replace_expected_text, replace_special_chars
from common.func_box import valid_img_extensions, vain_open_extensions
from common import gr_converter_html
from common.func_box import split_domain_url, extract_link_pf
from common.toolbox import update_ui, update_ui_lastest_msg, get_conf, trimmed_format_exc
from crazy_functions import crazy_utils
from crazy_functions.pdf_fns.breakdown_txt import breakdown_text_to_satisfy_token_limit
from request_llms import bridge_all
from common.path_handler import init_path
from crazy_functions import reader_fns
from common.logger_handler import logger


# <---------------------------------------写入文件方法----------------------------------------->






if __name__ == '__main__':
    test = [1, 2, 3, 4, [12], 33, 1]
