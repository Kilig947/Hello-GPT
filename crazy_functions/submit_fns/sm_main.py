# encoding: utf-8
# @Time   : 2024/7/19
# @Author : Spike
# @Descr   :
import os
import json

from common.func_box import replace_expected_text, long_name_processing
from common import gr_converter_html
from common.toolbox import update_ui, update_ui_lastest_msg, get_conf
from crazy_functions import crazy_utils
from crazy_functions.pdf_fns.breakdown_txt import breakdown_text_to_satisfy_token_limit
from request_llms import bridge_all
from crazy_functions.submit_fns.content_process import (
    find_index_inlist, json_args_return, input_retrieval_file,
    file_extraction_intype, content_clear_links, content_img_vision_analyze

)


# <---------------------------------------插件用了都说好方法----------------------------------------->
def split_list_token_limit(data, get_num, max_num=500):
    header_index = find_index_inlist(data_list=data, search_terms=['操作步骤', '前置条件', '预期结果'])
    header_data = data[header_index]
    max_num -= len(str(header_data))
    temp_list = []
    split_data = []
    for index in data[header_index + 1:]:
        if get_num(str(temp_list)) > max_num:
            temp_list.insert(0, header_data)
            split_data.append(json.dumps(temp_list, ensure_ascii=False))
            temp_list = []
        else:
            temp_list.append(index)
    return split_data


def split_content_limit(inputs: str, llm_kwargs, chatbot, history) -> list:
    """
    Args:
        inputs: 需要提取拆分的提问信息
        llm_kwargs: 调优参数
        chatbot: 对话组件
        history: 历史记录
    Returns: [拆分1， 拆分2]
    """
    model = llm_kwargs['llm_model']
    if model.find('&') != -1:  # 判断是否多模型，如果多模型，那么使用tokens最少的进行拆分
        models = str(model).split('&')
        _tokens = []
        _num_func = {}
        for _model in models:
            num_s = bridge_all.model_info[_model]['max_token']
            _tokens.append(num_s)
            _num_func[num_s] = _model
        all_tokens = min(_tokens)
        get_token_num = bridge_all.model_info[_num_func[all_tokens]]['token_cnt']
    else:
        all_tokens = bridge_all.model_info[model]['max_token']
        get_token_num = bridge_all.model_info[model]['token_cnt']
    max_token = all_tokens / 2  # 考虑到对话+回答会超过tokens,所以/2
    segments = []
    gpt_latest_msg = chatbot[-1][1]
    if type(inputs) is list:
        if get_token_num(str(inputs)) > max_token:
            bro_say = gpt_latest_msg + f'\n\n提交数据预计会超出`{all_tokens}' \
                                       f'token`限制, 将按照模型最大可接收token拆分为多线程运行\n\n---\n\n'
            yield from update_ui_lastest_msg(bro_say, chatbot, history)
            segments.extend(split_list_token_limit(data=inputs, get_num=get_token_num, max_num=max_token))
        else:
            segments.append(json.dumps(inputs, ensure_ascii=False))
    else:
        inputs = inputs.split('\n---\n')
        for input_ in inputs:
            if get_token_num(input_) > max_token:
                bro_say = gpt_latest_msg + f'\n\n{gr_converter_html.html_tag_color(input_[0][:20])} 对话数据预计会超出`{all_tokens}' \
                                           f'token`限制, 将按照模型最大可接收token拆分为多线程运行'
                yield from update_ui_lastest_msg(bro_say, chatbot, history)
                segments.extend(
                    breakdown_text_to_satisfy_token_limit(input_, max_token, llm_kwargs['llm_model']))
            else:
                segments.append(input_)
    yield from update_ui(chatbot, history)
    return segments


def input_output_processing(gpt_response_collection, llm_kwargs, plugin_kwargs, chatbot, history,
                            kwargs_prompt: str = False, knowledge_base: bool = False):
    """
    Args:
        gpt_response_collection:  多线程GPT的返回结果or文件读取处理后的结果
        plugin_kwargs: 对话使用的插件参数
        chatbot: 对话组件
        history: 历史对话
        llm_kwargs:  调优参数
        kwargs_prompt: Prompt名称, 如果为False，则不添加提示词
        knowledge_base: 是否启用知识库
    Returns: 下次使用？
        inputs_array， inputs_show_user_array
    """
    inputs_array = []
    inputs_show_user_array = []
    prompt_cls, = json_args_return(plugin_kwargs, ['提示词分类'])
    ipaddr = llm_kwargs['ipaddr']
    if kwargs_prompt:
        from common.db.repository import prompt_repository
        prompt = prompt_repository.query_prompt(kwargs_prompt, prompt_cls, ipaddr, quote_num=True)
        if prompt:
            prompt = prompt.value
        else:
            raise ValueError('指定的提示词不存在')
    else:
        prompt = '{{{v}}}'
    for inputs, you_say in zip(gpt_response_collection[1::2], gpt_response_collection[0::2]):
        content_limit = yield from split_content_limit(inputs, llm_kwargs, chatbot, history)
        try:
            plugin_kwargs['上阶段文件'] = you_say
            plugin_kwargs[you_say] = {}
            plugin_kwargs[you_say]['原测试用例数据'] = [json.loads(limit)[1:] for limit in content_limit]
            plugin_kwargs[you_say]['原测试用例表头'] = json.loads(content_limit[0])[0]
        except Exception as f:
            print(f'读取原测试用例报错 {f}')
        for limit in content_limit:
            # 拼接内容与提示词
            plugin_prompt = replace_expected_text(prompt, content=limit, expect='{{{v}}}')
            inputs_array.append(plugin_prompt)
            inputs_show_user_array.append(you_say)
    yield from update_ui(chatbot, history)
    return inputs_array, inputs_show_user_array


def submit_no_use_ui_task(txt_proc, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, *args):
    inputs_show_user = None  # 不重复展示
    gpt_say = yield from crazy_utils.request_gpt_model_in_new_thread_with_ui_alive(
        inputs=txt_proc, inputs_show_user=inputs_show_user,
        llm_kwargs=llm_kwargs, chatbot=chatbot, history=history,
        sys_prompt="", refresh_interval=0.1
    )
    gpt_response_collection = [txt_proc, gpt_say]
    history.extend(gpt_response_collection)


def submit_multithreaded_tasks(inputs_array, inputs_show_user_array, llm_kwargs, chatbot, history, plugin_kwargs):
    """
    Args: 提交多线程任务
        inputs_array: 需要提交给gpt的任务列表
        inputs_show_user_array: 显示在user页面上信息
        llm_kwargs: 调优参数
        chatbot: 对话组件
        history: 历史对话
        plugin_kwargs: 插件调优参数
    Returns:  将对话结果返回[输入, 输出]
    """
    apply_history, = json_args_return(plugin_kwargs, ['上下文关联'], True)
    if apply_history:
        history_array = [[history] for _ in range(len(inputs_array))]
    else:
        history_array = [[] for _ in range(len(inputs_array))]
    # 是否要多线程处理
    if len(inputs_array) == 1:
        inputs_show_user = None  # 不重复展示
        gpt_say = yield from crazy_utils.request_gpt_model_in_new_thread_with_ui_alive(
            inputs=inputs_array[0], inputs_show_user=inputs_show_user,
            llm_kwargs=llm_kwargs, chatbot=chatbot, history=history_array[0],
            sys_prompt="", refresh_interval=0.1
        )
        gpt_response_collection = [inputs_array[0], gpt_say]
        history.extend(gpt_response_collection)
    else:
        gpt_response_collection = yield from crazy_utils.request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array=inputs_array,
            inputs_show_user_array=inputs_show_user_array,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            history_array=history_array,
            sys_prompt_array=["" for _ in range(len(inputs_array))],
            # max_workers=5,  # OpenAI所允许的最大并行过载
        )
    if apply_history:
        history.extend(gpt_response_collection)
    return gpt_response_collection


def func_拆分与提问(file_limit, llm_kwargs, plugin_kwargs, chatbot, history, plugin_prompt, knowledge_base):
    many_llm = json_args_return(plugin_kwargs, ['多模型并行'], )
    if many_llm[0]:
        llm_kwargs['llm_model'] = "&".join([i for i in many_llm[0].split('&') if i])
    split_content_limit = yield from input_output_processing(file_limit, llm_kwargs, plugin_kwargs,
                                                             chatbot, history, kwargs_prompt=plugin_prompt,
                                                             knowledge_base=knowledge_base)
    inputs_array, inputs_show_user_array = split_content_limit
    gpt_response_collection = yield from submit_multithreaded_tasks(inputs_array, inputs_show_user_array,
                                                                    llm_kwargs, chatbot, history,
                                                                    plugin_kwargs)
    return gpt_response_collection


def user_input_embedding_content(user_input, chatbot, history, llm_kwargs, plugin_kwargs, valid_types):
    embedding_content = []  # 对话内容
    yield from update_ui(chatbot=chatbot, history=history, msg='🕵🏻‍超级侦探，正在办案～')
    if plugin_kwargs.get('embedding_content'):
        embedding_content = plugin_kwargs['embedding_content']
        plugin_kwargs['embedding_content'] = ''  # 用了即刻丢弃
    else:
        chatbot.append([user_input, ''])
        download_format = gr_converter_html.get_fold_panel()
        chatbot[-1][1] = download_format(title='检测提交是否存在需要解析的文件或链接...', content='')
        yield from update_ui(chatbot=chatbot, history=history, msg='Reader loading...')
        fp_mapping, download_status = input_retrieval_file(user_input, llm_kwargs, valid_types)
        download_status.update(fp_mapping)
        if fp_mapping:
            chatbot[-1][1] = download_format(title='链接解析完成', content=download_status, status='Done')
        elif download_status.get('status'):
            chatbot[-1][1] = download_format(title='解析链接失败，请检查报错', content=download_status.get('status'), status='Done')
        content_mapping = yield from file_extraction_intype(fp_mapping, chatbot, history, llm_kwargs, plugin_kwargs)
        for content_fp in content_mapping:  # 一个文件一个对话
            file_content = content_mapping[content_fp]
            # 将解析的数据提交到正文
            input_handle = user_input.replace(fp_mapping[content_fp], str(file_content))
            # 将其他文件链接清除
            user_clear = content_clear_links(input_handle, fp_mapping, content_mapping)
            # 识别图片链接内容
            complete_input = yield from content_img_vision_analyze(user_clear, chatbot, history,
                                                                   llm_kwargs, plugin_kwargs)
            embedding_content.extend([os.path.basename(content_fp), complete_input])
        if not content_mapping:
            if len(user_input) > 100:  # 没有探测到任何文件，并且提交大于50个字符，那么运行往下走
                chatbot[-1][1] = download_format(title='没有检测到任何文件', content=download_status, status='Done')
                yield from update_ui(chatbot=chatbot, history=history, msg='没有探测到文件')
                # 识别图片链接内容
                complete_input = yield from content_img_vision_analyze(user_input, chatbot, history,
                                                                       llm_kwargs, plugin_kwargs)
                embedding_content.extend([long_name_processing(user_input), complete_input])
            else:
                devs_document = get_conf('devs_document')
                status = '\n\n没有探测到任何文件，并且提交字符少于50，无法完成后续任务' \
                         f'请在输入框中输入需要解析的云文档链接或本地文件地址，如果有多个文档则用换行或空格隔开，然后再点击对应的插件\n\n' \
                         f'插件支持解析文档类型`{valid_types}`' \
                         f"有问题？请联系`@spike` or 查看开发文档{devs_document}"
                if chatbot[-1][1] is None:
                    chatbot[-1][1] = status
                chatbot[-1][1] += status
                yield from update_ui(chatbot=chatbot, history=history, msg='没有探测到数据')
                return []
        kb_upload, = json_args_return(plugin_kwargs, ['自动录入知识库'])
        files_list = [i for i in content_mapping if os.path.exists(i)]
        if kb_upload and files_list:
            from common.knowledge_base import kb_doc_api
            kb_doc_api.upload_docs_simple(files=files_list, knowledge_base_name=kb_upload)
    return embedding_content