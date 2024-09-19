# encoding: utf-8
# @Time   : 2023/6/15
# @Author : Spike
# @Descr   :
import os.path
from common.toolbox import update_ui, CatchException
from crazy_functions.submit_fns import (
    result_written_to_markdown, result_converter_to_flow_chart, result_extract_to_test_cases,
    result_supplementary_to_test_case, json_args_return, user_input_embedding_content,
    func_拆分与提问, file_extraction_intype, file_classification_to_dict
)

func_kwargs = {
    'Markdown转换为流程图': result_converter_to_flow_chart,
    '结果写入Markdown': result_written_to_markdown,
    '写入测试用例': result_extract_to_test_cases,
    '补充测试用例': result_supplementary_to_test_case
}


@CatchException
def Reader_多阶段生成回答(user_input, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    valid_type, = json_args_return(plugin_kwargs, keys=["处理文件类型"], default=[])
    embedding_limit = yield from user_input_embedding_content(user_input, chatbot, history,
                                                                        llm_kwargs, plugin_kwargs, valid_type)
    if not embedding_limit:
        return
    multi_stage_config, = json_args_return(plugin_kwargs, keys=['定制化流程'], default={})
    gpt_results_count = {}
    for stage in multi_stage_config:
        prompt = stage.get('提示词', False)
        func = stage.get('保存结果', False)
        knowledge = stage.get('关联知识库', False)
        chatbot[-1][1] += f'\n\n---\n\n```\n{stage}\n```'
        yield from update_ui(chatbot=chatbot, history=history, msg='提交到分词器')
        embedding_limit = yield from func_拆分与提问(embedding_limit, llm_kwargs, plugin_kwargs, chatbot,
                                                               history, plugin_prompt=prompt, knowledge_base=knowledge)
        if func and func_kwargs.get(func, False):
            gpt_results_count[prompt] = yield from func_kwargs[func](embedding_limit, llm_kwargs, plugin_kwargs,
                                                                     chatbot, history)
            embedding_limit = []
        else:
            if stage != [i for i in multi_stage_config][-1]:
                yield from update_ui(chatbot=chatbot, history=history, msg='你没有选择保存结果，将提取结果提交给下一阶段')
                content_limit = file_classification_to_dict(embedding_limit)
                embedding_limit = [[limit, "".join(content_limit[limit])] for limit in content_limit]
                yield from update_ui(chatbot=chatbot, history=history, msg='你没有选择保存结果，将提取结果提交给下一阶段')
        if stage != [i for i in multi_stage_config][-1]:
            chatbot.append(['进入下一步', ''])
            embedding_mapping = yield from file_extraction_intype(gpt_results_count[prompt], chatbot, history,
                                                                            llm_kwargs,
                                                                            plugin_kwargs)
            for i in embedding_mapping:
                embedding_limit.extend([os.path.basename(i), embedding_mapping[i]])
            yield from update_ui(chatbot=chatbot, history=history,
                                 msg=f'分词器处理，完成将启用「{len(embedding_mapping)}」个线程，转交LLM处理')
        yield from update_ui(chatbot=chatbot, history=history, msg=f'分词器处理完成')
    if not multi_stage_config:
        chatbot[-1][1] = chatbot[-1][
                             1] + f'!!!!! 自定义参数中的Json存在问题，请仔细检查以下配置是否符合JSON编码格式\n\n```\n{plugin_kwargs["advanced_arg"]}```'

