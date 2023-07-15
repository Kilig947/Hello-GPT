from comm_tools.toolbox import HotReload  # HotReload 的意思是热更新，修改函数插件后，不需要重启程序，代码直接生效


def get_crazy_functions():
    ###################### 第一组插件 ###########################
    from crazy_functions.读文章写摘要 import 读文章写摘要
    from crazy_functions.生成函数注释 import 批量生成函数注释
    from crazy_functions.解析项目源代码 import 解析项目本身
    from crazy_functions.解析项目源代码 import 解析一个Python项目
    from crazy_functions.解析项目源代码 import 解析一个C项目的头文件
    from crazy_functions.解析项目源代码 import 解析一个C项目
    from crazy_functions.解析项目源代码 import 解析一个Golang项目
    from crazy_functions.解析项目源代码 import 解析一个Rust项目
    from crazy_functions.解析项目源代码 import 解析一个Java项目
    from crazy_functions.解析项目源代码 import 解析一个前端项目
    from crazy_functions.高级功能函数模板 import 高阶功能模板函数
    from crazy_functions.Latex全文润色 import Latex英文润色
    from crazy_functions.询问多个大语言模型 import 同时问询
    from crazy_functions.解析项目源代码 import 解析一个Lua项目
    from crazy_functions.解析项目源代码 import 解析一个CSharp项目
    from crazy_functions.总结word文档 import 总结word文档
    from crazy_functions.辅助回答 import 猜你想问
    from crazy_functions.解析JupyterNotebook import 解析ipynb文件
    from crazy_functions.对话历史存档 import 对话历史存档
    from crazy_functions.对话历史存档 import 载入对话历史存档
    from crazy_functions.对话历史存档 import 删除所有本地对话历史记录
    from crazy_functions import KDOCS_轻文档分析
    from crazy_functions.批量Markdown翻译 import Markdown英译中
    function_plugins = {
        "猜你想问": {
            "Function": HotReload(猜你想问)
        },
        "解析整个Python项目": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,
            "Function": HotReload(解析一个Python项目)
        },

        "保存当前的对话": {
            "AsButton": True,
            "Function": HotReload(对话历史存档)
        },
        "载入对话历史存档（先上传存档或输入路径）": {
            "Color": "primary",
            "AsButton":False,
            "Function": HotReload(载入对话历史存档)
        },
        "Kdocs_多文件转测试用例(输入框输入文档链接)": {
            "Color": "primary",
            "AsButton": True,
            "Function": HotReload(KDOCS_轻文档分析.KDocs_转客户端测试用例),
            "AdvancedArgs": True,  # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "is_show 是否显示过程",  # 高级参数输入区的显示提示
            "Parameters": {
                "is_show": False,
                "prompt": '文档转测试用例',
                'img_ocr': False,
                "to_markdown": '文档转测试用例'
            }
        },
        "接口文档转测试用例(输入框输入需求文档)": {
            "Color": "primary",
            "AsButton": True,
            "Function": HotReload(KDOCS_轻文档分析.KDocs_转接口测试用例),
            "AdvancedArgs": True,  # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "is_show 是否显示过程",  # 高级参数输入区的显示提示
            "Parameters": {
                "is_show": False,
                "prompt": '接口文档转测试用例',
                'img_ocr': True,
                'to_markdown': '文档转Markdown_分割',
            }
        },
        "KDocs需求分析问答": {
            "Color": "primary",
            "AsButton": True,
            "Function": HotReload(KDOCS_轻文档分析.KDocs_需求分析问答),
            "AdvancedArgs": True,  # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "is_show 是否显示过程",  # 高级参数输入区的显示提示
            "Parameters": {
                "is_show": True,
                "prompt": '需求分析对话',
                'img_ocr': False,
                'to_markdown': '文档转Markdown',
            }
        },
        "KDocs文档转流程图": {
            "Color": "primary",
            "AsButton": True,
            "Function": HotReload(KDOCS_轻文档分析.KDocs_文档转流程图),
            "AdvancedArgs": True,  # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "is_show 是否显示过程",  # 高级参数输入区的显示提示
            "Parameters": {
                'to_markdown': '文档转Markdown',
                'img_ocr': True,
            }
        },

        "删除所有本地对话历史记录（请谨慎操作）": {
            "AsButton":False,
            "Function": HotReload(删除所有本地对话历史记录)
        },

        "[测试功能] 解析Jupyter Notebook文件": {
            "Color": "primary",
            "AsButton": False,
            "Function": HotReload(解析ipynb文件),
            "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "若输入0，则不解析notebook中的Markdown块", # 高级参数输入区的显示提示
        },
        "批量总结Word文档": {
            "AsButton": False,
            "Color": "primary",
            "Function": HotReload(总结word文档)
        },
        "解析整个C++项目头文件": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个C项目的头文件)
        },
        "解析整个C++项目（.cpp/.hpp/.c/.h）": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个C项目)
        },
        "解析整个Go项目": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个Golang项目)
        },
        "解析整个Rust项目": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个Rust项目)
        },
        "解析整个Java项目": {
            "Color": "primary",  # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个Java项目)
        },
        "解析整个前端项目（js,ts,css等）": {
            "Color": "primary",  # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个前端项目)
        },
        "解析整个Lua项目": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个Lua项目)
        },
        "解析整个CSharp项目": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(解析一个CSharp项目)
        },
        "读Tex论文写摘要": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(读文章写摘要)
        },
        "Markdown/Readme英译中": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,
            "Function": HotReload(Markdown英译中)
        },
        "批量生成函数注释": {
            "Color": "primary",    # 按钮颜色
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(批量生成函数注释)
        },
        "[多线程Demo] 解析此项目本身（源码自译解）": {
            "Function": HotReload(解析项目本身),
            "AsButton": False,  # 加入下拉菜单中
        },
        # "[老旧的Demo] 把本项目源代码切换成全英文": {
        #     # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
        #     "AsButton": False,  # 加入下拉菜单中
        #     "Function": HotReload(全项目切换英文)
        # },
        "[插件demo] 历史上的今天": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Function": HotReload(高阶功能模板函数),
            "AsButton": False,
        },

    }
    ###################### 第二组插件 ###########################
    # [第二组插件]: 经过充分测试
    from crazy_functions.批量总结PDF文档 import 批量总结PDF文档
    # from crazy_functions.批量总结PDF文档pdfminer import 批量总结PDF文档pdfminer
    from crazy_functions.批量翻译PDF文档_多线程 import 批量翻译PDF文档
    from crazy_functions.谷歌检索小助手 import 谷歌检索小助手
    from crazy_functions.理解PDF文档内容 import 理解PDF文档内容标准文件输入
    from crazy_functions.Latex全文润色 import Latex中文润色
    from crazy_functions.Latex全文润色 import Latex英文纠错
    from crazy_functions.Latex全文翻译 import Latex中译英
    from crazy_functions.Latex全文翻译 import Latex英译中
    from crazy_functions.批量Markdown翻译 import Markdown中译英

    function_plugins.update({
        "批量翻译PDF文档（多线程）": {
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(批量翻译PDF文档)
        },
        "询问多个GPT模型": {
            "Color": "primary",    # 按钮颜色
            "Function": HotReload(同时问询)
        },
        "[测试功能] 批量总结PDF文档": {
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Function": HotReload(批量总结PDF文档)
        },
        # "[测试功能] 批量总结PDF文档pdfminer": {
        #     "Color": "primary",
        #     "AsButton": False,  # 加入下拉菜单中
        #     "Function": HotReload(批量总结PDF文档pdfminer)
        # },
        "谷歌学术检索助手（输入谷歌学术搜索页url）": {
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(谷歌检索小助手)
        },
        "理解PDF文档内容 （模仿ChatPDF）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": True,  # 加入下拉菜单中
            "Function": HotReload(理解PDF文档内容标准文件输入)
        },
        "英文Latex项目全文润色（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Latex英文润色)
        },
        "英文Latex项目全文纠错（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Latex英文纠错)
        },
        "中文Latex项目全文润色（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Latex中文润色)
        },
        "Latex项目全文中译英（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Latex中译英)
        },
        "Latex项目全文英译中（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Latex英译中)
        },
        "批量Markdown中译英（输入路径或上传压缩包）": {
            # HotReload 的意思是热更新，修改函数插件代码后，不需要重启程序，代码直接生效
            "Color": "primary",
            "AsButton": False,  # 加入下拉菜单中
            "Function": HotReload(Markdown中译英)
        },


    })

    ###################### 第三组插件 ###########################
    # [第三组插件]: 尚未充分测试的函数插件
    try:
        from crazy_functions.下载arxiv论文翻译摘要 import 下载arxiv论文并翻译摘要
        function_plugins.update({
            "一键下载arxiv论文并翻译摘要（先在input输入编号，如1812.10695）": {
                "Color": "primary",
                "AsButton": False,  # 加入下拉菜单中
                "Function": HotReload(下载arxiv论文并翻译摘要)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.联网的ChatGPT import 连接网络回答问题
        function_plugins.update({
            "连接网络回答问题（输入问题后点击该插件，需要访问谷歌）": {
                "Color": "primary",
                "AsButton": False,  # 加入下拉菜单中
                "Function": HotReload(连接网络回答问题)
            }
        })
        from crazy_functions.联网的ChatGPT_bing版 import 连接bing搜索回答问题
        function_plugins.update({
            "连接网络回答问题（中文Bing版，输入问题后点击该插件）": {
                "Color": "primary",
                "AsButton": False,  # 加入下拉菜单中
                "Function": HotReload(连接bing搜索回答问题)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.解析项目源代码 import 解析任意code项目
        function_plugins.update({
            "解析项目源代码（手动指定和筛选源代码文件类型）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
                "ArgsReminder": "输入时用逗号隔开, *代表通配符, 加了^代表不匹配; 不输入代表全部匹配。例如: \"*.c, ^*.cpp, config.toml, ^*.toml\"", # 高级参数输入区的显示提示
                "Function": HotReload(解析任意code项目)
            },
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.询问多个大语言模型 import 同时问询_指定模型
        function_plugins.update({
            "询问多个GPT模型（手动指定询问哪些模型）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
                "ArgsReminder": "支持任意数量的llm接口，用&符号分隔。例如chatglm&gpt-3.5-turbo&api2d-gpt-4", # 高级参数输入区的显示提示
                "Function": HotReload(同时问询_指定模型)
            },
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.图片生成 import 图片生成
        function_plugins.update({
            "图片生成（先切换模型到openai或api2d）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
                "ArgsReminder": "在这里输入分辨率, 如256x256（默认）", # 高级参数输入区的显示提示
                "Function": HotReload(图片生成)
            },
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.总结音视频 import 总结音视频
        function_plugins.update({
            "批量总结音视频（输入路径或上传压缩包）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "调用openai api 使用whisper-1模型, 目前支持的格式:mp4, m4a, wav, mpga, mpeg, mp3。此处可以输入解析提示，例如：解析为简体中文（默认）。",
                "Function": HotReload(总结音视频)
            }
        })
    except:
        print('Load function plugin failed')

    from crazy_functions.解析项目源代码 import 解析任意code项目
    function_plugins.update({
        "解析项目源代码（手动指定和筛选源代码文件类型）": {
            "Color": "primary",
            "AsButton": False,
            "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "输入时用逗号隔开, *代表通配符, 加了^代表不匹配; 不输入代表全部匹配。例如: \"*.c, ^*.cpp, config.toml, ^*.toml\"", # 高级参数输入区的显示提示
            "Function": HotReload(解析任意code项目)
        },
    })
    from crazy_functions.询问多个大语言模型 import 同时问询_指定模型
    function_plugins.update({
        "询问多个GPT模型（手动指定询问哪些模型）": {
            "Color": "primary",
            "AsButton": False,
            "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "支持任意数量的llm接口，用&符号分隔。例如chatglm&gpt-3.5-turbo&api2d-gpt-4", # 高级参数输入区的显示提示
            "Function": HotReload(同时问询_指定模型)
        },
    })
    from crazy_functions.图片生成 import 图片生成
    function_plugins.update({
        "图片生成（先切换模型到openai或api2d）": {
            "Color": "primary",
            "AsButton": True,
            "AdvancedArgs": True, # 调用时，唤起高级参数输入区（默认False）
            "ArgsReminder": "在这里输入分辨率, 如'256x256'（默认）, '512x512', '1024x1024'", # 高级参数输入区的显示提示
            "Function": HotReload(图片生成)
        },
    })
    from crazy_functions.总结音视频 import 总结音视频
    function_plugins.update({
        "批量总结音视频（输入路径或上传压缩包）": {
            "Color": "primary",
            "AsButton": False,
            "AdvancedArgs": True,
            "ArgsReminder": "调用openai api 使用whisper-1模型, 目前支持的格式:mp4, m4a, wav, mpga, mpeg, mp3。此处可以输入解析提示，例如：解析为简体中文（默认）。",
            "Function": HotReload(总结音视频)
        }
    })
    try:
        from crazy_functions.数学动画生成manim import 动画生成
        function_plugins.update({
            "数学动画生成（Manim）": {
                "Color": "primary",
                "AsButton": False,
                "Function": HotReload(动画生成)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.批量Markdown翻译 import Markdown翻译指定语言
        function_plugins.update({
            "Markdown翻译（手动指定语言）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "请输入要翻译成哪种语言，默认为Chinese。",
                "Function": HotReload(Markdown翻译指定语言)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from comm_tools.Langchain知识库 import 知识库问答
        function_plugins.update({
            "[功能尚不稳定] 构建知识库（请先上传文件素材）": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "待注入的知识库名称id, 默认为default",
                "Function": HotReload(知识库问答)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from comm_tools.Langchain知识库 import 读取知识库作答
        function_plugins.update({
            "[功能尚不稳定] 知识库问答": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "待提取的知识库名称id, 默认为default, 您需要首先调用构建知识库",
                "Function": HotReload(读取知识库作答)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.Latex输出PDF结果 import Latex英文纠错加PDF对比
        function_plugins.update({
            "Latex英文纠错+高亮修正位置 [需Latex]": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "如果有必要, 请在此处追加更细致的矫错指令（使用英文）。",
                "Function": HotReload(Latex英文纠错加PDF对比)
            }
        })
        from crazy_functions.Latex输出PDF结果 import Latex翻译中文并重新编译PDF
        function_plugins.update({
            "Arixv翻译（输入arxivID）[需Latex]": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder":
                    "如果有必要, 请在此处给出自定义翻译命令, 解决部分词汇翻译不准确的问题。 "+
                    "例如当单词'agent'翻译不准确时, 请尝试把以下指令复制到高级参数区: " + 'If the term "agent" is used in this section, it should be translated to "智能体". ',
                "Function": HotReload(Latex翻译中文并重新编译PDF)
            }
        })
        function_plugins.update({
            "本地论文翻译（上传Latex压缩包）[需Latex]": {
                "Color": "primary",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder":
                    "如果有必要, 请在此处给出自定义翻译命令, 解决部分词汇翻译不准确的问题。 "+
                    "例如当单词'agent'翻译不准确时, 请尝试把以下指令复制到高级参数区: " + 'If the term "agent" is used in this section, it should be translated to "智能体". ',
                "Function": HotReload(Latex翻译中文并重新编译PDF)
            }
        })
    except:
        print('Load function plugin failed')

    # try:
    #     from crazy_functions.虚空终端 import 终端
    #     function_plugins.update({
    #         "超级终端": {
    #             "Color": "primary",
    #             "AsButton": False,
    #             # "AdvancedArgs": True,
    #             # "ArgsReminder": "",
    #             "Function": HotReload(终端)
    #         }
    #     })
    # except:
    #     print('Load function plugin failed')

    ###################### 第n组插件 ###########################
    return function_plugins
