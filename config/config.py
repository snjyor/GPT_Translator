"""
这个文件包含整个应用的配置参数
"""
import os
import openai


class Config:
    # ES服务器链接
    ELASTIC_SERVER = os.getenv("ELASTIC_SERVER")
    # ES用户名
    ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
    # ES密码
    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
    # 语料库ES索引
    INDEX = "translation_material_1.0"
    # openai密钥
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # 是否用Azure的openai?
    USE_AZURE_AI = True
    # 如果是的话，还需要配置以下参数(有默认值不用配)
    if USE_AZURE_AI:
        openai.api_type = os.getenv("AZURE_API_TYPE", "azure")
        openai.api_base = os.getenv("AZURE_API_BASE")
        openai.api_version = os.getenv("AZURE_API_VERSION", "2023-03-15-preview")
    # 默认使用Azure openai的gpt35模型，可选的模型名有gpt35,gpt4-8k,gpt4-32k,davinci
    AZURE_GPT_ENGINE = "gpt35"
    # 单次翻译文本限制长度, 这是对用户输入的文本长度而言
    TEXT_LENGTH_LIMIT = 2000
    # embedding向量维度
    VECTOR_DIM = 1536
    # 搜索最相似的K条数据返回
    DEFAULT_TOP_K = 1
    # 反馈数据插入数据库的最大尝试次数
    DATA_INSERT_TRY_NUM = 20
    # 是否要搜索术语对资料
    IS_SEARCH_TERM_DATA = False
    # 支持的语言
    SUPPORTED_LANGUAGES = [
        "English",
        "Chinese"
    ]
    # 不同模型有不同的MAX TOKENS
    ENGINE_TOKENS_MAPPING = {
        "gpt35": 4096,  # gpt-35-turbo
        "gpt4-8k": 8192,  # gpt-4
        "gpt4-32k": 32768,  # gpt-4-32k
        "davinci": 4097  # text-davinci-003
    }
    # GPT模型最大token数量
    MAX_TOKENS = 4096
    # 文本token限制, 这是对输入给GPT的文本token数量而言,即每次翻译大概TEXT_TOKEN_LIMIT的量, 这个变量应该至少小于MAX_TOKENS的一半以上, 最好是MAX_TOKENS的1/4
    TEXT_TOKEN_LIMIT = MAX_TOKENS // 4


cfg = Config()
