import dotenv
from loguru import logger

dotenv.load_dotenv()

logger.info("Configuration module loaded successfully.")

DASHSCOPE_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SILICONFLOW_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"


logger.info("API keys and URLs have been set.")
logger.debug(f"DASHSCOPE_API_KEY: {'set' if DASHSCOPE_API_KEY else 'not set'}")
logger.debug(f"SILICONFLOW_API_KEY: {'set' if SILICONFLOW_API_KEY else 'not set'}")

