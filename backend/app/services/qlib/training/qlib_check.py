"""
Qlib可用性检查模块
"""

from loguru import logger

# 检测Qlib可用性
try:
    pass

    QLIB_AVAILABLE = True
    logger.info("Qlib库已成功导入")
except ImportError as e:
    error_msg = str(e)
    missing_module = None

    # 检测缺失的模块
    if "setuptools_scm" in error_msg:
        missing_module = "setuptools_scm"
    elif "ruamel" in error_msg or "ruamel.yaml" in error_msg:
        missing_module = "ruamel.yaml"
    elif "cvxpy" in error_msg:
        missing_module = "cvxpy"
    elif "lightgbm" in error_msg:
        missing_module = "lightgbm"

    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}。导入错误: {e}\n"
            f"解决方法: pip install {missing_module}\n"
            f"如果还有其他依赖缺失，请运行修复脚本: ./fix_qlib_dependencies.sh\n"
            f"或手动安装所有依赖: pip install setuptools_scm cvxpy dill fire gym jupyter lightgbm matplotlib mlflow nbconvert pymongo python-redis-lock redis 'ruamel.yaml>=0.17.38'\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    else:
        logger.warning(
            f"Qlib未安装或缺少依赖。导入错误: {e}\n"
            f"安装方法: pip install git+https://github.com/microsoft/qlib.git\n"
            f"或使用 Gitee 镜像: pip install git+https://gitee.com/mirrors/qlib.git\n"
            f"如果已安装但缺少依赖，请运行: ./fix_qlib_dependencies.sh\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    QLIB_AVAILABLE = False
except Exception as e:
    logger.error(f"Qlib导入时发生未知错误: {e}")
    QLIB_AVAILABLE = False
