"""
文件服务路由 - 处理文件下载等操作
"""

import os
import tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

router = APIRouter(prefix="/files", tags=["文件服务"])


@router.get("/download/{filename}")
async def download_file(filename: str):
    """下载文件"""
    try:
        # 安全检查：只允许下载报告文件
        if not filename.endswith(('.pdf', '.xlsx', '.csv')):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 构建文件路径
        temp_dir = tempfile.gettempdir()
        reports_dir = os.path.join(temp_dir, "backtest_reports")
        file_path = os.path.join(reports_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 检查文件是否在允许的目录内（安全检查）
        if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
            raise HTTPException(status_code=403, detail="访问被拒绝")
        
        # 返回文件
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")