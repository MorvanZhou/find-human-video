"""
文件时间属性工具

提供跨平台的文件时间属性设置功能，用于将源视频的创建时间复制到输出文件。

支持平台：
- macOS: 支持设置创建时间（使用 SetFile 命令）
- Windows: 支持设置创建时间（需要 pywin32）
- Linux: 仅支持设置修改时间（系统限制）
"""

import os
import platform
import subprocess
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_file_times(file_path: str) -> dict | None:
    """
    获取文件的创建时间和修改时间
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含时间戳的字典：
        - created_time: 创建时间戳
        - modified_time: 修改时间戳
        失败返回 None
    """
    try:
        stat_info = os.stat(file_path)
        
        result = {
            'modified_time': stat_info.st_mtime,
        }
        
        system = platform.system()
        
        if system == 'Darwin':
            result['created_time'] = stat_info.st_birthtime
        elif system == 'Windows':
            result['created_time'] = stat_info.st_ctime
        else:
            # Linux: 使用修改时间作为替代
            result['created_time'] = stat_info.st_mtime
        
        return result
        
    except Exception as e:
        logger.warning(f"获取文件时间属性失败: {file_path}, 错误: {e}")
        return None


def set_file_times(file_path: str, times: dict) -> bool:
    """
    设置文件的创建时间和修改时间
    
    Args:
        file_path: 文件路径
        times: 包含时间戳的字典：
            - created_time: 创建时间戳（可选）
            - modified_time: 修改时间戳（可选）
            - accessed_time: 访问时间戳（可选）
        
    Returns:
        是否成功
    """
    try:
        # 设置访问时间和修改时间
        modified_time: float = times.get('modified_time')
        access_time: float = times.get('accessed_time', modified_time)
        
        
        if modified_time:
            os.utime(file_path, (access_time, modified_time))
        
        # 设置创建时间（平台相关）
        created_time = times.get('created_time')
        if created_time:
            system = platform.system()
            
            if system == 'Darwin':
                _set_macos_birth_time(file_path, created_time)
            elif system == 'Windows':
                _set_windows_birth_time(file_path, created_time)
        
        return True
        
    except Exception as e:
        logger.warning(f"设置文件时间属性失败: {file_path}, 错误: {e}")
        return False


def _set_macos_birth_time(file_path: str, birth_time: float) -> bool:
    """在 macOS 上设置文件的创建时间"""
    try:
        created_dt = datetime.fromtimestamp(birth_time)
        date_str = created_dt.strftime("%m/%d/%Y %H:%M:%S")
        
        result = subprocess.run(
            ["SetFile", "-d", date_str, file_path],
            capture_output=True,
            check=False
        )
        
        return result.returncode == 0
        
    except Exception as e:
        logger.debug(f"SetFile 设置创建时间失败: {file_path}, 错误: {e}")
        return False


def _set_windows_birth_time(file_path: str, birth_time: float) -> bool:
    """在 Windows 上设置文件的创建时间"""
    try:
        import pywintypes
        import win32file
        import win32con
        
        created_dt = datetime.fromtimestamp(birth_time)
        wintime = pywintypes.Time(created_dt)
        
        handle = win32file.CreateFile(
            file_path,
            win32con.GENERIC_WRITE,
            win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
            None,
            win32con.OPEN_EXISTING,
            win32con.FILE_ATTRIBUTE_NORMAL,
            None
        )
        
        try:
            win32file.SetFileTime(handle, wintime, None, None)
            return True
        finally:
            handle.Close()
            
    except ImportError:
        logger.debug("pywin32 未安装，无法设置 Windows 创建时间")
        return False
    except Exception as e:
        logger.debug(f"设置 Windows 创建时间失败: {file_path}, 错误: {e}")
        return False
