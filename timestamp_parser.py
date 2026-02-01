"""
监控视频文件名时间戳解析器

不同品牌的监控摄像头会使用不同的文件命名规则，本模块提供可扩展的时间戳解析器架构，
支持通过继承 BaseTimestampParser 来适配不同品牌的监控视频命名规则。

架构设计：
    BaseTimestampParser (抽象基类)
           │
    ┌──────┴──────────────────────────────┐
    │                                     │
    XiaomiTimestampParser          HikvisionTimestampParser
    (小米监控)                     (海康威视)
    │                                     │
    ├── 文件夹: YYYYMMDDHH                ├── 文件名: YYYYMMDDHHMMSS
    └── 文件名: MMmSSs_TIMESTAMP.mp4      └── ...

使用示例：
    from timestamp_parser import get_parser, XiaomiTimestampParser
    
    # 使用小米解析器
    parser = XiaomiTimestampParser()
    timestamp = parser.parse(Path("/path/to/2025060107/46M14S_1748735174.mp4"))
    
    # 通过品牌名获取解析器
    parser = get_parser("xiaomi")
    
    # 获取所有支持的品牌
    from timestamp_parser import SUPPORTED_BRANDS
    print(SUPPORTED_BRANDS)

扩展方法：
    1. 继承 BaseTimestampParser
    2. 实现 parse() 方法
    3. 在 PARSER_REGISTRY 中注册新的解析器
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Type


class BaseTimestampParser(ABC):
    """
    时间戳解析器抽象基类
    
    所有监控品牌的解析器都应该继承此类并实现 parse() 方法。
    
    Attributes:
        brand: 监控品牌名称
        description: 品牌命名规则描述
    """
    
    brand: str = "unknown"
    description: str = "未知品牌"
    
    @abstractmethod
    def parse(self, file_path: Path) -> float | None:
        """
        从文件路径解析时间戳
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳（秒），解析失败返回 None
        """
        pass
    
    def get_info(self) -> dict:
        """获取解析器信息"""
        return {
            "brand": self.brand,
            "description": self.description
        }


class XiaomiTimestampParser(BaseTimestampParser):
    """
    小米监控摄像头时间戳解析器
    
    文件命名规则：
    - 文件夹名: YYYYMMDDHH (年月日时)
    - 文件名格式: MMmSSs_TIMESTAMP.mp4 或 MMMSSS_TIMESTAMP.mp4
    
    示例：
    - 2025060107/46M14S_1748735174.mp4
      └── 2025年6月1日 07:46:14，时间戳 1748735174
    
    解析优先级：
    1. 优先从文件名中的 Unix 时间戳解析（最准确）
    2. 回退：从文件夹名+文件名组合解析（年月日时 + 分秒）
    """
    
    brand = "xiaomi"
    description = "小米监控: 文件夹 YYYYMMDDHH + 文件名 MMmSSs_TIMESTAMP.mp4"
    
    def parse(self, file_path: Path) -> float | None:
        """
        从小米监控文件名解析时间戳
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳，解析失败返回 None
        """
        filename = file_path.stem  # 去掉扩展名
        
        # 方法 1：从文件名中提取 Unix 时间戳
        # 格式: XXmYYs_TIMESTAMP 或 XXMYYS_TIMESTAMP
        match = re.search(r'_(\d{10,})$', filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 方法 2：从文件夹名和文件名组合解析
        # 文件夹: YYYYMMDDHH, 文件名: MMmSSs_xxx
        try:
            parent_name = file_path.parent.name
            if len(parent_name) == 10 and parent_name.isdigit():
                # 解析文件夹名: YYYYMMDDHH
                year = int(parent_name[0:4])
                month = int(parent_name[4:6])
                day = int(parent_name[6:8])
                hour = int(parent_name[8:10])
                
                # 解析文件名中的分钟和秒: XXmYYs 或 XXMYYS
                time_match = re.match(r'(\d{2})[Mm](\d{2})[Ss]', filename)
                if time_match:
                    minute = int(time_match.group(1))
                    second = int(time_match.group(2))
                    
                    dt = datetime(year, month, day, hour, minute, second)
                    return dt.timestamp()
        except Exception:
            pass
        
        return None


class HikvisionTimestampParser(BaseTimestampParser):
    """
    海康威视监控摄像头时间戳解析器
    
    常见文件命名规则：
    - 格式 1: YYYYMMDDHHMMSS.mp4 (14位数字)
    - 格式 2: ch01_YYYYMMDDHHMMSS.mp4 (带通道号)
    - 格式 3: IP_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.mp4 (起止时间)
    
    示例：
    - 20250601074614.mp4 → 2025年6月1日 07:46:14
    - ch01_20250601074614.mp4 → 2025年6月1日 07:46:14
    """
    
    brand = "hikvision"
    description = "海康威视: 文件名 YYYYMMDDHHMMSS.mp4 或带通道前缀"
    
    def parse(self, file_path: Path) -> float | None:
        """
        从海康威视监控文件名解析时间戳
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳，解析失败返回 None
        """
        filename = file_path.stem
        
        # 尝试匹配 14 位数字时间戳
        # 格式: YYYYMMDDHHMMSS (可能有前缀)
        match = re.search(r'(\d{14})', filename)
        if match:
            try:
                timestamp_str = match.group(1)
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        return None


class DahuaTimestampParser(BaseTimestampParser):
    """
    大华监控摄像头时间戳解析器
    
    常见文件命名规则：
    - 格式 1: YYYY-MM-DD HH-MM-SS.mp4
    - 格式 2: YYYYMMDD_HHMMSS.mp4
    - 格式 3: 带通道号 ch1_YYYYMMDD_HHMMSS.mp4
    
    示例：
    - 2025-06-01 07-46-14.mp4 → 2025年6月1日 07:46:14
    - 20250601_074614.mp4 → 2025年6月1日 07:46:14
    """
    
    brand = "dahua"
    description = "大华: 文件名 YYYY-MM-DD HH-MM-SS.mp4 或 YYYYMMDD_HHMMSS.mp4"
    
    def parse(self, file_path: Path) -> float | None:
        """
        从大华监控文件名解析时间戳
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳，解析失败返回 None
        """
        filename = file_path.stem
        
        # 格式 1: YYYY-MM-DD HH-MM-SS
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s*(\d{2})-(\d{2})-(\d{2})', filename)
        if match:
            try:
                dt = datetime(
                    int(match.group(1)),  # year
                    int(match.group(2)),  # month
                    int(match.group(3)),  # day
                    int(match.group(4)),  # hour
                    int(match.group(5)),  # minute
                    int(match.group(6))   # second
                )
                return dt.timestamp()
            except ValueError:
                pass
        
        # 格式 2: YYYYMMDD_HHMMSS
        match = re.search(r'(\d{8})[_\-](\d{6})', filename)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)
                dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        return None


class GenericTimestampParser(BaseTimestampParser):
    """
    通用时间戳解析器
    
    尝试从文件名中提取各种常见格式的时间戳。
    按优先级尝试多种格式，适用于未知品牌或自定义命名规则。
    
    支持的格式：
    - Unix 时间戳 (10-13 位数字)
    - YYYYMMDDHHMMSS (14位)
    - YYYYMMDD_HHMMSS
    - YYYY-MM-DD_HH-MM-SS
    - YYYY_MM_DD_HH_MM_SS
    """
    
    brand = "generic"
    description = "通用解析器: 尝试多种常见时间格式"
    
    def parse(self, file_path: Path) -> float | None:
        """
        尝试多种格式解析时间戳
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳，解析失败返回 None
        """
        filename = file_path.stem
        
        # 1. 尝试 Unix 时间戳 (10-13 位数字)
        match = re.search(r'[_\-]?(\d{10,13})[_\-\.]?', filename)
        if match:
            try:
                ts = float(match.group(1))
                # 13 位是毫秒级，转换为秒
                if ts > 1e12:
                    ts = ts / 1000
                # 验证时间戳范围 (2000-2100)
                if 946684800 <= ts <= 4102444800:
                    return ts
            except ValueError:
                pass
        
        # 2. 尝试 YYYYMMDDHHMMSS (14位)
        match = re.search(r'(\d{14})', filename)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        # 3. 尝试 YYYYMMDD_HHMMSS
        match = re.search(r'(\d{8})[_\-](\d{6})', filename)
        if match:
            try:
                dt = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        # 4. 尝试 YYYY-MM-DD_HH-MM-SS 或类似格式
        match = re.search(
            r'(\d{4})[_\-](\d{2})[_\-](\d{2})[_\-\s](\d{2})[_\-](\d{2})[_\-](\d{2})',
            filename
        )
        if match:
            try:
                dt = datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                    int(match.group(5)),
                    int(match.group(6))
                )
                return dt.timestamp()
            except ValueError:
                pass
        
        return None


# ============================================================
# 解析器注册表
# ============================================================

PARSER_REGISTRY: dict[str, Type[BaseTimestampParser]] = {
    "xiaomi": XiaomiTimestampParser,
    "hikvision": HikvisionTimestampParser,
    "dahua": DahuaTimestampParser,
    "generic": GenericTimestampParser,
}

# 品牌别名（便于用户输入）
BRAND_ALIASES: dict[str, str] = {
    "mi": "xiaomi",
    "小米": "xiaomi",
    "hik": "hikvision",
    "海康": "hikvision",
    "海康威视": "hikvision",
    "dh": "dahua",
    "大华": "dahua",
    "default": "generic",
    "auto": "generic",
    "通用": "generic",
}

# 支持的品牌列表
SUPPORTED_BRANDS = list(PARSER_REGISTRY.keys())


def get_parser(brand: str = "xiaomi") -> BaseTimestampParser:
    """
    获取指定品牌的时间戳解析器实例
    
    Args:
        brand: 监控品牌名称，支持别名
        
    Returns:
        对应品牌的解析器实例
        
    Raises:
        ValueError: 不支持的品牌
    
    Examples:
        >>> parser = get_parser("xiaomi")
        >>> parser = get_parser("小米")  # 使用别名
        >>> parser = get_parser("generic")  # 通用解析器
    """
    # 处理别名
    brand_lower = brand.lower()
    if brand_lower in BRAND_ALIASES:
        brand_lower = BRAND_ALIASES[brand_lower]
    
    if brand_lower not in PARSER_REGISTRY:
        supported = ", ".join(SUPPORTED_BRANDS)
        raise ValueError(
            f"不支持的监控品牌: {brand}。"
            f"支持的品牌: {supported}"
        )
    
    return PARSER_REGISTRY[brand_lower]()


def list_parsers() -> list[dict]:
    """
    列出所有可用的解析器
    
    Returns:
        解析器信息列表
    """
    result = []
    for brand, parser_class in PARSER_REGISTRY.items():
        # 对于 AutoDetectParser，不实例化（避免递归），直接使用类属性
        if brand == "auto":
            result.append({
                "brand": brand,
                "description": parser_class.description,
                "class": parser_class.__name__
            })
        else:
            parser = parser_class()
            result.append({
                "brand": brand,
                "description": parser.description,
                "class": parser_class.__name__
            })
    return result


def register_parser(brand: str, parser_class: Type[BaseTimestampParser], aliases: list[str] = None):
    """
    注册新的时间戳解析器
    
    用于扩展支持新的监控品牌。
    
    Args:
        brand: 品牌标识（小写）
        parser_class: 解析器类（必须继承 BaseTimestampParser）
        aliases: 品牌别名列表
        
    Examples:
        >>> class MyBrandParser(BaseTimestampParser):
        ...     brand = "mybrand"
        ...     description = "自定义品牌"
        ...     def parse(self, file_path):
        ...         return None
        >>> register_parser("mybrand", MyBrandParser, aliases=["mb", "我的品牌"])
    """
    if not issubclass(parser_class, BaseTimestampParser):
        raise TypeError("parser_class 必须继承 BaseTimestampParser")
    
    PARSER_REGISTRY[brand.lower()] = parser_class
    SUPPORTED_BRANDS.append(brand.lower())
    
    if aliases:
        for alias in aliases:
            BRAND_ALIASES[alias.lower()] = brand.lower()


# ============================================================
# 自动检测解析器
# ============================================================

class AutoDetectParser(BaseTimestampParser):
    """
    自动检测解析器
    
    尝试所有已注册的解析器，返回第一个成功解析的结果。
    适用于混合多个品牌监控的场景。
    """
    
    brand = "auto"
    description = "自动检测: 依次尝试所有解析器"
    
    def __init__(self, priority: list[str] = None):
        """
        初始化自动检测解析器
        
        Args:
            priority: 解析器优先级列表，默认按注册顺序
        """
        if priority:
            # 排除 auto 自身，避免递归
            self._parsers = [get_parser(b) for b in priority if b in PARSER_REGISTRY and b != "auto"]
        else:
            # 排除 AutoDetectParser 自身，避免无限递归
            self._parsers = [
                cls() for brand, cls in PARSER_REGISTRY.items() 
                if brand != "auto"
            ]
    
    def parse(self, file_path: Path) -> float | None:
        """
        依次尝试所有解析器
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            Unix 时间戳，所有解析器都失败返回 None
        """
        for parser in self._parsers:
            result = parser.parse(file_path)
            if result is not None:
                return result
        return None


# 注册自动检测解析器
PARSER_REGISTRY["auto"] = AutoDetectParser


# ============================================================
# 便捷函数
# ============================================================

def parse_timestamp(file_path: str | Path, brand: str = "xiaomi") -> float | None:
    """
    解析文件时间戳的便捷函数
    
    Args:
        file_path: 视频文件路径
        brand: 监控品牌名称
        
    Returns:
        Unix 时间戳，解析失败返回 None
        
    Examples:
        >>> ts = parse_timestamp("/path/to/2025060107/46M14S_1748735174.mp4", "xiaomi")
        >>> print(ts)  # 1748735174.0
    """
    parser = get_parser(brand)
    return parser.parse(Path(file_path))


if __name__ == "__main__":
    # 演示用法
    print("=" * 60)
    print("监控视频文件名时间戳解析器")
    print("=" * 60)
    
    print("\n可用的解析器:")
    for info in list_parsers():
        print(f"  - {info['brand']}: {info['description']}")
    
    print("\n示例解析:")
    test_files = [
        ("xiaomi", "2025060107/46M14S_1748735174.mp4"),
        ("hikvision", "ch01_20250601074614.mp4"),
        ("dahua", "2025-06-01 07-46-14.mp4"),
        ("generic", "video_1748735174_001.mp4"),
    ]
    
    for brand, filename in test_files:
        parser = get_parser(brand)
        ts = parser.parse(Path(filename))
        if ts:
            dt = datetime.fromtimestamp(ts)
            print(f"  [{brand}] {filename}")
            print(f"       → {dt.isoformat()}")
        else:
            print(f"  [{brand}] {filename} → 解析失败")
