"""IP 地理定位：查询出口公网 IP 所在位置，供 RAG 记忆与地理位置感知。

用 ipinfo.io（免费、无需 key）。返回英文城市/省份 + 国家代码，
常见地名翻译成中文（映射表兜底英文，DeepSeek 也能理解英文地名）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import httpx

from .utils.logging_setup import get_logger

log = get_logger("geo")

# 常见省份/城市 英文→中文 映射（ipinfo 返回英文，供 RAG 中文检索命中）
_CN_MAP = {
    "Beijing": "北京", "Shanghai": "上海", "Tianjin": "天津", "Chongqing": "重庆",
    "Guangdong": "广东", "Guangzhou": "广州", "Shenzhen": "深圳",
    "Zhejiang": "浙江", "Hangzhou": "杭州", "Ningbo": "宁波", "Wenzhou": "温州",
    "Jiangsu": "江苏", "Nanjing": "南京", "Suzhou": "苏州", "Wuxi": "无锡",
    "Fujian": "福建", "Fuzhou": "福州", "Xiamen": "厦门",
    "Shandong": "山东", "Jinan": "济南", "Qingdao": "青岛",
    "Sichuan": "四川", "Chengdu": "成都",
    "Hubei": "湖北", "Wuhan": "武汉",
    "Hunan": "湖南", "Changsha": "长沙",
    "Henan": "河南", "Zhengzhou": "郑州",
    "Hebei": "河北", "Shijiazhuang": "石家庄",
    "Anhui": "安徽", "Hefei": "合肥",
    "Jiangxi": "江西", "Nanchang": "南昌",
    "Shaanxi": "陕西", "Xian": "西安", "Xi'an": "西安",
    "Yunnan": "云南", "Kunming": "昆明",
    "Guizhou": "贵州", "Guiyang": "贵阳",
    "Guangxi": "广西", "Nanning": "南宁",
    "Hainan": "海南", "Haikou": "海口", "Sanya": "三亚",
    "Shanxi": "山西", "Taiyuan": "太原",
    "Liaoning": "辽宁", "Shenyang": "沈阳", "Dalian": "大连",
    "Jilin": "吉林", "Changchun": "长春",
    "Heilongjiang": "黑龙江", "Harbin": "哈尔滨",
    "Gansu": "甘肃", "Lanzhou": "兰州",
    "Qinghai": "青海", "Xining": "西宁",
    "Ningxia": "宁夏", "Yinchuan": "银川",
    "Xinjiang": "新疆", "Urumqi": "乌鲁木齐",
    "Inner Mongolia": "内蒙古", "Hohhot": "呼和浩特",
    "Tibet": "西藏", "Lhasa": "拉萨",
    "Macao": "澳门", "Macau": "澳门", "Hong Kong": "香港",
    "Taiwan": "台湾", "Taipei": "台北",
}

_COUNTRY_CN = {
    "CN": "中国", "US": "美国", "JP": "日本", "KR": "韩国", "SG": "新加坡",
    "HK": "中国香港", "MO": "中国澳门", "TW": "中国台湾",
    "GB": "英国", "DE": "德国", "FR": "法国", "CA": "加拿大", "AU": "澳大利亚",
}


def _zh(name: str) -> str:
    return _CN_MAP.get(name, name)


def detect_location(timeout: float = 10.0) -> Optional[Tuple[str, str, str, str]]:
    """查询出口 IP 位置，返回 (国家中文, 省份中文, 城市中文, ip)。

    失败返回 None（不阻塞启动）。
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get("https://ipinfo.io/json")
            r.raise_for_status()
            d = r.json()
    except Exception as e:
        log.warning("IP 定位失败: %s", e)
        return None
    country = _COUNTRY_CN.get(d.get("country", ""), d.get("country", ""))
    region = _zh(d.get("region", ""))
    city = _zh(d.get("city", ""))
    ip = d.get("ip", "")
    if not city and not region:
        log.warning("IP 定位结果无有效地理信息: %s", d)
        return None
    log.info("IP 定位: %s %s %s (ip=%s)", country, region, city, ip)
    return country, region, city, ip
