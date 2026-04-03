DEFAULT_INDUSTRY_CTR = 0.03
DEFAULT_INDUSTRY_CONVERSION = 0.025
CTR_DROP_THRESHOLD = 0.30
IMPRESSIONS_DROP_THRESHOLD = 0.30
CTR_VS_7D_THRESHOLD = 0.50
SALES_DROP_THRESHOLD = 0.30
DEFAULT_CONVERSION_BASIS = "订单商品数"

TAG_THRESHOLDS = [
    ("大爆款", 100000, 50.000001),
    ("爆款", 50000, 25),
    ("旺款", 10000, 10),
    ("常规款", 1000, 1),
]

CORE_TAG_ORDER = ["大爆款", "爆款", "旺款", "常规款", "新品", "滞制品"]
QUICK_TAG_OPTIONS = ["全部标签", "大爆款", "爆款", "旺款", "常规款", "新品", "滞制品", "上升趋势品"]