"""
统一示例入口：定义一批数据，运行五种真值发现方法并对比。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from five_methods_comparison import main


if __name__ == "__main__":
    main()
