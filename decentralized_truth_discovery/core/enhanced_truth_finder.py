"""
Enhanced TruthFinder with DECENTRUTH formulas
基于TruthFinder结构，应用DECENTRUTH论文中的真值发现公式

这个文件实现了增强版的真值发现算法，主要功能是：
1. 从多个可能不可靠的数据源中发现真实值
2. 自动评估每个数据源的可靠性
3. 检测和惩罚提供错误数据的恶意源（拜占庭源）
4. 支持批量处理和增量处理

核心改进：
1. 使用DECENTRUTH的源可靠性更新机制（公式4-6）
2. 使用基于误差和一致性的可靠性度量
3. 支持增量批次处理
4. 维护源可靠性历史状态
"""
import numpy as np
from numpy.linalg import norm
import math
import re
from scipy.special import erfc  # 互补误差函数，用于一致性度量计算
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple


class EnhancedTruthFinder:
    """
    增强版TruthFinder，应用DECENTRUTH的真值发现公式

    这个类的主要作用是：
    - 输入：多个数据源（website）提供的关于不同对象（object）的值（fact）
    - 输出：每个对象的真实值估计，以及每个数据源的可靠性评分

    数据结构：DataFrame with columns: website, fact, object
    - website: 数据源标识（例如："website_a"、"网站1"等）
    - fact: 输入值，可以是数值型（例如：100.5、50.2）或离散型（例如："Einstein"、"北京"）
    - object: 对象标识（例如："stock_A"、"股票A"等）

    支持的数据类型：
    1. 数值类型：算法使用加权平均计算真值，使用平方误差计算误差
       - 例如：多个网站报告股票价格，算法会计算加权平均价格
    2. 离散类型（字符串）：算法使用加权投票选择真值，使用0/1匹配误差
       - 例如：多个网站报告发现者姓名，算法会选择权重和最大的值

    举个例子（数值类型）：
    如果我们有多个网站报告同一只股票的价格：
    - website_a 说 stock_A 的价格是 100.5
    - website_b 说 stock_A 的价格是 101.0
    - website_c 说 stock_A 的价格是 99.8
    这个算法会自动判断哪个网站更可信，并给出股票的真实价格估计（比如 100.4）

    举个例子（离散类型）：
    如果我们有多个网站报告相对论的发现者：
    - website_a 说 "Einstein"（可靠性0.9）
    - website_b 说 "Einstein"（可靠性0.8）
    - website_c 说 "Newton"（可靠性0.5）
    这个算法会选择"Einstein"作为真值（因为它的权重和最大：0.9+0.8=1.7 > 0.5）
    """

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 100.0,
                 convergence_threshold: float = 1e-6,
                 max_iterations: int = 200,
                 initial_reliability: float = 0.5,
                 numeric_weight: float = 0.5,
                 enable_hybrid_mode: bool = False):
        """
        初始化EnhancedTruthFinder

        这个方法就像是给算法设置"参数"，告诉它如何工作。

        Args（参数）:
            alpha: 衰减因子，用于加权移动平均 (0 < alpha < 1)
                   - 这个值越大（接近1），算法越关注历史数据（长期趋势）
                   - 这个值越小（接近0），算法越关注当前数据（短期变化）
                   - 默认0.9表示算法更重视历史表现
                   - 例如：alpha=0.9意味着90%的权重给历史，10%给当前

            beta: 尺度因子，用于一致性度量计算
                  - 这个值控制算法对数据源行为突然变化的敏感度
                  - 值越大，算法越容易检测到恶意行为
                  - 默认100.0是一个经验值，适合大多数场景
                  - 例如：如果一个源的数据突然大幅偏离，beta越大，一致性评分下降越快

            convergence_threshold: 收敛阈值
                                  - 当源可靠性的变化小于这个值时，算法认为已经收敛，停止迭代
                                  - 默认1e-6表示0.000001，非常小的变化
                                  - 例如：如果上一轮可靠性是0.5，这一轮是0.5000001，变化小于阈值，就停止

            max_iterations: 最大迭代次数
                           - 即使没有收敛，最多迭代这么多次就停止
                           - 默认200次，通常足够了
                           - 例如：如果200次迭代后还没收敛，可能是数据有问题，强制停止避免无限循环

            initial_reliability: 初始源可靠性
                                - 刚开始时，假设所有数据源都有这个可靠性（0-1之间）
                                - 默认0.5表示中等可信度，随着迭代会调整
                                - 例如：刚开始时，我们不知道哪个源可信，所以都给0.5（50%可信），后来会根据表现调整

            numeric_weight: 数值误差权重（仅用于混合模式）
                          - 用于组合数值误差和离散误差：combined_error = α × numeric_error + (1-α) × discrete_error
                          - 默认0.5表示数值和离散误差同等重要
                          - 范围：0-1之间

            enable_hybrid_mode: 是否启用混合模式
                               - True: 支持从文本中提取数值和离散信息，使用多维度误差
                               - False: 使用原有的单一数据类型模式（数值或离散）
                               - 默认False，保持向后兼容
        """
        # 保存所有参数，供后续方法使用
        self.alpha = alpha
        self.beta = beta
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.initial_reliability = initial_reliability
        self.numeric_weight = numeric_weight
        self.enable_hybrid_mode = enable_hybrid_mode

        # 源可靠性历史状态
        # 这是一个字典，记录每个数据源的历史信息
        # 格式: {website名称: {'r': 可靠性值, 'e': 误差值, 'k': 一致性值}}
        # 例如: {'website_a': {'r': 0.95, 'e': 0.01, 'k': 0.99}}
        #      - r (reliability): 可靠性，值越大越可信（0-1之间）
        #      - e (error): 误差度量，值越小越好（表示数据与真值的偏差）
        #      - k (consistency): 一致性度量，值越大越好（表示数据源行为是否稳定）
        self.source_reliability_history: Dict[str, Dict[str, float]] = {}

        # 全局真值历史（用于增量处理）
        # 记录每个epoch（批次）计算的全局真值，用于增量处理时参考
        # 格式: {epoch编号: {object名称: 真值}}
        # 注意：真值可能是float（数值类型）或str（离散类型）
        # 例如: {0: {'stock_A': 100.5}, 1: {'stock_A': 100.6}}  # 数值类型
        # 例如: {0: {'theory_A': 'Einstein'}, 1: {'theory_A': 'Einstein'}}  # 离散类型
        # 注意：epoch编号在代码中使用int，但这里用str作为键（因为可能从字符串键转换）
        self.global_truth_history: Dict[int, Dict[str, Union[float, str]]] = {}

        # 数据类型标识
        # 用于标记当前处理的数据是数值类型还是离散类型
        # 'numeric': 数值类型，使用加权平均和平方误差
        # 'discrete': 离散类型，使用加权投票和0/1匹配误差
        # 'hybrid': 混合类型，从文本中提取数值和离散信息，使用多维度误差
        # None: 尚未检测（在第一次调用train或process_batch时自动检测）
        self.data_type: Optional[str] = None

        # TF-IDF向量化器（用于混合模式的语义相似度计算）
        # 在混合模式下，用于将文本转换为向量，计算余弦相似度
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

    def _initialize_source_reliability(self, dataframe: pd.DataFrame):
        """
        初始化源可靠性历史状态

        这个方法就像是给每个数据源建立"档案"，记录它们的初始状态。
        第一次遇到某个数据源时，给它设置初始值。

        工作原理：
        1. 遍历DataFrame中的所有website
        2. 如果这个website还没有记录，就创建一个新记录
        3. 设置初始值：可靠性0.5，误差0.0，一致性1.0

        Args（参数）:
            dataframe: 包含website列的DataFrame（数据表格）
                      - 这个DataFrame至少要有'website'这一列
                      - 我们会从这一列中提取所有的数据源名称

        举例说明：
        假设DataFrame中有这些website：['website_a', 'website_b', 'website_c']
        调用这个方法后，source_reliability_history会变成：
        {
            'website_a': {'r': 0.5, 'e': 0.0, 'k': 1.0},
            'website_b': {'r': 0.5, 'e': 0.0, 'k': 1.0},
            'website_c': {'r': 0.5, 'e': 0.0, 'k': 1.0}
        }
        """
        # 获取DataFrame中所有不重复的website名称
        # unique()方法返回一个数组，包含所有唯一的website名称
        # 例如：['website_a', 'website_b', 'website_c']
        websites = dataframe['website'].unique()

        # 遍历每个website
        for website in websites:
            # 检查这个website是否已经在历史记录中
            # 如果不在，说明是第一次遇到，需要初始化
            if website not in self.source_reliability_history:
                # 为这个website创建初始记录
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability,  # r = reliability（可靠性），初始值通常是0.5
                    'e': 0.0,  # e = error（误差），初始值0.0表示还没有误差数据
                    'k': 1.0   # k = consistency（一致性），初始值1.0表示完全一致（满分）
                }
                # 注意：第一次时，我们假设所有源都是可信的（r=0.5中等），误差为0，一致性为1（完美）

    def _detect_data_type(self, fact_series: pd.Series) -> str:
        """
        检测fact列的数据类型

        这个方法用于自动识别输入数据是数值类型还是离散类型（字符串）。
        检测逻辑：
        1. 尝试将所有值转换为数值
        2. 如果所有值都能成功转换，且没有NaN，则判定为数值类型
        3. 否则判定为离散类型

        为什么需要这个检测？
        - 数值类型和离散类型的真值计算方式不同
        - 数值类型：使用加权平均（可以计算平均值）
        - 离散类型：使用加权投票（不能计算平均值，只能选择最可能的值）

        Args（参数）:
            fact_series: fact列的Series（pandas的一列数据）
                        - 例如：pd.Series([100.5, 101.0, 99.8])
                        - 或者：pd.Series(['Einstein', 'Newton', 'Einstein'])

        Returns（返回值）:
            'numeric' 或 'discrete'
            - 'numeric': 所有值都能转换为数值，判定为数值类型
            - 'discrete': 包含无法转换为数值的值，判定为离散类型

        举例说明：
        1. 数值类型检测：
           fact_series = pd.Series([100.5, 101.0, 99.8])
           尝试转换：全部成功，没有NaN
           → 返回 'numeric'

        2. 离散类型检测：
           fact_series = pd.Series(['Einstein', 'Newton', 'Einstein'])
           尝试转换：'Einstein'无法转换为数值，产生NaN
           → 返回 'discrete'

        3. 边界情况（混合数据）：
           fact_series = pd.Series([100.5, 'Einstein', 99.8])
           尝试转换：'Einstein'无法转换，产生NaN
           → 返回 'discrete'（按离散类型处理）
        """
        # ========== 如果启用了混合模式，直接返回'hybrid' ==========
        if self.enable_hybrid_mode:
            # 混合模式：从文本中提取数值和离散信息
            # 检查fact列是否包含文本（字符串类型）
            if fact_series.dtype == 'object' or fact_series.dtype == 'string':
                # 检查是否包含较长的文本（通常混合模式用于处理完整文本）
                sample_text = str(fact_series.iloc[0]) if len(fact_series) > 0 else ''
                if len(sample_text) > 20:  # 如果文本长度超过20，认为是混合模式
                    return 'hybrid'

            # 如果文本太短，回退到普通检测
            # 继续下面的检测逻辑

        # ========== 普通数据类型检测 ==========
        # 尝试将所有值转换为数值
        # pd.to_numeric会自动尝试转换，如果失败就用NaN代替
        # errors='coerce'表示转换失败时不报错，而是用NaN代替
        numeric_values = pd.to_numeric(fact_series, errors='coerce')

        # 检查是否有NaN值
        # 如果所有值都能成功转换为数值，isna().any()会返回False
        # 如果有任何值无法转换（产生了NaN），isna().any()会返回True
        has_nan = numeric_values.isna().any()

        # 判断数据类型
        if not has_nan:
            # 没有NaN，说明所有值都能转换为数值
            # 判定为数值类型
            return 'numeric'
        else:
            # 有NaN，说明至少有一个值无法转换为数值
            # 判定为离散类型
            return 'discrete'

    def _extract_structured_info(self, text: str) -> Dict[str, Union[Dict[str, float], Dict[str, str]]]:
        """
        从文本中提取结构化信息（数值和离散信息）

        【老奶奶版解释】
        这个函数就像是一个"信息整理员"，它的工作是把一篇长文章（文本）整理成两个小盒子：
        - 第一个盒子装"数字"（比如价格、百分比）
        - 第二个盒子装"文字"（比如公司名、股票代码）

        想象一下：
        您收到一封信，信里写了很多内容，有数字、有文字，乱七八糟的。
        这个函数就像是一个细心的助手，帮您把信里的内容分类整理好：
        - 把所有带"$"符号的数字（比如$125）放到"数字盒子"里
        - 把所有公司名（比如"Agilent Technologies"）放到"文字盒子"里
        - 把所有股票代码（比如"A"）也放到"文字盒子"里

        这样整理好后，后面计算的时候就能方便地找到需要的信息了。

        【技术说明】
        这个方法使用正则表达式和简单的文本处理，从文本中提取：
        1. 数值信息：价格、百分比、数字等
        2. 离散信息：公司名、股票代码、关键词等

        提取策略：
        - 数值：使用正则表达式匹配货币符号、百分比、数字
        - 离散：提取公司名（大写字母+单词）、股票代码（Symbol: XXX）、关键词

        Args（参数）:
            text: 输入文本
                  - 例如："A new Agilent Technologies, Inc. (Symbol: A) option chain has opened with a $125 put priced at $4.50..."

        Returns（返回值）:
            包含数值和离散信息的字典：
            {
                'numeric': {
                    'put_price': 125.0,
                    'put_premium': 4.50,
                    'call_price': 150.0,
                    'call_premium': 8.10,
                    ...
                },
                'discrete': {
                    'company': 'Agilent Technologies',
                    'symbol': 'A',
                    'keywords': ['option', 'put', 'call', ...]
                }
            }
        """
        # ========== 第一步：准备两个空盒子 ==========
        # 【老奶奶版解释】
        # 就像您要整理东西，先准备两个空盒子：
        # - 一个盒子叫"numeric"（数字盒子），用来装所有数字
        # - 一个盒子叫"discrete"（文字盒子），用来装所有文字信息
        # 这两个盒子现在都是空的（{}表示空字典），等会儿我们会往里面放东西
        result = {
            'numeric': {},  # 数字盒子，现在是空的
            'discrete': {}  # 文字盒子，现在是空的
        }

        # ========== 第二步：从文本中找数字 ==========
        # 【老奶奶版解释】
        # 这一步就像是在一篇文章里找所有带"$"符号的钱数
        # 比如文章里写"$125"和"$4.50"，我们就把125和4.50找出来

        # 提取货币金额（$XXX.XX格式）
        # 【详细解释】
        # currency_pattern 就像是一个"模板"，告诉计算机要找什么样的数字
        # r'\$(\d+(?:\.\d+)?)' 这个模板的意思是：
        #   - \$ 表示要找"$"符号
        #   - \d+ 表示后面跟着一个或多个数字（比如125）
        #   - (?:\.\d+)? 表示可能还有小数点和小数（比如.50）
        #   所以 "$125" 和 "$4.50" 都能被找到
        currency_pattern = r'\$(\d+(?:\.\d+)?)'
        # re.findall 就像是用这个模板在文本里"搜索"，把所有符合模板的内容都找出来
        currency_matches = re.findall(currency_pattern, text)
        # 例如：如果文本是"$125 put priced at $4.50"
        # 那么 currency_matches = ['125', '4.50']
        # 注意：这里只提取数字部分，不包含"$"符号

        # 提取百分比（XX%格式）
        # 【老奶奶版解释】
        # 这一步是在文章里找所有带"%"符号的百分比
        # 比如文章里写"32%"，我们就把32找出来
        # percent_pattern 这个模板的意思是：
        #   - (\d+(?:\.\d+)?) 表示一个数字（可能带小数点）
        #   - % 表示后面跟着"%"符号
        percent_pattern = r'(\d+(?:\.\d+)?)%'
        percent_matches = re.findall(percent_pattern, text)
        # 例如：如果文本是"implied volatility of 32%"
        # 那么 percent_matches = ['32']

        # 提取普通数字（用于天数、数量等）
        # 【老奶奶版解释】
        # 这一步是在文章里找所有写"XX days"或"XX day"的数字
        # 比如文章里写"241 days"，我们就把241找出来
        # number_pattern 这个模板的意思是：
        #   - \b 表示单词边界（确保是完整的数字，不是其他数字的一部分）
        #   - (\d+) 表示一个或多个数字
        #   - \s* 表示可能有空格
        #   - (?:days?|day) 表示后面跟着"days"或"day"
        number_pattern = r'\b(\d+)\s*(?:days?|day)'
        number_matches = re.findall(number_pattern, text, re.IGNORECASE)
        # re.IGNORECASE 表示不区分大小写，所以"Days"、"DAYS"、"days"都能匹配
        # 例如：如果文本是"trading at 241 days until expiry"
        # 那么 number_matches = ['241']

        # 存储提取的数值（按出现顺序）
        # 【老奶奶版解释】
        # 现在我们把找到的所有数字都放到一个临时篮子里（all_numbers）
        # 这个篮子就像一个购物篮，我们把找到的数字都放进去
        all_numbers = []
        # 把找到的货币金额（比如125、4.50）放到篮子里
        for match in currency_matches:
            # float(match) 是把字符串"125"转换成数字125.0
            # 就像把"一百二十五"这个文字转换成数字125一样
            all_numbers.append(float(match))
        # 把找到的百分比（比如32）放到篮子里
        for match in percent_matches:
            all_numbers.append(float(match))
        # 把找到的天数（比如241）放到篮子里
        for match in number_matches:
            all_numbers.append(float(match))
        # 现在 all_numbers 这个篮子里可能有：[125.0, 4.5, 32.0, 241.0]

        # ========== 第三步：理解这些数字是什么意思 ==========
        # 【老奶奶版解释】
        # 现在我们找到了一堆数字，但是不知道它们分别代表什么
        # 比如我们找到了125，但不知道这是"价格"还是"溢价"
        # 所以我们要看看数字周围的文字，来理解这个数字的含义
        # 
        # 就像您看一封信，信里写"苹果5元"，您就知道5是苹果的价格
        # 如果写"苹果5斤"，您就知道5是苹果的重量
        # 我们也是通过看数字周围的文字来理解数字的含义

        # 把文本转换成小写，方便查找
        # 【详细解释】
        # text.lower() 是把所有大写字母变成小写
        # 比如"PUT"变成"put"，"Call"变成"call"
        # 这样做是为了查找时不分大小写，更容易找到关键词
        text_lower = text.lower()

        # 如果文本里同时包含"put"和"call"这两个词
        # 【老奶奶版解释】
        # "put"和"call"是期权交易的专业术语
        # 如果文章里同时提到这两个词，说明文章在讲期权交易
        # 这时候我们就要特别仔细地找"put"和"call"相关的价格和溢价
        if 'put' in text_lower and 'call' in text_lower:
            # 如果有put和call，尝试提取对应的价格和溢价
            # 【老奶奶版解释】
            # 这一步就像是在文章里找"put"和"call"这两个词出现在哪里
            # 然后看看它们附近有没有带"$"的数字
            # 如果"put"附近有"$125"，而且附近还有"price"这个词，我们就知道125是put的价格

            # 找到所有包含"put"的单词在文章中的位置
            # 【详细解释】
            # text.split() 是把文章按空格分成一个个单词，比如：
            #   "A new put option" 变成 ['A', 'new', 'put', 'option']
            # enumerate 是给每个单词编号，比如：
            #   [(0, 'A'), (1, 'new'), (2, 'put'), (3, 'option')]
            # 然后我们找出编号是几的单词包含"put"
            put_indices = [i for i, word in enumerate(text.split()) if 'put' in word.lower()]
            # 例如：如果文章是"A new put option with $125"
            # 那么 put_indices = [2]（因为"put"是第2个单词，从0开始数）

            # 同样地，找到所有包含"call"的单词的位置
            call_indices = [i for i, word in enumerate(text.split()) if 'call' in word.lower()]

            # 提取put相关的数值
            # 【老奶奶版解释】
            # 现在我们知道了"put"这个词在文章的哪个位置
            # 接下来我们要看看"put"附近（前后3个单词）有没有带"$"的数字
            # 如果有，再看看附近有没有"price"或"premium"这样的词
            # 如果有"price"，我们就知道这个数字是价格
            # 如果有"premium"，我们就知道这个数字是溢价
            for idx in put_indices:
                # 把文章分成单词
                words = text.split()
                # 看看"put"前后3个单词的范围
                # max(0, idx-3) 是确保不会越界（不会跑到文章开头之前）
                # min(len(words), idx+3) 是确保不会越界（不会跑到文章结尾之后）
                for i in range(max(0, idx-3), min(len(words), idx+3)):
                    # 如果这个单词包含"$"符号，说明是价格
                    if '$' in words[i] and i < len(currency_matches):
                        # 计算这是第几个"$"符号（因为可能有多个价格）
                        match_idx = text[:text.find(words[i])].count('$') - 1
                        if match_idx >= 0 and match_idx < len(currency_matches):
                            # 看看这个价格附近有没有"price"或"strike"这个词
                            # words[i-2:i+2] 是取前后2个单词的范围
                            if 'price' in words[i-2:i+2] or 'strike' in words[i-2:i+2]:
                                # 如果有"price"，我们就知道这是put的价格
                                # 把它存到数字盒子里，名字叫"put_price_0"（0表示第一个）
                                result['numeric'][f'put_price_{len(result["numeric"])}'] = float(currency_matches[match_idx])
                            elif 'premium' in words[i-2:i+2] or 'bid' in words[i-2:i+2]:
                                # 如果有"premium"，我们就知道这是put的溢价
                                # 把它存到数字盒子里，名字叫"put_premium_0"
                                result['numeric'][f'put_premium_{len(result["numeric"])}'] = float(currency_matches[match_idx])

            # 提取call相关的数值（和put的逻辑完全一样）
            # 【老奶奶版解释】
            # 同样的方法，我们找"call"附近的价格和溢价
            for idx in call_indices:
                words = text.split()
                for i in range(max(0, idx-3), min(len(words), idx+3)):
                    if '$' in words[i] and i < len(currency_matches):
                        match_idx = text[:text.find(words[i])].count('$') - 1
                        if match_idx >= 0 and match_idx < len(currency_matches):
                            if 'price' in words[i-2:i+2] or 'strike' in words[i-2:i+2]:
                                # 这是call的价格
                                result['numeric'][f'call_price_{len(result["numeric"])}'] = float(currency_matches[match_idx])
                            elif 'premium' in words[i-2:i+2] or 'bid' in words[i-2:i+2]:
                                # 这是call的溢价
                                result['numeric'][f'call_premium_{len(result["numeric"])}'] = float(currency_matches[match_idx])

        # 如果没有提取到结构化数值，就存储所有找到的数值
        # 【老奶奶版解释】
        # 如果上面的方法没有成功提取到"put_price"、"call_price"这样的结构化信息
        # 我们就简单地把所有找到的数字都存起来
        # 就像如果分不清哪个是价格、哪个是溢价，就把所有数字都放到盒子里
        # 最多存10个数字，避免盒子装太满
        if not result['numeric']:
            # all_numbers[:10] 表示只取前10个数字
            for i, num in enumerate(all_numbers[:10]):  # 最多存储10个数值
                # 存的时候给它们起名字：value_0, value_1, value_2...
                result['numeric'][f'value_{i}'] = num

        # 提取百分比（implied volatility等）
        # 【老奶奶版解释】
        # 把所有找到的百分比都存到数字盒子里
        # 比如找到"32%"，就存成"percentage_0": 32.0
        # 如果找到多个百分比，就存成"percentage_0", "percentage_1"...
        for i, pct in enumerate(percent_matches):
            result['numeric'][f'percentage_{i}'] = float(pct)

        # ========== 第四步：从文本中找文字信息 ==========
        # 【老奶奶版解释】
        # 现在我们要找文字信息了，比如公司名、股票代码、关键词等
        # 这些信息要放到"文字盒子"（discrete）里

        # 提取公司名（大写字母开头的多个单词，通常包含"Inc", "Technologies"等）
        # 【老奶奶版解释】
        # 公司名通常有几个特点：
        # 1. 第一个字母是大写的（比如"Agilent"）
        # 2. 后面可能跟着"Inc"、"Technologies"、"Corp"等词
        # 3. 比如"Agilent Technologies, Inc."就是一个完整的公司名
        # 
        # company_pattern 这个模板的意思是：
        #   - [A-Z] 表示第一个字母必须是大写
        #   - [a-zA-Z\s]+ 表示后面跟着多个字母或空格
        #   - (?:Inc|Technologies|Corp|...) 表示可能以这些词结尾
        company_pattern = r'([A-Z][a-zA-Z\s]+(?:Inc|Technologies|Corp|Corporation|Ltd|Limited)?)'
        company_matches = re.findall(company_pattern, text)
        if company_matches:
            # 如果找到多个可能的公司名，选择最长的那个
            # 【详细解释】
            # 比如可能找到"Agilent"和"Agilent Technologies, Inc."
            # 我们选择更长的"Agilent Technologies, Inc."，因为它更完整
            # max(..., key=len) 表示选择长度最长的
            # .strip() 是去掉首尾的空格
            result['discrete']['company'] = max(company_matches, key=len).strip()

        # 提取股票代码（Symbol: XXX格式）
        # 【老奶奶版解释】
        # 股票代码通常写在"Symbol: XXX"这样的格式里
        # 比如"Symbol: A"表示股票代码是"A"
        # symbol_pattern 这个模板的意思是：
        #   - Symbol: 表示要找"Symbol:"这个词（后面可能有空格）
        #   - \s* 表示可能有空格
        #   - ([A-Z]+) 表示后面跟着一个或多个大写字母（这就是股票代码）
        symbol_pattern = r'Symbol:\s*([A-Z]+)'
        symbol_match = re.search(symbol_pattern, text)
        if symbol_match:
            # 如果找到了，就把股票代码存起来
            # .group(1) 表示取第一个括号里匹配的内容（就是股票代码部分）
            result['discrete']['symbol'] = symbol_match.group(1)
            # 例如：如果文本是"Symbol: A"，那么 result['discrete']['symbol'] = 'A'

        # 提取关键词（put, call, option, premium等）
        # 【老奶奶版解释】
        # 关键词就是文章里提到的重要词汇
        # 比如如果文章提到"put"、"call"、"option"这些词，说明文章在讲期权交易
        # 我们把找到的关键词都存到一个列表里
        keywords = []
        # 这是我们关心的关键词列表
        key_terms = ['put', 'call', 'option', 'premium', 'strike', 'expiry', 'volatility', 'bid', 'ask']
        # 看看文章里有没有这些词
        for term in key_terms:
            if term.lower() in text_lower:
                # 如果找到了，就加到列表里
                keywords.append(term)
        # 把找到的所有关键词存到文字盒子里
        result['discrete']['keywords'] = keywords
        # 例如：如果文章提到"put"和"call"，那么 keywords = ['put', 'call']

        # 提取期权类型
        # 【老奶奶版解释】
        # 简单检查一下文章里有没有提到"put"或"call"
        # 如果有，就在文字盒子里做个标记
        if 'put' in text_lower:
            # 标记：这篇文章提到了put期权
            result['discrete']['has_put'] = 'true'
        if 'call' in text_lower:
            # 标记：这篇文章提到了call期权
            result['discrete']['has_call'] = 'true'

        # ========== 第五步：返回整理好的两个盒子 ==========
        # 【老奶奶版解释】
        # 现在我们已经把文章里的信息都整理好了：
        # - 数字盒子（numeric）里装着所有数字（价格、百分比等）
        # - 文字盒子（discrete）里装着所有文字信息（公司名、股票代码、关键词等）
        # 现在我们把这两个盒子返回给调用这个函数的人
        # 就像您整理好东西后，把两个盒子交给需要的人一样
        return result

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度（使用TF-IDF + 余弦相似度）

        这个方法使用TF-IDF向量化将文本转换为向量，然后计算余弦相似度。
        相似度范围：0-1，1表示完全相同，0表示完全不同。

        Args（参数）:
            text1: 第一个文本
            text2: 第二个文本

        Returns（返回值）:
            语义相似度得分（0-1之间）
            - 1.0: 完全相似
            - 0.0: 完全不同
        """
        if not text1 or not text2:
            return 0.0

        # 如果TF-IDF向量化器还没有初始化，就创建一个
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                max_features=1000
            )

        try:
            # 将两个文本转换为TF-IDF向量
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])

            # 计算余弦相似度
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            # 确保返回值在[0, 1]范围内
            return max(0.0, min(1.0, similarity))
        except Exception:
            # 如果计算失败（例如文本太短），使用简单的字符串匹配
            # 计算Jaccard相似度作为备选
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

    def _calculate_local_truth(self, df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        计算本地真值估计（按object分组）

        这个方法用于计算某个对象的真值估计。
        但是注意：在当前简化架构中，这个方法的逻辑和全局真值计算是一样的。
        因为在我们的架构中，website直接对应数据源，没有"节点"和"源"的区分。

        根据数据类型，使用不同的计算方式：
        1. 数值类型：使用加权平均（公式3的思想）
        真值 = Σ(源可靠性 × 源输入值) / Σ(源可靠性)
        2. 离散类型：使用加权投票
           真值 = argmax(Σ(源可靠性))，其中fact值相同

        通俗理解（数值类型）：
        - 如果website_a的可靠性是0.9，它提供的值是100.5
        - 如果website_b的可靠性是0.8，它提供的值是101.0
        - 那么真值 = (0.9×100.5 + 0.8×101.0) / (0.9+0.8) = (90.45 + 80.8) / 1.7 ≈ 100.7
        - 可以看到，可靠性高的源对最终结果影响更大

        通俗理解（离散类型）：
        - 如果website_a的可靠性是0.9，它提供的值是"Einstein"
        - 如果website_b的可靠性是0.8，它提供的值是"Einstein"
        - 如果website_c的可靠性是0.5，它提供的值是"Newton"
        - 那么真值 = "Einstein"（因为"Einstein"的权重和最大：0.9+0.8=1.7 > 0.5）

        Args（参数）:
            df: 单个object的数据子集
                - 这个DataFrame只包含一个object的所有数据
                - 例如：只包含'stock_A'这个对象的所有行的数据

        Returns（返回值）:
            {object: estimated_truth}  # 返回一个字典，包含对象名和估计的真值
            # 数值类型示例：{'stock_A': 100.5}
            # 离散类型示例：{'theory_A': 'Einstein'}
            # 混合类型示例：{'option_A': {'numeric': {...}, 'discrete': {...}}}
        """
        # 如果DataFrame是空的，直接返回空字典
        if df.empty:
            return {}

        # 获取这个object的名称（因为df只包含一个object，所以取第一个就行）
        object_id = df['object'].iloc[0]
        # iloc[0]表示第一行，['object']表示object这一列
        # 例如：如果df的第一行object列是'stock_A'，那么object_id = 'stock_A'

        # ========== 根据数据类型选择计算方式 ==========
        if self.data_type == 'hybrid':
            # ========== 混合类型：从文本中提取信息，分别计算数值和离散真值 ==========
            # 混合模式的思想：
            # 1. 从每个源的文本中提取结构化信息（数值+离散）
            # 2. 对数值部分使用加权平均
            # 3. 对离散部分使用加权投票或语义相似度
            # 4. 组合成结构化真值

            # 收集所有源提取的信息
            extracted_infos = []  # 存储每个源提取的信息和可靠性
            for _, row in df.iterrows():
                website = row['website']
                fact_text = str(row['fact'])
                r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)

                # 提取结构化信息
                info = self._extract_structured_info(fact_text)
                extracted_infos.append({
                    'info': info,
                    'weight': r,
                    'text': fact_text
                })

            # 计算数值真值（对每个数值字段使用加权平均）
            numeric_truth = {}
            for key in set().union(*[item['info']['numeric'].keys() for item in extracted_infos]):
                # 对每个数值字段，计算加权平均
                weighted_sum = 0.0
                total_weight = 0.0
                for item in extracted_infos:
                    if key in item['info']['numeric']:
                        value = item['info']['numeric'][key]
                        weight = item['weight']
                        weighted_sum += weight * value
                        total_weight += weight

                if total_weight > 0:
                    numeric_truth[key] = weighted_sum / total_weight

            # 计算离散真值（使用加权投票，选择权重和最大的值）
            discrete_truth = {}
            for key in set().union(*[item['info']['discrete'].keys() for item in extracted_infos]):
                # 对每个离散字段，使用加权投票
                candidate_weights = {}
                for item in extracted_infos:
                    if key in item['info']['discrete']:
                        value = str(item['info']['discrete'][key])
                        weight = item['weight']
                        if value not in candidate_weights:
                            candidate_weights[value] = 0.0
                        candidate_weights[value] += weight

                if candidate_weights:
                    discrete_truth[key] = max(candidate_weights, key=candidate_weights.get)

            # 组合成结构化真值
            truth_value = {
                'numeric': numeric_truth,
                'discrete': discrete_truth
            }

        elif self.data_type == 'numeric':
            # ========== 数值类型：使用加权平均 ==========
            # 加权平均 = 所有(权重×值)的总和 / 所有权重的总和
            weighted_sum = 0.0    # 累加：可靠性 × 输入值
            total_weight = 0.0    # 累加：可靠性（权重总和）

            # 遍历这个object的所有数据行
            for _, row in df.iterrows():
                # row是一行数据，例如：website='website_a', fact=100.5, object='stock_A'
                website = row['website']  # 获取数据源名称，例如：'website_a'
                fact_value = float(row['fact'])  # 获取输入值并转为浮点数，例如：100.5

                # 获取这个数据源的可靠性
                # get方法：如果website在历史记录中，就取'r'值；否则用初始可靠性
                # 例如：如果'website_a'的可靠性是0.95，那么r = 0.95
                r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)

                # 累加：权重×值
                weighted_sum += r * fact_value
                # 例如：如果r=0.95, fact_value=100.5，那么weighted_sum += 0.95 * 100.5 = 95.475

                # 累加：权重
                total_weight += r
                # 例如：如果r=0.95，那么total_weight += 0.95

            # 计算最终的真值估计
            if total_weight > 0:
                # 加权平均公式：总和 / 权重总和
                truth_value = weighted_sum / total_weight
                # 例如：如果weighted_sum=95.475, total_weight=0.95，那么truth_value = 95.475 / 0.95 = 100.5
            else:
                # 如果没有权重（所有源的可靠性都是0），就用简单平均
                # 这通常不会发生，但为了安全起见，保留这个备选方案
                truth_value = float(df['fact'].mean())
                # mean()方法计算平均值，例如：[100.5, 101.0, 99.8]的平均值是100.43

        else:
            # ========== 离散类型：使用加权投票 ==========
            # 加权投票的思想：对于每个候选值，累加所有提供该值的源的可靠性
            # 然后选择权重和最大的值作为真值

            # 这个字典用于记录每个候选值的权重和
            # 格式：{候选值: 权重和}
            # 例如：{'Einstein': 1.7, 'Newton': 0.5}
            # 表示：提供'Einstein'的源的可靠性总和是1.7，提供'Newton'的源的可靠性总和是0.5
            candidate_weights: Dict[str, float] = {}

            # 遍历这个object的所有数据行
            for _, row in df.iterrows():
                # row是一行数据，例如：website='website_a', fact='Einstein', object='theory_A'
                website = row['website']  # 获取数据源名称，例如：'website_a'
                fact_value = str(row['fact'])  # 获取输入值并转为字符串，例如：'Einstein'

                # 获取这个数据源的可靠性
                r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)
                # 例如：如果'website_a'的可靠性是0.9，那么r = 0.9

                # 累加这个候选值的权重
                # 如果这个候选值还没有记录，就初始化为0，然后加上当前源的可靠性
                if fact_value not in candidate_weights:
                    candidate_weights[fact_value] = 0.0
                candidate_weights[fact_value] += r
                # 例如：如果fact_value='Einstein', r=0.9
                # 那么：candidate_weights['Einstein'] += 0.9
                # 如果之前已经有0.8，现在变成1.7

            # 选择权重和最大的候选值作为真值
            if candidate_weights:
                # max函数可以找到字典中值最大的键
                # key参数指定比较的依据：按照字典的值（权重和）进行比较
                truth_value = max(candidate_weights, key=candidate_weights.get)
                # 例如：如果candidate_weights = {'Einstein': 1.7, 'Newton': 0.5}
                # 那么：truth_value = 'Einstein'（因为1.7 > 0.5）
            else:
                # 如果没有候选值（这通常不会发生），使用众数（出现次数最多的值）
                truth_value = str(df['fact'].mode().iloc[0] if not df['fact'].mode().empty else df['fact'].iloc[0])
                # mode()方法返回出现次数最多的值，例如：['Einstein', 'Einstein', 'Newton']的众数是'Einstein'

        # 返回结果：一个字典，键是object名称，值是估计的真值
        return {object_id: truth_value}
        # 数值类型示例：返回 {'stock_A': 100.5}
        # 离散类型示例：返回 {'theory_A': 'Einstein'}

    def _update_source_reliability(self,
                                  dataframe: pd.DataFrame,
                                  global_truth: Dict[str, Union[float, str]]):
        """
        更新源可靠性度量

        这是算法的核心方法之一！它根据当前的真值估计，更新每个数据源的可靠性。
        这个过程包括三个步骤，对应论文的公式(4), (5), (6)：

        步骤1：计算误差度量 e_s（公式4）
        - 计算这个源提供的值与真值的偏差（均方误差）
        - 使用加权移动平均，结合历史误差和当前误差

        步骤2：计算一致性度量 k_s（公式5）
        - 检测源的行为是否突然改变（拜占庭行为检测）
        - 如果误差突然大幅变化，一致性会降低

        步骤3：更新可靠性度量 r_s（公式6）
        - 综合误差和一致性，计算新的可靠性
        - 误差小且一致性好 → 可靠性高

        通俗理解：
        假设website_a之前一直提供准确数据，可靠性是0.95
        现在它突然提供了一个明显错误的数据
        - 误差e会增大（因为与真值偏差大）
        - 一致性k会降低（因为行为突然改变了）
        - 可靠性r会降低（因为误差大且一致性差）
        这样，算法就能自动"惩罚"提供错误数据的源

        Args（参数）:
            dataframe: 完整数据DataFrame
                      - 包含所有website、fact、object的数据
            global_truth: 全局真值估计
                         - 格式：{object名称: 真值}
                         - 数值类型示例：{'stock_A': 100.5, 'stock_B': 50.2}
                         - 离散类型示例：{'theory_A': 'Einstein', 'theory_B': 'Newton'}
                         - 这是上一轮计算出的真值估计
                         - 注意：真值可能是float（数值类型）或str（离散类型）

        Returns（返回值）:
            无（void），这个方法直接修改self.source_reliability_history
        """
        # 按website分组，对每个数据源分别更新可靠性
        # unique()获取所有唯一的website名称
        for website in dataframe['website'].unique():
            # 筛选出这个website的所有数据行
            # 例如：如果website='website_a'，source_data就是website_a的所有行
            source_data = dataframe[dataframe['website'] == website]

            # 如果这个website还没有历史记录，先初始化
            if website not in self.source_reliability_history:
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability,
                    'e': 0.0,
                    'k': 1.0
                }

            # ========== 步骤1：计算误差度量 e_s（公式4）==========
            # 收集该源参与的所有对象及其输入值与真值的误差
            # 根据数据类型，使用不同的误差计算方式：
            # - 数值类型：使用平方误差 (input_value - truth_value)^2
            # - 离散类型：使用0/1匹配误差（匹配=0，不匹配=1）

            errors = []  # 存储每个对象的误差
            for _, row in source_data.iterrows():
                # 遍历这个源提供的所有数据
                object_id = row['object']      # 对象名称，例如：'stock_A' 或 'theory_A'
                input_value = row['fact']      # 源提供的值，可能是数值或字符串

                # 如果全局真值中有这个对象的估计值
                if object_id in global_truth:
                    truth_value = global_truth[object_id]  # 真值估计，可能是数值或字符串

                    # ========== 根据数据类型计算误差 ==========
                    if self.data_type == 'hybrid':
                        # ========== 混合类型：多维度误差计算 ==========
                        # 混合模式的思想：
                        # 1. 从输入文本中提取结构化信息（数值+离散）
                        # 2. 分别计算数值误差和离散误差
                        # 3. 使用加权组合得到综合误差
                        # 4. 综合误差 = α × 数值误差 + (1-α) × 离散误差

                        input_text = str(input_value)  # 源提供的文本
                        truth_dict = truth_value  # 真值字典，格式：{'numeric': {...}, 'discrete': {...}}

                        # 从输入文本中提取结构化信息
                        extracted_info = self._extract_structured_info(input_text)

                        # ========== 计算数值误差 ==========
                        numeric_errors = []
                        if 'numeric' in truth_dict and truth_dict['numeric']:
                            # 对每个数值字段，计算平方误差
                            for key, truth_val in truth_dict['numeric'].items():
                                if key in extracted_info['numeric']:
                                    extracted_val = extracted_info['numeric'][key]
                                    # 计算归一化的平方误差（避免不同量级的数值影响）
                                    # 归一化：使用相对误差或除以真值
                                    if abs(truth_val) > 1e-6:  # 避免除零
                                        relative_error = abs(extracted_val - truth_val) / abs(truth_val)
                                        numeric_errors.append(relative_error ** 2)
                                    else:
                                        # 如果真值接近0，使用绝对误差
                                        numeric_errors.append((extracted_val - truth_val) ** 2)

                        # 计算平均数值误差
                        numeric_error = np.mean(numeric_errors) if numeric_errors else 1.0
                        # 如果没有提取到数值或没有匹配的字段，认为误差最大（1.0）

                        # ========== 计算离散误差（使用语义相似度）==========
                        discrete_errors = []
                        if 'discrete' in truth_dict and truth_dict['discrete']:
                            # 方法1：对每个离散字段，计算匹配度
                            for key, truth_val in truth_dict['discrete'].items():
                                if key in extracted_info['discrete']:
                                    extracted_val = str(extracted_info['discrete'][key])
                                    truth_val_str = str(truth_val)

                                    if extracted_val == truth_val_str:
                                        # 完全匹配，误差为0
                                        discrete_errors.append(0.0)
                                    else:
                                        # 不完全匹配，使用语义相似度计算误差
                                        # 误差 = 1 - 相似度
                                        similarity = self._calculate_semantic_similarity(extracted_val, truth_val_str)
                                        discrete_errors.append(1.0 - similarity)

                            # 方法2：整体文本的语义相似度
                            # 构建真值文本摘要（用于整体相似度计算）
                            truth_text_parts = []
                            if 'company' in truth_dict['discrete']:
                                truth_text_parts.append(truth_dict['discrete']['company'])
                            if 'symbol' in truth_dict['discrete']:
                                truth_text_parts.append(truth_dict['discrete']['symbol'])
                            if 'keywords' in truth_dict['discrete']:
                                if isinstance(truth_dict['discrete']['keywords'], list):
                                    truth_text_parts.extend(truth_dict['discrete']['keywords'])

                            truth_text_summary = ' '.join(truth_text_parts)

                            # 计算整体文本相似度
                            if truth_text_summary:
                                overall_similarity = self._calculate_semantic_similarity(input_text, truth_text_summary)
                                # 将整体相似度也加入离散误差（权重较低）
                                discrete_errors.append(1.0 - overall_similarity)

                        # 计算平均离散误差
                        discrete_error = np.mean(discrete_errors) if discrete_errors else 1.0
                        # 如果没有提取到离散信息或没有匹配的字段，认为误差最大（1.0）

                        # ========== 加权组合综合误差 ==========
                        # 综合误差 = α × 数值误差 + (1-α) × 离散误差
                        # α是数值误差的权重（self.numeric_weight）
                        combined_error = self.numeric_weight * numeric_error + (1 - self.numeric_weight) * discrete_error

                        error = combined_error
                        # 例如：如果numeric_weight=0.5, numeric_error=0.1, discrete_error=0.2
                        # 那么：combined_error = 0.5×0.1 + 0.5×0.2 = 0.15

                    elif self.data_type == 'numeric':
                        # ========== 数值类型：使用平方误差 ==========
                    # 平方误差的优点：无论偏差是正还是负，都是正数，且大的偏差会被放大
                        input_float = float(input_value)  # 转换为浮点数，例如：100.5
                        truth_float = float(truth_value)  # 转换为浮点数，例如：100.2
                        error = (input_float - truth_float) ** 2
                        # 例如：如果input_float=100.5, truth_float=100.2，那么error = (0.3)^2 = 0.09
                        # 平方误差的特性：
                        # - 当输入值等于真值时，误差为0（完美匹配）
                        # - 当输入值偏离真值时，误差增大，且偏离越大，误差增长越快（平方效应）
                        # - 例如：偏离1单位，误差=1；偏离2单位，误差=4；偏离3单位，误差=9

                    else:
                        # ========== 离散类型：使用0/1匹配误差 ==========
                        # 0/1匹配误差的思想：
                        # - 如果输入值等于真值（完全匹配），误差为0
                        # - 如果输入值不等于真值（不匹配），误差为1
                        # 这样，误差就表示"不匹配率"，值越大表示越不可靠
                        input_str = str(input_value)  # 转换为字符串，例如：'Einstein'
                        truth_str = str(truth_value)   # 转换为字符串，例如：'Einstein'

                        if input_str == truth_str:
                            # 完全匹配，误差为0
                            error = 0.0
                            # 例如：input_str='Einstein', truth_str='Einstein' → error = 0.0
                        else:
                            # 不匹配，误差为1
                            error = 1.0
                            # 例如：input_str='Newton', truth_str='Einstein' → error = 1.0
                            # 注意：这里使用简单的0/1误差，不考虑字符串相似度
                            # 未来可以扩展为使用编辑距离等更复杂的度量

                    errors.append(error)  # 保存这个误差

            # 如果没有误差数据（这个源没有提供任何有效输入），跳过更新
            if not errors:
                continue

            # 计算平均误差（MSE - Mean Squared Error 或 MAE - Mean Absolute Error）
            # 对于数值类型：MSE = 所有平方误差的平均值
            # 对于离散类型：MAE = 所有0/1误差的平均值（实际上就是不匹配率）
            mse = np.mean(errors)
            # 数值类型示例：如果errors = [0.01, 0.04, 0.09]，那么mse = (0.01+0.04+0.09)/3 = 0.047
            # 离散类型示例：如果errors = [0.0, 0.0, 1.0]，那么mse = (0.0+0.0+1.0)/3 = 0.333
            # 离散类型的mse实际上表示"不匹配率"：0.333表示33.3%的数据不匹配

            # 获取历史值（上一轮的误差和一致性）
            prev_e = self.source_reliability_history[website]['e']  # 上一轮的误差，例如：0.05
            prev_k = self.source_reliability_history[website]['k']  # 上一轮的一致性，例如：0.98

            # ========== 更新误差度量（加权移动平均）==========
            # 公式(4)的思想：e_s = α × e_s_prev + (1-α) × MSE
            # 加权移动平均的好处：
            # - 如果α=0.9，那么90%的权重给历史，10%给当前
            # - 这样可以平滑短期波动，关注长期趋势
            # - 如果数据源偶尔犯错，不会立即被判定为不可信
            new_e = self.alpha * prev_e + (1 - self.alpha) * mse
            # 例如：如果alpha=0.9, prev_e=0.05, mse=0.1
            # 那么：new_e = 0.9×0.05 + 0.1×0.1 = 0.045 + 0.01 = 0.055

            # ========== 步骤2：计算一致性度量 k_s（公式5）==========
            # 一致性度量的作用：检测拜占庭行为（突然的大幅变化）

            # 计算误差的变化量：|新误差 - 旧误差|
            error_change = abs(new_e - prev_e)
            # 例如：如果new_e=0.055, prev_e=0.05，那么error_change = |0.055-0.05| = 0.005

            # 使用互补误差函数erfc计算一致性
            # erfc是erf的补函数，取值范围是(0, 2)
            # 公式(5)：k_s = erfc(β × |e_s - e_s_prev|)
            # erfc函数的特性：
            # - 当输入=0时，erfc(0) = 1（完全一致）
            # - 当输入增大时，erfc值减小（一致性变差）
            # - β是尺度因子，控制敏感度：β越大，对变化越敏感
            new_k = erfc(self.beta * error_change)
            # 例如：如果beta=100.0, error_change=0.005
            # 那么：new_k = erfc(100.0 × 0.005) = erfc(0.5) ≈ 0.48
            # 如果error_change很大（比如0.1），那么new_k会很小（接近0），表示行为突然改变

            # ========== 更新一致性度量（加权移动平均）==========
            # 同样使用加权移动平均，平滑一致性度量
            k_updated = self.alpha * prev_k + (1 - self.alpha) * new_k
            # 例如：如果alpha=0.9, prev_k=0.98, new_k=0.48
            # 那么：k_updated = 0.9×0.98 + 0.1×0.48 = 0.882 + 0.048 = 0.93

            # ========== 步骤3：更新可靠性度量 r_s（公式6）==========
            # 可靠性 = f(误差, 一致性)
            # 误差越小，可靠性越高；一致性越好，可靠性越高

            # 计算误差分量：误差越小，这个分量越大
            # 使用 1/(1+误差) 的形式，确保：
            # - 当误差=0时，分量=1（完美）
            # - 当误差增大时，分量减小
            # - 加上1避免除零错误
            if new_e > 0:
                error_component = 1.0 / (1.0 + new_e)
            else:
                error_component = 1.0  # 误差为0，完美
            # 例如：如果new_e=0.055，那么error_component = 1/(1+0.055) = 1/1.055 ≈ 0.948
            # 如果new_e=1.0（误差很大），那么error_component = 1/(1+1.0) = 0.5

            # 计算最终的可靠性：误差分量 × 一致性度量
            new_r = error_component * k_updated
            # 例如：如果error_component=0.948, k_updated=0.93
            # 那么：new_r = 0.948 × 0.93 ≈ 0.882
            # 这个值在0-1之间，越大表示越可靠

            # ========== 更新历史状态 ==========
            # 保存新计算的值，供下一轮使用
            self.source_reliability_history[website]['e'] = new_e      # 更新误差
            self.source_reliability_history[website]['k'] = k_updated  # 更新一致性
            self.source_reliability_history[website]['r'] = new_r      # 更新可靠性

    def _calculate_global_truth(self, dataframe: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        计算全局真值估计

        这个方法计算每个对象的全局真值估计。
        根据DECENTRUTH论文公式(9)的思想，使用基于源可靠性的加权平均。

        公式(9)的简化版本：
        x̂_i = (Σ_{n∈CS} w_n × x̃_n) / (Σ_{n∈CS} w_n)

        在我们的简化架构中：
        - 没有"节点提案"的概念，website直接对应数据源
        - 使用源可靠性 r_s 作为权重 w_n
        - 所以公式简化为：x̂_i = (Σ r_s × x_{s,i}) / (Σ r_s)

        通俗理解：
        对于同一个对象（比如stock_A），多个源提供了不同的值：
        - website_a提供100.5，可靠性0.95
        - website_b提供101.0，可靠性0.90
        - website_c提供99.8，可靠性0.70（可能不太可靠）

        全局真值 = (0.95×100.5 + 0.90×101.0 + 0.70×99.8) / (0.95+0.90+0.70)
                 = (95.475 + 90.9 + 69.86) / 2.55
                 = 256.235 / 2.55
                 ≈ 100.5

        可以看到，可靠性高的源（website_a）对结果影响更大。

        Args（参数）:
            dataframe: 完整数据DataFrame
                      - 包含所有website、fact、object的数据

        Returns（返回值）:
            {object: truth_value}  # 返回每个对象的真值估计
            # 数值类型示例：{'stock_A': 100.5, 'stock_B': 50.2}
            # 离散类型示例：{'theory_A': 'Einstein', 'theory_B': 'Newton'}
            # 混合类型示例：{'option_A': {'numeric': {...}, 'discrete': {...}}}
        """
        global_truth = {}  # 存储结果：{对象名: 真值}

        # 按object分组计算，每个对象分别计算真值
        for object_id in dataframe['object'].unique():
            # 筛选出这个对象的所有数据
            # 例如：如果object_id='stock_A'，obj_data就是所有stock_A的数据行
            obj_data = dataframe[dataframe['object'] == object_id]

            # ========== 根据数据类型选择计算方式 ==========
            if self.data_type == 'hybrid':
                # ========== 混合类型：从文本中提取信息，分别计算数值和离散真值 ==========
                # 收集所有源提取的信息
                extracted_infos = []
                for _, row in obj_data.iterrows():
                    website = row['website']
                    fact_text = str(row['fact'])
                    r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)

                    # 提取结构化信息
                    info = self._extract_structured_info(fact_text)
                    extracted_infos.append({
                        'info': info,
                        'weight': r
                    })

                # 计算数值真值（对每个数值字段使用加权平均）
                numeric_truth = {}
                for key in set().union(*[item['info']['numeric'].keys() for item in extracted_infos]):
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for item in extracted_infos:
                        if key in item['info']['numeric']:
                            value = item['info']['numeric'][key]
                            weight = item['weight']
                            weighted_sum += weight * value
                            total_weight += weight

                    if total_weight > 0:
                        numeric_truth[key] = weighted_sum / total_weight

                # 计算离散真值（使用加权投票）
                discrete_truth = {}
                for key in set().union(*[item['info']['discrete'].keys() for item in extracted_infos]):
                    candidate_weights = {}
                    for item in extracted_infos:
                        if key in item['info']['discrete']:
                            value = str(item['info']['discrete'][key])
                            weight = item['weight']
                            if value not in candidate_weights:
                                candidate_weights[value] = 0.0
                            candidate_weights[value] += weight

                    if candidate_weights:
                        discrete_truth[key] = max(candidate_weights, key=candidate_weights.get)

                # 组合成结构化真值
                global_truth[object_id] = {
                    'numeric': numeric_truth,
                    'discrete': discrete_truth
                }

            elif self.data_type == 'numeric':
                # ========== 数值类型：使用加权平均 ==========
                weighted_sum = 0.0    # 累加：可靠性 × 输入值
                total_weight = 0.0    # 累加：可靠性（权重总和）

                # 遍历这个对象的所有数据行
                for _, row in obj_data.iterrows():
                    website = row['website']      # 数据源名称
                    fact_value = float(row['fact'])  # 输入值，转换为浮点数

                    # 获取这个数据源的可靠性作为权重
                    # 可靠性越高，权重越大
                    r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)
                    # 例如：如果website='website_a'，它的可靠性是0.95，那么r=0.95

                    # 累加：权重 × 值
                    weighted_sum += r * fact_value
                    # 例如：如果r=0.95, fact_value=100.5，那么weighted_sum += 95.475

                    # 累加：权重
                    total_weight += r
                    # 例如：total_weight += 0.95

                # 计算最终的真值估计
                if total_weight > 0:
                    # 加权平均公式：总和 / 权重总和
                    global_truth[object_id] = weighted_sum / total_weight
                    # 例如：如果weighted_sum=256.235, total_weight=2.55
                    # 那么：global_truth['stock_A'] = 256.235 / 2.55 ≈ 100.5
                else:
                    # 如果所有权重都是0（所有源都不可靠），使用简单平均
                    # 这通常不会发生，但为了安全起见保留
                    global_truth[object_id] = float(obj_data['fact'].mean())

            else:
                # ========== 离散类型：使用加权投票 ==========
                # 加权投票的思想：对于每个候选值，累加所有提供该值的源的可靠性
                # 然后选择权重和最大的值作为真值

                # 这个字典用于记录每个候选值的权重和
                # 格式：{候选值: 权重和}
                candidate_weights: Dict[str, float] = {}

                # 遍历这个对象的所有数据行
                for _, row in obj_data.iterrows():
                    website = row['website']      # 数据源名称
                    fact_value = str(row['fact'])  # 输入值，转换为字符串

                    # 获取这个数据源的可靠性
                    r = self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)

                    # 累加这个候选值的权重
                    if fact_value not in candidate_weights:
                        candidate_weights[fact_value] = 0.0
                    candidate_weights[fact_value] += r
                    # 例如：如果fact_value='Einstein', r=0.9
                    # 那么：candidate_weights['Einstein'] += 0.9

                # 选择权重和最大的候选值作为真值
                if candidate_weights:
                    global_truth[object_id] = max(candidate_weights, key=candidate_weights.get)
                    # 例如：如果candidate_weights = {'Einstein': 1.7, 'Newton': 0.5}
                    # 那么：global_truth['theory_A'] = 'Einstein'（因为1.7 > 0.5）
                else:
                    # 如果没有候选值（这通常不会发生），使用众数
                    global_truth[object_id] = str(obj_data['fact'].mode().iloc[0] if not obj_data['fact'].mode().empty else obj_data['fact'].iloc[0])

        # 返回所有对象的真值估计
        return global_truth
        # 数值类型示例：返回 {'stock_A': 100.5, 'stock_B': 50.2}
        # 离散类型示例：返回 {'theory_A': 'Einstein', 'theory_B': 'Newton'}

    def _update_dataframe(self, dataframe: pd.DataFrame, global_truth: Dict[str, float]):
        """
        更新DataFrame，添加新列

        这个方法不会改变原有的数据，只是在DataFrame中添加新的列，
        这些列包含计算出的可靠性度量和真值估计。

        添加的列：
        1. source_reliability（必须）：每个源的可靠性 r_s
        2. error_metric（可选）：每个源的误差度量 e_s
        3. consistency_metric（可选）：每个源的一致性度量 k_s
        4. global_truth（参考）：每个对象的全局真值估计

        这些新列可以让我们方便地查看结果，了解：
        - 哪些源更可靠
        - 哪些源的误差大
        - 哪些源的行为不一致（可能有问题）
        - 每个对象的真值估计是多少

        Args（参数）:
            dataframe: 原始DataFrame（会被修改，添加新列）
            global_truth: 全局真值估计 {object: truth_value}
                        - 数值类型示例：{'stock_A': 100.5, 'stock_B': 50.2}
                        - 离散类型示例：{'theory_A': 'Einstein', 'theory_B': 'Newton'}
        """
        # ========== 添加source_reliability列（必须）==========
        # 这一列显示每个数据源的可靠性评分（0-1之间）
        # map方法：对每个website，查找它的可靠性值
        dataframe['source_reliability'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('r', self.initial_reliability)
        )
        # lambda是匿名函数，w是website名称
        # 例如：如果website='website_a'，就从history中查找'r'值，如果找不到就用初始值
        # 结果：每一行的source_reliability列都会显示对应website的可靠性

        # ========== 添加error_metric列（可选）==========
        # 这一列显示每个数据源的误差度量（越小越好）
        dataframe['error_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('e', 0.0)
        )
        # 同样使用map方法，查找每个website的误差值'e'

        # ========== 添加consistency_metric列（可选）==========
        # 这一列显示每个数据源的一致性度量（越大越好，0-1之间）
        dataframe['consistency_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('k', 1.0)
        )
        # 查找每个website的一致性值'k'

        # ========== 添加global_truth列（参考）==========
        # 这一列显示每个对象对应的全局真值估计（方便对比）
        dataframe['global_truth'] = dataframe['object'].map(global_truth)
        # 对每个object，查找它的真值估计
        # 例如：如果object='stock_A'，那么global_truth列会显示100.5（真值估计）
        # 这样我们可以对比：website提供的值 vs 真值估计，看看偏差有多大

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        执行一次迭代

        这是一个完整的工作流程，执行一次"迭代"：
        1. 计算全局真值估计（基于当前源可靠性）
        2. 更新源可靠性（基于真值估计）
        3. 更新DataFrame（添加新列）

        迭代的意义：
        - 第一次迭代：使用初始可靠性计算真值，然后根据真值更新可靠性
        - 第二次迭代：使用更新后的可靠性重新计算真值，然后再更新可靠性
        - 这样反复进行，直到可靠性稳定（收敛）

        这个过程就像是"鸡生蛋、蛋生鸡"的问题：
        - 我们需要真值来评估源的可靠性
        - 但我们需要源的可靠性来计算真值
        - 所以通过迭代，逐步改进两者的估计

        Args（参数）:
            dataframe: 输入DataFrame（会被修改）

        Returns（返回值）:
            更新后的DataFrame（已添加新列）
        """
        # ========== 步骤1：计算全局真值估计 ==========
        # 使用当前的源可靠性，计算每个对象的真值估计
        # 第一次调用时，使用初始可靠性（0.5）
        # 后续调用时，使用上一轮更新后的可靠性
        global_truth = self._calculate_global_truth(dataframe)
        # 返回：{'stock_A': 100.5, 'stock_B': 50.2} 这样的字典

        # ========== 步骤2：更新源可靠性 ==========
        # 根据计算出的真值估计，更新每个源的可靠性
        # 这一步会修改self.source_reliability_history
        # 如果某个源提供的值与真值偏差大，它的可靠性会降低
        self._update_source_reliability(dataframe, global_truth)

        # ========== 步骤3：更新DataFrame ==========
        # 在DataFrame中添加新列，包含计算出的所有指标
        # 这一步会修改dataframe，添加source_reliability、error_metric等列
        self._update_dataframe(dataframe, global_truth)

        # 返回更新后的DataFrame
        return dataframe

    def train(self,
              dataframe: pd.DataFrame,
              max_iterations: Optional[int] = None,
              threshold: Optional[float] = None) -> pd.DataFrame:
        """
        训练模型（批量处理，与TruthFinder兼容）

        这是主要的训练方法，用于批量处理所有数据。
        它会反复迭代，直到：
        - 源可靠性不再显著变化（收敛）
        - 或者达到最大迭代次数

        使用场景：
        - 一次性处理所有数据（批量处理）
        - 与原始TruthFinder的train()方法兼容
        - 适合静态数据集

        Args（参数）:
            dataframe: 输入DataFrame，必须包含website, fact, object列
                      - website: 数据源标识（字符串）
                      - fact: 输入值，可以是数值型（例如：100.5）或离散型（例如："Einstein"）
                      - object: 对象标识（字符串）
                      - 注意：算法会自动检测fact列的数据类型（数值或离散）
            max_iterations: 最大迭代次数（可选）
                           - 如果不提供，使用初始化时设置的值
            threshold: 收敛阈值（可选）
                      - 如果不提供，使用初始化时设置的值

        Returns（返回值）:
            更新后的DataFrame，包含原有列和新添加的列：
            - source_reliability: 源可靠性（必须）
            - error_metric: 误差度量（可选）
            - consistency_metric: 一致性度量（可选）
            - global_truth: 全局真值估计（参考）

        注意：实际迭代次数存储在self.last_iteration_count中
        """
        # ========== 使用默认值或传入值 ==========
        # 如果调用者提供了参数，就用提供的；否则用初始化时的默认值
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        conv_threshold = threshold if threshold is not None else self.convergence_threshold

        # ========== 数据预处理 ==========
        # 创建一个副本，避免修改原始数据
        dataframe = dataframe.copy()

        # 初始化迭代计数
        self.last_iteration_count = 0

        # ========== 检测数据类型 ==========
        # 自动检测fact列是数值类型还是离散类型
        # 这个检测只在第一次调用时进行，后续使用self.data_type
        if self.data_type is None:
            self.data_type = self._detect_data_type(dataframe['fact'])
            # 例如：如果fact列是[100.5, 101.0, 99.8]，返回'numeric'
            # 例如：如果fact列是['Einstein', 'Newton', 'Einstein']，返回'discrete'

        # ========== 根据数据类型进行预处理 ==========
        if self.data_type == 'hybrid':
            # 混合类型：保持为文本，不进行转换
            # 混合模式需要从文本中动态提取信息，所以保持原始文本格式
            dataframe['fact'] = dataframe['fact'].astype(str)
            # 例如：保持完整的文本，如"A new Agilent Technologies..."
        elif self.data_type == 'numeric':
            # 数值类型：转换为浮点数
            # to_numeric会自动转换字符串为数字
            # errors='coerce'表示如果转换失败，就用NaN（而不是报错）
            dataframe['fact'] = pd.to_numeric(dataframe['fact'], errors='coerce')
            # 例如：'100.5' → 100.5, '50' → 50.0, 'abc' → NaN
            # 注意：如果检测为数值类型，理论上不应该有NaN，但保留这个处理以防万一
        else:
            # 离散类型：保持为字符串，不进行数值转换
            # 确保所有值都是字符串类型（统一格式）
            dataframe['fact'] = dataframe['fact'].astype(str)
            # 例如：'Einstein' → 'Einstein', 100.5 → '100.5'（如果混入了数值，转为字符串）
            # 注意：如果数据是混合类型（数值和字符串），会被统一处理为离散类型

        # ========== 初始化源可靠性 ==========
        # 为所有新出现的website创建初始记录
        self._initialize_source_reliability(dataframe)

        # ========== 迭代直到收敛 ==========
        # 反复执行迭代，直到满足停止条件
        for i in range(max_iter):
            print('Iteration {}/{}'.format(i + 1, max_iter))
            # 保存上一轮的源可靠性（用于判断是否收敛）
            # 我们只保存'r'值（可靠性），因为这是判断收敛的主要指标
            prev_reliability = {
                w: self.source_reliability_history[w]['r']
                for w in self.source_reliability_history.keys()
            }
            # 例如：{'website_a': 0.5, 'website_b': 0.5, 'website_c': 0.5}

            # ========== 执行一次迭代 ==========
            # 这一步会：
            # 1. 计算全局真值
            # 2. 更新源可靠性
            # 3. 更新DataFrame
            dataframe = self.iteration(dataframe)

            # ========== 检查收敛 ==========
            # 获取当前轮的源可靠性
            current_reliability = {
                w: self.source_reliability_history[w]['r']
                for w in self.source_reliability_history.keys()
            }
            # 例如：{'website_a': 0.95, 'website_b': 0.90, 'website_c': 0.70}

            # 计算每个源的可靠性变化量
            # 取所有变化量中最大的，作为"最大变化"
            changes = []
            # 获取所有website的并集（可能有些源在第一轮出现，有些在后续出现）
            all_websites = set(list(prev_reliability.keys()) + list(current_reliability.keys()))

            for w in all_websites:
                # 计算这个源可靠性的变化：|当前值 - 上一轮值|
                prev_val = prev_reliability.get(w, 0)      # 如果上一轮没有，假设为0
                curr_val = current_reliability.get(w, 0)   # 如果当前轮没有，假设为0
                change = abs(curr_val - prev_val)          # 变化量的绝对值
                changes.append(change)

            # 找出最大变化量
            max_change = max(changes) if changes else 0
            # 例如：如果changes = [0.45, 0.40, 0.20]，那么max_change = 0.45

            # ========== 判断是否收敛 ==========
            # 如果最大变化量小于阈值，说明已经收敛（可靠性稳定了）
            if max_change < conv_threshold:
                # 提前退出循环，不需要继续迭代
                print('Convergence reached at iteration {}/{}'.format(i + 1, max_iter))
                self.last_iteration_count = i + 1
                break
            # 例如：如果max_change=0.0000001, conv_threshold=0.000001
            # 那么0.0000001 < 0.000001，满足条件，退出循环

        # 如果没有提前退出，说明达到了最大迭代次数
        if self.last_iteration_count == 0:
            self.last_iteration_count = max_iter

        # 返回最终结果
        return dataframe
        # DataFrame现在包含原有数据和新添加的列（source_reliability等）

    def process_batch(self,
                     dataframe: pd.DataFrame,
                     epoch: Optional[int] = None) -> pd.DataFrame:
        """
        处理一个批次的数据（增量处理）

        这个方法用于增量处理场景，适用于：
        - 流式数据（数据持续到达）
        - 分批处理大数据集
        - 需要维护历史状态的情况

        与train()的区别：
        - train()：批量处理，一次性处理所有数据，多次迭代直到收敛
        - process_batch()：增量处理，每次处理一批数据，使用历史状态，只迭代一次

        增量处理的优势：
        - 可以处理不断到达的新数据
        - 历史状态会保留，源可靠性的评估会考虑历史表现
        - 适合在线场景（实时处理）

        Args（参数）:
            dataframe: 当前批次的数据
                      - 格式和train()一样，包含website, fact, object列
            epoch: 当前epoch编号（可选，用于历史记录）
                   - epoch就是"批次编号"，从0开始
                   - 如果不提供，不会保存历史真值

        Returns（返回值）:
            更新后的DataFrame（格式和train()返回的一样）
        """
        # ========== 数据预处理 ==========
        # 创建副本，避免修改原始数据
        dataframe = dataframe.copy()

        # ========== 检测数据类型 ==========
        # 自动检测fact列是数值类型还是离散类型
        # 这个检测只在第一次调用时进行，后续使用self.data_type
        # 注意：如果之前已经检测过（self.data_type不为None），就使用之前的结果
        # 这样可以确保同一实例处理的数据类型一致
        if self.data_type is None:
            self.data_type = self._detect_data_type(dataframe['fact'])
            # 例如：如果fact列是[100.5, 101.0, 99.8]，返回'numeric'
            # 例如：如果fact列是['Einstein', 'Newton', 'Einstein']，返回'discrete'

        # ========== 根据数据类型进行预处理 ==========
        if self.data_type == 'hybrid':
            # 混合类型：保持为文本，不进行转换
            dataframe['fact'] = dataframe['fact'].astype(str)
        elif self.data_type == 'numeric':
            # 数值类型：转换为浮点数（和train()一样）
            dataframe['fact'] = pd.to_numeric(dataframe['fact'], errors='coerce')
            # 例如：'100.5' → 100.5, '50' → 50.0
        else:
            # 离散类型：保持为字符串，不进行数值转换
            dataframe['fact'] = dataframe['fact'].astype(str)
            # 例如：'Einstein' → 'Einstein'

        # ========== 初始化源可靠性 ==========
        # 如果遇到新的数据源，为它们创建初始记录
        # 如果数据源已经在历史记录中，不会覆盖（保留历史状态）
        self._initialize_source_reliability(dataframe)

        # ========== 获取上一epoch的全局真值（如果有）==========
        # 在增量处理中，我们可以参考上一批次计算的真值
        # 这对于某些算法可能有用，虽然当前实现中还没有直接使用
        previous_truth = None
        if epoch is not None and epoch > 0 and (epoch - 1) in self.global_truth_history:
            previous_truth = self.global_truth_history[epoch - 1]
            # 例如：如果epoch=1，上一epoch是0，就查找epoch=0时的真值
            # 返回：{'stock_A': 100.5, 'stock_B': 50.2}

        # ========== 执行迭代 ==========
        # 只迭代一次（不像train()那样多次迭代直到收敛）
        # 因为增量处理假设数据是连续的，每次处理时可靠性已经基本稳定
        dataframe = self.iteration(dataframe)

        # ========== 保存当前epoch的真值 ==========
        # 计算当前批次的全局真值，并保存到历史记录中
        # 这样下一批次可以引用
        current_truth = self._calculate_global_truth(dataframe)
        if epoch is not None:
            self.global_truth_history[epoch] = current_truth
            # 例如：保存 {0: {'stock_A': 100.5}, 1: {'stock_A': 100.6}}

        return dataframe

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:
        """
        获取源可靠性信息

        这是一个查询方法，用于查看某个数据源或所有数据源的可靠性信息。

        用途：
        - 调试：查看某个源的可靠性是否合理
        - 分析：比较不同源的可靠性
        - 监控：跟踪源可靠性的变化

        Args（参数）:
            website: 指定源（可选）
                    - 如果提供，只返回这个源的信息
                    - 如果不提供（None），返回所有源的信息

        Returns（返回值）:
            如果指定website，返回这个源的信息字典：
            {'r': 0.95, 'e': 0.01, 'k': 0.99}

            如果不指定，返回所有源的信息字典：
            {
                'website_a': {'r': 0.95, 'e': 0.01, 'k': 0.99},
                'website_b': {'r': 0.90, 'e': 0.02, 'k': 0.98},
                ...
            }
        """
        if website is not None:
            # 只返回指定源的信息
            # 使用copy()避免外部修改影响内部状态
            return self.source_reliability_history.get(website, {}).copy()
        else:
            # 返回所有源的信息
            # 对每个源的信息也使用copy()，确保返回的是副本
            return self.source_reliability_history.copy()

    def reset_history(self):
        """
        重置历史状态（用于新的训练）

        这个方法用于清除所有历史记录，让算法从"干净"的状态重新开始。

        使用场景：
        - 处理完全独立的新数据集
        - 测试：确保每次测试从相同状态开始
        - 切换数据集：处理完一个数据集后，清除状态再处理另一个

        注意：
        - 这个方法会清除所有历史，包括源可靠性和全局真值历史
        - 调用后，算法会忘记之前学到的所有信息
        - 如果只是想处理新数据但保留历史，不要调用这个方法
        """
        # 清空源可靠性历史记录
        self.source_reliability_history = {}
        # 例如：调用前是 {'website_a': {...}, 'website_b': {...}}
        # 调用后变成 {}

        # 清空全局真值历史记录
        self.global_truth_history = {}
        # 例如：调用前是 {0: {...}, 1: {...}}
        # 调用后变成 {}
