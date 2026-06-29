#!/usr/bin/env python3
"""Replace plain-text variable names in tracked-change paragraphs with OMML inline equations."""

import copy
from lxml import etree
from docx import Document

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def qw(tag):
    return f'{{{W}}}{tag}'

def qm(tag):
    return f'{{{M}}}{tag}'


# ── OMML building blocks ──────────────────────────────────────────

def _mr(text, italic=True):
    """Single math run (m:r)."""
    mr = etree.Element(qm('r'))
    if not italic:
        rPr = etree.SubElement(mr, qm('rPr'))
        etree.SubElement(rPr, qm('sty')).set(qm('val'), 'p')
    wrPr = etree.SubElement(mr, qw('rPr'))
    rf = etree.SubElement(wrPr, qw('rFonts'))
    rf.set(qw('ascii'), 'Cambria Math')
    rf.set(qw('hAnsi'), 'Cambria Math')
    sz = etree.SubElement(wrPr, qw('sz'))
    sz.set(qw('val'), '24')
    mt = etree.SubElement(mr, qm('t'))
    mt.text = text
    if text and (' ' in text):
        mt.set(XML_SPACE, 'preserve')
    return mr


def _omath(*children):
    om = etree.Element(qm('oMath'))
    for c in children:
        om.append(c)
    return om


def _ssub(base_el, sub_el):
    ss = etree.Element(qm('sSub'))
    etree.SubElement(ss, qm('sSubPr'))
    e = etree.SubElement(ss, qm('e'))
    e.append(base_el)
    s = etree.SubElement(ss, qm('sub'))
    s.append(sub_el)
    return ss


def _ssup(base_el, sup_el):
    ss = etree.Element(qm('sSup'))
    etree.SubElement(ss, qm('sSupPr'))
    e = etree.SubElement(ss, qm('e'))
    e.append(base_el)
    s = etree.SubElement(ss, qm('sup'))
    s.append(sup_el)
    return ss


def _ssubsup(base_el, sub_el, sup_el):
    sss = etree.Element(qm('sSubSup'))
    etree.SubElement(sss, qm('sSubSupPr'))
    e = etree.SubElement(sss, qm('e'))
    e.append(base_el)
    sb = etree.SubElement(sss, qm('sub'))
    sb.append(sub_el)
    sp = etree.SubElement(sss, qm('sup'))
    sp.append(sup_el)
    return sss


def _acc(base_el, char='̂'):
    """Accent element. Default is hat (̂)."""
    acc = etree.Element(qm('acc'))
    accPr = etree.SubElement(acc, qm('accPr'))
    if char != '̂':
        etree.SubElement(accPr, qm('chr')).set(qm('val'), char)
    e_el = etree.SubElement(acc, qm('e'))
    e_el.append(base_el)
    return acc


def _del(beg, end, *content_els):
    """Delimiter (parentheses, brackets, braces, etc.)."""
    d = etree.Element(qm('d'))
    dPr = etree.SubElement(d, qm('dPr'))
    etree.SubElement(dPr, qm('begChr')).set(qm('val'), beg)
    etree.SubElement(dPr, qm('endChr')).set(qm('val'), end)
    e_el = etree.SubElement(d, qm('e'))
    for c in content_els:
        e_el.append(c)
    return d


def _bar(base_el):
    """Bar accent (ε̄)."""
    return _acc(base_el, '̄')


def _nary(ch, sub_el, sup_el, body_el):
    """N-ary operator like Σ."""
    n = etree.Element(qm('nary'))
    nPr = etree.SubElement(n, qm('naryPr'))
    etree.SubElement(nPr, qm('chr')).set(qm('val'), ch)
    s = etree.SubElement(n, qm('sub'))
    if sub_el is not None:
        s.append(sub_el)
    sp = etree.SubElement(n, qm('sup'))
    if sup_el is not None:
        sp.append(sup_el)
    e = etree.SubElement(n, qm('e'))
    if body_el is not None:
        e.append(body_el)
    return n


def _frac(num_el, den_el):
    f = etree.Element(qm('f'))
    etree.SubElement(f, qm('fPr'))
    n = etree.SubElement(f, qm('num'))
    n.append(num_el)
    d = etree.SubElement(f, qm('den'))
    d.append(den_el)
    return f


def _rad(deg_el, body_el):
    """Radical (square root)."""
    r = etree.Element(qm('rad'))
    rPr = etree.SubElement(r, qm('radPr'))
    if deg_el is None:
        etree.SubElement(rPr, qm('degHide')).set(qm('val'), '1')
    deg = etree.SubElement(r, qm('deg'))
    if deg_el is not None:
        deg.append(deg_el)
    e = etree.SubElement(r, qm('e'))
    e.append(body_el)
    return r


# ── Convenience constructors ──────────────────────────────────────

def VAR(letter):
    """Simple italic variable."""
    return _omath(_mr(letter))


def TXT(text):
    """Plain (upright) math text."""
    return _omath(_mr(text, italic=False))


def SUB(base, sub, base_italic=True, sub_italic=None):
    """x_s. Auto-detects if sub should be italic."""
    if sub_italic is None:
        sub_italic = not (len(sub) > 1 and sub.isalpha() and sub not in ('si',))
    return _omath(_ssub(_mr(base, base_italic), _mr(sub, sub_italic)))


def SUP(base, sup, base_italic=True, sup_italic=None):
    if sup_italic is None:
        sup_italic = not (len(sup) > 1 and sup.isalpha())
    return _omath(_ssup(_mr(base, base_italic), _mr(sup, sup_italic)))


def SUBSUP(base, sub, sup, base_italic=True, sub_italic=True, sup_italic=None):
    if sup_italic is None:
        sup_italic = not (len(sup) > 1 and all(c.isalpha() or c == ',' for c in sup)
                          and not all(c == ',' or len(c) == 1 for c in sup.split(',')))
    return _omath(_ssubsup(_mr(base, base_italic), _mr(sub, sub_italic), _mr(sup, sup_italic)))


def NESTEDSUB(base, sub_base, sub_sub):
    """r_{s_i}: base with nested subscript."""
    inner = _ssub(_mr(sub_base), _mr(sub_sub))
    return _omath(_ssub(_mr(base), inner))


def HATSUB(base, sub):
    """x̂_o: hat accent with subscript."""
    return _omath(_ssub(_acc(_mr(base)), _mr(sub)))


def BARSUB(base, sub):
    """ε̄_o: bar accent with subscript."""
    return _omath(_ssub(_bar(_mr(base)), _mr(sub)))


def HATVAR(base):
    """x̂: just hat, no subscript."""
    return _omath(_acc(_mr(base)))


# ── Complex expression builders (per-paragraph) ──────────────────

def expr_alpha_beta_init():
    """(α₀, β₀) = (1, 1)"""
    om = _omath(
        _del('(', ')',
             _ssub(_mr('α'), _mr('0')),
             _mr(',', False),
             _ssub(_mr('β'), _mr('0'))),
        _mr(' = ', False),
        _del('(', ')',
             _mr('1', False),
             _mr(',', False),
             _mr('1', False)),
    )
    return om


def expr_w_eq_r(sub_w, sub_r_base, sub_r_sub):
    """w_i = r_{s_i}"""
    inner_sub = _ssub(_mr(sub_r_base), _mr(sub_r_sub))
    om = _omath(
        _ssub(_mr('w'), _mr(sub_w)),
        _mr('=', False),
        _ssub(_mr('r'), inner_sub),
    )
    return om


def expr_d_eq_abs():
    """d_o^i = |x_o^i − m|"""
    om = _omath(
        _ssubsup(_mr('d'), _mr('o'), _mr('i')),
        _mr('=', False),
        _mr('|', False),
        _ssubsup(_mr('x'), _mr('o'), _mr('i')),
        _mr('−', False),
        _mr('m'),
        _mr('|', False),
    )
    return om


def expr_d_order_stat():
    """d_o^{(j)}"""
    om = _omath(
        _ssubsup(_mr('d'), _mr('o'), _mr('(j)')),
    )
    return om


def expr_d_set_range():
    """{d_o^i}_{i=1}^{n}"""
    inner = _ssubsup(
        _del('{', '}', _ssubsup(_mr('d'), _mr('o'), _mr('i'))),
        _mr('i=1', False),
        _mr('n'),
    )
    om = _omath(inner)
    return om


def expr_sigma_def():
    """σ_o = δ_o / Φ^{−1}(0.75) ≈ 1.4826 δ_o"""
    om = _omath(
        _ssub(_mr('σ'), _mr('o')),
        _mr('=', False),
        _ssub(_mr('δ'), _mr('o')),
        _mr('/', False),
        _ssup(_mr('Φ', False), _mr('−1', False)),
        _del('(', ')', _mr('0.75', False)),
        _mr('≈', False),
        _mr('1.4826', False),
        _ssub(_mr('δ'), _mr('o')),
    )
    return om


def expr_phi_inv():
    """Φ^{−1}(0.75)"""
    om = _omath(
        _ssup(_mr('Φ', False), _mr('−1', False)),
        _del('(', ')', _mr('0.75', False)),
    )
    return om


def expr_u_def():
    """u_o^i = (x_o^i − m) / (c · σ_o)"""
    om = _omath(
        _ssubsup(_mr('u'), _mr('o'), _mr('i')),
        _mr('=', False),
        _del('(', ')',
             _ssubsup(_mr('x'), _mr('o'), _mr('i')),
             _mr('−', False),
             _mr('m')),
        _mr('/', False),
        _del('(', ')',
             _mr('c'),
             _mr('·', False),
             _ssub(_mr('σ'), _mr('o'))),
    )
    return om


def expr_c_eq():
    """c = 4.685"""
    om = _omath(
        _mr('c'),
        _mr(' = ', False),
        _mr('4.685', False),
    )
    return om


def expr_abs_u_le_1():
    """|u_o^i| ≤ 1"""
    om = _omath(
        _mr('|', False),
        _ssubsup(_mr('u'), _mr('o'), _mr('i')),
        _mr('|', False),
        _mr(' ≤ ', False),
        _mr('1', False),
    )
    return om


def expr_tukey_value():
    """(1 − (u_o^i)²)²"""
    om = _omath(
        _ssup(
            _del('(', ')',
                 _mr('1', False),
                 _mr('−', False),
                 _ssup(
                     _del('(', ')', _ssubsup(_mr('u'), _mr('o'), _mr('i'))),
                     _mr('2', False))),
            _mr('2', False)),
    )
    return om


def expr_wbase_eq():
    """w_i^{base} = r_{s_i}"""
    inner_sub = _ssub(_mr('s'), _mr('i'))
    om = _omath(
        _ssubsup(_mr('w'), _mr('i'), _mr('base', False)),
        _mr('=', False),
        _ssub(_mr('r'), inner_sub),
    )
    return om


def expr_eps_eq():
    """ε_o^i = x_o^i − x̂_o"""
    om = _omath(
        _ssubsup(_mr('ε'), _mr('o'), _mr('i')),
        _mr('=', False),
        _ssubsup(_mr('x'), _mr('o'), _mr('i')),
        _mr('−', False),
        _ssub(_acc(_mr('x')), _mr('o')),
    )
    return om


def expr_gamma_def():
    """γ_o 的定义: max(1.4826 δ_o^{res}, √(1/n · Σ(ε_o^i − ε̄_o)²), ε_{tol})"""
    om = _omath(
        _ssub(_mr('γ'), _mr('o')),
    )
    return om


def expr_gamma_full():
    """max(1.4826 δ_o^{res}, √(...), ε_{tol})"""
    om = _omath(
        _mr('max', False),
        _del('(', ')',
             _mr('1.4826', False),
             _ssubsup(_mr('δ'), _mr('o'), _mr('res', False)),
             _mr(',', False),
             _rad(None,
                  _frac(
                      _mr('1', False),
                      _mr('n'))),
             _mr('·', False),
             _mr('Σ', False),
             _ssup(
                 _del('(', ')',
                      _ssubsup(_mr('ε'), _mr('o'), _mr('i')),
                      _mr('−', False),
                      _ssub(_bar(_mr('ε')), _mr('o'))),
                 _mr('2', False)),
             _mr(',', False),
             _ssub(_mr('ε'), _mr('tol', False))),
    )
    return om


def expr_delta_res():
    """δ_o^{res}"""
    return _omath(_ssubsup(_mr('δ'), _mr('o'), _mr('res', False)))


def expr_eps_set():
    """{ε_o^i}"""
    return _omath(_del('{', '}', _ssubsup(_mr('ε'), _mr('o'), _mr('i'))))


def expr_ebar_eq():
    """ε̄_o = 1/n · Σ ε_o^i"""
    om = _omath(
        _ssub(_bar(_mr('ε')), _mr('o')),
        _mr('=', False),
        _mr('1', False),
        _mr('/', False),
        _mr('n'),
        _mr(' · ', False),
        _mr('Σ', False),
        _ssubsup(_mr('ε'), _mr('o'), _mr('i')),
    )
    return om


def expr_phi_map():
    """φ: Σ* → ℝ^d"""
    om = _omath(
        _mr('φ'),
        _mr(': ', False),
        _ssup(_mr('Σ', False), _mr('*', False)),
        _mr(' → ', False),
        _ssup(_mr('ℝ', False), _mr('d')),
    )
    return om


def expr_norm2():
    """‖·‖_2"""
    om = _omath(
        _ssub(
            _del('‖', '‖', _mr('·', False)),
            _mr('2', False)),
    )
    return om


def expr_sim_def():
    """sim_o^{i,j} = (v_o^i)^⊤ · v_o^j"""
    om = _omath(
        _ssubsup(_mr('sim', False), _mr('o'), _mr('i,j')),
        _mr('=', False),
        _ssup(
            _del('(', ')', _ssubsup(_mr('v'), _mr('o'), _mr('i'))),
            _mr('⊤', False)),
        _mr(' · ', False),
        _ssubsup(_mr('v'), _mr('o'), _mr('j')),
    )
    return om


def expr_alpha_in():
    """α ∈ [0, 1]"""
    om = _omath(
        _mr('α'),
        _mr(' ∈ ', False),
        _del('[', ']', _mr('0', False), _mr(',', False), _mr('1', False)),
    )
    return om


def expr_alpha_eq():
    """α = 0.5"""
    om = _omath(
        _mr('α'),
        _mr(' = ', False),
        _mr('0.5', False),
    )
    return om


def expr_rs_set():
    """{r_s}_{s∈S}"""
    inner = _ssub(
        _del('{', '}', _ssub(_mr('r'), _mr('s'))),
        _mr('s∈S'),
    )
    return _omath(inner)


def expr_hs_set():
    """{h_o^s}"""
    return _omath(_del('{', '}', _ssubsup(_mr('h'), _mr('o'), _mr('s'))))


def expr_is_valid():
    """is_valid = 1"""
    om = _omath(
        _mr('is_valid', False),
        _mr(' = ', False),
        _mr('1', False),
    )
    return om


# ── Paragraph content definitions ─────────────────────────────────
# Each is a list of ('t', text) or ('m', omath_generator_func)

P68 = [
    ('t', '其中，'),
    ('m', lambda: SUB('r', 's')),
    ('t', ' 表示智能体生成源 '),
    ('m', lambda: VAR('s')),
    ('t', ' 的可信度，取值范围为 [0, 1]；'),
    ('m', lambda: SUB('α', 's')),
    ('t', ' 表示源 '),
    ('m', lambda: VAR('s')),
    ('t', ' 累积的正向证据参数；'),
    ('m', lambda: SUB('β', 's')),
    ('t', ' 表示源 '),
    ('m', lambda: VAR('s')),
    ('t', ' 累积的负向证据参数；初始值设为 '),
    ('m', expr_alpha_beta_init),
    ('t', '，对应均匀先验。'),
]

P71 = [
    ('t', '其中，'),
    ('m', lambda: VAR('m')),
    ('t', ' 为加权中位数，作为数值真值的鲁棒初始估计；'),
    ('m', lambda: VAR('z')),
    ('t', ' 为候选中心值；'),
    ('m', lambda: expr_w_eq_r('i', 's', 'i')),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个保留源的权重，取其可信度得分；'),
    ('m', lambda: SUBSUP('x', 'o', 'i')),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个保留源对对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 提供的数值报告；'),
    ('m', lambda: VAR('n')),
    ('t', ' 为经可靠性门控后保留的源数量。'),
]

P73 = [
    ('t', '其中，'),
    ('m', lambda: SUB('σ', 'o')),
    ('t', ' 为对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 的鲁棒尺度估计；'),
    ('m', lambda: SUB('δ', 'o')),
    ('t', ' 为保留源数值报告相对于加权中位数 '),
    ('m', lambda: VAR('m')),
    ('t', ' 的中位数绝对偏差（MAD）；'),
    ('m', expr_phi_inv),
    ('t', ' ≈ 0.67449 为标准正态分布的 0.75 分位数；1.4826 为将 MAD 转换为与标准差一致的尺度因子。'),
]

P75 = [
    ('t', '其中，'),
    ('m', lambda: SUB('σ', 'o')),
    ('t', ' 为鲁棒尺度估计，定义为 '),
    ('m', expr_sigma_def),
    ('t', '；'),
    ('m', expr_phi_inv),
    ('t', ' ≈ 0.67449 为标准正态分布的 0.75 分位数；'),
    ('m', expr_u_def),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个源的标准化残差；'),
    ('m', expr_c_eq),
    ('t', ' 为鲁棒统计学中的标准调谐常数；'),
    ('m', lambda: SUBSUP('w', 'i', 'Tukey', sup_italic=False)),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个源的 Tukey bisquare 权重，当 '),
    ('m', expr_abs_u_le_1),
    ('t', ' 时取值 '),
    ('m', expr_tukey_value),
    ('t', '，否则为 0。'),
]

P77 = [
    ('t', '其中，'),
    ('m', lambda: HATSUB('x', 'o')),
    ('t', ' 为对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 经 IRLS 迭代后的鲁棒数值真值估计；'),
    ('m', expr_wbase_eq),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个保留源继承自跨批次可靠性模型的基础权重（即源可信度）；'),
    ('m', lambda: SUBSUP('w', 'i', 'Tukey', sup_italic=False)),
    ('t', ' 为式（4）所得的 Tukey bisquare 权重；'),
    ('m', lambda: SUBSUP('x', 'o', 'i')),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个保留源的数值报告。该过程重复迭代直至收敛或达到最大迭代次数；当 '),
    ('m', lambda: SUB('σ', 'o')),
    ('t', ' 低于容差阈值 '),
    ('m', lambda: SUB('ε', 'tol', sub_italic=False)),
    ('t', ' 时，直接返回加权中位数以避免 Tukey bisquare 权重中的除零问题。'),
]

P80 = [
    ('t', '其中，'),
    ('m', lambda: SUBSUP('c', 'o', 'i,num', sup_italic=False)),
    ('t', ' 为源 '),
    ('m', lambda: VAR('i')),
    ('t', ' 在对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 上的数值一致性得分，取值范围为 [0, 1]；'),
    ('m', expr_eps_eq),
    ('t', ' 为源 '),
    ('m', lambda: VAR('i')),
    ('t', ' 的数值报告相对于估计真值的残差；'),
    ('m', lambda: SUB('γ', 'o')),
    ('t', ' 为一致性尺度，定义为 '),
    ('m', expr_gamma_full),
    ('t', '，其中 '),
    ('m', expr_delta_res),
    ('t', ' 为残差序列 '),
    ('m', expr_eps_set),
    ('t', ' 的中位数绝对偏差，'),
    ('m', expr_ebar_eq),
    ('t', ' 为残差均值，'),
    ('m', lambda: SUB('ε', 'tol', sub_italic=False)),
    ('t', ' 为数值稳定性容差。'),
]

P83 = [
    ('t', '其中，'),
    ('m', lambda: SUBSUP('v', 'o', 'i')),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 条文本声明经 SBERT 编码后的单位长度语义向量；'),
    ('m', expr_phi_map),
    ('t', ' 为句子级嵌入函数，由 Sentence-BERT（SBERT）模型实例化；'),
    ('m', lambda: SUBSUP('t', 'o', 'i')),
    ('t', ' 为第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 个保留源对对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 提供的文本声明；'),
    ('m', expr_norm2),
    ('t', ' 为 L2 范数，用于将嵌入向量归一化为单位长度。两条文本声明之间的语义相似度定义为其语义向量的余弦相似度 '),
    ('m', expr_sim_def),
    ('t', '。'),
]

P85 = [
    ('t', '其中，'),
    ('m', lambda: SUBSUP('A', 'o', 'i,j')),
    ('t', ' 为对象 '),
    ('m', lambda: VAR('o')),
    ('t', ' 上第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 条与第 '),
    ('m', lambda: VAR('j')),
    ('t', ' 条文本声明之间的混合亲和度；'),
    ('m', lambda: SUBSUP('sim', 'o', 'i,j', base_italic=False)),
    ('t', ' 为二者的语义相似度（余弦相似度）；'),
    ('m', lambda: SUBSUP('num', 'o', 'i,j', base_italic=False)),
    ('t', ' 为二者的数值一致性，通过对各自文本中正则抽取的有序数值序列以最近匹配的相对偏差经高斯衰减后取平均并对称化得到；'),
    ('m', expr_alpha_in),
    ('t', ' 为语义与数值信号之间的平衡因子，本发明中设为 '),
    ('m', expr_alpha_eq),
    ('t', '；'),
    ('m', lambda: SUBSUP('L', 'o', 'i')),
    ('t', ' 和 '),
    ('m', lambda: SUBSUP('L', 'o', 'j')),
    ('t', ' 分别为从第 '),
    ('m', lambda: VAR('i')),
    ('t', ' 条和第 '),
    ('m', lambda: VAR('j')),
    ('t', ' 条文本声明中通过正则表达式 '),
    ('m', lambda: SUB('π', 'num', sub_italic=False)),
    ('t', ' 抽取的有序数值序列；当任一序列为空集时，混合亲和度退化为仅采用语义相似度。'),
]

P91 = [
    ('t', '其中，'),
    ('m', lambda: NESTEDSUB('α', 's', 'i')),
    ('t', ' 和 '),
    ('m', lambda: NESTEDSUB('β', 's', 'i')),
    ('t', ' 分别为源 '),
    ('m', lambda: SUB('s', 'i')),
    ('t', ' 的 Beta 后验正向参数和负向参数；'),
    ('m', lambda: VAR('η')),
    ('t', ' 为学习率，控制单批次证据对后验更新的影响幅度；'),
    ('m', lambda: SUBSUP('c', 'e', 'i')),
    ('t', ' 为源 '),
    ('m', lambda: VAR('i')),
    ('t', ' 在事件批次 '),
    ('m', lambda: VAR('e')),
    ('t', ' 中的批次级一致性得分，由该源在该批次所有对象上的数值通道得分与文本通道得分按可用情况取平均后再在所有对象上取平均得到。更新后的可信度按式（1）重新计算。'),
]

P96 = [
    ('t', '其中，'),
    ('m', lambda: VAR('C')),
    ('t', ' 为整体 Circom 算术电路；'),
    ('m', lambda: SUB('C', 'NUM', sub_italic=False)),
    ('t', ' 为数值通道电路，实现可靠性门控、加权中位数初始化、Tukey bisquare 权重下的 IRLS 迭代以及数值一致性评分；'),
    ('m', lambda: SUB('C', 'TXT', sub_italic=False)),
    ('t', ' 为文本通道电路，实现语义余弦相似度计算、正则数值抽取、混合亲和度矩阵构建、软投票文本输出选择与文本一致性评分；'),
    ('m', lambda: SUB('C', 'REL', sub_italic=False)),
    ('t', ' 为源可靠性更新电路，基于通道级一致性得分实现 Beta 后验参数更新；∧ 表示逻辑合取，即电路输出有效信号 '),
    ('m', expr_is_valid),
    ('t', ' 当且仅当三个子模块的约束全部成立。'),
]

P99 = [
    ('t', '其中，'),
    ('m', lambda: TXT('pub')),
    ('t', ' 为公共输入，包含每个对象的可信数值输出 '),
    ('m', lambda: HATSUB('x', 'o')),
    ('t', '、可信文本输出 '),
    ('m', lambda: HATSUB('t', 'o')),
    ('t', '，以及更新后的源可信度 '),
    ('m', expr_rs_set),
    ('t', '；'),
    ('m', lambda: TXT('wit')),
    ('t', ' 为见证（私有输入），包含各智能体的原始混合声明 '),
    ('m', expr_hs_set),
    ('t', '，以及中位数 '),
    ('m', lambda: VAR('m')),
    ('t', '、鲁棒尺度 '),
    ('m', lambda: SUB('σ', 'o')),
    ('t', '、文本语义向量 '),
    ('m', lambda: SUBSUP('v', 'o', 'i')),
    ('t', '、抽取数值序列 '),
    ('m', lambda: SUBSUP('L', 'o', 'i')),
    ('t', '、混合亲和度 '),
    ('m', lambda: SUBSUP('A', 'o', 'i,j')),
    ('t', '、文本通道得分 '),
    ('m', lambda: SUBSUP('c', 'o', 'i,txt', sup_italic=False)),
    ('t', '、批次级得分 '),
    ('m', lambda: SUBSUP('c', 'e', 'i')),
    ('t', ' 与更新前后的 Beta 后验参数 '),
    ('m', lambda: SUB('α', 's')),
    ('t', '、'),
    ('m', lambda: SUB('β', 's')),
    ('t', ' 等全部中间量；'),
    ('m', lambda: VAR('C')),
    ('t', ' 为式（10）定义的整体电路。该命题断言存在一组合法的见证，使得电路在给定公共输入与见证下的全部约束成立。'),
]

PARA_DEFS = {
    68: P68,
    71: P71,
    73: P73,
    75: P75,
    77: P77,
    80: P80,
    83: P83,
    85: P85,
    91: P91,
    96: P96,
    99: P99,
}


# ── Document processing ──────────────────────────────────────────

def make_text_run(text, rpr_template=None):
    """Create a w:r text run."""
    wr = etree.Element(qw('r'))
    if rpr_template is not None:
        wr.append(copy.deepcopy(rpr_template))
    else:
        rPr = etree.SubElement(wr, qw('rPr'))
        rf = etree.SubElement(rPr, qw('rFonts'))
        rf.set(qw('eastAsia'), '宋体')
        sz = etree.SubElement(rPr, qw('sz'))
        sz.set(qw('val'), '24')
        szCs = etree.SubElement(rPr, qw('szCs'))
        szCs.set(qw('val'), '24')
    wt = etree.SubElement(wr, qw('t'))
    wt.text = text
    if text and (text[0] == ' ' or text[-1] == ' '):
        wt.set(XML_SPACE, 'preserve')
    return wr


def rebuild_paragraph(para_element, content_def, next_id):
    """Rebuild the tracked-change paragraph with proper OMML."""
    # Find the w:ins element that contains the content
    ins_el = None
    for child in para_element:
        if child.tag == qw('ins'):
            ins_el = child
            break

    if ins_el is None:
        print("  WARNING: no w:ins found")
        return next_id

    # Get author and date from the ins element
    author = ins_el.get(qw('author'), 'Claude')
    date = ins_el.get(qw('date'), '2026-06-17T23:44:35Z')

    # Get the text run properties from existing content
    rpr_template = None
    for r in ins_el.findall(f'.//{qw("r")}'):
        rpr = r.find(qw('rPr'))
        if rpr is not None:
            rpr_template = copy.deepcopy(rpr)
            break

    # Clear existing content in ins
    for child in list(ins_el):
        ins_el.remove(child)

    # Build new content
    for seg_type, seg_content in content_def:
        if seg_type == 't':
            # Text run inside the same w:ins
            wr = make_text_run(seg_content, rpr_template)
            ins_el.append(wr)
        elif seg_type == 'm':
            # Math element - we need to create a new w:ins for it
            # Actually, m:oMath can go directly inside w:ins
            omath = seg_content()
            ins_el.append(omath)

    return next_id


def main():
    src = '/Users/huabowen/TruthFinder/docs/副本-【发明】【北京邮电大学】专利书-LASO.docx建议.docx'
    dst = src  # overwrite

    doc = Document(src)
    next_id = 200  # start new IDs from 200

    for pidx, content_def in PARA_DEFS.items():
        print(f"Processing P{pidx}...")
        para = doc.paragraphs[pidx]
        next_id = rebuild_paragraph(para._element, content_def, next_id)

    doc.save(dst)
    print(f"\nSaved to: {dst}")


if __name__ == '__main__':
    main()
