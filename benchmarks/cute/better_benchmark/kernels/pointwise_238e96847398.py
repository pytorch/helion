"""better-benchmark repro pointwise_238e96847398 (torchbench pyhpc_equation_of_state, inference).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_div_log_mul_pow_sqrt_sub_0``: pyhpc's ``gsw_dHdT`` (the
derivative of TEOS-10 dynamic enthalpy with respect to Conservative
Temperature) of bf16 Conservative Temperature ``ct`` and Absolute Salinity
``sa`` (contiguous ``[204, 204, 26]``) and a pressure profile ``p``
(``[1, 1, 26]``, broadcast over the last dim), stored in bf16. Inductor
evaluates the 372-op graph in fp32 as 334 operations: it drops the ``* 1.0``
no-ops, lowers ``reciprocal`` to ``1.0 / x`` and ``pow(x, 2)`` to ``x * x``,
and CSEs the repeated subexpressions. It indexes the dense fields as one flat
dimension (two loads and one store per element) and gathers the pressure with
``index % 26``. The port tiles the same flat view with the same index math and
evaluates the same 334 operations in the graph's order (statements are named
after the FX graph's nodes in ``repro.py``). The randn inputs make about half
of the outputs NaN (``sqrt`` of negative salinity), as in eager, and the fp32
result is dominated by cancellation (its error against the float64 reference
is comparable to the output's magnitude, for Inductor too), so only the
fp64-anchored gate is meaningful.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gsw_dhdt(
    temperature: torch.Tensor, salinity: torch.Tensor, pressure: torch.Tensor
) -> torch.Tensor:
    # temperature, salinity: [..., levels]; pressure: [1, ..., 1, levels].
    levels = pressure.size(-1)
    ct_flat = temperature.view(-1)
    sa_flat = salinity.view(-1)
    p_flat = pressure.view(-1)
    n = ct_flat.size(0)
    out = torch.empty([n], dtype=temperature.dtype, device=temperature.device)
    for tile in hl.tile(n):
        ct = ct_flat[tile].to(torch.float32)
        sa = sa_flat[tile].to(torch.float32)
        p = p_flat[tile.index % levels].to(torch.float32)
        mul = p * 10000.0
        mul_1 = ct * -1.8648264253656e-14
        mul_2 = mul_1 * 2.0
        add = mul_2 + -1.941660213148725e-11
        mul_3 = sa * 1.119522344879478e-14
        add_1 = add + mul_3
        mul_4 = ct * 3.726050720345733e-06
        add_2 = mul_4 + -0.000343609007985188
        mul_5 = ct * add_2
        mul_6 = mul_5 * 0.5
        add_3 = mul_6 + -0.011166348136762635
        mul_7 = ct * 6.876837219536232e-07
        add_4 = mul_7 + -0.0001806789763745328
        mul_8 = sa * add_4
        mul_9 = mul_8 * 0.5
        add_5 = add_3 + mul_9
        mul_10 = add_5 * 1.2115804975093732e-16
        mul_11 = ct * -1.061519070296458e-11
        add_6 = mul_11 + -1.988366587925593e-08
        mul_12 = ct * add_6
        add_7 = mul_12 + -3.087032500374211e-07
        mul_13 = sa * 1.55093272922008e-10
        add_8 = add_7 + mul_13
        reciprocal = torch.reciprocal(add_8)
        mul_15 = mul_10 * reciprocal
        sub = add_1 - mul_15
        mul_16 = ct * 6.057902487546866e-17
        add_9 = mul_16 + -1.200507748551599e-15
        mul_17 = add_9 * 2.0
        add_10 = mul_4 + -0.000171804503992594
        mul_19 = sa * 6.876837219536232e-07
        mul_20 = mul_19 * 0.5
        add_11 = add_10 + mul_20
        mul_21 = mul_17 * add_11
        mul_22 = mul_21 * reciprocal
        sub_1 = sub - mul_22
        mul_23 = add_9 * add_5
        mul_24 = mul_23 * 2.0
        pow_1 = torch.pow(add_8, 2)
        reciprocal_1 = torch.reciprocal(pow_1)
        mul_26 = mul_11 * 2.0
        add_12 = mul_26 + -1.988366587925593e-08
        mul_27 = reciprocal_1 * add_12
        mul_28 = mul_24 * mul_27
        add_13 = sub_1 + mul_28
        mul_29 = p * 3.028951243773433e-17
        add_14 = add_13 + mul_29
        mul_30 = mul * add_14
        mul_31 = mul_30 * reciprocal
        add_15 = mul_1 + -1.941660213148725e-11
        add_16 = add_15 + mul_3
        mul_33 = ct * add_16
        add_17 = mul_33 + -1.11901159287511e-10
        mul_35 = mul_24 * reciprocal
        sub_2 = add_17 - mul_35
        mul_36 = add_9 * 0.5
        mul_37 = mul_36 * p
        add_18 = sub_2 + mul_37
        mul_38 = mul * add_18
        mul_39 = mul_38 * mul_27
        sub_3 = mul_31 - mul_39
        mul_40 = ct * -1.931012931541776e-12
        add_19 = mul_40 + 3.191413910561627e-09
        mul_41 = ct * add_19
        add_20 = mul_41 + -3.212746477974189e-07
        mul_42 = mul_40 * 2.0
        add_21 = mul_42 + 3.191413910561627e-09
        mul_43 = ct * add_21
        add_22 = add_20 + mul_43
        mul_44 = sa * 6.211426728363857e-10
        add_23 = add_22 + mul_44
        pow_2 = torch.pow(add_5, 2)
        mul_45 = pow_2 * 2.4231609950187464e-16
        mul_46 = mul_45 * reciprocal
        mul_47 = mul_23 * 8.0
        mul_48 = reciprocal * add_11
        mul_49 = mul_47 * mul_48
        add_24 = mul_46 + mul_49
        mul_50 = add_9 * pow_2
        mul_51 = mul_50 * 4.0
        mul_52 = mul_51 * mul_27
        sub_4 = add_24 - mul_52
        mul_53 = ct * 0.001181805545074306
        add_25 = mul_53 + -0.03147759265588511
        mul_54 = ct * add_25
        add_26 = mul_54 + 2.839940833161907
        mul_55 = ct * add_26
        add_27 = mul_55 + 999.8420897506056
        mul_56 = ct * 0.0002327859407479162
        add_28 = mul_56 + -0.02986498947203215
        mul_57 = ct * add_28
        add_29 = mul_57 + -6.698001071123802
        sqrt = torch.sqrt(sa)
        mul_58 = ct * 1.645039373682922e-07
        add_30 = mul_58 + -1.426984671633621e-05
        mul_59 = ct * add_30
        add_31 = mul_59 + 0.00050954225738805
        mul_60 = ct * add_31
        add_32 = mul_60 + -0.0398882237896849
        mul_61 = sqrt * add_32
        add_33 = add_29 + mul_61
        mul_62 = sa * add_33
        add_34 = add_27 + mul_62
        mul_63 = add_34 * 6.057902487546866e-17
        sub_5 = sub_4 - mul_63
        mul_64 = mul_53 * 2.0
        add_36 = mul_64 + -0.03147759265588511
        mul_65 = ct * add_36
        add_37 = add_26 + mul_65
        mul_66 = mul_56 * 2.0
        add_38 = mul_66 + -0.02986498947203215
        mul_67 = mul_58 * 2.0
        add_40 = mul_67 + -1.426984671633621e-05
        mul_68 = ct * add_40
        add_41 = add_31 + mul_68
        mul_69 = sqrt * add_41
        add_42 = add_38 + mul_69
        mul_70 = sa * add_42
        add_43 = add_37 + mul_70
        mul_71 = add_9 * add_43
        sub_6 = sub_5 - mul_71
        mul_72 = add_1 * 2.0
        mul_73 = mul_72 * add_5
        sub_7 = sub_6 - mul_73
        mul_74 = add_17 * 2.0
        mul_75 = mul_74 * add_11
        sub_8 = sub_7 - mul_75
        mul_76 = sub_8 * reciprocal
        add_47 = add_23 + mul_76
        mul_78 = mul_51 * reciprocal
        mul_79 = add_9 * add_34
        sub_9 = mul_78 - mul_79
        mul_81 = mul_74 * add_5
        sub_10 = sub_9 - mul_81
        mul_82 = sub_10 * reciprocal_1
        mul_83 = mul_82 * add_12
        sub_11 = add_47 - mul_83
        mul_84 = sub_11 * 5000.0
        mul_85 = mul_84 * reciprocal
        add_48 = mul_5 + -0.02233269627352527
        add_49 = add_48 + mul_8
        mul_88 = add_8 * p
        add_50 = add_49 + mul_88
        mul_89 = p * add_50
        reciprocal_2 = torch.reciprocal(add_34)
        mul_91 = mul_89 * reciprocal_2
        add_51 = mul_91 + 1.0
        log = torch.log(add_51)
        mul_92 = mul_85 * log
        add_52 = sub_3 + mul_92
        mul_93 = ct * add_20
        add_54 = mul_93 + -2.742185394906099e-05
        mul_94 = ct * 6.211426728363857e-10
        add_55 = mul_94 + -1.105097577149576e-07
        mul_95 = sa * add_55
        add_56 = add_54 + mul_95
        mul_96 = sub_10 * reciprocal
        add_57 = add_56 + mul_96
        mul_97 = add_57 * 5000.0
        mul_98 = mul_97 * reciprocal_1
        mul_99 = mul_98 * log
        mul_100 = mul_99 * add_12
        sub_12 = add_52 - mul_100
        mul_102 = mul_97 * reciprocal
        mul_103 = mul_4 * 2.0
        add_58 = mul_103 + -0.000343609007985188
        add_59 = add_58 + mul_19
        mul_105 = add_12 * p
        add_60 = add_59 + mul_105
        mul_106 = p * add_60
        mul_107 = mul_106 * reciprocal_2
        pow_3 = torch.pow(add_34, 2)
        div = mul_89 / pow_3
        mul_108 = div * add_43
        sub_13 = mul_107 - mul_108
        mul_109 = mul_102 * sub_13
        div_1 = mul_109 / add_51
        add_61 = sub_12 + div_1
        mul_110 = ct * 6.743689325042773e-10
        add_62 = mul_110 + 1.119513357486743e-06
        mul_111 = ct * add_62
        add_63 = mul_111 + -2.349607444135925e-05
        mul_112 = ct * add_63
        add_64 = mul_112 + 0.002775927747785646
        mul_113 = mul_110 * 2.0
        add_66 = mul_113 + 1.119513357486743e-06
        mul_114 = ct * add_66
        add_67 = add_63 + mul_114
        mul_115 = ct * add_67
        add_68 = add_64 + mul_115
        mul_116 = ct * -1.811147201949891e-11
        add_69 = mul_116 + 9.527875081696435e-10
        mul_117 = ct * add_69
        add_70 = mul_117 + 1.262937315098546e-07
        mul_118 = ct * add_70
        add_71 = mul_118 + -2.764306979894411e-05
        mul_119 = mul_116 * 2.0
        add_73 = mul_119 + 9.527875081696435e-10
        mul_120 = ct * add_73
        add_74 = add_70 + mul_120
        mul_121 = ct * add_74
        add_75 = add_71 + mul_121
        mul_122 = ct * 2.681097235569143e-12
        add_76 = mul_122 + -4.634182341116144e-11
        mul_123 = ct * add_76
        add_77 = mul_123 + -7.672876869259043e-09
        mul_124 = ct * add_77
        add_78 = mul_124 + 3.801564588876298e-07
        mul_125 = mul_122 * 2.0
        add_80 = mul_125 + -4.634182341116144e-11
        mul_126 = ct * add_80
        add_81 = add_77 + mul_126
        mul_127 = ct * add_81
        add_82 = add_78 + mul_127
        mul_128 = sqrt * add_82
        add_83 = add_75 + mul_128
        mul_129 = sa * add_83
        add_84 = add_68 + mul_129
        mul_130 = mul_63 * 2.0
        mul_131 = add_5 * reciprocal
        mul_132 = mul_130 * mul_131
        mul_133 = mul_71 * 2.0
        mul_134 = mul_133 * mul_131
        add_85 = mul_132 + mul_134
        mul_135 = mul_79 * 2.0
        mul_136 = mul_135 * mul_48
        add_86 = add_85 + mul_136
        mul_138 = mul_135 * add_5
        mul_139 = mul_138 * reciprocal_1
        mul_140 = mul_139 * add_12
        sub_14 = add_86 - mul_140
        mul_141 = add_1 * add_34
        sub_15 = sub_14 - mul_141
        mul_142 = add_17 * add_43
        sub_16 = sub_15 - mul_142
        mul_143 = sub_16 * reciprocal
        add_87 = add_84 + mul_143
        mul_145 = mul_135 * mul_131
        mul_146 = add_17 * add_34
        sub_17 = mul_145 - mul_146
        mul_147 = sub_17 * reciprocal_1
        mul_148 = mul_147 * add_12
        sub_18 = add_87 - mul_148
        mul_149 = sub_18 * add_8
        mul_150 = ct * add_64
        add_89 = mul_150 + 1.0
        mul_151 = ct * add_71
        add_91 = mul_151 + -0.007521448093615448
        mul_152 = sa * 5.41932655114874e-06
        add_92 = add_91 + mul_152
        mul_153 = ct * add_78
        add_94 = mul_153 + -3.303308871386421e-05
        mul_154 = sqrt * add_94
        add_95 = add_92 + mul_154
        mul_155 = sa * add_95
        add_96 = add_89 + mul_155
        mul_156 = sub_17 * reciprocal
        add_97 = add_96 + mul_156
        mul_157 = add_97 * add_12
        add_98 = mul_149 + mul_157
        mul_158 = sub_11 * add_5
        sub_19 = add_98 - mul_158
        mul_159 = add_57 * add_11
        sub_20 = sub_19 - mul_159
        mul_160 = sub_20 * 5000.0
        mul_161 = mul_160 * reciprocal
        mul_162 = add_34 * add_8
        sub_21 = pow_2 - mul_162
        sqrt_1 = torch.sqrt(sub_21)
        reciprocal_3 = torch.reciprocal(sqrt_1)
        mul_164 = mul_88 * 2.0
        sub_22 = add_5 - sqrt_1
        reciprocal_4 = torch.reciprocal(sub_22)
        mul_166 = sqrt_1 * reciprocal_4
        add_103 = add_5 + sqrt_1
        add_104 = add_103 + mul_88
        reciprocal_5 = torch.reciprocal(add_104)
        mul_168 = mul_166 * reciprocal_5
        mul_169 = mul_164 * mul_168
        add_105 = mul_169 + 1.0
        log_1 = torch.log(add_105)
        mul_170 = reciprocal_3 * log_1
        mul_171 = mul_161 * mul_170
        add_106 = add_61 + mul_171
        mul_172 = add_97 * add_8
        mul_173 = add_57 * add_5
        sub_23 = mul_172 - mul_173
        mul_174 = sub_23 * 5000.0
        mul_175 = mul_174 * reciprocal_1
        mul_176 = mul_175 * mul_170
        mul_177 = mul_176 * add_12
        sub_24 = add_106 - mul_177
        mul_178 = sub_23 * reciprocal
        mul_179 = mul_178 * 2500.0
        div_2 = mul_179 / sqrt_1
        div_3 = div_2 / sub_21
        mul_180 = div_3 * log_1
        mul_181 = add_5 * 2.0
        mul_182 = mul_181 * add_11
        mul_183 = add_43 * add_8
        sub_25 = mul_182 - mul_183
        mul_184 = add_34 * add_12
        sub_26 = sub_25 - mul_184
        mul_185 = mul_180 * sub_26
        sub_27 = sub_24 - mul_185
        mul_186 = mul_178 * 5000.0
        mul_187 = mul_186 * reciprocal_3
        mul_188 = mul_105 * 2.0
        mul_189 = mul_188 * mul_168
        mul_190 = mul_88 * reciprocal_3
        mul_191 = mul_190 * reciprocal_4
        mul_192 = mul_191 * reciprocal_5
        mul_193 = mul_192 * sub_26
        add_107 = mul_189 + mul_193
        mul_194 = mul_88 * sqrt_1
        mul_195 = mul_194 * 2.0
        pow_4 = torch.pow(sub_22, 2)
        div_4 = mul_195 / pow_4
        mul_196 = div_4 * reciprocal_5
        mul_197 = reciprocal_3 * sub_26
        div_5 = mul_197 / 2.0
        sub_28 = add_11 - div_5
        mul_198 = mul_196 * sub_28
        sub_29 = add_107 - mul_198
        mul_200 = mul_195 * reciprocal_4
        pow_5 = torch.pow(add_104, 2)
        div_6 = mul_200 / pow_5
        add_112 = add_11 + div_5
        add_113 = add_112 + mul_105
        mul_201 = div_6 * add_113
        sub_30 = sub_29 - mul_201
        mul_202 = mul_187 * sub_30
        div_7 = mul_202 / add_105
        add_114 = sub_27 + div_7
        out[tile] = add_114.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    pressure, temperature, salinity = inputs[:3]
    out = gsw_dhdt(temperature, salinity, pressure)
    return (out.view(temperature.shape),)
