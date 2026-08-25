# 音速轨迹数值验证报告

总判定: PASS (21/21)

| 检查 | 结果 | 备注 |
|---|---|---|
| 锚点 beta_CJ = 30.384 度 | PASS | 30.384156 |
| 锚点 theta_CJ = 11.006 度（交接值系 11.0055 的进位舍入） | PASS | 11.005497 |
| 锚点 theta_max = 40.3829 度 | PASS | 40.382949 |
| 锚点 beta_max = 67.4017 度 | PASS | 67.401690 |
| 锚点参数恰有 1 个合法音速点 | PASS | count=1 |
| 高压缩支全段恰有 1 个音速穿越 | PASS | count=1 |
| 闭式 beta_s 与 Brent 之差 < 1e-9 度 | PASS | err=1.272e-14 度 |
| 闭式 theta_s 与 Brent 之差 < 1e-9 度 | PASS | err=0.000e+00 度 |
| 音速点在高压缩支 (R>0) | PASS | R=34.7515 |
| 音速点在弱支：beta_s < beta_max | PASS | beta_s=67.152656 < beta_max=67.401690 |
| theta_CJ < theta_s < theta_max | PASS | 11.0055 < 40.380691 < 40.382949 |
| 脱体点处 M2 < 1 | PASS | M2=0.990632 |
| CJ 点 M2n = 1 | PASS | M2n=1.00000000000000 |
| CJ 点 M2 > 1（切向分量） | PASS | M2=3.013775 |
| 强支全段 M2 < 1 | PASS | max M2=0.990632 |
| 低压缩支全段 M2 > 1 | PASS | min M2=3.013808 |
| 扫描全部参数组均恰有 1 个高压缩支音速点 | PASS | 288/288 组有音速点; 多根组数=0 |
| 扫描全域 theta_CJ < theta_s < theta_max | PASS | 违例 0/288 |
| 扫描全域音速点在弱支 (beta_s < beta_max) | PASS | 违例 0/288 |
| 渐近残差随 M 单调衰减（O(1/m) 收敛） | PASS | M=200 时最大 |残差|=1.736e-03 |
| 隐式二次因子在音速点上的归一化残差 < 1e-8 | PASS | max=1.787e-13 |

关键数值（gamma=1.3, M=7, Q=10）：
- u_s = 41.612799988981
- beta_s = 67.15265586 度
- theta_s = 40.38069105 度
- theta_max - theta_s = 0.002258 度
- 脱体点 M2 = 0.990632
- CJ 点 M2 = 3.013775

输出: sonic_scan.csv, sonic_asymptotics.csv, sonic_locus.png