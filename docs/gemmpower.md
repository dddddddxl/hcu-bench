# 可选 gemmpower EDPP 有限循环

归属operators，不是新suite；主模板及专用模板均enabled:false，不随默认all/operators启动。
显式选择operators并不会自动开启disabled块；必须由用户主动启用。

```bash
# 先在YAML填好仓库/输出目录、明确分配且空闲的单张HCU；在准备好的Linux DTK环境中：
bash bench.sh configs/gemmpower.template.yaml operators --set tests.gemmpower.enabled=true
```

随包固定官方HYGON-AI/hcu-dcgm提交b3d37387be4264fa25325922d01dd056ef999a64的
NMZ/BW1100 gfx938 loader gemmPower_edpp和fp8_nmz_edpp.co，原始LICENSE随附，文件SHA256固定校验。
来源：[官方目录](https://github.com/HYGON-AI/hcu-dcgm/tree/b3d37387be4264fa25325922d01dd056ef999a64/pkg/dcgm/resources/gemmpoweredpp/nmz)。
不下载运行库、不改变频率/电压/功耗限制/宿主NUMA；原件只读，仅复制自有loader并加执行权限。

先用已有hipcc/SDK在case目录编译只读HIP设备探针（不依赖Torch），确认恰好一张可见设备并读架构/PCI。
不是gfx938时记skipped，在调用loader或加载co前停止，不视为通过；环境缺失/探针失败记失败。
拒绝HSA_OVERRIDE等架构伪装与冲突可见性设置，不能只改YAML让这份co在gfx92a上运行。
换架构须拿到对应工具/co或可编译的负载源码，并新增独立验证及哈希，当前不支持自动重编译kernel。

固定m2048/n2048/k12032/g22/t1/-w/-vs，默认-l4、原生超时30秒，外层240秒包含探针编译。
g22是GEMM类型而非卡号，节点hcus负责选卡。OMP4/PASSIVE仅作用于本测试。
目前仅集成有限循环smoke，iterations可显式配置1..1000；大次数需要用户同步增加超时。
超时/取消/非零退出/循环不完整/原生error均失败，不把计划杀进程当自然完成。

记录执行命令、设备、输入哈希、原生日志和前/中/后可读sysfs功耗/温度/利用率/显存/RAS原始值。
各hwmon字段按原路径保存，功率通常为微瓦、温度通常为毫摄氏度，不盲目将某个power字段命名板卡功耗；
传感器不可读时不编造0值。标准指标仅为原生进程墙钟时间，含CPU参考/初始化，不冒充GPU计算时间或TFLOPS。
没有稳态/峰值功耗指标，也未集成此前180秒采样的完整验收流程；短窗口传感器数据不用于长稳结论。
correctness=not_checked，原生CPU CHECK耗时行不是数值通过证明。

这是官方EDPP负载，不是Excel原始fp8_nmz_mmac.v9.co，也没有对应完整kernel源码。
此前同一输入在BW1100/gfx938有限循环自然退出0；gfx92a曾模块加载成功但查找内核符号失败，不能据此跨架构使用。
本次仓库接入的CPU合同测试不等同于重新完成实机功耗或峰值性能验收。
