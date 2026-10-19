import csv
import re
import os
import subprocess

try:
    gc_freq = float(input('请输入GC频率(MHz)，默认1500: ') or 1500)
except ValueError:
    gc_freq = 1500

# 自动运行shell命令，生成pattern输出文件
shell_cmds = [
    f'MODEL={int(gc_freq)} ./run_pattern.sh -p 1 -n > pattern1_output.txt',
    './run_pattern.sh -p 3 -n > pattern3_output.txt',
    './run_pattern.sh -p 4 -n > pattern4_output.txt',
    './run_pattern.sh -p 7 -n > pattern7_output.txt',
]

for cmd in shell_cmds:
    print(f'正在执行: {cmd}')
    subprocess.run(cmd, shell=True)

# 获取用户输入

try:
    mem_freq = float(input('请输入内存频率(MHz)，默认1800: ') or 1800)
except ValueError:
    mem_freq = 1800
try:
    cu_num = int(input('请输入CU个数，默认80: ') or 80)
except ValueError:
    cu_num = 80

# 理论值计算公式（示例，实际请根据硬件公式调整）
THEORY = mem_freq * 1024 / 1000  # 示例公式
L1_read_theory = gc_freq * cu_num * 64 / 1000
L2_read_theory = gc_freq * 128 * 32 / 1000
L2_write_theory = gc_freq * 64 * 32 / 1000
LDS_read_theory = gc_freq * cu_num * 128 / 1000
LDS_write_theory = gc_freq * cu_num * 128 / 1000
# 你可以根据实际需求修改上面各项理论值

def get_theory(name):
    if name == 'L1_read':
        return L1_read_theory
    elif name == 'L2_read':
        return L2_read_theory
    elif name == 'L2_write':
        return L2_write_theory
    elif name == 'LDS_read':
        return LDS_read_theory
    elif name == 'LDS_write':
        return LDS_write_theory
    else:
        return THEORY

# pattern1/3的测试项顺序
pattern1_names = [
    'L1_read', 'L2_read', 'L2_write', 'LDS_read', 'LDS_write',  
    'memory_read', 'memory_write', 'memory_copy',
    'memory_to_lds', 'lds_to_memory', 
    'matrix_read', 'matrix_write', 'matrix_copy',
    'mapping_matrix_write', 'tiling_read',
    'memory_read_16wave', 'memory_write_16wave', 'memory_copy_16wave',
    'memory_read_4wave_64x4',
    '',  # 空行
    '',  # 空行
]

pattern3_names = [
    'memory_read', 'memory_write', 'memory_copy',
    'matrix_read', 'matrix_write', 'matrix_copy',
    '',  # 空行
    'memory_read_8wave', 'memory_write_8wave', 'memory_copy_8wave',
    'matrix_read_8wave', 'matrix_write_8wave', 'matrix_copy_8wave',
    '',  # 空行
    'memory_read_16wave', 'memory_write_16wave', 'memory_copy_16wave',
    'matrix_read_16wave', 'matrix_write_16wave', 'matrix_copy_16wave',
    '',  # 空行
]

# pattern4的测试项顺序
pattern4_names = [
    'memory_read_4wave', 'memory_read_4wave_8', 'memory_read_4wave_16',
    '',
    'memory_read_2wave_2', 'memory_read_2wave_2_8', 'memory_read_2wave_2_16',
    '',
    'memory_read_1wave_4', 'memory_read_1wave_4_8', 'memory_read_1wave_4_16',
    '',
    'memory_read_4wave_16', 'memory_read_2wave_16', 'memory_read_1wave_16',
    '',  # 空行
]

pattern7_names = [
    'memory_read_4wave_4globalx4-6K LDS',
    'memory_read_4wave_4globalx4-8K LDS',
    'memory_read_4wave_4globalx4-9K LDS',
    'memory_read_4wave_4globalx4-10K LDS',
    'memory_read_4wave_4globalx4-12K LDS',
    'memory_read_4wave_4globalx4-16K LDS',
    'memory_read_4wave_4globalx4-20K LDS',
    'memory_read_4wave_4globalx4-32K LDS',
    'memory_read_4wave_4globalx4-64K LDS',
    '',  # 空行
]

pattern_files = [
    ('pattern1_output.txt', pattern1_names),
    ('pattern3_output.txt', pattern3_names),   #4wave 8wave 16wave
    ('pattern4_output.txt', pattern4_names),   #dwordx1 dwordx2 dwordx4
    ('pattern7_output.txt', pattern7_names),   #LDS TG/CU
]

rows = []
for fname, name_list in pattern_files:
    if not os.path.exists(fname):
        print(f'警告: {fname} 不存在，跳过')
        continue
    values = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if re.match(r'^-?\d+(\.\d+)?$', line):
                values.append(float(line))
    vi = 0
    for name in name_list:
        if name == '':
            rows.append(['', '', '', ''])
        else:
            theory_val = get_theory(name)
            if vi >= len(values):
                rows.append([name, f'{theory_val:,.2f}', '', ''])
            else:
                real = values[vi]
                percent = f'{real / theory_val * 100:.2f} %'
                rows.append([name, f'{theory_val:,.2f}', f'{real:.2f}', percent])
                vi += 1

with open('all_patterns.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['测试项', '80CU理论值', '实测值', '效率'])
    for row in rows:
        writer.writerow(row)

print('已生成 all_patterns.csv') 