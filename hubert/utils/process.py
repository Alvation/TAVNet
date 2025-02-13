import os

# 指定原始TSV文件所在目录
input_directory = '/workspace/AVTSR/hubert/30h_data'
# 指定新TSV文件保存的目录
output_directory = '/workspace/AVTSR/hubert/30h_data/processed'

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 处理的文件名列表
filenames = ['train.tsv', 'test.tsv', 'valid.tsv']

# 遍历处理每一个文件
for filename in filenames:
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        # 读取并写入根目录标识
        root_directory = infile.readline().strip()
        outfile.write(root_directory + '\n')

        # 处理接下来的每一行
        for line in infile:
            parts = line.strip().split()
            # 只提取第一列、第三列和最后一列
            new_line = f"{parts[2]}\t{parts[-1]}\n"
            outfile.write(new_line)

print("处理完毕，新文件已保存在指定的输出目录。")