import os

# 定义目标目录
target_dir = "/home/zt/PapersCode/sam2-main/result/zt-sam2.1-test5-0.6549-0.8649"

# 初始化计数器
zero_file_count = 0

# 存储包含全零行的文件名
zero_file_names = []

# 遍历目标目录下的所有文件
for filename in os.listdir(target_dir):
    # 检查文件是否为 .txt 文件
    if filename.endswith(".txt"):
        file_path = os.path.join(target_dir, filename)
        
        # 打开文件并逐行读取
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                
                # 检查是否为全零行
                if line == "0\t0\t0\t0":
                    zero_file_count += 1
                    zero_file_names.append(filename)  # 记录文件名
                    break  # 找到全零行后跳出循环，继续下一个文件

# 输出结果
print(f"包含全零行的 .txt 文件数量: {zero_file_count}")
print("包含全零行的 .txt 文件名称:")
for name in zero_file_names:
    print(name)