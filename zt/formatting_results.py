import os

# 定义文件夹路径和对应的前缀
folders = {
    "/home/zt/PapersCode/sam2-main/result/HSI-VIS": "vis",
    "/home/zt/PapersCode/sam2-main/result/HSI-NIR": "nir",
    "/home/zt/PapersCode/sam2-main/result/HSI-RedNIR": "rednir",
}

# 目标文件夹
output_folder = "/home/zt/PapersCode/sam2-main/result/zt"
# 确保目标文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历每个文件夹
for folder_path, prefix in folders.items():
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        continue
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否是 .txt 文件
        if filename.endswith(".txt"):
            old_path = os.path.join(folder_path, filename)
            new_filename = f"{prefix}-{filename}"
            new_path = os.path.join(output_folder, new_filename)
            
            # 复制文件并重命名
            with open(old_path, 'r') as src, open(new_path, 'w') as dst:
                dst.write(src.read())
            print(f"已保存: {filename} -> {new_filename}")
    
    print("===============================================")
    print(f"{folder_path} 目录下所有 .txt 文件已复制并重命名！")
    print("===============================================")



