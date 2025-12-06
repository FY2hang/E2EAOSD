import os, glob, cv2, tqdm
from prettytable import PrettyTable
import numpy as np # 用于方便地计算最大值

# 颜色常量
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

# --- 数据集配置 ---
image_postfix = ['jpg', 'png', 'bmp', 'tif']
# images_folder_path = [
#                     #   '/home/featurize/zfy/dataset/VisDrone/images/train', 
#                     #   '/home/featurize/zfy/dataset/VisDrone/images/val',
#                       '/home/featurize/zfy/dataset/VisDrone/images/test']
# labels_folder_path = [
#                     #   '/home/featurize/zfy/dataset/VisDrone/labels/train',
#                     #   '/home/featurize/zfy/dataset/VisDrone/labels/val',
#                       '/home/featurize/zfy/dataset/VisDrone/labels/test']
# # 数据集类别列表
# classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
# images_folder_path = [
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/images/train', 
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/images/val',
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/images/test']
# labels_folder_path = [
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/labels/train',
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/labels/val',
#                       '/home/featurize/zfy/dataset/AI-TOD/aitodtoolkit/aitod/labels/test']
# # 数据集类别列表
# # classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
# classes = ['airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle', 'person', 'wind-mill']

# images_folder_path = [
#                       '/home/featurize/zfy/dataset/SIMD/images/train', 
#                       '/home/featurize/zfy/dataset/SIMD/images/val',
#                       '/home/featurize/zfy/dataset/SIMD/images/test']
# labels_folder_path = [
#                       '/home/featurize/zfy/dataset/SIMD/labels/train',
#                       '/home/featurize/zfy/dataset/SIMD/labels/val',
#                       '/home/featurize/zfy/dataset/SIMD/labels/test']
# # 数据集类别列表
# classes = ['car', 'truck', 'van', 'longvehicle', 'bus', 'airliner', 'propeller', 'trainer', 'chartered', 'fighter', 'other', 'stairtruck', 'pushbacktruck', 'helicopter', 'boat']

images_folder_path = [
                      '/home/featurize/zfy/dataset/DOTA_YOLO_split/images/train',
                      '/home/featurize/zfy/dataset/DOTA_YOLO_split/images/val',
                    #   '/home/featurize/zfy/dataset/DOTA_YOLO_split/images/test']
]
labels_folder_path = [
                      '/home/featurize/zfy/dataset/DOTA_YOLO_split/labels/train',
                      '/home/featurize/zfy/dataset/DOTA_YOLO_split/labels/val',
                    #   '/home/featurize/zfy/dataset/DOTA_YOLO_split/labels/test']
]
# 数据集类别列表
# classes = ['car', 'truck', 'van', 'longvehicle', 'bus', 'airliner', 'propeller', 'trainer', 'chartered', 'fighter', 'other', 'stairtruck', 'pushbacktruck', 'helicopter', 'boat']
# classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
classes = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool']
 

# 目标大小划分边界 (COCO 标准): S < 32*32, 32*32 <= M <= 96*96, L > 96*96
object_info = [32*32, 96*96] 
# --------------------


def get_images_and_labels_path(images_folder_path, labels_folder_path):
    """
    扫描图片和标签路径，并将有对应标签的图片路径进行匹配。
    """
    labels_filename = {}
    for folder_path in labels_folder_path:
        glob_list = glob.glob(os.path.join(folder_path, '*.txt'))
        filename = {os.path.splitext(os.path.basename(i))[0]:i for i in glob_list}
        labels_filename.update(filename)
    
    images_filename = {}
    for folder_path in images_folder_path:
        for p in image_postfix:
            glob_list = glob.glob(os.path.join(folder_path, f'*.{p}'))
            filename = {os.path.splitext(os.path.basename(i))[0]:i for i in glob_list}
            images_filename.update(filename)
    
    print(ORANGE + f'image_path_length:{len(images_filename)} label_path_length:{len(labels_filename)}' + RESET)

    image_label_dict = {}
    for i in labels_filename:
        if i in images_filename:
            image_label_dict[labels_filename[i]] = images_filename[i]
    
    print(GREEN + f'After matching. data_length:{len(image_label_dict)}' + RESET)

    return image_label_dict

def show_dataset_info(image_label_dict):
    """
    统计数据集信息，并找出拥有最多 S/M/L 数量和总标注数量的图像。
    """
    # 1. 类别统计字典
    classes_dict = {cls:{'s':0, 'm':0, 'l':0, 'num':0} for cls in classes}
    
    # 2. 图片统计列表 (用于计算平均值)
    image_size_stats = [] 
    
    # 3. 最大值跟踪字典 (新增)
    max_stats = {
        'max_s': {'count': -1, 'path': ''},
        'max_m': {'count': -1, 'path': ''},
        'max_l': {'count': -1, 'path': ''},
        'max_total': {'count': -1, 'path': ''}
    }

    print(YELLOW + 'Starting dataset statistics and finding maximum count images...' + RESET)
    
    for label_path in tqdm.tqdm(image_label_dict, desc='Processing labels'):
        image_path = image_label_dict[label_path]

        image = cv2.imread(image_path)
        try:
            h, w = image.shape[:2]
        except:
            print(RED + f'{image_path} read failure. skip.' + RESET)
            continue
        
        with open(label_path) as f:
            label = list(map(lambda x:x.strip().split(), f.readlines()))
        
        current_image_stats = {'s': 0, 'm': 0, 'l': 0}
        
        for cls_id_str,x_c,y_c,width_norm,height_norm in label:
            cls_id = int(float(cls_id_str))
            
            if cls_id >= len(classes):
                continue

            category_name = classes[cls_id]
            
            width_pix = float(width_norm) * w
            height_pix = float(height_norm) * h
            obj_area = width_pix * height_pix

            # --- 类别统计 ---
            classes_dict[category_name]['num'] += 1
            
            # --- 图片和类别同时进行 S/M/L 统计 ---
            if obj_area < object_info[0]: 
                classes_dict[category_name]['s'] += 1
                current_image_stats['s'] += 1 
            elif obj_area > object_info[1]: 
                classes_dict[category_name]['l'] += 1
                current_image_stats['l'] += 1 
            else:
                classes_dict[category_name]['m'] += 1
                current_image_stats['m'] += 1 
        
        # 将当前图片的统计结果添加到列表
        current_total = sum(current_image_stats.values())
        if current_total > 0:
             image_size_stats.append(current_image_stats)

        # --- 更新最大值跟踪 (新增逻辑) ---
        
        # 1. 最多小目标 (S) 图像
        if current_image_stats['s'] > max_stats['max_s']['count']:
            max_stats['max_s'] = {'count': current_image_stats['s'], 'path': image_path}
            
        # 2. 最多中目标 (M) 图像
        if current_image_stats['m'] > max_stats['max_m']['count']:
            max_stats['max_m'] = {'count': current_image_stats['m'], 'path': image_path}
            
        # 3. 最多大目标 (L) 图像
        if current_image_stats['l'] > max_stats['max_l']['count']:
            max_stats['max_l'] = {'count': current_image_stats['l'], 'path': image_path}

        # 4. 最多总标注 (Total) 图像
        if current_total > max_stats['max_total']['count']:
            max_stats['max_total'] = {'count': current_total, 'path': image_path}


    # --- 1. 类别统计结果展示 (保持不变) ---
    print(BLUE + "\n--- 1. Per-Class Object Size Distribution ---\n" + RESET)
    total_s = sum(v['s'] for v in classes_dict.values())
    total_m = sum(v['m'] for v in classes_dict.values())
    total_l = sum(v['l'] for v in classes_dict.values())
    total_num = sum(v['num'] for v in classes_dict.values())

    table_class = PrettyTable()
    table_class.title = "Dataset Object Size Distribution (YOLO Format)"
    table_class.field_names = ["Category", "Small (s)", "Medium (m)", "Large (l)", "Total (num)"]

    for category, values in classes_dict.items():
        s, m, l, num = values['s'], values['m'], values['l'], values['num']
        
        if num == 0:
            s_percent, m_percent, l_percent = "0.0%", "0.0%", "0.0%"
        else:
            s_percent = f"{s/num:.1%}"
            m_percent = f"{m/num:.1%}"
            l_percent = f"{l/num:.1%}"
            
        row = [category, f"{s} ({s_percent})", f"{m} ({m_percent})", f"{l} ({l_percent})", num]
        table_class.add_row(row)

    if total_num == 0:
        total_s_percent, total_m_percent, total_l_percent = "0.0%", "0.0%", "0.0%"
    else:
        total_s_percent = f"{total_s/total_num:.1%}"
        total_m_percent = f"{total_m/total_num:.1%}"
        total_l_percent = f"{total_l/total_num:.1%}"

    row_total = ["All", f"{total_s} ({total_s_percent})", f"{total_m} ({total_m_percent})", f"{total_l} ({total_l_percent})", total_num]
    table_class.add_row(row_total)
    table_class.align["Category"] = "l"

    print(table_class.get_string())
    
    
    # --- 2. 图像最大目标数量统计 (新增输出) ---
    print(BLUE + "\n--- 2. Images with Maximum Object Counts ---\n" + RESET)
    
    table_max = PrettyTable()
    table_max.title = "Maximum Object Counts in a Single Image"
    table_max.field_names = ["Max Count Type", "Count", "Image Path"]
    table_max.align = "l"
    
    # 填充表格数据
    table_max.add_row(["Max Small (S) Count", max_stats['max_s']['count'], max_stats['max_s']['path']])
    table_max.add_row(["Max Medium (M) Count", max_stats['max_m']['count'], max_stats['max_m']['path']])
    table_max.add_row(["Max Large (L) Count", max_stats['max_l']['count'], max_stats['max_l']['path']])
    table_max.add_row(["Max Total Annotations", max_stats['max_total']['count'], max_stats['max_total']['path']])

    print(table_max.get_string())
    
    
    # --- 3. 每张图片平均目标数量统计 (保持不变) ---
    print(BLUE + "\n--- 3. Average Objects Per Image ---\n" + RESET)
    
    num_images_with_objects = len(image_size_stats)
    
    if num_images_with_objects > 0:
        total_s_per_image = sum(d['s'] for d in image_size_stats)
        total_m_per_image = sum(d['m'] for d in image_size_stats)
        total_l_per_image = sum(d['l'] for d in image_size_stats)
        
        avg_s = total_s_per_image / num_images_with_objects
        avg_m = total_m_per_image / num_images_with_objects
        avg_l = total_l_per_image / num_images_with_objects
        avg_total = (total_s_per_image + total_m_per_image + total_l_per_image) / num_images_with_objects

        table_avg = PrettyTable()
        table_avg.title = f"Average Object Count Per Image (Total Images with Objects: {num_images_with_objects})"
        table_avg.field_names = ["Size Category", "Average Count Per Image"]
        table_avg.align["Size Category"] = "l"
        table_avg.align["Average Count Per Image"] = "r"

        table_avg.add_row(["Small (s)", f"{avg_s:.2f}"])
        table_avg.add_row(["Medium (m)", f"{avg_m:.2f}"])
        table_avg.add_row(["Large (l)", f"{avg_l:.2f}"])
        table_avg.add_row(["Total Objects", f"{avg_total:.2f}"])
        
        print(table_avg.get_string())
    else:
        print(ORANGE + "No images with objects found in the dataset to calculate average counts." + RESET)


if __name__ == '__main__':
    image_label_dict = get_images_and_labels_path(images_folder_path, labels_folder_path)
    
    show_dataset_info(image_label_dict)