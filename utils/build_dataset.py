import os
import yaml
import numpy as np
from scapy.all import PcapReader
from tqdm import tqdm
from utils.preprocess import packet_to_fixed_bytes, bytes_to_image

##get_class_map(cfg)函数的作用是生成当前任务对应的标签映射规则
##在此函数中，定义了三种工作模式及其工作规则：二分类，十分类二十分类
def get_class_map(cfg):#cfg为配置字典，即config内的参数
    normal_10 = cfg["dataset_map"]["normal_10"]#从config中取出正常样本类别标签合集
    malware_10 = cfg["dataset_map"]["malware_10"]
    task_mode = cfg["task"]["mode"]#读取任务模式，20分类

    if task_mode == "binary":
        class_map = {}#创建二分类的标签映射表
        for k in normal_10.keys():#找到normal_10中的所有类别，每一类映射0
            class_map[k] = 0
        for k in malware_10.keys():
            class_map[k] = 1
        num_classes = 2#类别数 2分类

    elif task_mode == "10class":#10分类只对正常的做分类，再config内的映射成0~9
        class_map = normal_10.copy()
        num_classes = 10

    elif task_mode == "20class":#20分类分别将正常映射成0~9，恶意映射成10~19，再config内已完成映射
        class_map = {}
        class_map.update(normal_10)
        class_map.update(malware_10)
        num_classes = 20

    else:
        raise ValueError("task.mode must be binary / 10class / 20class")

    return class_map, num_classes, normal_10, malware_10

#find_pcap_files(root_dir)函数用于根据目录查找所有pcap包，合成pcap包的文件路径
def find_pcap_files(root_dir):#输入为根目录
    """
    递归查找抓包文件，兼容:
    .pcap / .cap / .pcapng
    """
    valid_exts = (".pcap", ".cap", ".pcapng")#标定识别文件的类型
    pcap_paths = []#保存文件路劲

    for current_root, _, files in os.walk(root_dir):#按顺序打开或识别根目录内的子文件
        for fname in files:#打开或识别子文件内的子文件
            if fname.lower().endswith(valid_exts):#判断文件后缀是否合法
                pcap_paths.append(os.path.join(current_root, fname))#合法文件拼接文件路劲

    pcap_paths.sort()#按照开包顺序排序
    return pcap_paths

##parse_pcap_to_samples_stream（）函数将单个pacp包中的每一个数据包提取出来，逐包遍历，将每个包打开并处理成特征序列和特征灰度图像，并打包成样本字典，
# 字典中包含seq序列标签和img图像标签，label类别标签，和样本来源"group_id"
def parse_pcap_to_samples_stream(
    pcap_path,
    label,
    group_id,
    seq_len,
    image_size,
    max_packets_per_pcap=-1
):
    """
    低内存流式读取 pcap
    """
    samples = []#空列表，存放样本
    packet_count = 0#初始化包数计数器

    try:
        with PcapReader(pcap_path) as pcap_reader:#按路径找到网站pcap，依次读取
            for pkt in pcap_reader:#找到并对pcap的每个数据包进行处理
                try:
                    raw_bytes = bytes(pkt)#原始流量转换成unit8
                    seq = packet_to_fixed_bytes(raw_bytes, seq_len=seq_len)#unit8数据流处理成定长序列
                    img = bytes_to_image(seq, image_size=image_size)#序列变成方阵

                    samples.append({
                        "seq": seq.astype(np.int64),
                        "img": img.astype(np.float32),
                        "label": int(label),
                       "group_id": str(group_id)
                    })#创建样本字典列表，按读取的数据包顺序依次添加

                    packet_count += 1#记录报数
                    if max_packets_per_pcap > 0 and packet_count >= max_packets_per_pcap:
                        break

                except Exception as e:
                    print(f"[WARN] packet parse error in {pcap_path}: {e}")
                    continue

    except Exception as e:
        print(f"[WARN] pcap read error: {pcap_path}, reason: {e}")

    return samples#返回单个pcap的所有样本列表

##把样本列表保存在npz文件内
def save_samples_npz(samples, save_path):#输入为全量样本和config样本保存路径
    if len(samples) == 0:
        raise ValueError("No samples to save.")

    seqs = np.stack([s["seq"] for s in samples], axis=0)#堆叠样本序列组并保存在seqs
    imgs = np.stack([s["img"] for s in samples], axis=0)
    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    groups = np.array([s["group_id"] for s in samples])

    np.savez_compressed(save_path, seq=seqs, img=imgs, label=labels, group_id=groups)#保存全量样本


#思路：先找子文件内的pcap包，再找同名目录下的pcap包，拼接全量样本
def process_class_path(
    class_name,
    current_label,
    base_dir,
    prefix,
    seq_len,
    image_size,
    max_packets_per_pcap,
    all_samples,
    max_total_samples
):
    """
    统一处理一个类别:
    - 如果有同名单文件，优先读单文件
    - 否则递归读取同名目录内的所有抓包文件
    """
    pcap_file_path = os.path.join(base_dir, f"{class_name}.pcap")#假设根目录中有直接的某类别的pcap包，那么构造某类别pcap包的路径
    cap_file_path = os.path.join(base_dir, f"{class_name}.cap")
    pcapng_file_path = os.path.join(base_dir, f"{class_name}.pcapng")
    dir_path = os.path.join(base_dir, class_name)#若根目录中没有现成的某类别的pcap包，那就找某类别的同名子文件，子文件找pcap包，构造目录路径

    def reach_global_limit():
        return max_total_samples > 0 and len(all_samples) >= max_total_samples

    # 1) 优先处理单文件类
    single_file = None#单文件变量设为空，初始化
    if os.path.isfile(pcap_file_path):#若实际文件中有pcap单文件的路劲能够与构造的路径匹配，则判断文件存在，并将路劲保存在变量中
        single_file = pcap_file_path
    elif os.path.isfile(cap_file_path):
        single_file = cap_file_path
    elif os.path.isfile(pcapng_file_path):
        single_file = pcapng_file_path

    if single_file is not None:#如果有文件，则打印正在处理文件，并将处理的文件的全量样本数据保存在samples内存
        print(f"[INFO] Processing {prefix.lower()} file: {single_file}")
        samples = parse_pcap_to_samples_stream(
            pcap_path=single_file,
            label=current_label,
            group_id=f"{prefix}_{class_name}",
            seq_len=seq_len,
            image_size=image_size,
            max_packets_per_pcap=max_packets_per_pcap
        )
        all_samples.extend(samples)#将每个pcap包的所有样本拼接成全量样本

        if len(samples) == 0:
            print(f"[WARN] No valid samples extracted from file: {single_file}")

        if reach_global_limit():
            del all_samples[max_total_samples:]

        return

    # 2) 否则处理目录类
    if os.path.isdir(dir_path):#对于根目录内的子文件，若实际有文件路劲与构造文件的路径匹配
        print(f"[INFO] Processing {prefix.lower()} folder: {dir_path}")
        pcap_files = find_pcap_files(dir_path)#找到子文件内的pcap文件，并将路径赋值给内存
        print(f"[INFO] Found {len(pcap_files)} capture files under {dir_path}")

        if len(pcap_files) == 0:
            print(f"[WARN] No .pcap/.cap/.pcapng files found in folder: {dir_path}")
            return

        class_sample_count = 0#初始化当前类别样本数

        for pcap_path in tqdm(pcap_files, desc=f"{class_name}", leave=False):#逐个获取当前文件路径下的所有pcpa包，将包路径保存在内存中
            if reach_global_limit():
                break

            rel_name = os.path.relpath(pcap_path, dir_path)#计算相对路径
            rel_name = rel_name.replace("\\", "_").replace("/", "_")

            samples = parse_pcap_to_samples_stream(
                pcap_path=pcap_path,
                label=current_label,
                group_id=f"{prefix}_{class_name}_{rel_name}",#样本来源为类别路径下的相对路径
                seq_len=seq_len,
                image_size=image_size,
                max_packets_per_pcap=max_packets_per_pcap
            )

            all_samples.extend(samples)#目录类的pcap样本加入放到全量样本
            class_sample_count += len(samples)

            if reach_global_limit():
                del all_samples[max_total_samples:]
                break

        if class_sample_count == 0:
            print(f"[WARN] Folder processed but no valid samples extracted: {dir_path}")
        else:
            print(f"[INFO] Collected {class_sample_count} samples for class: {class_name}")

        return

    # 3) 文件和目录都不存在
    print(f"[WARN] class not found: {class_name} under {base_dir}")

##build_dataset()函数是数据构建入口
##逻辑：1、从config中读取配置文件，得到各种全局参数  2、取出关键参数，如序列尺寸图像尺寸 3、创建输出目录 4、生成标签映射：十分类、二十分类的标签
  #    5、处理 Benign 类别， 6、处理 Malware 类别  7、第七步：汇总统计 8、保存到 all_samples.npz

def build_dataset():
    with open("configs/config_seq_only.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]#原始数据目录
    processed_dir = cfg["data"]["processed_dir"]#处理后数据目录
    all_file = cfg["data"]["all_file"]#全量样本npz文件名
    seq_len = cfg["data"]["seq_len"]#序列长度
    image_size = cfg["data"]["image_size"]#图片尺寸

    max_packets_per_pcap = cfg["data"].get("max_packets_per_pcap", -1)#读取所有pcap最大包数5000
    max_total_samples = cfg["data"].get("max_total_samples", -1)#样本数不限，总样本数小于或等于5000xpcap数

    os.makedirs(processed_dir, exist_ok=True)#创建全量样本输出路径
    save_path = os.path.join(processed_dir, all_file)#拼接全量样本保存路径

    class_map, num_classes, normal_10, malware_10 = get_class_map(cfg)

    all_samples = []

    benign_root = os.path.join(raw_dir, "Benign")#构造正常类路径
    malware_root = os.path.join(raw_dir, "Malware")#构造恶意类路径

    # 1. 处理 Benign
    for class_name, label in normal_10.items():#找到并获取类别名和对应的映射值
        if max_total_samples > 0 and len(all_samples) >= max_total_samples:
            print("[INFO] Reached global sample limit, stop processing benign.")
            break

        current_label = label if cfg["task"]["mode"] != "binary" else 0

        process_class_path(
            class_name=class_name,
            current_label=current_label,
            base_dir=benign_root,
            prefix="Benign",
            seq_len=seq_len,
            image_size=image_size,
            max_packets_per_pcap=max_packets_per_pcap,
            all_samples=all_samples,
            max_total_samples=max_total_samples
        )#将正常类所有样本拼接到全量样本

    # 2. 处理 Malware
    if cfg["task"]["mode"] != "10class":
        for class_name, label in malware_10.items():
            if max_total_samples > 0 and len(all_samples) >= max_total_samples:
                print("[INFO] Reached global sample limit, stop processing malware.")
                break

            current_label = label if cfg["task"]["mode"] != "binary" else 1

            process_class_path(
                class_name=class_name,
                current_label=current_label,
                base_dir=malware_root,
                prefix="Malware",
                seq_len=seq_len,
                image_size=image_size,
                max_packets_per_pcap=max_packets_per_pcap,
                all_samples=all_samples,
                max_total_samples=max_total_samples
            )#恶意类样本全拼接

    print(f"Total samples collected: {len(all_samples)}")#输出全量样本数量

    if len(all_samples) == 0:
        raise ValueError("No samples to save, please check data/raw directory structure and class names.")

    save_samples_npz(all_samples, save_path)
    print(f"[INFO] Saved processed dataset to: {save_path}")

    labels = np.array([s["label"] for s in all_samples], dtype=np.int64)#提取标签0~19
    print("[INFO] Final label distribution:", np.unique(labels, return_counts=True))