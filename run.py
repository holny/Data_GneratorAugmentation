import argparse
import os
import hwr_data
total_line_augment = "lpbnecfg"
total_char_augment = "mra"
"""
line单行数据增强: l-画线; p-仿射变换; b-模糊(高斯、均匀、中值中随机); n-噪声(高斯噪声、弹性变换中随机); e-(浮雕、对比度中随机); c-像素(Dropout、加减、乘除中随机); 
    f-行倾斜随机角度; g-图片背景;
char单字符数据增强: m-单字符随机上下;  r-单字符随机大小; a-单字符倾斜随机角度; d-对字符数据腐蚀或膨胀一次;
0-用于test_mode 不使用任何数据增强的情况(true_write_type测试)
"""
default_args = {
    "test_mode":True,           # 使用测试模式,用matplotlib查看数据增强效果.效果保存在data/test_mode目录下
    "train_data_num":124,       # 要生成的训练集数据量
    "test_data_num":64,         # 要生成的测试集数据量
    "valid_data_num":32,        # 要生成的验证集数据量
    "write_types":3,            # HCL HWDB 写法(同一行字符生成图片几次)

    "data_augment":"0g",         # 在训练集开启数据增强
    "data_augment_percent":1,         # 在训练集开启数据增强的占比
    "true_write_type":True,           # True-真实化字体处理字符，让字符图片向真实笔记靠拢,建议使用HCL数据集，效果好.
    "true_write_type_ratio":1,           # 真实化字体数据占总体数据比重
    "data_dir":"./data",        # 生成的数据存储目的地址

    "background_dir":"./res/background",        # 训练集添加的背景图片地址
    "default_bg_path":"./res/background/11.png",        # Train&Test&Valid 数据使用的默认图片背景(一般为纯白色背景)

    "off_corpus":False,                         # True-为停止使用语料库生成文本,文本随机生成。
    "corpus_path":"./res/text_dict.txt",        # 语料库地址，需第一次由init_text_check_path构建生成。若已经构建好的，直接指定即可

    "hcl_dir":"/Users/hly/Documents/github/HYL_HWR/hwr/data_generator/res/hcl_writer_rgba/hcl_writer_rgba",     # HCL资源来源地址
    "hwdb_trn_dir":"./res/hwdb/hwdb_train",     # HWDB train set 资源来源地址
    "hwdb_tst_dir":"./res/hwdb/hwdb_test",      # HWDB test set 资源来源地址
    "hcl_ratio":1,                              # 生成的数据中HCL数据所占比(HCL和HWDB混合)
    "line_mix":False,                           # HCL和HWDB混合是行内混合(True)，还是行外混合(False)

    "const_char_num":False,                     # True-固定每行数据中字符个数为max_char_num。(停止使用语料库才有效-off_corpus)
    "max_char_num":13,                          # 每行数据图片中 最大字符个数
    "chars_gap_width":1,                        # 每行数据图片中 字符间距
    "image_height":64,                          # 生成的数据图片 高
    "image_width":0,                            # 生成的数据图片 宽 (宽小于等于100时, 宽度就随机)

    ## 第一次使用需要构建HWDB和语料库，如果已有构建成功的HWDB数据与语料库，直接指定即可 无需 is_first=True
    "is_first":False,
    # 构建目的地址 hwdb_trn_dir
    "init_hwdb_trn_dir":"/Users/hly/Documents/github/HYL_HWR/hwr/data_generator/res/HWDB1.1trn_gnt",
    # 构建目的地址 hwdb_tst_dir
    "init_hwdb_tst_dir":"/Users/hly/Documents/github/HYL_HWR/hwr/data_generator/res/HWDB1.1tst_gnt",
    # 构建目的地址 corpus_path
    "init_text_check_path":"./res/zhonghuayuwen_3755.csv",

}


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description="生成数据&数据增强.(第一次使用(-first)需构建HWDB和语料库,若已有构建好的，直接指定即可)")
    parser.add_argument(
        "-test_mode",
        "--test_mode",
        action="store_true",
        help="使用测试模式,用matplotlib查看数据增强效果.效果保存在data/test_mode目录下",
        default=default_args["test_mode"]
    )
    parser.add_argument(
        "-train",
        "--train_data_num",
        type=int,
        nargs="?",
        help="要生成的训练集数据量",
        default=default_args["train_data_num"]
    )
    parser.add_argument(
        "-test",
        "--test_data_num",
        type=int,
        nargs="?",
        help="要生成测试集数据量",
        default=default_args["test_data_num"]
    )
    parser.add_argument(
        "-valid",
        "--valid_data_num",
        type=int,
        nargs="?",
        help="要生成的验证集数据量",
        default=default_args["valid_data_num"]
    )
    parser.add_argument(
        "-wt",
        "--write_types",
        type=int,
        nargs="?",
        help="HCL HWDB 写法",
        default=default_args["write_types"]
    )

    """
    line单行数据增强: l-画线; p-仿射变换; b-模糊(高斯、均匀、中值中随机); n-噪声(高斯噪声、弹性变换中随机); e-(浮雕、对比度中随机); c-像素(Dropout、加减、乘除中随机).
    char单字符数据增强: m-单字符随机上下;  r-单字符随机大小; a-单字符随机倾斜
    """
    parser.add_argument(
        "-aug",
        "--data_augment",
        type=str,
        nargs="?",
        help="在训练集开启数据增强.line单行数据增强: l-画线; p-仿射变换; b-模糊(高斯、均匀、中值中随机); n-噪声(高斯噪声、弹性变换中随机); e-(浮雕、对比度中随机); c-像素(Dropout、加减、乘除中随机).char单字符数据增强: m-单字符随机上下;  r-单字符随机大小; a-单字符随机倾斜",
        # default=""
        default=default_args["data_augment"]
    )
    parser.add_argument(
        "-data_augment_percent",
        "--data_augment_percent",
        type=float,
        nargs="?",
        help="Train中数据增强数据的比例,默认0.5",
        default=default_args["data_augment_percent"]
    )
    parser.add_argument(
        "-true",
        "--true_write_type",
        action="store_true",
        help="True-真实化处理字符，让字符图片向真实数据字体靠拢.",
        default=default_args["true_write_type"]
    )
    parser.add_argument(
        "-true_ratio",
        "--true_write_type_ratio",
        type=float,
        nargs="?",
        help="真实化字体数据占所有数据的比重,默认0.5.",
        default=default_args["true_write_type_ratio"]
    )
    parser.add_argument(
        "-dir",
        "--data_dir",
        type=str,
        nargs="?",
        help="生成的数据存储目标目录地址",
        default=default_args["data_dir"]
    )

    parser.add_argument(
        "-background_dir",
        "--background_dir",
        type=str,
        nargs="?",
        help="图片背景资源地址",
        default=default_args["background_dir"]
    )
    parser.add_argument(
        "-default_bg_path",
        "--default_bg_path",
        type=str,
        nargs="?",
        help="Train&Test&Valid 数据使用的默认图片背景(一般为纯白色背景).",
        default=default_args["default_bg_path"]
    )

    ## 语料库-begin
    parser.add_argument(
        "-off_corpus",
        "--off_corpus",
        action="store_true",
        help="停止使用语料库来生成数据,默认使用语料库.",
        default=default_args["off_corpus"]
    )
    parser.add_argument(
        "-corpus_path",
        "--corpus_path",
        type=str,
        nargs="?",
        help="语料库资源地址,这个语料库需要是第一次运行时被检查过在dict范围内后生成的text_dict.txt",
        default=default_args["corpus_path"]
    )
    ## 语料库-end

    parser.add_argument(
        "-hcl_dir",
        "--hcl_dir",
        type=str,
        nargs="?",
        help="HCL资源地址(hcl_writer_rgba)",
        default=default_args["hcl_dir"]
    )

    ## 第一运行-begin
    parser.add_argument(
        "-first",
        "--is_first",
        action="store_true",
        help="是否第一次使用,第一次使用要构建HWDB.若存在已久构建的HWDB,则无需开启,然后直接指定-hwdb_trn_path和-hwdb_tst_path",
        default=default_args["is_first"]
    )
    parser.add_argument(
        "-init_hwdb_trn_dir",
        "--init_hwdb_trn_dir",
        type=str,
        nargs="?",
        help="第一次使用需要构建HWDB时,HWDB的资源地址(HWDB1.1trn_gnt)",
        default=default_args["init_hwdb_trn_dir"]
    )
    parser.add_argument(
        "-init_hwdb_tst_dir",
        "--init_hwdb_tst_dir",
        type=str,
        nargs="?",
        help="第一次使用需要构建HWDB时,HWDB的资源地址(HWDB1.1tst_gnt)",
        default=default_args["init_hwdb_tst_dir"]
    )
    parser.add_argument(
        "-init_text_check_path",
        "--init_text_check_path",
        type=str,
        nargs="?",
        help="第一次使用需要(zhonghuayuwen.csv)检查语料库是否在dict范围,检查结果生成覆盖-corpus_path",
        default=default_args["init_text_check_path"]
    )
    ## 第一运行-end

    ## 第一运行构建HWDB数据，生成的train与test地址-begin
    parser.add_argument(
        "-hwdb_trn_dir",
        "--hwdb_trn_dir",
        type=str,
        nargs="?",
        help="第一次构建HWDB完成后,HWDB的训练集资源地址.",
        default=default_args["hwdb_trn_dir"]
    )
    parser.add_argument(
        "-hwdb_tst_dir",
        "--hwdb_tst_dir",
        type=str,
        nargs="?",
        help="第一次构建HWDB完成后,HWDB的测试集资源地址.",
        default=default_args["hwdb_tst_dir"]
    )
    ## 第一运行构建HWDB数据，生成的train与test地址-end

    parser.add_argument(
        "-ratio",
        "--hcl_ratio",
        type=float,
        nargs="?",
        help="HCL数据在所有生成的数据中所占的比例,默认0.5",
        default=default_args["hcl_ratio"]
    )
    parser.add_argument(
        "-mix",
        "--line_mix",
        action="store_true",
        help="开启单图片中HCL HWDB混合,默认单行内不混合",
        default=default_args["line_mix"]
    )

    parser.add_argument(
        "-const_char_num",
        "--const_char_num",
        action="store_true",
        help="固定单个图片中字符个数,个数为-max_char_num 指定的个数.只有不使用语料库时有效.默认随机个数.",
        default=default_args["const_char_num"]
    )
    parser.add_argument(
        "-max_char_num",
        "--max_char_num",
        type=int,
        nargs="?",
        help="单个图片中字符最大个数,当-const_char_num后,个数固定为最大(不随机).",
        default=default_args["max_char_num"]
    )
    parser.add_argument(
        "-chars_gap_width",
        "--chars_gap_width",
        type=int,
        nargs="?",
        help="图片中字符之间间隔距离.",
        default=default_args["chars_gap_width"]
    )

    parser.add_argument(
        "-height",
        "--image_height",
        type=int,
        nargs="?",
        help="图片高度,默认32.",
        default=default_args["image_height"]
    )
    parser.add_argument(
        "-width",
        "--image_width",
        type=int,
        nargs="?",
        help="图片宽度,宽度小于100就当做默认不打开,不定宽.",
        default=default_args["image_width"]
    )

    return parser.parse_args()

def main():
    """

    """
    args = parse_arguments()
    # 控制台输入参数判断
    is_continue = True
    if not args.test_mode:
        if args.train_data_num < 1 and args.test_data_num < 1  and args.valid_data_num < 1 :
            is_continue = False
            print("Train&Test&Valid数据量输入有误！")
        if args.write_types < 1:
            is_continue = False
            print("write_types输入有误！")


    else:
        args.train_data_num = 1
        args.test_data_num = 0
        args.valid_data_num = 0
        args.write_types = 1
        print("进入测试模式!")
        if args.is_first:
            if not os.path.exists(args.init_hwdb_trn_dir):
                is_continue = False
                print(
                    "第一使用需要构建HWDB,但init_hwdb_trn_dir 路径目录不存在,请重新指定!若已有构建好的HWDB,请指定-is_first,-hwdb_trn_dir和-hwdb_tst_dir")
            if not os.path.exists(args.init_hwdb_tst_dir):
                is_continue = False
                print(
                    "第一使用需要构建HWDB,init_hwdb_tst_dir 路径目录不存在,请重新指定!若已有构建好的HWDB,请指定-is_first,-hwdb_trn_dir和-hwdb_tst_dir")
            if args.off_corpus == False and not os.path.exists(args.init_text_check_path):
                is_continue = False
                print("第一次使用,但init_text_check_path 路径文件不存在,请重新指定!如果不想开启语料库,请指定--off_corpus")

        else:
            if args.off_corpus == False and not os.path.exists(args.corpus_path):
                is_continue = False
                print("corpus_path 路径文件不存在,请重新指定!请使用第一次使用text check后生成的与语料库.如果不想开启语料库,请指定--off_corpus")
            if args.hcl_ratio != 1 and not os.path.exists(args.hwdb_trn_dir):
                is_continue = False
                print("hwdb_trn_dir 路径目录不存在,请重新指定!若没有,请按第一次构建来,如果不想使用HWDB,请指定-hcl_ratio 1")
            if args.hcl_ratio != 1 and not os.path.exists(args.hwdb_tst_dir):
                is_continue = False
                print("hwdb_tst_dir 路径目录不存在,请重新指定!若没有,请按第一次构建来,如果不想使用HWDB,请指定-hcl_ratio 1")

        if args.hcl_ratio > 0 and not os.path.exists(args.hcl_dir):
            is_continue = False
            print("hcl_dir 路径目录不存在,请重新指定!如果不想使用HCL,请指定-hcl_ratio 0")
        if not os.path.exists(args.default_bg_path):
            is_continue = False
            print("default_bg_path 默认背景图片不存在,请重新指定! 一般纯白背景即可.")

    if not is_continue:
        return 0
    else:
        hwr_data.run(args)







if __name__ == "__main__":
    main()

