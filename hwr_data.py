import res.dictionary as dictionary
import os
from os import walk
import numpy as np
import struct
from PIL import Image,ImageDraw
import pickle
import re
import time
import pandas as pd
import ast
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import cv2
from imgaug import augmenters as iaa
import sys
RDM = np.random

class DataGenerator:
    def __init__(self,hwdb_trn_dir,hwdb_tst_dir,hcl_dir,background_dir,default_bg_path,corpus_path,off_corpus,hcl_ratio,img_height,img_width,const_char_num=False,max_char_num=12,line_mix=False,test_mode=False,true_write_type=True):
        self.char_dict = dict(dictionary.char_dict)
        self.char_dict_reverse = {v: k for k, v in self.char_dict.items()}

        ## 第一次构建HWDB
        self.hwdb_trn_dir = hwdb_trn_dir    # HWDB train 资源目录(第一使用构建HWDB生成)
        self.hwdb_tst_dir = hwdb_tst_dir    # HWDB test 资源目录(第一使用构建HWDB生成)
        self.hcl_dir = hcl_dir
        self.background_dir = background_dir
        self.default_bg_path = default_bg_path
        self.corpus_path = corpus_path
        self.hcl_ratio = hcl_ratio
        self.img_height = img_height
        self.img_width = img_width
        self.const_char_num = const_char_num
        self.max_char_num = max_char_num
        self.line_mix=line_mix
        self.off_corpus = off_corpus
        self.test_mode = test_mode
        self.true_write_type = true_write_type

        print("DataGenerator init...")
        print("---hwdb_trn_dir:", self.hwdb_trn_dir)
        print("---hwdb_tst_dir:", self.hwdb_tst_dir)
        print("---hcl_dir:", self.hcl_dir)
        print("---background_dir:", self.background_dir)
        print("---default_bg_path:", self.default_bg_path)
        print("---corpus_path:", self.corpus_path)
        print("---hcl_ratio:", self.hcl_ratio)
        print("---img_height:", self.img_height)
        print("---img_width:", self.img_width)
        print("---const_char_num:", self.const_char_num)
        print("---max_char_num:", self.max_char_num)
        print("---line_mix:", self.line_mix)
        print("---off_corpus:", self.off_corpus)
        print("---true_write_type:", self.true_write_type)
        print("---test_mode:", self.test_mode)

        self.which = ""
        self.gen_data_num = 0
        self.data_dir = ""
        self.data_augment = ""
        self.temp_data_augment = ""

    def gen_single_character_from_HWDB(self,init_hwdb_trn_dir,init_hwdb_tst_dir):
        '''
        获取单张图片数据集
        HWDB训练数据集路径train_data_dir
        HWDB测试数据集路径test_data_dir
        :return:
        '''
        train_counter = 0
        test_counter = 0
        print("gen_single_character_from_HWDB--第一次需要构建HWDB")

        train_data_dir = init_hwdb_trn_dir
        test_data_dir = init_hwdb_tst_dir

        train_save_path = self.hwdb_trn_dir
        test_save_path = self.hwdb_tst_dir
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)

        for image, tagcode in self._read_from_gnt_dir(gnt_dir=train_data_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            im = Image.fromarray(image)
            # %0.5d
            dir_name = os.path.join(train_save_path ,'%d' % self.char_dict[tagcode_unicode])
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            im.convert('RGB').save(os.path.join(dir_name ,str(train_counter) + '.png'))
            print("train_counter=", train_counter)
            train_counter += 1
            if is_test:
                if train_counter > 10240:
                    break
            # 0-897757

        for image, tagcode in self._read_from_gnt_dir(gnt_dir=test_data_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            im = Image.fromarray(image)
            dir_name = os.path.join(test_save_path , '%d' % self.char_dict[tagcode_unicode])
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            im.convert('RGB').save(os.path.join(dir_name , str(test_counter) + '.png'))
            print("test_counter=", test_counter)
            test_counter += 1
            if is_test:
                if test_counter > 10240:
                    break
            # 0-223990
        print("gen_single_character_from_HWDB--构建HWDB完成")

    def _read_from_gnt_dir(self, gnt_dir):
        '''
        获取HWDB1.1trn_gnt和HWDB1.1tst_gnt的图片数据
        :param gnt_dir:gnt路径
        :return:image, tagcode
        '''
        def one_file(f):
            header_size = 10
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size: break
                sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)
                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                yield image, tagcode

        for file_name in os.listdir(gnt_dir):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(gnt_dir, file_name)
                with open(file_path, 'rb') as f:
                    for image, tagcode in one_file(f):
                        yield image, tagcode

    def _get_root_dir_file(self, file_dir):
        '''
        遍历文件路径
        :param file_dir:
        :return:
        '''
        for root, dirs, files in walk(file_dir):
            return root, dirs, files

    def _normalizer_image(self, img_path):
        img = cv2.imread(img_path, 0)
        ar = np.array(img)
        max_width = 568
        # for hcl max 15 cha width = 484
        shape_offset = (max_width - ar.shape[1])
        # BLACK = [0,0,0]
        img = cv2.copyMakeBorder(img, 0, 0, shape_offset, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # 根据需求再改
        return img

    def _get_scrapy_random_text(self):
        '''
        从爬虫获取的语料库选取文本数据
        :return:
        '''
        text_dict_path = self.corpus_path
        text_dict = {}
        with open(text_dict_path, 'r', encoding='utf-8') as f1:
            text_dict = eval(f1.read())
        return text_dict

    def _trans_augment_parameter(self, aug):
        """
            line单行数据增强: l-画线; p-仿射变换; b-模糊(高斯、均匀、中值中随机); n-噪声(高斯噪声、弹性变换中随机); e-(浮雕、对比度中随机); c-像素(Dropout、加减、乘除中随机).
            char单字符数据增强: m-单字符随机上下;  r-单字符随机大小; a-单字符随机倾斜
        """
        aug_str = ""
        if aug.find("l")>=0:
            aug_str += "-画线"
        if aug.find("p")>=0:
            aug_str += "-仿射变换"
        if aug.find("b")>=0:
            aug_str += "-模糊(高斯、均匀、中值中随机)"
        if aug.find("n")>=0:
            aug_str += "-噪声(高斯噪声、弹性变换中随机)"
        if aug.find("e")>=0:
            aug_str += "-浮雕、对比度中随机"
        if aug.find("c")>=0:
            aug_str += "-像素(Dropout、加减、乘除中随机)"
        if aug.find("m")>=0:
            aug_str += "-单字符随机上下"
        if aug.find("r")>=0:
            aug_str += "-单字符随机大小"
        if aug.find("a")>=0:
            aug_str += "-单字符随机倾斜"
        if aug.find("0")>=0:
            aug_str += "-单字符真实化"
        return aug_str

    def _show_image(self,image_list,label_list,filename="test_mode"):
        # if image.shape[2] ==1:
        #     image = np.squeeze(image,axis=2)
        if len(image_list)==1:
            image = image_list[0]
            image_label = label_list[0]
            f = plt.figure()
            ax = f.add_subplot(111)
            zhfont = matplotlib.font_manager.FontProperties(
                fname="/Library/Fonts/Songti.ttc")  # 字体

            ax.text(0.1, 0.9, image_label, ha='center', va='center',
                    transform=ax.transAxes, fontproperties=zhfont)
            plt.imshow(image)
            if self.test_mode:

                img_name = image_label + '.png'
                # save_addr = img_gen_dir_path + '/' + img_name
                save_addr = os.path.join(self.gen_image_dir, img_name)
                # print(save_addr)
                plt.savefig(save_addr)
            plt.show()
        elif len(image_list)==2:
            # img_org,img_aug = image_list[0],image_list[1]
            # label_org, label_aug = label_list[0], label_list[1]
            # f = plt.figure()
            zhfont = matplotlib.font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")  # 字体
            # 行、列、索引
            for i in range(len(image_list)):
                # ax = f.add_subplot(211)
                # ax.text(0.1, 0.9, image_label, ha='center', va='center',transform=ax.transAxes, fontproperties=zhfont)
                # plt.imshow(image)

                plt.subplot(len(image_list), 1, i + 1)
                plt.subplots_adjust(bottom=0.1,top=0.9,wspace=0.01, hspace=0.01)
                plt.imshow(image_list[i])
                plt.title(label_list[i], fontsize=8,fontproperties=zhfont)
                plt.xticks([])
                plt.yticks([])

            if self.test_mode:
                img_name = filename + '.png'
                # save_addr = img_gen_dir_path + '/' + img_name
                save_addr = os.path.join(self.gen_image_dir, img_name)
                # print(save_addr)
                plt.savefig(save_addr)
            plt.show()

    def _read_data_from_res(self, path,mode="None"):

        if mode == "Image":
            try:
                image = Image.open(path)
                return image
            except FileNotFoundError:
                print("Error, No such file or directory:",path)
                sys.exit(1)
        else:
            print("Error, wrong mode :", mode)
            return None


    def _get_char_img(self, char,char_type="HWDB"):
        '''
        生成单字图片
        :param char:
        :return: char_img
        '''
        # hcl
        # xx001-xx700  训练集
        # hh001-hh300   测试集
        char_dir = str(self.char_dict[char])
        ## 字符图片矫正，
        # 替换HCL里的 一 为 HWDB中的一
        if char == "一":
            char = '一'
            # print("矫正，char:",char)
            if self.which == 'test':
                char_dir, _, char_file = self._get_root_dir_file(os.path.join(self.hwdb_tst_dir, char_dir))
            else:
                char_dir, _, char_file = self._get_root_dir_file(os.path.join(self.hwdb_trn_dir, char_dir))
            char_path = os.path.join(char_dir, RDM.choice(char_file))

            # print("get_char_img--train--HCL,char_dir:", char_dir)
            # index = RDM.randint(1, 701)
            # index = str(index).zfill(3)
            # # char_path = hcl_char_img_root+'/xx' + index + '/xx' + index + '_' + char + '.png'
            # char_path = os.path.join(self.hcl_dir, "xx" + index, "xx" + index + '_' + char + '.png')
            char_type="HWDB"
        else:
            if self.line_mix:
                ## 根据预先设置的hcl ration 来源比例，随机个数字，数字小于threshold，就从HCL取，否则从HWDB取
                threshold = 1000 * self.hcl_ratio
                rdm = RDM.randint(0, 999)
                if rdm>=threshold:
                    char_type = "HWDB"
                else:
                    char_type = "HCL"

            if self.which == 'train':
                if char_type=="HWDB":
                    # print("get_char_img--train--HWDB,char_dir:", char_dir)
                    # char_dir, _, char_file = self.get_root_dir_file(hwdb_train_char_img_root + '/' + char_dir)
                    char_dir, _, char_file = self._get_root_dir_file(os.path.join(self.hwdb_trn_dir,char_dir))

                    # char_path = char_dir + '/' + RDM.choice(char_file)
                    if not self.test_mode:
                        char_path = os.path.join(char_dir, RDM.choice(char_file))
                    else:
                        char_path = os.path.join(char_dir, char_file[10])

                else:
                    # print("get_char_img--train--HCL,char_dir:", char_dir)

                    if not self.test_mode:
                        index = RDM.randint(1, 701)
                        index = str(index).zfill(3)
                        # char_path = hcl_char_img_root+'/xx' + index + '/xx' + index + '_' + char + '.png'
                        char_path = os.path.join(self.hcl_dir, "xx" + index, "xx" + index + '_' + char + '.png')
                    else:
                        index = 256
                        index = str(index).zfill(3)
                        char_path = os.path.join(self.hcl_dir,"xx"+index,"xx"+index+'_' + char + '.png')

            elif self.which == 'test':
                if char_type=="HWDB":
                    # print("get_char_img--test--HWDB,char_dir:", char_dir)
                    # char_dir, _, char_file = self.get_root_dir_file('./data/train' + '/' + char_dir)
                    char_dir, _, char_file = self._get_root_dir_file(os.path.join(self.hwdb_tst_dir,char_dir))

                    # char_path = char_dir + '/' + RDM.choice(char_file)
                    char_path = os.path.join(char_dir, RDM.choice(char_file))
                else:
                    # print("get_char_img--test--HCL,char_dir:", char_dir)
                    index = RDM.randint(1, 301)
                    index = str(index).zfill(3)
                    # char_path = '/Users/xbb1973/PycharmProjects/res/hcl_writer_rgba/xx' + index + '/xx' + index + '_' + char + '.png'
                    char_path = os.path.join(self.hcl_dir, "xx" + index, "xx" + index + '_' + char + '.png')

            elif self.which == 'valid':
                if char_type=="HWDB":
                    # print("get_char_img--valid--HWDB,char_dir:", char_dir)
                    char_dir, _, char_file = self._get_root_dir_file(os.path.join(self.hwdb_trn_dir,char_dir))
                    char_path = os.path.join(char_dir,RDM.choice(char_file))
                else:
                    # print("get_char_img--valid--HCL,char_dir:", char_dir)
                    index = RDM.randint(1, 701)
                    index = str(index).zfill(3)
                    # char_path = hcl_char_img_root+'/xx' + index + '/xx' + index + '_' + char + '.png'
                    char_path = os.path.join(self.hcl_dir, "xx" + index, "xx" + index + '_' + char + '.png')
            else:
                print("Error,get_char_img-Unknown which:",self.which)
                exit(0)

        # print(char_path)

        # char_img = Image.open(char_path)
        char_img = self._read_data_from_res(path=char_path,mode="Image")
        # char_img = char_img.convert("RGBA")
        # r, g, b, a = char_img.split()
        # char_img.paste(char_img, (0, 0), mask=a)
        # img = np.array(char_img)
        # img[:,:,3] = 126
        # print("char_img.shape:",img.shape)
        # char_img = Image.fromarray(img,mode="RGBA")
        char_size = int(self.img_height - 1)
        if char_img.height > char_img.width:
            size_rate = char_size / char_img.height
        else:
            size_rate = char_size / char_img.width
        char_img = char_img.resize(
            (int(char_img.width * size_rate), int(char_img.height * size_rate)))

        # 拼接RGBA文件需要原图片的alph通道提取出的mask
        background = Image.new('RGBA', (int(char_img.width), char_img.height),
                         (255, 255, 255, 0))
        Image.isImageType('RGBA')
        try:
            r, g, b, alph = char_img.split()
            background.paste(char_img, mask=alph)
        except:
            background.paste(char_img)
        return background,char_type

    def _get_str_img(self, str_gen ):
        '''
        生成文本行图片
        :param str_gen:
        :return: str_img
        '''
        # width=字符串长度*2*self.height*2
        # str_img_width = int(len(str_gen) * (2)) * self.height * 2
        str_img_width = len(str_gen) * self.img_height  + (len(str_gen) - 1) * self.chars_gap_width
        if len(str_gen) <= 4:
            str_img_width+= (self.img_height//2)
        str_img_height = self.img_height
        str_img = Image.new('RGBA',
                            (str_img_width, str_img_height),
                            (255, 255, 255, 0))
        str_img_aug = str_img.copy()
        if self.img_width>=100:
            bg_width = self.img_width
        else:
            bg_width = len(str_gen) * self.img_height  + (len(str_gen) - 1) * self.chars_gap_width
        bg_height = self.img_height
        # print("-------str_img_width=",str_img_width)

        ## 背景处理
        if not self.test_mode:      ## 正常模式
            if self.which=="train":
                if self.temp_data_augment.find("g") >= 0:
                    bg_filename_list = os.listdir(self.background_dir)
                    # print("get_str_img---,bg_filename_list:",bg_filename_list)
                    bg_filename = RDM.choice(bg_filename_list)
                    back = self._read_data_from_res(path=os.path.join(self.background_dir, bg_filename),mode="Image")
                    # back = Image.open(os.path.join(self.background_dir, bg_filename))
                else:
                    # 指定白色背景
                    # bg_filename = str(back_index) + '.png'
                    back = self._read_data_from_res(path=self.default_bg_path, mode="Image")
                    # back = Image.open(self.default_bg_path)
                    # default_bg_path
                # print("get_str_img---,bg_filename:", bg_filename)
                # back = Image.open(os.path.join(self.background_dir, bg_filename))
                ## 从背景图片中随机截取背景区域
                back = back.resize((back.width +bg_width+6,back.height + bg_height+6))
                bg_rdm_height = RDM.randint(5,back.height - bg_height-6)
                bg_rdm_width = RDM.randint(5,back.width - bg_width-6)
                # crop(x1,y1,x2,y2) x-width,y-height
                back = back.crop((bg_rdm_width, bg_rdm_height, bg_rdm_width+bg_width, bg_rdm_height+bg_height))
                # print("-------back.size=",back.size)
            else:
                # 非训练集指定白色背景
                # back_index = 11
                # back_filename = str(back_index) + '.png'
                # back = Image.open(os.path.join(self.background_dir, back_filename))
                back = self._read_data_from_res(path=self.default_bg_path, mode="Image")

                bg_rdm_height = RDM.randint(5, back.height - bg_height - 6)
                bg_rdm_width = RDM.randint(5, back.width - bg_width - 6)
                # crop(x1,y1,x2,y2) x-width,y-height
                back = back.crop(
                    (bg_rdm_width, bg_rdm_height, bg_rdm_width + bg_width, bg_rdm_height + bg_height))
                # print("-------back.size=", back.size)
            back_aug = back.crop()
        else:       ## test_mode 测试模式
            if self.temp_data_augment.find("g") >= 0:
                bg_filename_list = os.listdir(self.background_dir)
                # print("get_str_img---,bg_filename_list:",bg_filename_list)
                bg_filename = RDM.choice(bg_filename_list)
                back_aug = self._read_data_from_res(path=os.path.join(self.background_dir, bg_filename), mode="Image")
            else:
                # 指定白色背景
                # back_index = 11
                # bg_filename = str(back_index) + '.png'
                back_aug = self._read_data_from_res(path=self.default_bg_path, mode="Image")

            # back_aug = Image.open(os.path.join(self.background_dir, bg_filename))
            ## 从背景图片中随机截取背景区域
            back_aug = back_aug.resize((back_aug.width + bg_width + 6, back_aug.height + bg_height + 6))
            bg_rdm_height = RDM.randint(5, back_aug.height - bg_height - 6)
            bg_rdm_width = RDM.randint(5, back_aug.width - bg_width - 6)
            # crop(x1,y1,x2,y2) x-width,y-height
            back_aug = back_aug.crop((bg_rdm_width, bg_rdm_height, bg_rdm_width + bg_width, bg_rdm_height + bg_height))
            # print("-------back_aug.size=",back_aug.size)

            # 指定白色背景
            # back_index = 11
            # back_filename = str(back_index) + '.png'
            # back = Image.open(os.path.join(self.background_dir, back_filename))
            back = self._read_data_from_res(path=self.default_bg_path, mode="Image")

            bg_rdm_height = RDM.randint(5, back.height - bg_height - 6)
            bg_rdm_width = RDM.randint(5, back.width - bg_width - 6)
            # crop(x1,y1,x2,y2) x-width,y-height
            back = back.crop(
                (bg_rdm_width, bg_rdm_height, bg_rdm_width + bg_width, bg_rdm_height + bg_height))
            # print("-------back.size=", back.size)

        ## 上面获取到了行背景和文字背景后，往文字背景贴字。贴完字后，把文字背景贴到行背景，最终生成数据。
        # char width
        char_width = 0
        char_aug_width = 0
        ## 根据预先设置的hcl ration 来源比例，随机个数字，数字小于threshold，就从HCL取，否则从HWDB取
        threshold = 1000 * self.hcl_ratio
        rdm = RDM.randint(0, 999)
        for char in str_gen:
            if not self.line_mix:

                if rdm >= threshold:
                    char_img,char_from = self._get_char_img(char,"HWDB")
                else:
                    char_img,char_from = self._get_char_img(char, "HCL")
            else:
                char_img,char_from = self._get_char_img(char,"line_mix")

            ## 对单字进行数据增强
            char_img_aug = char_img.copy()
            char_img_aug = self._char_image_data_augment(char_img_aug,char_from)
            char_top_margin = 0
            if self.temp_data_augment.find("m")>=0:
                char_top_margin = RDM.randint(1,str_img_aug.height//8)
            try:
                r, g, b, alph = char_img.split()
                # 拼接汉字字符，这里可以控制间距进行数据增强
                # augment place
                str_img.paste(char_img, (char_width, int((str_img.height - char_img.height) / 2)), mask=alph)
                str_img_aug.paste(char_img_aug, (char_aug_width, int((str_img.height - char_img.height) / 2)+char_top_margin), mask=alph)
                # back.paste(char_img, (char_width, int((str_img.height - char_img.height) / 2)), mask=alph)
            except:
                str_img.paste(char_img, (char_width,  int((str_img.height - char_img.height) / 2)))
                str_img_aug.paste(char_img_aug, (char_aug_width, int((str_img.height - char_img.height) / 2)+char_top_margin))
                # back.paste(char_img, (char_width,  int((str_img.height - char_img.height) / 2)))
            char_width += char_img.width+self.chars_gap_width
            char_aug_width += char_img_aug.width+self.chars_gap_width

        str_img_aug = str_img_aug.resize(back_aug.size)
        r, g, b, a = str_img_aug.split()
        back_aug.paste(str_img_aug, (5, 0), mask=a)

        str_img = str_img.resize(back.size)
        r, g, b, a = str_img.split()
        back.paste(str_img, (5, 0),mask = a)

        return back_aug,back

    def _get_char_list(self, str):
        '''
        分解str，获得单个char，再根据char_dict得到每一个char的对应编号，最终等到char_list
        :param str:
        :return:char_list:str中每个char的编号列表
        '''
        char_list = []
        new_str = ''
        for char in str:
            try:
                char_list.append(self.char_dict[char])
                new_str += char
            except:
                rd = RDM.randint(0, 3755)
                char_list.append(rd)
                new_str += list(self.char_dict.keys())[list(self.char_dict.values()).index(rd)]

        return new_str, char_list

    def _true_write_type_data(self,img,char_from):
        augmenter = []
        if char_from=="HWDB":       ## HWDB数据真实化效果不好，先跳过。只对HCL进行真实化
            return img

        # aug_pwa = iaa.Resize((0.7, 1))  # 将w和h在0.5-1.5倍范围内resize
        augmenter.append(iaa.Sharpen(alpha=(1, 1), lightness=(1, 1)))      # 锐化处理
        # augmenter.append(iaa.ContrastNormalization((1.5, 1.5), per_channel=0.5))  # 对比度

        seq = iaa.Sequential(augmenter)
        # img为opencv，image为PIL，二者进行转化
        # PIL->opencv
        # img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)
        image_aug = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2GRAY)

        # + cv2.THRESH_OTSU
        # print("_true_write_type_data--char_from=",char_from)
        ret3, image_aug = cv2.threshold(image_aug, 245, 255, cv2.THRESH_BINARY)
        # ret3, image_aug = cv2.threshold(image_aug, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imglist = []
        imglist.append(image_aug)
        images_aug = seq.augment_images(imglist)
        image_aug = images_aug[0]

        #
        kernel = np.ones((2, 2), np.uint8)
        # image_aug = cv2.erode(image_aug, kernel, iterations=1)
        # image_aug = cv2.dilate(image_aug, kernel, iterations=2)

        image_aug = cv2.dilate(image_aug, kernel, iterations=2)
        image_aug = cv2.erode(image_aug, kernel, iterations=1)
        #
        # image_aug = cv2.Canny(image_aug, 0, 100)
        #
        # seq = iaa.Sequential([iaa.Invert(1, per_channel=True)])
        # images_aug = seq.augment_images([image_aug])
        # image_aug = images_aug[0]
        # image_aug = cv2.morphologyEx(image_aug, cv2.MORPH_OPEN, kernel)

        # image_aug = Image.fromarray(cv2.cvtColor(image_aug, cv2.COLOR_BGRA2RGBA))
        image_aug = Image.fromarray(cv2.cvtColor(image_aug, cv2.COLOR_GRAY2RGBA))

        return image_aug

    def _char_image_data_augment(self, img,char_from):
        if self.true_write_type:    # 首先对单字符图片进行真实化处理
            img = self._true_write_type_data(img,char_from)

        height = img.height
        width = img.width
        # print("_char_image_data_augment,image.height=",height," ,width=",width)
        augmenter = []

        if self.temp_data_augment.find("r")>=0:
            aug_pwa = iaa.Resize((0.7,1))  # 将w和h在0.5-1.5倍范围内resize
            augmenter.append(aug_pwa)
        if self.temp_data_augment.find("a")>=0:
            aug_pwa = iaa.Affine(                          #对一部分图像做仿射变换
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},#图像缩放为80%到120%之间
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
                # rotate=(-20, 20),   #旋转±45度之间
                shear=(-25, 25),    #剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],   #使用最邻近差值或者双线性差值
                # cval=(0, 255),  #全白全黑填充
            )
            augmenter.append(aug_pwa)


        seq = iaa.Sequential(augmenter)

        # img为opencv，image为PIL，二者进行转化
        # PIL->opencv
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)
        imglist = []
        imglist.append(img_cv)

        images_aug = seq.augment_images(imglist)
        image = Image.fromarray(cv2.cvtColor(images_aug[0], cv2.COLOR_BGRA2RGBA))

        return image

    def _line_image_data_augment(self,img,char_num):
        height = img.height
        width = img.width
        # print("_line_image_data_augment,image.height=",height," ,width=",width)
        augmenter = []

        if self.temp_data_augment.find("l")>=0:
            image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # print("_line_image_data_augment---line")
            line_num = RDM.randint(4,8)
            line_length = RDM.randint(width//4, width//2)
            # draw = ImageDraw.Draw(img)
            for x in range(line_num):
                startX = RDM.randint(0,int(width*(3/4)))
                startY = RDM.randint(0,height)

                if startX+line_length > width:
                    endX = width
                else:
                    endX = startX+line_length
                endY = RDM.randint(0,height)
                # draw.line([(startX, startY), (endX, endY)], fill="gray", width=1)
                cv2.line(image, (startX, startY), (endX, endY), (190, 190, 190),thickness=1, lineType=8)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.temp_data_augment.find("p")>=0:
            width_num = char_num + 2
            height_num = 6
            if self.test_mode:
                draw = ImageDraw.Draw(img)
                for x in range(int(width_num)):
                    startX = img.size[0] / int(width_num) * (x + 1)
                    startY = 0
                    endX = startX
                    endY = img.size[1]
                    draw.line([(startX, startY), (endX, endY)], fill="gray", width=1)

                for x in range(int(height_num)):
                    startX = 0
                    startY = img.size[1] / int(height_num) * (x + 1)
                    endX = img.size[0]
                    endY = startY
                    draw.line([(startX, startY), (endX, endY)], fill="gray", width=1)

            # 整体流程为：定义变换序列（Sequential）→读入图片（imread）→执行变换（augment_images）→保存图片（imwrite）
            # imgaug test
            # StochasticParameter
            aug_pwa = iaa.PiecewiseAffine(scale=(0.02, 0.04), nb_rows=(2, height_num), nb_cols=(2, width_num),
                                          order=1, cval=0, mode='constant',
                                          absolute_scale=False, polygon_recoverer=None, name=None,
                                          deterministic=False, random_state=None)
            augmenter.append(aug_pwa)

        # 在每个图像上放置规则的点网格，然后将每个点随机地移动2 - 3％
        # aug = iaa.PiecewiseAffine(scale=(0.02, 0.03))
        if self.temp_data_augment.find("b")>=0:
            # 用高斯模糊，均值模糊，中值模糊中的一种增强。
            aug_pwa = iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.4,0.7)),  # 高斯模糊
                iaa.AverageBlur(k=(2, 7)),  # 均匀模糊，核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                iaa.MedianBlur(k=(3, 11)) # 中值模糊
            ])
            augmenter.append(aug_pwa)

        if self.temp_data_augment.find("n") >= 0:
            aug_pwa = iaa.OneOf([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),      # 高斯噪声
                iaa.ElasticTransformation(alpha=(0.4, 0.4), sigma=0.25),        # 弹性变换，把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
            ])
            augmenter.append(aug_pwa)

        if self.temp_data_augment.find("e") >= 0:
            aug_pwa = iaa.OneOf([
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),     #浮雕效果
                iaa.ContrastNormalization((0.5, 1.2), per_channel=0.5)   # 对比度

            ])
            augmenter.append(aug_pwa)

        if self.temp_data_augment.find("c") >= 0:
            aug_pwa = iaa.OneOf([
                iaa.Dropout((0.01, 0.02), per_channel=0.5),     # 将1%到2%的像素设置为黑色
                iaa.Add((-10, 10), per_channel=0.5),     # 每个像素随机加减-10到10之间的数
                iaa.Multiply((0.8, 1.2), per_channel=0.5)   # 像素乘上0.8或者1.2之间的数字.
            ])
            augmenter.append(aug_pwa)

        if self.temp_data_augment.find("f")>=0:
            aug_pwa = iaa.Affine(                          #对一部分图像做仿射变换
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},#图像缩放为80%到120%之间
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
                # rotate=(-20, 20),   #旋转±45度之间
                shear=(-25, 25),    #剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],   #使用最邻近差值或者双线性差值
                # cval=(0, 255),  #全白全黑填充
            )
            augmenter.append(aug_pwa)

        seq = iaa.Sequential(augmenter)


        # img = cv2.imread('./gen_images/0.png')
        # img为opencv，image为PIL，二者进行转化
        # PIL->opencv
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)
        imglist = []
        imglist.append(img_cv)
        # img_arr = np.asarray(img)
        images_aug = seq.augment_images(imglist)
        image = Image.fromarray(cv2.cvtColor(images_aug[0], cv2.COLOR_BGRA2RGBA))

        return image

    def _gen_data_with_img_and_label(self, str):
        '''
        获取最终数据
        :param str:文本行
        :param img_name:图片保存名称
        :param data_path:图片保存地址
        :param augment:是否启用数据增强
        :param is_test:是否为测试状态
        :return:img, label
        '''

        str, char_list = self._get_char_list(str)
        # if self.test_mode:
            # str = "数据增强效果图"
            # str = "刽寝镇勒"

        img_aug,img = self._get_str_img(str)
        image_aug = self._line_image_data_augment(img_aug,len(str))

        # print("gen_data_with_img_and_label,image_aug.shape=",image_aug.shape)
        if self.test_mode:
            image_aug = np.array(image_aug, dtype=np.int32)
            image_list = []
            image_list.append(img)
            image_list.append(image_aug)
            label_list = []
            aug_str = self._trans_augment_parameter(self.temp_data_augment)
            label_list.append(str+"_"+"original")
            label_list.append(str+"_"+"augment_"+aug_str)
            self._show_image(image_list,label_list,filename=aug_str+"_"+str)
            # self._show_image(image_aug, str+"_"+"augment_"+self.temp_data_augment)

        return image_aug,str, char_list

    def run(self, which, gen_data_num, gen_image_dir, gen_info_path, gen_frequency_list_path,write_types,
                         chars_gap_width, data_augment,data_augment_percent):

        self.which = which
        self.gen_data_num = gen_data_num
        self.gen_image_dir = gen_image_dir
        self.gen_info_path = gen_info_path
        self.gen_frequency_list_path = gen_frequency_list_path
        self.write_types = write_types
        self.chars_gap_width = chars_gap_width
        self.data_augment = data_augment                # 是整个Train上应用的效果
        self.temp_data_augment = self.data_augment      # 临时作为每个样本的增强效果
        self.data_augment_percent = data_augment_percent

        if not os.path.exists(self.gen_image_dir):
            os.makedirs(self.gen_image_dir)

        data_dir = self.gen_info_path.rsplit("/", 1)[0]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_dir = self.gen_frequency_list_path.rsplit("/", 1)[0]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        print("which:",self.which)
        print("gen_data_num:", self.gen_data_num)
        print("gen_image_dir:", self.gen_image_dir)
        print("gen_info_path:", self.gen_info_path)
        print("gen_frequency_list_path:", self.gen_frequency_list_path)
        print("write_types:", self.write_types)
        print("chars_gap_width:", self.chars_gap_width)
        print("data_augment:", self.data_augment)
        print("data_augment_percent:", self.data_augment_percent)

        word_frequency_dict = dict()
        # 初始化word_frequency_dict，否则会出现key error
        for key in self.char_dict.keys():
            word_frequency_dict[key] = 0
        word_frequency_dict['keyError'] = 0
        if not self.off_corpus:         # 使用语料库
            text_dict = self._get_scrapy_random_text()

        # while gen_data_count < self.gen_data_num:
        gen_data_count = 0
        # epochs = (self.gen_data_num+1)//self.write_types
        # if epochs == 0:
        #     epochs=1
        with tqdm(range(self.gen_data_num)) as pbar:
            ## for write_types
            str_gen=""
            for gen_data_count,_ in enumerate(pbar):

                if self.const_char_num:
                    ## 固定单个图片中字符个数
                    random_char_max_num = self.max_char_num
                else:
                    ## 否则随机个数
                    random_char_max_num = RDM.randint(2, self.max_char_num)


                labels = []
                if gen_data_count % self.write_types == 0:
                    if not self.off_corpus:          # 使用语料库
                        char_num = gen_data_count % 3755
                        text_list = list(text_dict[self.char_dict_reverse[char_num]])
                        random_text_pos = RDM.randint(0, len(text_list))
                        str_gen = text_list[random_text_pos]
                    else:
                        str_gen = ''
                        for i in range(random_char_max_num):
                            str_gen += self.char_dict_reverse[RDM.randint(0, 3755)]

                # 异常字符处理，暂时随机生成
                key_error_count = 0
                new_str = ''
                # print("test----gen_data_count=",gen_data_count," ,str_gen=",str_gen)
                # continue
                for char_item in str_gen:
                    try:
                        char_file_dir = self.char_dict[char_item]
                        new_str += char_item
                        try:
                            word_frequency_dict[char_item] += 1
                        except KeyError:
                            word_frequency_dict['keyError'] += 1
                        except:
                            pass
                    except KeyError:
                        key_error_count += 1
                        # print('key_error_count--begin')
                        # print(str_gen)
                        # print(char_item)
                        # print(key_error_count)
                        # print('key_error_count--end')
                        char_file_dir = RDM.randint(0, 3755)
                        char_item = self.char_dict_reverse[char_file_dir]
                        new_str += char_item
                        try:
                            word_frequency_dict[char_item] += 1
                        except KeyError:
                            word_frequency_dict['keyError'] += 1
                        except:
                            pass
                str_gen = new_str


                # # 同一个文本行输出write_types种不同写法，暂时不加入数据增强部分。
                # for i in range(write_types):
                ## 随机选取 data_augment_of_all 中参数应用效果,应用在data_augment
                if not self.test_mode and self.data_augment!="":
                    if self.data_augment_percent<=1 and self.data_augment_percent>=0:
                        threshold = 1000 * self.data_augment_percent
                        rdm = RDM.randint(0, 999)
                        if rdm>=threshold:
                            self.temp_data_augment=""
                        else:
                            if len(self.data_augment)>=2:
                                self.temp_data_augment = RDM.sample(self.data_augment,2)
                            else:
                                self.temp_data_augment = self.data_augment
                    else:
                        print("Error, data_augment_percent设置错误，data_augment_percent=",self.data_augment_percent)

                img, new_str,char_list = self._gen_data_with_img_and_label(str=str_gen)
                img_name = str(gen_data_count) + '_' + str_gen + '.png'
                # save_addr = img_gen_dir_path + '/' + img_name
                save_addr = os.path.join(self.gen_image_dir,img_name)
                # print(save_addr)
                if self.test_mode:
                    continue
                img.save(save_addr)
                if True:
                    line = str(img.width) + ' ' + str(img.height) + ' ' + img_name + ' ' + str(char_list).split('[')[-1].split(']')[0].replace(' ', '')
                    labels.append(line)
                    # print(line)
                else:
                    line = [save_addr, char_list]
                    labels.append(line)

                pbar.set_description(f'---Which:{self.which} ,Count:{gen_data_count+1:7d} ,Augment:{self.temp_data_augment} ')
                # pbar.set_description(f'---Which:{self.which} ,Count:{gen_data_count + 1:7d} ')
                # 最后将标签写入文件，考虑到数据太多，如果最后再写可能会出问题，现在每一次循环追加的形式写一次
                # 32位的py list可以存储5000w个元素，64位的py list可以存储......
                with open(self.gen_info_path, 'a', encoding='UTF-8') as f1:
                    for i in range(0, len(labels)):
                        line1 = labels[i]
                        # print(line1)
                        # print(str(line1))
                        f1.write(str(line1))
                        f1.write('\n')

        with open(self.gen_frequency_list_path, 'a', encoding='UTF-8') as f1:
                f1.write(str(word_frequency_dict))
                f1.write('\n')


def text_check(init_text_check_path):
    # 文本数据处理
    # csv_file_path = '../../res/zhonghuayuwen_3755.csv'
    csv_file_path = init_text_check_path
    df = pd.read_csv(csv_file_path)
    # char_item char2text_list_len char2text_list
    cols = ['char_item', 'char2text_list_len']
    df = pd.DataFrame(df, columns=cols)
    # # 查看没有文本行的数据
    # print(df[df['char2text_list_len'] <= 10])
    # # 89
    # print(df[df['char2text_list_len'] <= 25])
    # # 579
    # print(df[df['char2text_list_len'] <= 50])
    # # 1102
    # print(df[df['char2text_list_len'] <= 100])
    # # 1772
    #      char_item  char2text_list_len
    # 3651         捌                   0
    # 获取charitem，检查是否有遗漏
    char_item = df.get('char_item')
    print(char_item)
    # 0       丁
    # 1       一
    # 2       丈
    # 3       七
    # 4       万
    #        ..
    # 3749    兽
    # 3750    入
    # 3751    党
    # 3752    匡
    # 3753    凯
    # Name: char_item, Length: 3754, dtype: object
    # char_item_dict = dict(char_item)
    # print(char_item_dict)
    # {0: '丁', 1: '一', 2: '丈', 3: '七', 4: '万', 5: '丑', 6: '三', 7: '丘', 8: '上', 9: '与', 10: '下', 11: '专', 12: '丙', 13: '不', 14: '且', 15: '世', 16: '丛', 17: '业', 18: '东', 19: '丧', 20: '丝', 21: '丫', 22: '丢', 23: '严', 24: '两', 25: '丰', 26: '丸', 27: '个', 28: '丹', 29: '临', 30: '串', 31: '中', 32: '丽', 33: '为', 34: '主', 35: '举', 36: '乃', 37: '乌', 38: '乍', 39: '久', 40: '么', 41: '乏', 42: '义', 43: '乒', 44: '乎', 45: '之', 46: '乔', 47: '乖', 48: '乐', 49: '乓', 50: '乞', 51: '乘', 52: '习', 53: '乙', 54: '九', 55: '也', 56: '乾', 57: '乡', 58: '书', 59: '乳', 60: '买', 61: '乱', 62: '予', 63: '了', 64: '争', 65: '亏', 66: '事', 67: '二', 68: '互', 69: '于', 70: '亚', 71: '云', 72: '井', 73: '亢', 74: '亡', 75: '五', 76: '亥', 77: '些', 78: '亨', 79: '交', 80: '享', 81: '亦', 82: '产', 83: '亭', 84: '京', 85: '亩', 86: '什', 87: '亮', 88: '仁', 89: '亲', 90: '仆', 91: '仇', 92: '人', 93: '亿', 94: '介', 95: '仅', 96: '仓', 97: '今', 98: '仔', 99: '仕', 100: '仍', 101: '从', 102: '仗', 103: '仑', 104: '仙', 105: '付', 106: '他', 107: '仪', 108: '仟', 109: '代', 110: '令', 111: '仰', 112: '仲', 113: '以', 114: '们', 115: '仿', 116: '企', 117: '件', 118: '伊', 119: '伍', 120: '伎', 121: '任', 122: '价', 123: '份', 124: '伐', 125: '休', 126: '伏', 127: '优', 128: '伙', 129: '众', 130: '伞', 131: '伟', 132: '伦', 133: '会', 134: '估', 135: '伪', 136: '传', 137: '伯', 138: '伶', 139: '伤', 140: '伴', 141: '伸', 142: '佃', 143: '似', 144: '伺', 145: '佐', 146: '佑', 147: '但', 148: '位', 149: '住', 150: '低', 151: '佛', 152: '佩', 153: '体', 154: '何', 155: '佬', 156: '余', 157: '佯', 158: '作', 159: '你', 160: '佳', 161: '侄', 162: '侈', 163: '侍', 164: '佣', 165: '侗', 166: '使', 167: '佰', 168: '侠', 169: '例', 170: '侦', 171: '供', 172: '依', 173: '侮', 174: '侧', 175: '侯', 176: '侵', 177: '侥', 178: '促', 179: '侣', 180: '俊', 181: '俏', 182: '俄', 183: '便', 184: '侩', 185: '俘', 186: '侨', 187: '俗', 188: '俞', 189: '保', 190: '俯', 191: '俩', 192: '信', 193: '俱', 194: '修', 195: '俐', 196: '倔', 197: '俺', 198: '倘', 199: '候', 200: '倍', 201: '倚', 202: '倡', 203: '倒', 204: '倦', 205: '债', 206: '借', 207: '俭', 208: '倾', 209: '值', 210: '健', 211: '偏', 212: '偶', 213: '倪', 214: '假', 215: '做', 216: '偿', 217: '停', 218: '傅', 219: '偷', 220: '傍', 221: '傣', 222: '储', 223: '僚', 224: '催', 225: '僧', 226: '傲', 227: '傻', 228: '像', 229: '僵', 230: '僻', 231: '儒', 232: '傀', 233: '允', 234: '兆', 235: '充', 236: '儿', 237: '元', 238: '兄', 239: '僳', 240: '兑', 241: '先', 242: '光', 243: '免', 244: '克', 245: '兔', 246: '兜', 247: '傈', 248: '儡', 249: '兰', 250: '兴', 251: '六', 252: '共', 253: '典', 254: '兹', 255: '兵', 256: '关', 257: '具', 258: '其', 259: '公', 260: '冀', 261: '冈', 262: '养', 263: '八', 264: '兼', 265: '全', 266: '冒', 267: '冕', 268: '再', 269: '写', 270: '冤', 271: '农', 272: '冠', 273: '军', 274: '冯', 275: '冶', 276: '况', 277: '冬', 278: '决', 279: '冰', 280: '册', 281: '冲', 282: '冗', 283: '凄', 284: '冻', 285: '凉', 286: '凋', 287: '净', 288: '准', 289: '减', 290: '冷', 291: '凝', 292: '凌', 293: '凑', 294: '凤', 295: '几', 296: '凳', 297: '凡', 298: '凸', 299: '凶', 300: '凭', 301: '凹', 302: '凛', 303: '凿', 304: '出', 305: '刁', 306: '函', 307: '刃', 308: '刀', 309: '击', 310: '冉', 311: '分', 312: '划', 313: '刑', 314: '列', 315: '凰', 316: '刘', 317: '删', 318: '创', 319: '则', 320: '刨', 321: '判', 322: '初', 323: '券', 324: '刮', 325: '刚', 326: '刹', 327: '刷', 328: '到', 329: '剁', 330: '制', 331: '刺', 332: '刻', 333: '剃', 334: '削', 335: '剐', 336: '剔', 337: '剂', 338: '剑', 339: '前', 340: '剥', 341: '剪', 342: '剧', 343: '别', 344: '剩', 345: '剖', 346: '刽', 347: '劈', 348: '副', 349: '割', 350: '务', 351: '力', 352: '劝', 353: '劣', 354: '努', 355: '办', 356: '劫', 357: '功', 358: '助', 359: '剿', 360: '励', 361: '加', 362: '勃', 363: '勒', 364: '动', 365: '劳', 366: '勘', 367: '勋', 368: '勉', 369: '势', 370: '募', 371: '勺', 372: '勤', 373: '勇', 374: '勿', 375: '匀', 376: '勾', 377: '匈', 378: '匙', 379: '包', 380: '匝', 381: '化', 382: '北', 383: '劲', 384: '匠', 385: '匣', 386: '匪', 387: '利', 388: '匿', 389: '匹', 390: '医', 391: '卉', 392: '区', 393: '卑', 394: '十', 395: '协', 396: '匆', 397: '卒', 398: '卓', 399: '华', 400: '半', 401: '午', 402: '卜', 403: '卢', 404: '单', 405: '卖', 406: '卡', 407: '南', 408: '卤', 409: '卧', 410: '博', 411: '卫', 412: '卯', 413: '印', 414: '占', 415: '卞', 416: '升', 417: '卿', 418: '千', 419: '卷', 420: '卵', 421: '卸', 422: '厉', 423: '历', 424: '厂', 425: '厅', 426: '厌', 427: '却', 428: '厢', 429: '厦', 430: '厘', 431: '厩', 432: '厚', 433: '原', 434: '厨', 435: '厕', 436: '参', 437: '厄', 438: '去', 439: '叉', 440: '压', 441: '友', 442: '及', 443: '又', 444: '叔', 445: '叁', 446: '双', 447: '发', 448: '反', 449: '叛', 450: '取', 451: '叠', 452: '受', 453: '变', 454: '口', 455: '叮', 456: '叭', 457: '古', 458: '召', 459: '另', 460: '叫', 461: '可', 462: '台', 463: '右', 464: '司', 465: '吁', 466: '叼', 467: '史', 468: '号', 469: '叶', 470: '只', 471: '吉', 472: '叹', 473: '吏', 474: '各', 475: '合', 476: '吊', 477: '吃', 478: '名', 479: '吐', 480: '吕', 481: '向', 482: '吓', 483: '后', 484: '吞', 485: '同', 486: '君', 487: '吗', 488: '吠', 489: '否', 490: '吟', 491: '吭', 492: '吧', 493: '吮', 494: '启', 495: '吱', 496: '听', 497: '吝', 498: '含', 499: '吵', 500: '吻', 501: '吸', 502: '吼', 503: '吴', 504: '吹', 505: '吩', 506: '吨', 507: '呆', 508: '呐', 509: '句', 510: '呕', 511: '告', 512: '呈', 513: '呛', 514: '呸', 515: '员', 516: '呢', 517: '味', 518: '呵', 519: '咀', 520: '呼', 521: '咋', 522: '呜', 523: '咎', 524: '命', 525: '咏', 526: '咒', 527: '周', 528: '咕', 529: '呻', 530: '和', 531: '咬', 532: '咳', 533: '咆', 534: '咽', 535: '咸', 536: '哀', 537: '咱', 538: '哄', 539: '哉', 540: '品', 541: '哇', 542: '哎', 543: '咨', 544: '咖', 545: '响', 546: '哥', 547: '哨', 548: '哦', 549: '咐', 550: '哩', 551: '哟', 552: '咙', 553: '哪', 554: '哭', 555: '哼', 556: '唤', 557: '唇', 558: '售', 559: '唐', 560: '唉', 561: '唬', 562: '唯', 563: '啄', 564: '啃', 565: '唾', 566: '唱', 567: '商', 568: '唆', 569: '啪', 570: '啮', 571: '啊', 572: '啸', 573: '啼', 574: '啥', 575: '唁', 576: '啦', 577: '喉', 578: '喀', 579: '喂', 580: '喧', 581: '善', 582: '喳', 583: '喘', 584: '喊', 585: '喻', 586: '嗅', 587: '喷', 588: '嗓', 589: '喝', 590: '嗡', 591: '啤', 592: '哺', 593: '嘉', 594: '嘘', 595: '喇', 596: '嘱', 597: '嘎', 598: '嘲', 599: '嘶', 600: '嘻', 601: '嘛', 602: '噎', 603: '嘿', 604: '噪', 605: '噬', 606: '嗽', 607: '嗣', 608: '嘴', 609: '嚎', 610: '器', 611: '嚣', 612: '嚼', 613: '囚', 614: '嚷', 615: '囊', 616: '囤', 617: '四', 618: '园', 619: '困', 620: '回', 621: '因', 622: '团', 623: '围', 624: '固', 625: '嚏', 626: '国', 627: '圆', 628: '圈', 629: '址', 630: '土', 631: '囱', 632: '坍', 633: '场', 634: '地', 635: '坑', 636: '噶', 637: '坚', 638: '均', 639: '圭', 640: '坐', 641: '坛', 642: '坝', 643: '块', 644: '坟', 645: '圾', 646: '坠', 647: '坤', 648: '坦', 649: '坯', 650: '坡', 651: '坪', 652: '垄', 653: '垂', 654: '垒', 655: '垢', 656: '垛', 657: '垣', 658: '垦', 659: '坏', 660: '垮', 661: '垫', 662: '坎', 663: '埂', 664: '型', 665: '埃', 666: '垃', 667: '域', 668: '坞', 669: '城', 670: '堑', 671: '基', 672: '堕', 673: '堂', 674: '堤', 675: '堪', 676: '堡', 677: '堰', 678: '培', 679: '堆', 680: '塑', 681: '堵', 682: '境', 683: '塞', 684: '埔', 685: '填', 686: '塘', 687: '塔', 688: '墟', 689: '增', 690: '墩', 691: '墙', 692: '墨', 693: '壤', 694: '壕', 695: '壁', 696: '壮', 697: '墒', 698: '壳', 699: '士', 700: '声', 701: '夕', 702: '备', 703: '复', 704: '夏', 705: '壬', 706: '外', 707: '处', 708: '夯', 709: '多', 710: '夜', 711: '大', 712: '壹', 713: '太', 714: '夫', 715: '夷', 716: '奈', 717: '天', 718: '奇', 719: '奋', 720: '头', 721: '奉', 722: '奏', 723: '契', 724: '奖', 725: '奠', 726: '奔', 727: '央', 728: '奥', 729: '夺', 730: '奢', 731: '奴', 732: '套', 733: '奶', 734: '奸', 735: '女', 736: '奎', 737: '奄', 738: '妆', 739: '妇', 740: '她', 741: '好', 742: '妓', 743: '妈', 744: '妖', 745: '妒', 746: '妙', 747: '妄', 748: '妨', 749: '姆', 750: '妹', 751: '姑', 752: '始', 753: '妊', 754: '姐', 755: '委', 756: '姜', 757: '姓', 758: '姚', 759: '妻', 760: '姨', 761: '姬', 762: '妥', 763: '姻', 764: '威', 765: '娟', 766: '娇', 767: '妮', 768: '娱', 769: '娃', 770: '娘', 771: '姥', 772: '娶', 773: '婉', 774: '婴', 775: '婿', 776: '婚', 777: '婶', 778: '媒', 779: '媚', 780: '娠', 781: '娩', 782: '娄', 783: '嫌', 784: '娜', 785: '嫡', 786: '嫩', 787: '孔', 788: '子', 789: '孝', 790: '存', 791: '字', 792: '孕', 793: '孟', 794: '孙', 795: '孤', 796: '孩', 797: '孰', 798: '孵', 799: '孽', 800: '宁', 801: '宅', 802: '宇', 803: '孜', 804: '宏', 805: '它', 806: '安', 807: '宋', 808: '守', 809: '完', 810: '宗', 811: '孺', 812: '学', 813: '官', 814: '宜', 815: '宝', 816: '宦', 817: '审', 818: '定', 819: '宪', 820: '宣', 821: '孪', 822: '宰', 823: '宴', 824: '室', 825: '宫', 826: '容', 827: '宾', 828: '害', 829: '宿', 830: '宽', 831: '宵', 832: '寂', 833: '宙', 834: '寅', 835: '家', 836: '客', 837: '寐', 838: '密', 839: '寝', 840: '察', 841: '富', 842: '寡', 843: '寒', 844: '寨', 845: '寓', 846: '寸', 847: '寺', 848: '寻', 849: '寿', 850: '导', 851: '对', 852: '射', 853: '封', 854: '寇', 855: '寞', 856: '尊', 857: '将', 858: '寥', 859: '尝', 860: '少', 861: '尧', 862: '小', 863: '尤', 864: '尼', 865: '尚', 866: '尉', 867: '就', 868: '尺', 869: '尿', 870: '屁', 871: '尾', 872: '屈', 873: '尽', 874: '屉', 875: '局', 876: '层', 877: '屎', 878: '屑', 879: '展', 880: '居', 881: '届', 882: '屠', 883: '屋', 884: '屡', 885: '履', 886: '屏', 887: '属', 888: '屯', 889: '尹', 890: '岂', 891: '山', 892: '岗', 893: '岔', 894: '岁', 895: '岛', 896: '尸', 897: '岩', 898: '岳', 899: '岭', 900: '峙', 901: '峡', 902: '峦', 903: '岸', 904: '峭', 905: '峻', 906: '峰', 907: '屿', 908: '崇', 909: '屹', 910: '崖', 911: '崩', 912: '崔', 913: '嵌', 914: '岿', 915: '巡', 916: '州', 917: '川', 918: '巧', 919: '巨', 920: '巢', 921: '左', 922: '崎', 923: '巫', 924: '工', 925: '巳', 926: '差', 927: '巷', 928: '己', 929: '币', 930: '崭', 931: '已', 932: '帅', 933: '希', 934: '帐', 935: '布', 936: '帕', 937: '帖', 938: '帆', 939: '师', 940: '帘', 941: '帛', 942: '帜', 943: '帚', 944: '帝', 945: '帧', 946: '席', 947: '幂', 948: '带', 949: '帽', 950: '市', 951: '幌', 952: '巾', 953: '幕', 954: '幅', 955: '常', 956: '幢', 957: '帮', 958: '幸', 959: '平', 960: '幽', 961: '年', 962: '庄', 963: '干', 964: '广', 965: '庐', 966: '幼', 967: '库', 968: '床', 969: '序', 970: '庆', 971: '庚', 972: '庞', 973: '府', 974: '应', 975: '底', 976: '庙', 977: '废', 978: '庇', 979: '庶', 980: '度', 981: '店', 982: '庭', 983: '康', 984: '廉', 985: '庸', 986: '廓', 987: '廖', 988: '廊', 989: '廷', 990: '延', 991: '弃', 992: '建', 993: '弊', 994: '异', 995: '弄', 996: '幻', 997: '弓', 998: '开', 999: '弗', 1000: '弛', 1001: '座', 1002: '弘', 1003: '式', 1004: '引', 1005: '弦', 1006: '张', 1007: '弯', 1008: '弹', 1009: '录', 1010: '彝', 1011: '弱', 1012: '当', 1013: '归', 1014: '弧', 1015: '彰', 1016: '强', 1017: '形', 1018: '彩', 1019: '役', 1020: '彻', 1021: '彼', 1022: '影', 1023: '彭', 1024: '征', 1025: '径', 1026: '彤', 1027: '往', 1028: '彦', 1029: '徐', 1030: '很', 1031: '律', 1032: '彬', 1033: '循', 1034: '待', 1035: '御', 1036: '徽', 1037: '得', 1038: '微', 1039: '忠', 1040: '忍', 1041: '心', 1042: '德', 1043: '志', 1044: '必', 1045: '徊', 1046: '忧', 1047: '忙', 1048: '忘', 1049: '徘', 1050: '忿', 1051: '怎', 1052: '念', 1053: '怔', 1054: '怒', 1055: '怜', 1056: '态', 1057: '怀', 1058: '忽', 1059: '怕', 1060: '怠', 1061: '忱', 1062: '怯', 1063: '忻', 1064: '思', 1065: '急', 1066: '怂', 1067: '恋', 1068: '性', 1069: '恐', 1070: '怪', 1071: '恒', 1072: '恤', 1073: '怖', 1074: '恕', 1075: '恭', 1076: '恨', 1077: '息', 1078: '恰', 1079: '恼', 1080: '恩', 1081: '恶', 1082: '悔', 1083: '恍', 1084: '悉', 1085: '悍', 1086: '悟', 1087: '恬', 1088: '悦', 1089: '恫', 1090: '恢', 1091: '悲', 1092: '患', 1093: '您', 1094: '惑', 1095: '恳', 1096: '恿', 1097: '惊', 1098: '情', 1099: '惦', 1100: '惟', 1101: '惠', 1102: '惩', 1103: '惯', 1104: '惨', 1105: '惶', 1106: '悼', 1107: '惰', 1108: '惹', 1109: '惧', 1110: '想', 1111: '惕', 1112: '愚', 1113: '惫', 1114: '惭', 1115: '慈', 1116: '惮', 1117: '意', 1118: '慎', 1119: '愈', 1120: '慕', 1121: '慌', 1122: '慧', 1123: '慰', 1124: '惺', 1125: '憎', 1126: '慢', 1127: '憾', 1128: '憨', 1129: '愉', 1130: '憋', 1131: '懂', 1132: '戈', 1133: '戊', 1134: '戌', 1135: '戎', 1136: '慑', 1137: '戒', 1138: '戚', 1139: '懊', 1140: '戏', 1141: '成', 1142: '战', 1143: '懈', 1144: '或', 1145: '戳', 1146: '截', 1147: '懦', 1148: '慷', 1149: '扁', 1150: '扒', 1151: '扎', 1152: '房', 1153: '所', 1154: '扔', 1155: '戮', 1156: '扑', 1157: '扦', 1158: '打', 1159: '扣', 1160: '才', 1161: '执', 1162: '扛', 1163: '扩', 1164: '扮', 1165: '扫', 1166: '扰', 1167: '扯', 1168: '扭', 1169: '扳', 1170: '扼', 1171: '扶', 1172: '承', 1173: '扬', 1174: '抄', 1175: '找', 1176: '技', 1177: '抑', 1178: '批', 1179: '托', 1180: '抖', 1181: '抒', 1182: '投', 1183: '抚', 1184: '抠', 1185: '抓', 1186: '把', 1187: '抡', 1188: '抗', 1189: '护', 1190: '抛', 1191: '抉', 1192: '抢', 1193: '披', 1194: '押', 1195: '报', 1196: '抿', 1197: '抵', 1198: '抬', 1199: '抹', 1200: '拄', 1201: '拈', 1202: '抽', 1203: '抱', 1204: '担', 1205: '拆', 1206: '拎', 1207: '拐', 1208: '拉', 1209: '拒', 1210: '拓', 1211: '拌', 1212: '拂', 1213: '拍', 1214: '招', 1215: '拜', 1216: '抨', 1217: '拙', 1218: '拘', 1219: '拖', 1220: '拟', 1221: '拢', 1222: '拇', 1223: '拦', 1224: '拧', 1225: '拨', 1226: '拥', 1227: '括', 1228: '择', 1229: '拴', 1230: '拷', 1231: '拳', 1232: '拱', 1233: '拭', 1234: '拼', 1235: '拽', 1236: '拾', 1237: '拿', 1238: '持', 1239: '拣', 1240: '指', 1241: '挎', 1242: '按', 1243: '挂', 1244: '拯', 1245: '挑', 1246: '挟', 1247: '挡', 1248: '挫', 1249: '振', 1250: '挽', 1251: '挤', 1252: '挪', 1253: '捂', 1254: '挛', 1255: '挨', 1256: '挺', 1257: '捅', 1258: '挥', 1259: '捆', 1260: '挞', 1261: '挚', 1262: '挝', 1263: '捐', 1264: '损', 1265: '捞', 1266: '捡', 1267: '捕', 1268: '捣', 1269: '捧', 1270: '捷', 1271: '捏', 1272: '捻', 1273: '捶', 1274: '换', 1275: '掀', 1276: '掂', 1277: '捎', 1278: '据', 1279: '授', 1280: '掐', 1281: '掏', 1282: '掉', 1283: '掠', 1284: '掌', 1285: '探', 1286: '掘', 1287: '掣', 1288: '排', 1289: '控', 1290: '掩', 1291: '接', 1292: '掖', 1293: '掳', 1294: '掸', 1295: '掺', 1296: '推', 1297: '描', 1298: '揍', 1299: '揉', 1300: '揖', 1301: '插', 1302: '掷', 1303: '握', 1304: '揪', 1305: '提', 1306: '揭', 1307: '揩', 1308: '揣', 1309: '援', 1310: '搀', 1311: '揽', 1312: '搂', 1313: '搐', 1314: '搅', 1315: '措', 1316: '搔', 1317: '搜', 1318: '搪', 1319: '搓', 1320: '搏', 1321: '搞', 1322: '搬', 1323: '搽', 1324: '搁', 1325: '搭', 1326: '摄', 1327: '摊', 1328: '摆', 1329: '摘', 1330: '摧', 1331: '摩', 1332: '摔', 1333: '撂', 1334: '摇', 1335: '撅', 1336: '撇', 1337: '摸', 1338: '撑', 1339: '撕', 1340: '撒', 1341: '撞', 1342: '撬', 1343: '撤', 1344: '撮', 1345: '撰', 1346: '撵', 1347: '播', 1348: '摈', 1349: '撼', 1350: '撩', 1351: '擂', 1352: '擎', 1353: '操', 1354: '擒', 1355: '攀', 1356: '擅', 1357: '攒', 1358: '攘', 1359: '擦', 1360: '摹', 1361: '攫', 1362: '收', 1363: '攻', 1364: '支', 1365: '效', 1366: '敏', 1367: '改', 1368: '政', 1369: '擞', 1370: '放', 1371: '故', 1372: '敛', 1373: '敝', 1374: '敞', 1375: '教', 1376: '救', 1377: '敷', 1378: '敲', 1379: '整', 1380: '斋', 1381: '敬', 1382: '敦', 1383: '数', 1384: '文', 1385: '敖', 1386: '斟', 1387: '散', 1388: '斗', 1389: '斧', 1390: '斥', 1391: '斜', 1392: '斯', 1393: '斤', 1394: '斩', 1395: '断', 1396: '新', 1397: '斌', 1398: '方', 1399: '施', 1400: '旁', 1401: '旗', 1402: '旋', 1403: '无', 1404: '族', 1405: '旨', 1406: '既', 1407: '旦', 1408: '日', 1409: '旷', 1410: '斡', 1411: '旺', 1412: '昂', 1413: '旧', 1414: '早', 1415: '昌', 1416: '昏', 1417: '昔', 1418: '旱', 1419: '明', 1420: '易', 1421: '昧', 1422: '映', 1423: '星', 1424: '昨', 1425: '时', 1426: '春', 1427: '旬', 1428: '昭', 1429: '晌', 1430: '旅', 1431: '晕', 1432: '显', 1433: '晓', 1434: '是', 1435: '晤', 1436: '晦', 1437: '晒', 1438: '晚', 1439: '晋', 1440: '晨', 1441: '晃', 1442: '旭', 1443: '普', 1444: '景', 1445: '智', 1446: '晾', 1447: '暂', 1448: '暇', 1449: '晶', 1450: '暑', 1451: '晴', 1452: '暴', 1453: '暮', 1454: '曝', 1455: '暗', 1456: '曳', 1457: '暖', 1458: '曼', 1459: '曲', 1460: '曹', 1461: '曰', 1462: '朋', 1463: '更', 1464: '曾', 1465: '朗', 1466: '替', 1467: '月', 1468: '曙', 1469: '服', 1470: '有', 1471: '朝', 1472: '札', 1473: '木', 1474: '朴', 1475: '末', 1476: '未', 1477: '期', 1478: '朽', 1479: '本', 1480: '朵', 1481: '机', 1482: '杂', 1483: '杉', 1484: '杀', 1485: '杏', 1486: '材', 1487: '晰', 1488: '杖', 1489: '权', 1490: '杠', 1491: '杜', 1492: '村', 1493: '李', 1494: '杆', 1495: '束', 1496: '杭', 1497: '杰', 1498: '条', 1499: '杨', 1500: '朱', 1501: '杯', 1502: '来', 1503: '构', 1504: '析', 1505: '枉', 1506: '板', 1507: '枕', 1508: '极', 1509: '枣', 1510: '枚', 1511: '林', 1512: '术', 1513: '松', 1514: '枯', 1515: '枫', 1516: '枢', 1517: '枪', 1518: '柄', 1519: '架', 1520: '柏', 1521: '柔', 1522: '枝', 1523: '柜', 1524: '某', 1525: '柑', 1526: '染', 1527: '柬', 1528: '查', 1529: '柱', 1530: '柿', 1531: '柴', 1532: '枷', 1533: '栅', 1534: '柳', 1535: '柒', 1536: '栈', 1537: '栋', 1538: '标', 1539: '栓', 1540: '柞', 1541: '柠', 1542: '栗', 1543: '树', 1544: '样', 1545: '栖', 1546: '校', 1547: '核', 1548: '根', 1549: '株', 1550: '格', 1551: '栽', 1552: '桃', 1553: '桂', 1554: '框', 1555: '桐', 1556: '桌', 1557: '案', 1558: '桑', 1559: '档', 1560: '桶', 1561: '梁', 1562: '桨', 1563: '梅', 1564: '梆', 1565: '桩', 1566: '桔', 1567: '梗', 1568: '桥', 1569: '梧', 1570: '梭', 1571: '梢', 1572: '梨', 1573: '梯', 1574: '梦', 1575: '梳', 1576: '检', 1577: '棋', 1578: '械', 1579: '桅', 1580: '棉', 1581: '棘', 1582: '棒', 1583: '棕', 1584: '棱', 1585: '桓', 1586: '棺', 1587: '椅', 1588: '森', 1589: '椒', 1590: '植', 1591: '椭', 1592: '椎', 1593: '棵', 1594: '椰', 1595: '棚', 1596: '椿', 1597: '楚', 1598: '楔', 1599: '楞', 1600: '棠', 1601: '榆', 1602: '榨', 1603: '榜', 1604: '楼', 1605: '概', 1606: '槐', 1607: '槽', 1608: '模', 1609: '橙', 1610: '楷', 1611: '榔', 1612: '横', 1613: '橱', 1614: '榷', 1615: '欣', 1616: '欧', 1617: '欢', 1618: '槛', 1619: '次', 1620: '樊', 1621: '歉', 1622: '歇', 1623: '橡', 1624: '止', 1625: '檀', 1626: '正', 1627: '檬', 1628: '檄', 1629: '歹', 1630: '歌', 1631: '此', 1632: '歼', 1633: '殆', 1634: '殉', 1635: '殊', 1636: '殷', 1637: '殿', 1638: '残', 1639: '死', 1640: '毁', 1641: '毋', 1642: '殖', 1643: '段', 1644: '殃', 1645: '歪', 1646: '母', 1647: '毙', 1648: '每', 1649: '毡', 1650: '毯', 1651: '比', 1652: '毫', 1653: '毛', 1654: '氖', 1655: '氟', 1656: '毅', 1657: '氦', 1658: '民', 1659: '毖', 1660: '氢', 1661: '氨', 1662: '氰', 1663: '气', 1664: '氯', 1665: '氮', 1666: '汁', 1667: '永', 1668: '氧', 1669: '水', 1670: '求', 1671: '汛', 1672: '汝', 1673: '汞', 1674: '氛', 1675: '汉', 1676: '汗', 1677: '污', 1678: '汰', 1679: '池', 1680: '江', 1681: '汪', 1682: '汲', 1683: '汽', 1684: '汤', 1685: '汕', 1686: '沁', 1687: '沈', 1688: '沏', 1689: '沉', 1690: '汐', 1691: '沙', 1692: '沤', 1693: '沦', 1694: '沟', 1695: '没', 1696: '沂', 1697: '沫', 1698: '沼', 1699: '沪', 1700: '河', 1701: '沛', 1702: '油', 1703: '沾', 1704: '泄', 1705: '汾', 1706: '泉', 1707: '治', 1708: '泊', 1709: '泌', 1710: '沧', 1711: '泣', 1712: '泛', 1713: '泰', 1714: '沿', 1715: '泳', 1716: '泪', 1717: '泅', 1718: '波', 1719: '注', 1720: '沮', 1721: '泻', 1722: '泵', 1723: '泥', 1724: '泽', 1725: '泼', 1726: '洒', 1727: '津', 1728: '洲', 1729: '洪', 1730: '洗', 1731: '洼', 1732: '洞', 1733: '洽', 1734: '活', 1735: '洁', 1736: '洋', 1737: '浊', 1738: '浇', 1739: '派', 1740: '浅', 1741: '流', 1742: '浙', 1743: '浑', 1744: '济', 1745: '浦', 1746: '浓', 1747: '洛', 1748: '浩', 1749: '浸', 1750: '浴', 1751: '涅', 1752: '海', 1753: '消', 1754: '涉', 1755: '涂', 1756: '涎', 1757: '浮', 1758: '涛', 1759: '涌', 1760: '涕', 1761: '涝', 1762: '洱', 1763: '浚', 1764: '浪', 1765: '涤', 1766: '润', 1767: '涩', 1768: '涨', 1769: '涵', 1770: '涧', 1771: '涡', 1772: '涸', 1773: '液', 1774: '淋', 1775: '淌', 1776: '淘', 1777: '淤', 1778: '淫', 1779: '涣', 1780: '涯', 1781: '淹', 1782: '混', 1783: '深', 1784: '渍', 1785: '添', 1786: '渊', 1787: '淆', 1788: '渐', 1789: '淄', 1790: '淀', 1791: '淖', 1792: '渡', 1793: '淑', 1794: '港', 1795: '渺', 1796: '淳', 1797: '渴', 1798: '游', 1799: '湛', 1800: '湖', 1801: '湾', 1802: '湘', 1803: '溃', 1804: '溅', 1805: '渣', 1806: '湿', 1807: '温', 1808: '溢', 1809: '溜', 1810: '渭', 1811: '溪', 1812: '溺', 1813: '滋', 1814: '湃', 1815: '溶', 1816: '湍', 1817: '滇', 1818: '滑', 1819: '滤', 1820: '滥', 1821: '滚', 1822: '滞', 1823: '滨', 1824: '满', 1825: '滦', 1826: '漆', 1827: '漂', 1828: '滩', 1829: '漏', 1830: '滓', 1831: '滴', 1832: '漾', 1833: '漱', 1834: '演', 1835: '漫', 1836: '滔', 1837: '潜', 1838: '潘', 1839: '潭', 1840: '澄', 1841: '澈', 1842: '潮', 1843: '澜', 1844: '澡', 1845: '漓', 1846: '澳', 1847: '漳', 1848: '濒', 1849: '瀑', 1850: '潍', 1851: '灭', 1852: '灌', 1853: '灯', 1854: '灸', 1855: '灶', 1856: '灵', 1857: '火', 1858: '灿', 1859: '澎', 1860: '炊', 1861: '灾', 1862: '灰', 1863: '炎', 1864: '炉', 1865: '炒', 1866: '炳', 1867: '炕', 1868: '炭', 1869: '炬', 1870: '炮', 1871: '炸', 1872: '炙', 1873: '炼', 1874: '炽', 1875: '点', 1876: '烈', 1877: '烘', 1878: '烙', 1879: '烂', 1880: '烃', 1881: '烛', 1882: '烦', 1883: '烤', 1884: '烬', 1885: '烧', 1886: '烫', 1887: '烹', 1888: '热', 1889: '焉', 1890: '烷', 1891: '烁', 1892: '焙', 1893: '焊', 1894: '炔', 1895: '焰', 1896: '煞', 1897: '烩', 1898: '然', 1899: '烽', 1900: '煤', 1901: '烯', 1902: '熄', 1903: '熏', 1904: '熊', 1905: '照', 1906: '熔', 1907: '焕', 1908: '燃', 1909: '燎', 1910: '熬', 1911: '爆', 1912: '爪', 1913: '爵', 1914: '煌', 1915: '燥', 1916: '爱', 1917: '父', 1918: '爷', 1919: '爸', 1920: '爬', 1921: '燕', 1922: '煽', 1923: '版', 1924: '爽', 1925: '牌', 1926: '牢', 1927: '片', 1928: '牧', 1929: '牲', 1930: '特', 1931: '牵', 1932: '犬', 1933: '犁', 1934: '物', 1935: '犹', 1936: '牛', 1937: '犊', 1938: '狄', 1939: '犯', 1940: '牡', 1941: '状', 1942: '狐', 1943: '狞', 1944: '狭', 1945: '狗', 1946: '犀', 1947: '狸', 1948: '狱', 1949: '狮', 1950: '牺', 1951: '猎', 1952: '狼', 1953: '狈', 1954: '独', 1955: '狙', 1956: '猜', 1957: '献', 1958: '猴', 1959: '猿', 1960: '狰', 1961: '猪', 1962: '玄', 1963: '猫', 1964: '玉', 1965: '猖', 1966: '王', 1967: '玻', 1968: '玩', 1969: '环', 1970: '猾', 1971: '珍', 1972: '珠', 1973: '獭', 1974: '班', 1975: '琴', 1976: '球', 1977: '玫', 1978: '理', 1979: '琢', 1980: '玛', 1981: '珐', 1982: '瑟', 1983: '珊', 1984: '玲', 1985: '瓜', 1986: '瓣', 1987: '瓤', 1988: '瓦', 1989: '瓢', 1990: '瓮', 1991: '瑰', 1992: '瑶', 1993: '琵', 1994: '瓷', 1995: '琳', 1996: '甜', 1997: '甘', 1998: '甚', 1999: '甩', 2000: '甫', 2001: '生', 2002: '甭', 2003: '璃', 2004: '用', 2005: '申', 2006: '田', 2007: '畏', 2008: '甲', 2009: '男', 2010: '由', 2011: '电', 2012: '畔', 2013: '界', 2014: '畦', 2015: '畜', 2016: '畅', 2017: '留', 2018: '疆', 2019: '番', 2020: '略', 2021: '疑', 2022: '疗', 2023: '疏', 2024: '疟', 2025: '疤', 2026: '畸', 2027: '甥', 2028: '疮', 2029: '疫', 2030: '疯', 2031: '疲', 2032: '畴', 2033: '疵', 2034: '疹', 2035: '疽', 2036: '疼', 2037: '疙', 2038: '疚', 2039: '画', 2040: '疡', 2041: '疥', 2042: '痒', 2043: '病', 2044: '痘', 2045: '痕', 2046: '痞', 2047: '痢', 2048: '痹', 2049: '痛', 2050: '痰', 2051: '瘤', 2052: '瘟', 2053: '瘦', 2054: '瘪', 2055: '痴', 2056: '痊', 2057: '瘫', 2058: '痔', 2059: '癣', 2060: '癌', 2061: '皂', 2062: '登', 2063: '白', 2064: '百', 2065: '瘁', 2066: '瘩', 2067: '皇', 2068: '皆', 2069: '皖', 2070: '盂', 2071: '皿', 2072: '盅', 2073: '皮', 2074: '皱', 2075: '盏', 2076: '盔', 2077: '皋', 2078: '盒', 2079: '盐', 2080: '监', 2081: '的', 2082: '盗', 2083: '盟', 2084: '盖', 2085: '盘', 2086: '盲', 2087: '盛', 2088: '盯', 2089: '盼', 2090: '目', 2091: '眉', 2092: '直', 2093: '眠', 2094: '盾', 2095: '眩', 2096: '省', 2097: '看', 2098: '眶', 2099: '眨', 2100: '真', 2101: '眷', 2102: '眯', 2103: '睛', 2104: '督', 2105: '眼', 2106: '睁', 2107: '着', 2108: '睬', 2109: '睡', 2110: '睹', 2111: '瞄', 2112: '眺', 2113: '盎', 2114: '瞅', 2115: '瞒', 2116: '瞳', 2117: '瞻', 2118: '瞬', 2119: '相', 2120: '瞪', 2121: '瞧', 2122: '瞥', 2123: '睦', 2124: '睫', 2125: '矛', 2126: '矢', 2127: '矫', 2128: '矣', 2129: '矽', 2130: '矮', 2131: '矾', 2132: '矿', 2133: '码', 2134: '石', 2135: '矩', 2136: '砂', 2137: '瞩', 2138: '短', 2139: '砍', 2140: '研', 2141: '砚', 2142: '砌', 2143: '砰', 2144: '砷', 2145: '砧', 2146: '破', 2147: '硒', 2148: '硅', 2149: '硝', 2150: '砒', 2151: '硷', 2152: '确', 2153: '硫', 2154: '砸', 2155: '碍', 2156: '硬', 2157: '碑', 2158: '碎', 2159: '碘', 2160: '砾', 2161: '碗', 2162: '碧', 2163: '碴', 2164: '础', 2165: '硕', 2166: '磅', 2167: '碳', 2168: '碱', 2169: '碉', 2170: '磁', 2171: '磕', 2172: '礁', 2173: '碌', 2174: '磷', 2175: '礼', 2176: '示', 2177: '祝', 2178: '祖', 2179: '磨', 2180: '社', 2181: '祥', 2182: '磊', 2183: '票', 2184: '祷', 2185: '祭', 2186: '磺', 2187: '禹', 2188: '神', 2189: '祁', 2190: '福', 2191: '禾', 2192: '秀', 2193: '离', 2194: '私', 2195: '禽', 2196: '禄', 2197: '秃', 2198: '禁', 2199: '秘', 2200: '秋', 2201: '租', 2202: '秤', 2203: '秒', 2204: '科', 2205: '秧', 2206: '秦', 2207: '积', 2208: '祟', 2209: '秽', 2210: '称', 2211: '稗', 2212: '种', 2213: '稀', 2214: '稠', 2215: '移', 2216: '税', 2217: '稚', 2218: '稳', 2219: '稼', 2220: '稻', 2221: '秉', 2222: '穗', 2223: '穆', 2224: '穴', 2225: '稍', 2226: '突', 2227: '究', 2228: '穷', 2229: '稿', 2230: '空', 2231: '窃', 2232: '穿', 2233: '窍', 2234: '窑', 2235: '窄', 2236: '秩', 2237: '窘', 2238: '窜', 2239: '窗', 2240: '窟', 2241: '窿', 2242: '竖', 2243: '窥', 2244: '立', 2245: '竞', 2246: '窝', 2247: '站', 2248: '窖', 2249: '窒', 2250: '竭', 2251: '章', 2252: '竟', 2253: '竿', 2254: '竹', 2255: '符', 2256: '笛', 2257: '端', 2258: '笺', 2259: '笔', 2260: '笑', 2261: '笨', 2262: '笼', 2263: '筋', 2264: '竣', 2265: '筏', 2266: '第', 2267: '筐', 2268: '筷', 2269: '筑', 2270: '筒', 2271: '筛', 2272: '筹', 2273: '等', 2274: '签', 2275: '箍', 2276: '笆', 2277: '简', 2278: '箔', 2279: '箕', 2280: '策', 2281: '箩', 2282: '篆', 2283: '算', 2284: '篓', 2285: '管', 2286: '箱', 2287: '篙', 2288: '篇', 2289: '篮', 2290: '箭', 2291: '篱', 2292: '篷', 2293: '簧', 2294: '簿', 2295: '答', 2296: '籽', 2297: '米', 2298: '类', 2299: '籍', 2300: '篡', 2301: '粒', 2302: '粉', 2303: '粤', 2304: '粟', 2305: '粗', 2306: '粥', 2307: '粘', 2308: '粹', 2309: '糊', 2310: '糕', 2311: '粮', 2312: '糜', 2313: '粕', 2314: '糟', 2315: '糠', 2316: '糖', 2317: '索', 2318: '系', 2319: '素', 2320: '粱', 2321: '絮', 2322: '精', 2323: '繁', 2324: '紧', 2325: '纠', 2326: '纂', 2327: '累', 2328: '紫', 2329: '纤', 2330: '纬', 2331: '纱', 2332: '红', 2333: '纲', 2334: '纵', 2335: '纷', 2336: '纳', 2337: '紊', 2338: '纪', 2339: '纹', 2340: '纽', 2341: '纺', 2342: '纸', 2343: '纯', 2344: '纫', 2345: '绅', 2346: '线', 2347: '织', 2348: '纶', 2349: '绊', 2350: '细', 2351: '终', 2352: '绒', 2353: '绑', 2354: '结', 2355: '绘', 2356: '绕', 2357: '统', 2358: '绞', 2359: '给', 2360: '绢', 2361: '绣', 2362: '经', 2363: '绥', 2364: '绝', 2365: '绩', 2366: '络', 2367: '续', 2368: '绍', 2369: '绪', 2370: '绳', 2371: '绚', 2372: '绷', 2373: '绸', 2374: '综', 2375: '绵', 2376: '缀', 2377: '绽', 2378: '缅', 2379: '缆', 2380: '绿', 2381: '缎', 2382: '维', 2383: '缓', 2384: '绰', 2385: '缕', 2386: '缔', 2387: '编', 2388: '缘', 2389: '缚', 2390: '缨', 2391: '缠', 2392: '缄', 2393: '缩', 2394: '缸', 2395: '缴', 2396: '罕', 2397: '缺', 2398: '罗', 2399: '网', 2400: '罩', 2401: '罚', 2402: '罢', 2403: '罪', 2404: '置', 2405: '署', 2406: '羌', 2407: '缮', 2408: '羡', 2409: '羹', 2410: '羊', 2411: '美', 2412: '羽', 2413: '翅', 2414: '翌', 2415: '翁', 2416: '翘', 2417: '翔', 2418: '羞', 2419: '翠', 2420: '群', 2421: '羚', 2422: '耀', 2423: '翼', 2424: '翻', 2425: '考', 2426: '耍', 2427: '老', 2428: '耐', 2429: '耙', 2430: '耕', 2431: '者', 2432: '翱', 2433: '耸', 2434: '耻', 2435: '而', 2436: '耶', 2437: '聂', 2438: '聊', 2439: '聋', 2440: '聘', 2441: '联', 2442: '聚', 2443: '翰', 2444: '耿', 2445: '职', 2446: '肃', 2447: '肆', 2448: '耘', 2449: '耪', 2450: '肋', 2451: '肘', 2452: '肖', 2453: '肉', 2454: '肠', 2455: '肝', 2456: '肚', 2457: '肤', 2458: '肢', 2459: '股', 2460: '聪', 2461: '肩', 2462: '肥', 2463: '育', 2464: '肿', 2465: '肾', 2466: '肛', 2467: '胆', 2468: '肺', 2469: '胃', 2470: '胎', 2471: '胚', 2472: '背', 2473: '胞', 2474: '胜', 2475: '胖', 2476: '胡', 2477: '肪', 2478: '胳', 2479: '胶', 2480: '胺', 2481: '肮', 2482: '胸', 2483: '脂', 2484: '脊', 2485: '能', 2486: '脉', 2487: '脏', 2488: '脓', 2489: '脐', 2490: '脖', 2491: '胰', 2492: '脑', 2493: '脆', 2494: '胯', 2495: '脚', 2496: '腊', 2497: '腑', 2498: '腐', 2499: '脾', 2500: '腕', 2501: '腋', 2502: '腔', 2503: '腥', 2504: '腹', 2505: '腮', 2506: '脸', 2507: '腻', 2508: '腺', 2509: '腰', 2510: '脯', 2511: '膏', 2512: '腾', 2513: '膊', 2514: '膛', 2515: '腆', 2516: '腿', 2517: '膝', 2518: '膨', 2519: '膜', 2520: '臀', 2521: '臂', 2522: '臣', 2523: '臭', 2524: '膳', 2525: '臼', 2526: '舀', 2527: '臻', 2528: '舆', 2529: '自', 2530: '至', 2531: '舅', 2532: '致', 2533: '臃', 2534: '舌', 2535: '舒', 2536: '舔', 2537: '舍', 2538: '航', 2539: '舱', 2540: '舵', 2541: '舷', 2542: '舞', 2543: '般', 2544: '舰', 2545: '艇', 2546: '舟', 2547: '艰', 2548: '艘', 2549: '船', 2550: '良', 2551: '艺', 2552: '芋', 2553: '色', 2554: '艾', 2555: '芒', 2556: '节', 2557: '芜', 2558: '舶', 2559: '芦', 2560: '芬', 2561: '芯', 2562: '苇', 2563: '花', 2564: '芍', 2565: '芽', 2566: '苔', 2567: '苍', 2568: '苑', 2569: '苏', 2570: '苛', 2571: '芳', 2572: '苟', 2573: '苫', 2574: '苗', 2575: '芝', 2576: '芥', 2577: '芭', 2578: '芹', 2579: '茁', 2580: '茄', 2581: '茂', 2582: '范', 2583: '茅', 2584: '茧', 2585: '茎', 2586: '茸', 2587: '茬', 2588: '茫', 2589: '茶', 2590: '荐', 2591: '荒', 2592: '苹', 2593: '荚', 2594: '荤', 2595: '英', 2596: '草', 2597: '荣', 2598: '荷', 2599: '茨', 2600: '茵', 2601: '茹', 2602: '莹', 2603: '莲', 2604: '莽', 2605: '荔', 2606: '获', 2607: '莱', 2608: '菊', 2609: '菌', 2610: '莫', 2611: '菱', 2612: '萌', 2613: '荧', 2614: '萎', 2615: '萝', 2616: '菜', 2617: '萍', 2618: '萤', 2619: '莉', 2620: '萨', 2621: '葛', 2622: '著', 2623: '菠', 2624: '营', 2625: '菩', 2626: '董', 2627: '葱', 2628: '葬', 2629: '蒂', 2630: '蒙', 2631: '蒜', 2632: '葡', 2633: '蒋', 2634: '萄', 2635: '蓄', 2636: '蓝', 2637: '蓬', 2638: '蒸', 2639: '蔓', 2640: '蔚', 2641: '蔡', 2642: '蔫', 2643: '葫', 2644: '蔽', 2645: '蓑', 2646: '蓉', 2647: '蕾', 2648: '蓟', 2649: '薛', 2650: '薪', 2651: '蕴', 2652: '藉', 2653: '蔑', 2654: '薄', 2655: '藤', 2656: '藩', 2657: '蔼', 2658: '藏', 2659: '蘸', 2660: '藻', 2661: '蔷', 2662: '虏', 2663: '虎', 2664: '虐', 2665: '虞', 2666: '虱', 2667: '虹', 2668: '虚', 2669: '藐', 2670: '蚀', 2671: '虽', 2672: '蚌', 2673: '蚁', 2674: '蚊', 2675: '虫', 2676: '蚕', 2677: '虾', 2678: '蛆', 2679: '蛊', 2680: '蛙', 2681: '蛤', 2682: '蛋', 2683: '蛮', 2684: '蛰', 2685: '蛾', 2686: '蚂', 2687: '蛹', 2688: '蚜', 2689: '蛇', 2690: '蚤', 2691: '蜕', 2692: '蜜', 2693: '蛔', 2694: '蝇', 2695: '蜡', 2696: '蝉', 2697: '蝶', 2698: '蛛', 2699: '融', 2700: '蟹', 2701: '螺', 2702: '蝗', 2703: '蜒', 2704: '蜘', 2705: '衔', 2706: '衙', 2707: '衡', 2708: '血', 2709: '街', 2710: '行', 2711: '蝴', 2712: '螟', 2713: '衰', 2714: '衬', 2715: '衣', 2716: '衫', 2717: '补', 2718: '袒', 2719: '袋', 2720: '袖', 2721: '袜', 2722: '袍', 2723: '袭', 2724: '裔', 2725: '被', 2726: '裕', 2727: '裂', 2728: '裁', 2729: '袄', 2730: '装', 2731: '裙', 2732: '裤', 2733: '裸', 2734: '裴', 2735: '裳', 2736: '衷', 2737: '褐', 2738: '褒', 2739: '褂', 2740: '襟', 2741: '褪', 2742: '袱', 2743: '覆', 2744: '褥', 2745: '觅', 2746: '规', 2747: '要', 2748: '西', 2749: '览', 2750: '观', 2751: '视', 2752: '见', 2753: '襄', 2754: '觉', 2755: '誓', 2756: '角', 2757: '解', 2758: '警', 2759: '言', 2760: '订', 2761: '计', 2762: '认', 2763: '讨', 2764: '训', 2765: '詹', 2766: '讯', 2767: '议', 2768: '讳', 2769: '记', 2770: '讣', 2771: '讹', 2772: '讲', 2773: '讼', 2774: '让', 2775: '讽', 2776: '诀', 2777: '论', 2778: '讫', 2779: '譬', 2780: '访', 2781: '识', 2782: '设', 2783: '诉', 2784: '诊', 2785: '诈', 2786: '译', 2787: '诛', 2788: '讶', 2789: '词', 2790: '诞', 2791: '诚', 2792: '诡', 2793: '试', 2794: '询', 2795: '诗', 2796: '话', 2797: '诅', 2798: '详', 2799: '误', 2800: '诱', 2801: '该', 2802: '诵', 2803: '诬', 2804: '语', 2805: '说', 2806: '诫', 2807: '请', 2808: '诣', 2809: '诸', 2810: '读', 2811: '诌', 2812: '谊', 2813: '谁', 2814: '诲', 2815: '谋', 2816: '谐', 2817: '谈', 2818: '谎', 2819: '谚', 2820: '谓', 2821: '谜', 2822: '谣', 2823: '诽', 2824: '谦', 2825: '谆', 2826: '谬', 2827: '谭', 2828: '谨', 2829: '谱', 2830: '豆', 2831: '豁', 2832: '谷', 2833: '谤', 2834: '豹', 2835: '豪', 2836: '豫', 2837: '诺', 2838: '谩', 2839: '貌', 2840: '谰', 2841: '谴', 2842: '责', 2843: '财', 2844: '负', 2845: '败', 2846: '贤', 2847: '账', 2848: '豌', 2849: '豢', 2850: '贩', 2851: '贫', 2852: '贪', 2853: '贬', 2854: '贮', 2855: '豺', 2856: '贱', 2857: '贷', 2858: '质', 2859: '贴', 2860: '贵', 2861: '贺', 2862: '贯', 2863: '贼', 2864: '费', 2865: '赃', 2866: '资', 2867: '赌', 2868: '赎', 2869: '赋', 2870: '赊', 2871: '贰', 2872: '赐', 2873: '赘', 2874: '贸', 2875: '赚', 2876: '赞', 2877: '赛', 2878: '赖', 2879: '赂', 2880: '贿', 2881: '赠', 2882: '赣', 2883: '赦', 2884: '赫', 2885: '赤', 2886: '赢', 2887: '赴', 2888: '赵', 2889: '趁', 2890: '赶', 2891: '趋', 2892: '超', 2893: '趣', 2894: '起', 2895: '越', 2896: '趟', 2897: '趾', 2898: '跃', 2899: '趴', 2900: '走', 2901: '跌', 2902: '足', 2903: '践', 2904: '跨', 2905: '跺', 2906: '跪', 2907: '路', 2908: '跳', 2909: '踞', 2910: '踏', 2911: '跟', 2912: '踪', 2913: '踢', 2914: '踩', 2915: '蹄', 2916: '蹈', 2917: '距', 2918: '蹬', 2919: '蹭', 2920: '蹿', 2921: '躁', 2922: '蹦', 2923: '蹲', 2924: '踌', 2925: '躯', 2926: '踊', 2927: '躬', 2928: '躲', 2929: '蹋', 2930: '轧', 2931: '轩', 2932: '躺', 2933: '轨', 2934: '车', 2935: '轰', 2936: '轮', 2937: '躇', 2938: '软', 2939: '轴', 2940: '轿', 2941: '辉', 2942: '辈', 2943: '轻', 2944: '转', 2945: '辐', 2946: '辑', 2947: '辅', 2948: '辕', 2949: '辖', 2950: '辆', 2951: '辙', 2952: '辞', 2953: '辛', 2954: '辟', 2955: '辗', 2956: '辣', 2957: '辨', 2958: '辩', 2959: '辰', 2960: '输', 2961: '辊', 2962: '辱', 2963: '辽', 2964: '辫', 2965: '迂', 2966: '迄', 2967: '边', 2968: '迅', 2969: '辜', 2970: '达', 2971: '迈', 2972: '返', 2973: '过', 2974: '违', 2975: '近', 2976: '运', 2977: '迎', 2978: '还', 2979: '进', 2980: '迟', 2981: '连', 2982: '迫', 2983: '迭', 2984: '述', 2985: '远', 2986: '迷', 2987: '适', 2988: '追', 2989: '迹', 2990: '逃', 2991: '送', 2992: '迢', 2993: '退', 2994: '逆', 2995: '逐', 2996: '递', 2997: '逊', 2998: '选', 2999: '逗', 3000: '透', 3001: '逛', 3002: '逝', 3003: '逞', 3004: '通', 3005: '逢', 3006: '逮', 3007: '速', 3008: '逸', 3009: '造', 3010: '逾', 3011: '迸', 3012: '途', 3013: '逼', 3014: '遇', 3015: '遁', 3016: '遏', 3017: '遗', 3018: '遥', 3019: '遣', 3020: '遂', 3021: '遍', 3022: '遵', 3023: '遮', 3024: '遭', 3025: '邑', 3026: '邀', 3027: '道', 3028: '邓', 3029: '邦', 3030: '邪', 3031: '邱', 3032: '逻', 3033: '邮', 3034: '邢', 3035: '那', 3036: '避', 3037: '郁', 3038: '郊', 3039: '邻', 3040: '郎', 3041: '邹', 3042: '郡', 3043: '郝', 3044: '郭', 3045: '邯', 3046: '酉', 3047: '鄂', 3048: '鄙', 3049: '部', 3050: '酋', 3051: '酌', 3052: '都', 3053: '郑', 3054: '酚', 3055: '配', 3056: '郴', 3057: '酬', 3058: '酒', 3059: '酱', 3060: '郸', 3061: '酮', 3062: '酷', 3063: '酶', 3064: '酿', 3065: '醉', 3066: '醇', 3067: '酗', 3068: '醋', 3069: '酞', 3070: '醒', 3071: '醛', 3072: '释', 3073: '酪', 3074: '采', 3075: '酵', 3076: '鉴', 3077: '里', 3078: '野', 3079: '釜', 3080: '量', 3081: '钉', 3082: '金', 3083: '钒', 3084: '钓', 3085: '钝', 3086: '针', 3087: '醚', 3088: '钠', 3089: '钧', 3090: '钦', 3091: '钟', 3092: '钮', 3093: '钢', 3094: '钩', 3095: '钨', 3096: '钡', 3097: '钞', 3098: '钵', 3099: '钱', 3100: '钳', 3101: '铂', 3102: '钙', 3103: '铃', 3104: '钻', 3105: '铆', 3106: '钥', 3107: '铅', 3108: '铡', 3109: '铁', 3110: '铣', 3111: '铭', 3112: '铝', 3113: '铜', 3114: '铱', 3115: '铬', 3116: '铰', 3117: '铸', 3118: '链', 3119: '银', 3120: '销', 3121: '铺', 3122: '锁', 3123: '锈', 3124: '锅', 3125: '锄', 3126: '锐', 3127: '锌', 3128: '锋', 3129: '锑', 3130: '铲', 3131: '锗', 3132: '铀', 3133: '锡', 3134: '锤', 3135: '锣', 3136: '锦', 3137: '锥', 3138: '锨', 3139: '锚', 3140: '锭', 3141: '锰', 3142: '锯', 3143: '键', 3144: '锻', 3145: '锹', 3146: '镁', 3147: '错', 3148: '镍', 3149: '镐', 3150: '镑', 3151: '镀', 3152: '镇', 3153: '镰', 3154: '镜', 3155: '镶', 3156: '闰', 3157: '长', 3158: '门', 3159: '闭', 3160: '镊', 3161: '闪', 3162: '闯', 3163: '闲', 3164: '闸', 3165: '闷', 3166: '闺', 3167: '阀', 3168: '阅', 3169: '间', 3170: '阉', 3171: '问', 3172: '闻', 3173: '阎', 3174: '阐', 3175: '阁', 3176: '镣', 3177: '阔', 3178: '防', 3179: '阶', 3180: '阴', 3181: '队', 3182: '阜', 3183: '阳', 3184: '阵', 3185: '阻', 3186: '闽', 3187: '阿', 3188: '阂', 3189: '附', 3190: '阑', 3191: '陋', 3192: '陕', 3193: '陡', 3194: '陈', 3195: '限', 3196: '降', 3197: '院', 3198: '险', 3199: '阮', 3200: '除', 3201: '陪', 3202: '隅', 3203: '陷', 3204: '陶', 3205: '隆', 3206: '陀', 3207: '隋', 3208: '隘', 3209: '陛', 3210: '障', 3211: '随', 3212: '隙', 3213: '陨', 3214: '雁', 3215: '雅', 3216: '雄', 3217: '隶', 3218: '雀', 3219: '难', 3220: '雏', 3221: '雕', 3222: '集', 3223: '隔', 3224: '雌', 3225: '隧', 3226: '雷', 3227: '霄', 3228: '雨', 3229: '雪', 3230: '雹', 3231: '震', 3232: '霓', 3233: '霉', 3234: '雾', 3235: '需', 3236: '霞', 3237: '霸', 3238: '雍', 3239: '露', 3240: '靖', 3241: '青', 3242: '靡', 3243: '静', 3244: '革', 3245: '霍', 3246: '非', 3247: '靠', 3248: '靶', 3249: '霜', 3250: '面', 3251: '鞍', 3252: '鞘', 3253: '霖', 3254: '鞠', 3255: '鞭', 3256: '靛', 3257: '韩', 3258: '霹', 3259: '韵', 3260: '韧', 3261: '顷', 3262: '音', 3263: '页', 3264: '韦', 3265: '顶', 3266: '顺', 3267: '顽', 3268: '颁', 3269: '顾', 3270: '项', 3271: '须', 3272: '韭', 3273: '韶', 3274: '颐', 3275: '颊', 3276: '顿', 3277: '频', 3278: '颈', 3279: '领', 3280: '颖', 3281: '颜', 3282: '颗', 3283: '颠', 3284: '颓', 3285: '题', 3286: '额', 3287: '颤', 3288: '颧', 3289: '飘', 3290: '餐', 3291: '饥', 3292: '飞', 3293: '风', 3294: '食', 3295: '饲', 3296: '颇', 3297: '饰', 3298: '饱', 3299: '饭', 3300: '饵', 3301: '饶', 3302: '颅', 3303: '饮', 3304: '馆', 3305: '馋', 3306: '饿', 3307: '馏', 3308: '馅', 3309: '饯', 3310: '馁', 3311: '香', 3312: '驮', 3313: '驯', 3314: '马', 3315: '驱', 3316: '首', 3317: '驳', 3318: '驰', 3319: '驴', 3320: '驶', 3321: '驹', 3322: '馈', 3323: '驼', 3324: '骇', 3325: '驻', 3326: '验', 3327: '驭', 3328: '骄', 3329: '骏', 3330: '骚', 3331: '骤', 3332: '骑', 3333: '骗', 3334: '骡', 3335: '髓', 3336: '鬃', 3337: '骨', 3338: '骂', 3339: '骆', 3340: '高', 3341: '魂', 3342: '魄', 3343: '魔', 3344: '鲁', 3345: '鲍', 3346: '鬼', 3347: '鲤', 3348: '骋', 3349: '鳖', 3350: '鳞', 3351: '鱼', 3352: '鳃', 3353: '鸣', 3354: '鸦', 3355: '鸥', 3356: '魁', 3357: '鸡', 3358: '鸭', 3359: '鸿', 3360: '鸟', 3361: '鸽', 3362: '鹃', 3363: '鹅', 3364: '麓', 3365: '鹿', 3366: '鹰', 3367: '麦', 3368: '麻', 3369: '鹤', 3370: '鸳', 3371: '黄', 3372: '黎', 3373: '黑', 3374: '黔', 3375: '鸵', 3376: '默', 3377: '鼎', 3378: '鼓', 3379: '齿', 3380: '齐', 3381: '鹏', 3382: '鼻', 3383: '龙', 3384: '龚', 3385: '龋', 3386: '龟', 3387: '龄', 3388: '鼠', 3389: '鹊', 3390: '黍', 3391: '鲸', 3392: '鸯', 3393: '鲜', 3394: '魏', 3395: '驾', 3396: '饼', 3397: '馒', 3398: '预', 3399: '颂', 3400: '鞋', 3401: '靴', 3402: '靳', 3403: '零', 3404: '雇', 3405: '隐', 3406: '陵', 3407: '陇', 3408: '陆', 3409: '际', 3410: '骸', 3411: '闹', 3412: '镭', 3413: '钾', 3414: '钎', 3415: '饺', 3416: '重', 3417: '釉', 3418: '酸', 3419: '酥', 3420: '酣', 3421: '郧', 3422: '邵', 3423: '迪', 3424: '陌', 3425: '这', 3426: '迁', 3427: '较', 3428: '载', 3429: '身', 3430: '跑', 3431: '跋', 3432: '赔', 3433: '赁', 3434: '赏', 3435: '贾', 3436: '购', 3437: '酝', 3438: '货', 3439: '贡', 3440: '贞', 3441: '贝', 3442: '象', 3443: '谗', 3444: '谢', 3445: '谅', 3446: '调', 3447: '课', 3448: '赡', 3449: '评', 3450: '证', 3451: '许', 3452: '讥', 3453: '誉', 3454: '触', 3455: '裹', 3456: '貉', 3457: '袁', 3458: '蠕', 3459: '表', 3460: '衍', 3461: '谍', 3462: '衅', 3463: '蠢', 3464: '蝎', 3465: '诧', 3466: '蜂', 3467: '蜀', 3468: '蛀', 3469: '虑', 3470: '誊', 3471: '藕', 3472: '薯', 3473: '蕊', 3474: '蓖', 3475: '蕉', 3476: '蔬', 3477: '蔗', 3478: '蒲', 3479: '葵', 3480: '落', 3481: '萧', 3482: '蜗', 3483: '菇', 3484: '莆', 3485: '蘑', 3486: '药', 3487: '荫', 3488: '荡', 3489: '荆', 3490: '苯', 3491: '苦', 3492: '若', 3493: '苞', 3494: '艳', 3495: '舜', 3496: '膘', 3497: '臆', 3498: '膀', 3499: '菲', 3500: '菏', 3501: '胁', 3502: '胀', 3503: '莎', 3504: '脱', 3505: '肯', 3506: '肇', 3507: '肌', 3508: '耽', 3509: '耳', 3510: '羔', 3511: '翟', 3512: '耗', 3513: '罐', 3514: '缉', 3515: '缝', 3516: '继', 3517: '绎', 3518: '组', 3519: '练', 3520: '级', 3521: '糯', 3522: '约', 3523: '肄', 3524: '糙', 3525: '粪', 3526: '簇', 3527: '笋', 3528: '童', 3529: '稽', 3530: '程', 3531: '秸', 3532: '秆', 3533: '绦', 3534: '祈', 3535: '祸', 3536: '碾', 3537: '碰', 3538: '碟', 3539: '粳', 3540: '硼', 3541: '砖', 3542: '知', 3543: '矗', 3544: '瞎', 3545: '益', 3546: '盈', 3547: '癸', 3548: '盆', 3549: '瘸', 3550: '瘴', 3551: '磐', 3552: '磋', 3553: '痈', 3554: '症', 3555: '疾', 3556: '甸', 3557: '甄', 3558: '瓶', 3559: '琶', 3560: '瑞', 3561: '琼', 3562: '皑', 3563: '痪', 3564: '现', 3565: '痉', 3566: '率', 3567: '猛', 3568: '狠', 3569: '狂', 3570: '牟', 3571: '牙', 3572: '瑚', 3573: '爹', 3574: '琐', 3575: '琉', 3576: '熟', 3577: '琅', 3578: '煮', 3579: '煎', 3580: '玖', 3581: '焦', 3582: '猩', 3583: '焚', 3584: '狡', 3585: '灼', 3586: '炯', 3587: '烟', 3588: '激', 3589: '漠', 3590: '溯', 3591: '源', 3592: '熙', 3593: '渤', 3594: '渠', 3595: '渝', 3596: '渗', 3597: '渔', 3598: '清', 3599: '淮', 3600: '涪', 3601: '潦', 3602: '潞', 3603: '淡', 3604: '滁', 3605: '测', 3606: '浆', 3607: '泡', 3608: '溉', 3609: '法', 3610: '沸', 3611: '沃', 3612: '汹', 3613: '沥', 3614: '淬', 3615: '汇', 3616: '汀', 3617: '毗', 3618: '涟', 3619: '殴', 3620: '氏', 3621: '毕', 3622: '毒', 3623: '歧', 3624: '泞', 3625: '武', 3626: '沽', 3627: '步', 3628: '款', 3629: '欺', 3630: '樟', 3631: '欲', 3632: '欠', 3633: '榴', 3634: '氓', 3635: '椽', 3636: '棍', 3637: '栏', 3638: '柯', 3639: '果', 3640: '望', 3641: '朔', 3642: '最', 3643: '昼', 3644: '料', 3645: '昆', 3646: '斑', 3647: '敢', 3648: '敌', 3649: '携', 3650: '橇', 3651: '捌', 3652: '掇', 3653: '樱', 3654: '捉', 3655: '挣', 3656: '挠', 3657: '挖', 3658: '拔', 3659: '折', 3660: '手', 3661: '扇', 3662: '户', 3663: '戴', 3664: '我', 3665: '戍', 3666: '懒', 3667: '慨', 3668: '愿', 3669: '愧', 3670: '愤', 3671: '捍', 3672: '感', 3673: '愁', 3674: '惜', 3675: '惋', 3676: '悬', 3677: '悠', 3678: '悄', 3679: '恃', 3680: '总', 3681: '怨', 3682: '快', 3683: '忌', 3684: '忆', 3685: '徒', 3686: '弥', 3687: '弟', 3688: '并', 3689: '巴', 3690: '悸', 3691: '峪', 3692: '悯', 3693: '尘', 3694: '尖', 3695: '尔', 3696: '寄', 3697: '宠', 3698: '实', 3699: '宛', 3700: '彪', 3701: '季', 3702: '嫉', 3703: '婪', 3704: '嫂', 3705: '娥', 3706: '嫁', 3707: '巩', 3708: '媳', 3709: '巍', 3710: '婆', 3711: '峨', 3712: '姿', 3713: '如', 3714: '夹', 3715: '夸', 3716: '失', 3717: '够', 3718: '壶', 3719: '墓', 3720: '墅', 3721: '塌', 3722: '埠', 3723: '埋', 3724: '坊', 3725: '坷', 3726: '在', 3727: '圣', 3728: '圃', 3729: '图', 3730: '嗜', 3731: '喜', 3732: '哲', 3733: '哮', 3734: '哆', 3735: '哗', 3736: '哑', 3737: '哈', 3738: '咯', 3739: '呀', 3740: '吾', 3741: '叙', 3742: '县', 3743: '即', 3744: '危', 3745: '切', 3746: '啡', 3747: '内', 3748: '兢', 3749: '兽', 3750: '入', 3751: '党', 3752: '匡', 3753: '凯'}
    char_item_list = list(char_item)
    # print(char_item_list)
    # ['丁', '一', '丈', '七', '万', '丑', '三', '丘', '上', '与', '下', '专', '丙', '不', '且', '世', '丛', '业', '东', '丧', '丝', '丫', '丢', '严', '两', '丰', '丸', '个', '丹', '临', '串', '中', '丽', '为', '主', '举', '乃', '乌', '乍', '久', '么', '乏', '义', '乒', '乎', '之', '乔', '乖', '乐', '乓', '乞', '乘', '习', '乙', '九', '也', '乾', '乡', '书', '乳', '买', '乱', '予', '了', '争', '亏', '事', '二', '互', '于', '亚', '云', '井', '亢', '亡', '五', '亥', '些', '亨', '交', '享', '亦', '产', '亭', '京', '亩', '什', '亮', '仁', '亲', '仆', '仇', '人', '亿', '介', '仅', '仓', '今', '仔', '仕', '仍', '从', '仗', '仑', '仙', '付', '他', '仪', '仟', '代', '令', '仰', '仲', '以', '们', '仿', '企', '件', '伊', '伍', '伎', '任', '价', '份', '伐', '休', '伏', '优', '伙', '众', '伞', '伟', '伦', '会', '估', '伪', '传', '伯', '伶', '伤', '伴', '伸', '佃', '似', '伺', '佐', '佑', '但', '位', '住', '低', '佛', '佩', '体', '何', '佬', '余', '佯', '作', '你', '佳', '侄', '侈', '侍', '佣', '侗', '使', '佰', '侠', '例', '侦', '供', '依', '侮', '侧', '侯', '侵', '侥', '促', '侣', '俊', '俏', '俄', '便', '侩', '俘', '侨', '俗', '俞', '保', '俯', '俩', '信', '俱', '修', '俐', '倔', '俺', '倘', '候', '倍', '倚', '倡', '倒', '倦', '债', '借', '俭', '倾', '值', '健', '偏', '偶', '倪', '假', '做', '偿', '停', '傅', '偷', '傍', '傣', '储', '僚', '催', '僧', '傲', '傻', '像', '僵', '僻', '儒', '傀', '允', '兆', '充', '儿', '元', '兄', '僳', '兑', '先', '光', '免', '克', '兔', '兜', '傈', '儡', '兰', '兴', '六', '共', '典', '兹', '兵', '关', '具', '其', '公', '冀', '冈', '养', '八', '兼', '全', '冒', '冕', '再', '写', '冤', '农', '冠', '军', '冯', '冶', '况', '冬', '决', '冰', '册', '冲', '冗', '凄', '冻', '凉', '凋', '净', '准', '减', '冷', '凝', '凌', '凑', '凤', '几', '凳', '凡', '凸', '凶', '凭', '凹', '凛', '凿', '出', '刁', '函', '刃', '刀', '击', '冉', '分', '划', '刑', '列', '凰', '刘', '删', '创', '则', '刨', '判', '初', '券', '刮', '刚', '刹', '刷', '到', '剁', '制', '刺', '刻', '剃', '削', '剐', '剔', '剂', '剑', '前', '剥', '剪', '剧', '别', '剩', '剖', '刽', '劈', '副', '割', '务', '力', '劝', '劣', '努', '办', '劫', '功', '助', '剿', '励', '加', '勃', '勒', '动', '劳', '勘', '勋', '勉', '势', '募', '勺', '勤', '勇', '勿', '匀', '勾', '匈', '匙', '包', '匝', '化', '北', '劲', '匠', '匣', '匪', '利', '匿', '匹', '医', '卉', '区', '卑', '十', '协', '匆', '卒', '卓', '华', '半', '午', '卜', '卢', '单', '卖', '卡', '南', '卤', '卧', '博', '卫', '卯', '印', '占', '卞', '升', '卿', '千', '卷', '卵', '卸', '厉', '历', '厂', '厅', '厌', '却', '厢', '厦', '厘', '厩', '厚', '原', '厨', '厕', '参', '厄', '去', '叉', '压', '友', '及', '又', '叔', '叁', '双', '发', '反', '叛', '取', '叠', '受', '变', '口', '叮', '叭', '古', '召', '另', '叫', '可', '台', '右', '司', '吁', '叼', '史', '号', '叶', '只', '吉', '叹', '吏', '各', '合', '吊', '吃', '名', '吐', '吕', '向', '吓', '后', '吞', '同', '君', '吗', '吠', '否', '吟', '吭', '吧', '吮', '启', '吱', '听', '吝', '含', '吵', '吻', '吸', '吼', '吴', '吹', '吩', '吨', '呆', '呐', '句', '呕', '告', '呈', '呛', '呸', '员', '呢', '味', '呵', '咀', '呼', '咋', '呜', '咎', '命', '咏', '咒', '周', '咕', '呻', '和', '咬', '咳', '咆', '咽', '咸', '哀', '咱', '哄', '哉', '品', '哇', '哎', '咨', '咖', '响', '哥', '哨', '哦', '咐', '哩', '哟', '咙', '哪', '哭', '哼', '唤', '唇', '售', '唐', '唉', '唬', '唯', '啄', '啃', '唾', '唱', '商', '唆', '啪', '啮', '啊', '啸', '啼', '啥', '唁', '啦', '喉', '喀', '喂', '喧', '善', '喳', '喘', '喊', '喻', '嗅', '喷', '嗓', '喝', '嗡', '啤', '哺', '嘉', '嘘', '喇', '嘱', '嘎', '嘲', '嘶', '嘻', '嘛', '噎', '嘿', '噪', '噬', '嗽', '嗣', '嘴', '嚎', '器', '嚣', '嚼', '囚', '嚷', '囊', '囤', '四', '园', '困', '回', '因', '团', '围', '固', '嚏', '国', '圆', '圈', '址', '土', '囱', '坍', '场', '地', '坑', '噶', '坚', '均', '圭', '坐', '坛', '坝', '块', '坟', '圾', '坠', '坤', '坦', '坯', '坡', '坪', '垄', '垂', '垒', '垢', '垛', '垣', '垦', '坏', '垮', '垫', '坎', '埂', '型', '埃', '垃', '域', '坞', '城', '堑', '基', '堕', '堂', '堤', '堪', '堡', '堰', '培', '堆', '塑', '堵', '境', '塞', '埔', '填', '塘', '塔', '墟', '增', '墩', '墙', '墨', '壤', '壕', '壁', '壮', '墒', '壳', '士', '声', '夕', '备', '复', '夏', '壬', '外', '处', '夯', '多', '夜', '大', '壹', '太', '夫', '夷', '奈', '天', '奇', '奋', '头', '奉', '奏', '契', '奖', '奠', '奔', '央', '奥', '夺', '奢', '奴', '套', '奶', '奸', '女', '奎', '奄', '妆', '妇', '她', '好', '妓', '妈', '妖', '妒', '妙', '妄', '妨', '姆', '妹', '姑', '始', '妊', '姐', '委', '姜', '姓', '姚', '妻', '姨', '姬', '妥', '姻', '威', '娟', '娇', '妮', '娱', '娃', '娘', '姥', '娶', '婉', '婴', '婿', '婚', '婶', '媒', '媚', '娠', '娩', '娄', '嫌', '娜', '嫡', '嫩', '孔', '子', '孝', '存', '字', '孕', '孟', '孙', '孤', '孩', '孰', '孵', '孽', '宁', '宅', '宇', '孜', '宏', '它', '安', '宋', '守', '完', '宗', '孺', '学', '官', '宜', '宝', '宦', '审', '定', '宪', '宣', '孪', '宰', '宴', '室', '宫', '容', '宾', '害', '宿', '宽', '宵', '寂', '宙', '寅', '家', '客', '寐', '密', '寝', '察', '富', '寡', '寒', '寨', '寓', '寸', '寺', '寻', '寿', '导', '对', '射', '封', '寇', '寞', '尊', '将', '寥', '尝', '少', '尧', '小', '尤', '尼', '尚', '尉', '就', '尺', '尿', '屁', '尾', '屈', '尽', '屉', '局', '层', '屎', '屑', '展', '居', '届', '屠', '屋', '屡', '履', '屏', '属', '屯', '尹', '岂', '山', '岗', '岔', '岁', '岛', '尸', '岩', '岳', '岭', '峙', '峡', '峦', '岸', '峭', '峻', '峰', '屿', '崇', '屹', '崖', '崩', '崔', '嵌', '岿', '巡', '州', '川', '巧', '巨', '巢', '左', '崎', '巫', '工', '巳', '差', '巷', '己', '币', '崭', '已', '帅', '希', '帐', '布', '帕', '帖', '帆', '师', '帘', '帛', '帜', '帚', '帝', '帧', '席', '幂', '带', '帽', '市', '幌', '巾', '幕', '幅', '常', '幢', '帮', '幸', '平', '幽', '年', '庄', '干', '广', '庐', '幼', '库', '床', '序', '庆', '庚', '庞', '府', '应', '底', '庙', '废', '庇', '庶', '度', '店', '庭', '康', '廉', '庸', '廓', '廖', '廊', '廷', '延', '弃', '建', '弊', '异', '弄', '幻', '弓', '开', '弗', '弛', '座', '弘', '式', '引', '弦', '张', '弯', '弹', '录', '彝', '弱', '当', '归', '弧', '彰', '强', '形', '彩', '役', '彻', '彼', '影', '彭', '征', '径', '彤', '往', '彦', '徐', '很', '律', '彬', '循', '待', '御', '徽', '得', '微', '忠', '忍', '心', '德', '志', '必', '徊', '忧', '忙', '忘', '徘', '忿', '怎', '念', '怔', '怒', '怜', '态', '怀', '忽', '怕', '怠', '忱', '怯', '忻', '思', '急', '怂', '恋', '性', '恐', '怪', '恒', '恤', '怖', '恕', '恭', '恨', '息', '恰', '恼', '恩', '恶', '悔', '恍', '悉', '悍', '悟', '恬', '悦', '恫', '恢', '悲', '患', '您', '惑', '恳', '恿', '惊', '情', '惦', '惟', '惠', '惩', '惯', '惨', '惶', '悼', '惰', '惹', '惧', '想', '惕', '愚', '惫', '惭', '慈', '惮', '意', '慎', '愈', '慕', '慌', '慧', '慰', '惺', '憎', '慢', '憾', '憨', '愉', '憋', '懂', '戈', '戊', '戌', '戎', '慑', '戒', '戚', '懊', '戏', '成', '战', '懈', '或', '戳', '截', '懦', '慷', '扁', '扒', '扎', '房', '所', '扔', '戮', '扑', '扦', '打', '扣', '才', '执', '扛', '扩', '扮', '扫', '扰', '扯', '扭', '扳', '扼', '扶', '承', '扬', '抄', '找', '技', '抑', '批', '托', '抖', '抒', '投', '抚', '抠', '抓', '把', '抡', '抗', '护', '抛', '抉', '抢', '披', '押', '报', '抿', '抵', '抬', '抹', '拄', '拈', '抽', '抱', '担', '拆', '拎', '拐', '拉', '拒', '拓', '拌', '拂', '拍', '招', '拜', '抨', '拙', '拘', '拖', '拟', '拢', '拇', '拦', '拧', '拨', '拥', '括', '择', '拴', '拷', '拳', '拱', '拭', '拼', '拽', '拾', '拿', '持', '拣', '指', '挎', '按', '挂', '拯', '挑', '挟', '挡', '挫', '振', '挽', '挤', '挪', '捂', '挛', '挨', '挺', '捅', '挥', '捆', '挞', '挚', '挝', '捐', '损', '捞', '捡', '捕', '捣', '捧', '捷', '捏', '捻', '捶', '换', '掀', '掂', '捎', '据', '授', '掐', '掏', '掉', '掠', '掌', '探', '掘', '掣', '排', '控', '掩', '接', '掖', '掳', '掸', '掺', '推', '描', '揍', '揉', '揖', '插', '掷', '握', '揪', '提', '揭', '揩', '揣', '援', '搀', '揽', '搂', '搐', '搅', '措', '搔', '搜', '搪', '搓', '搏', '搞', '搬', '搽', '搁', '搭', '摄', '摊', '摆', '摘', '摧', '摩', '摔', '撂', '摇', '撅', '撇', '摸', '撑', '撕', '撒', '撞', '撬', '撤', '撮', '撰', '撵', '播', '摈', '撼', '撩', '擂', '擎', '操', '擒', '攀', '擅', '攒', '攘', '擦', '摹', '攫', '收', '攻', '支', '效', '敏', '改', '政', '擞', '放', '故', '敛', '敝', '敞', '教', '救', '敷', '敲', '整', '斋', '敬', '敦', '数', '文', '敖', '斟', '散', '斗', '斧', '斥', '斜', '斯', '斤', '斩', '断', '新', '斌', '方', '施', '旁', '旗', '旋', '无', '族', '旨', '既', '旦', '日', '旷', '斡', '旺', '昂', '旧', '早', '昌', '昏', '昔', '旱', '明', '易', '昧', '映', '星', '昨', '时', '春', '旬', '昭', '晌', '旅', '晕', '显', '晓', '是', '晤', '晦', '晒', '晚', '晋', '晨', '晃', '旭', '普', '景', '智', '晾', '暂', '暇', '晶', '暑', '晴', '暴', '暮', '曝', '暗', '曳', '暖', '曼', '曲', '曹', '曰', '朋', '更', '曾', '朗', '替', '月', '曙', '服', '有', '朝', '札', '木', '朴', '末', '未', '期', '朽', '本', '朵', '机', '杂', '杉', '杀', '杏', '材', '晰', '杖', '权', '杠', '杜', '村', '李', '杆', '束', '杭', '杰', '条', '杨', '朱', '杯', '来', '构', '析', '枉', '板', '枕', '极', '枣', '枚', '林', '术', '松', '枯', '枫', '枢', '枪', '柄', '架', '柏', '柔', '枝', '柜', '某', '柑', '染', '柬', '查', '柱', '柿', '柴', '枷', '栅', '柳', '柒', '栈', '栋', '标', '栓', '柞', '柠', '栗', '树', '样', '栖', '校', '核', '根', '株', '格', '栽', '桃', '桂', '框', '桐', '桌', '案', '桑', '档', '桶', '梁', '桨', '梅', '梆', '桩', '桔', '梗', '桥', '梧', '梭', '梢', '梨', '梯', '梦', '梳', '检', '棋', '械', '桅', '棉', '棘', '棒', '棕', '棱', '桓', '棺', '椅', '森', '椒', '植', '椭', '椎', '棵', '椰', '棚', '椿', '楚', '楔', '楞', '棠', '榆', '榨', '榜', '楼', '概', '槐', '槽', '模', '橙', '楷', '榔', '横', '橱', '榷', '欣', '欧', '欢', '槛', '次', '樊', '歉', '歇', '橡', '止', '檀', '正', '檬', '檄', '歹', '歌', '此', '歼', '殆', '殉', '殊', '殷', '殿', '残', '死', '毁', '毋', '殖', '段', '殃', '歪', '母', '毙', '每', '毡', '毯', '比', '毫', '毛', '氖', '氟', '毅', '氦', '民', '毖', '氢', '氨', '氰', '气', '氯', '氮', '汁', '永', '氧', '水', '求', '汛', '汝', '汞', '氛', '汉', '汗', '污', '汰', '池', '江', '汪', '汲', '汽', '汤', '汕', '沁', '沈', '沏', '沉', '汐', '沙', '沤', '沦', '沟', '没', '沂', '沫', '沼', '沪', '河', '沛', '油', '沾', '泄', '汾', '泉', '治', '泊', '泌', '沧', '泣', '泛', '泰', '沿', '泳', '泪', '泅', '波', '注', '沮', '泻', '泵', '泥', '泽', '泼', '洒', '津', '洲', '洪', '洗', '洼', '洞', '洽', '活', '洁', '洋', '浊', '浇', '派', '浅', '流', '浙', '浑', '济', '浦', '浓', '洛', '浩', '浸', '浴', '涅', '海', '消', '涉', '涂', '涎', '浮', '涛', '涌', '涕', '涝', '洱', '浚', '浪', '涤', '润', '涩', '涨', '涵', '涧', '涡', '涸', '液', '淋', '淌', '淘', '淤', '淫', '涣', '涯', '淹', '混', '深', '渍', '添', '渊', '淆', '渐', '淄', '淀', '淖', '渡', '淑', '港', '渺', '淳', '渴', '游', '湛', '湖', '湾', '湘', '溃', '溅', '渣', '湿', '温', '溢', '溜', '渭', '溪', '溺', '滋', '湃', '溶', '湍', '滇', '滑', '滤', '滥', '滚', '滞', '滨', '满', '滦', '漆', '漂', '滩', '漏', '滓', '滴', '漾', '漱', '演', '漫', '滔', '潜', '潘', '潭', '澄', '澈', '潮', '澜', '澡', '漓', '澳', '漳', '濒', '瀑', '潍', '灭', '灌', '灯', '灸', '灶', '灵', '火', '灿', '澎', '炊', '灾', '灰', '炎', '炉', '炒', '炳', '炕', '炭', '炬', '炮', '炸', '炙', '炼', '炽', '点', '烈', '烘', '烙', '烂', '烃', '烛', '烦', '烤', '烬', '烧', '烫', '烹', '热', '焉', '烷', '烁', '焙', '焊', '炔', '焰', '煞', '烩', '然', '烽', '煤', '烯', '熄', '熏', '熊', '照', '熔', '焕', '燃', '燎', '熬', '爆', '爪', '爵', '煌', '燥', '爱', '父', '爷', '爸', '爬', '燕', '煽', '版', '爽', '牌', '牢', '片', '牧', '牲', '特', '牵', '犬', '犁', '物', '犹', '牛', '犊', '狄', '犯', '牡', '状', '狐', '狞', '狭', '狗', '犀', '狸', '狱', '狮', '牺', '猎', '狼', '狈', '独', '狙', '猜', '献', '猴', '猿', '狰', '猪', '玄', '猫', '玉', '猖', '王', '玻', '玩', '环', '猾', '珍', '珠', '獭', '班', '琴', '球', '玫', '理', '琢', '玛', '珐', '瑟', '珊', '玲', '瓜', '瓣', '瓤', '瓦', '瓢', '瓮', '瑰', '瑶', '琵', '瓷', '琳', '甜', '甘', '甚', '甩', '甫', '生', '甭', '璃', '用', '申', '田', '畏', '甲', '男', '由', '电', '畔', '界', '畦', '畜', '畅', '留', '疆', '番', '略', '疑', '疗', '疏', '疟', '疤', '畸', '甥', '疮', '疫', '疯', '疲', '畴', '疵', '疹', '疽', '疼', '疙', '疚', '画', '疡', '疥', '痒', '病', '痘', '痕', '痞', '痢', '痹', '痛', '痰', '瘤', '瘟', '瘦', '瘪', '痴', '痊', '瘫', '痔', '癣', '癌', '皂', '登', '白', '百', '瘁', '瘩', '皇', '皆', '皖', '盂', '皿', '盅', '皮', '皱', '盏', '盔', '皋', '盒', '盐', '监', '的', '盗', '盟', '盖', '盘', '盲', '盛', '盯', '盼', '目', '眉', '直', '眠', '盾', '眩', '省', '看', '眶', '眨', '真', '眷', '眯', '睛', '督', '眼', '睁', '着', '睬', '睡', '睹', '瞄', '眺', '盎', '瞅', '瞒', '瞳', '瞻', '瞬', '相', '瞪', '瞧', '瞥', '睦', '睫', '矛', '矢', '矫', '矣', '矽', '矮', '矾', '矿', '码', '石', '矩', '砂', '瞩', '短', '砍', '研', '砚', '砌', '砰', '砷', '砧', '破', '硒', '硅', '硝', '砒', '硷', '确', '硫', '砸', '碍', '硬', '碑', '碎', '碘', '砾', '碗', '碧', '碴', '础', '硕', '磅', '碳', '碱', '碉', '磁', '磕', '礁', '碌', '磷', '礼', '示', '祝', '祖', '磨', '社', '祥', '磊', '票', '祷', '祭', '磺', '禹', '神', '祁', '福', '禾', '秀', '离', '私', '禽', '禄', '秃', '禁', '秘', '秋', '租', '秤', '秒', '科', '秧', '秦', '积', '祟', '秽', '称', '稗', '种', '稀', '稠', '移', '税', '稚', '稳', '稼', '稻', '秉', '穗', '穆', '穴', '稍', '突', '究', '穷', '稿', '空', '窃', '穿', '窍', '窑', '窄', '秩', '窘', '窜', '窗', '窟', '窿', '竖', '窥', '立', '竞', '窝', '站', '窖', '窒', '竭', '章', '竟', '竿', '竹', '符', '笛', '端', '笺', '笔', '笑', '笨', '笼', '筋', '竣', '筏', '第', '筐', '筷', '筑', '筒', '筛', '筹', '等', '签', '箍', '笆', '简', '箔', '箕', '策', '箩', '篆', '算', '篓', '管', '箱', '篙', '篇', '篮', '箭', '篱', '篷', '簧', '簿', '答', '籽', '米', '类', '籍', '篡', '粒', '粉', '粤', '粟', '粗', '粥', '粘', '粹', '糊', '糕', '粮', '糜', '粕', '糟', '糠', '糖', '索', '系', '素', '粱', '絮', '精', '繁', '紧', '纠', '纂', '累', '紫', '纤', '纬', '纱', '红', '纲', '纵', '纷', '纳', '紊', '纪', '纹', '纽', '纺', '纸', '纯', '纫', '绅', '线', '织', '纶', '绊', '细', '终', '绒', '绑', '结', '绘', '绕', '统', '绞', '给', '绢', '绣', '经', '绥', '绝', '绩', '络', '续', '绍', '绪', '绳', '绚', '绷', '绸', '综', '绵', '缀', '绽', '缅', '缆', '绿', '缎', '维', '缓', '绰', '缕', '缔', '编', '缘', '缚', '缨', '缠', '缄', '缩', '缸', '缴', '罕', '缺', '罗', '网', '罩', '罚', '罢', '罪', '置', '署', '羌', '缮', '羡', '羹', '羊', '美', '羽', '翅', '翌', '翁', '翘', '翔', '羞', '翠', '群', '羚', '耀', '翼', '翻', '考', '耍', '老', '耐', '耙', '耕', '者', '翱', '耸', '耻', '而', '耶', '聂', '聊', '聋', '聘', '联', '聚', '翰', '耿', '职', '肃', '肆', '耘', '耪', '肋', '肘', '肖', '肉', '肠', '肝', '肚', '肤', '肢', '股', '聪', '肩', '肥', '育', '肿', '肾', '肛', '胆', '肺', '胃', '胎', '胚', '背', '胞', '胜', '胖', '胡', '肪', '胳', '胶', '胺', '肮', '胸', '脂', '脊', '能', '脉', '脏', '脓', '脐', '脖', '胰', '脑', '脆', '胯', '脚', '腊', '腑', '腐', '脾', '腕', '腋', '腔', '腥', '腹', '腮', '脸', '腻', '腺', '腰', '脯', '膏', '腾', '膊', '膛', '腆', '腿', '膝', '膨', '膜', '臀', '臂', '臣', '臭', '膳', '臼', '舀', '臻', '舆', '自', '至', '舅', '致', '臃', '舌', '舒', '舔', '舍', '航', '舱', '舵', '舷', '舞', '般', '舰', '艇', '舟', '艰', '艘', '船', '良', '艺', '芋', '色', '艾', '芒', '节', '芜', '舶', '芦', '芬', '芯', '苇', '花', '芍', '芽', '苔', '苍', '苑', '苏', '苛', '芳', '苟', '苫', '苗', '芝', '芥', '芭', '芹', '茁', '茄', '茂', '范', '茅', '茧', '茎', '茸', '茬', '茫', '茶', '荐', '荒', '苹', '荚', '荤', '英', '草', '荣', '荷', '茨', '茵', '茹', '莹', '莲', '莽', '荔', '获', '莱', '菊', '菌', '莫', '菱', '萌', '荧', '萎', '萝', '菜', '萍', '萤', '莉', '萨', '葛', '著', '菠', '营', '菩', '董', '葱', '葬', '蒂', '蒙', '蒜', '葡', '蒋', '萄', '蓄', '蓝', '蓬', '蒸', '蔓', '蔚', '蔡', '蔫', '葫', '蔽', '蓑', '蓉', '蕾', '蓟', '薛', '薪', '蕴', '藉', '蔑', '薄', '藤', '藩', '蔼', '藏', '蘸', '藻', '蔷', '虏', '虎', '虐', '虞', '虱', '虹', '虚', '藐', '蚀', '虽', '蚌', '蚁', '蚊', '虫', '蚕', '虾', '蛆', '蛊', '蛙', '蛤', '蛋', '蛮', '蛰', '蛾', '蚂', '蛹', '蚜', '蛇', '蚤', '蜕', '蜜', '蛔', '蝇', '蜡', '蝉', '蝶', '蛛', '融', '蟹', '螺', '蝗', '蜒', '蜘', '衔', '衙', '衡', '血', '街', '行', '蝴', '螟', '衰', '衬', '衣', '衫', '补', '袒', '袋', '袖', '袜', '袍', '袭', '裔', '被', '裕', '裂', '裁', '袄', '装', '裙', '裤', '裸', '裴', '裳', '衷', '褐', '褒', '褂', '襟', '褪', '袱', '覆', '褥', '觅', '规', '要', '西', '览', '观', '视', '见', '襄', '觉', '誓', '角', '解', '警', '言', '订', '计', '认', '讨', '训', '詹', '讯', '议', '讳', '记', '讣', '讹', '讲', '讼', '让', '讽', '诀', '论', '讫', '譬', '访', '识', '设', '诉', '诊', '诈', '译', '诛', '讶', '词', '诞', '诚', '诡', '试', '询', '诗', '话', '诅', '详', '误', '诱', '该', '诵', '诬', '语', '说', '诫', '请', '诣', '诸', '读', '诌', '谊', '谁', '诲', '谋', '谐', '谈', '谎', '谚', '谓', '谜', '谣', '诽', '谦', '谆', '谬', '谭', '谨', '谱', '豆', '豁', '谷', '谤', '豹', '豪', '豫', '诺', '谩', '貌', '谰', '谴', '责', '财', '负', '败', '贤', '账', '豌', '豢', '贩', '贫', '贪', '贬', '贮', '豺', '贱', '贷', '质', '贴', '贵', '贺', '贯', '贼', '费', '赃', '资', '赌', '赎', '赋', '赊', '贰', '赐', '赘', '贸', '赚', '赞', '赛', '赖', '赂', '贿', '赠', '赣', '赦', '赫', '赤', '赢', '赴', '赵', '趁', '赶', '趋', '超', '趣', '起', '越', '趟', '趾', '跃', '趴', '走', '跌', '足', '践', '跨', '跺', '跪', '路', '跳', '踞', '踏', '跟', '踪', '踢', '踩', '蹄', '蹈', '距', '蹬', '蹭', '蹿', '躁', '蹦', '蹲', '踌', '躯', '踊', '躬', '躲', '蹋', '轧', '轩', '躺', '轨', '车', '轰', '轮', '躇', '软', '轴', '轿', '辉', '辈', '轻', '转', '辐', '辑', '辅', '辕', '辖', '辆', '辙', '辞', '辛', '辟', '辗', '辣', '辨', '辩', '辰', '输', '辊', '辱', '辽', '辫', '迂', '迄', '边', '迅', '辜', '达', '迈', '返', '过', '违', '近', '运', '迎', '还', '进', '迟', '连', '迫', '迭', '述', '远', '迷', '适', '追', '迹', '逃', '送', '迢', '退', '逆', '逐', '递', '逊', '选', '逗', '透', '逛', '逝', '逞', '通', '逢', '逮', '速', '逸', '造', '逾', '迸', '途', '逼', '遇', '遁', '遏', '遗', '遥', '遣', '遂', '遍', '遵', '遮', '遭', '邑', '邀', '道', '邓', '邦', '邪', '邱', '逻', '邮', '邢', '那', '避', '郁', '郊', '邻', '郎', '邹', '郡', '郝', '郭', '邯', '酉', '鄂', '鄙', '部', '酋', '酌', '都', '郑', '酚', '配', '郴', '酬', '酒', '酱', '郸', '酮', '酷', '酶', '酿', '醉', '醇', '酗', '醋', '酞', '醒', '醛', '释', '酪', '采', '酵', '鉴', '里', '野', '釜', '量', '钉', '金', '钒', '钓', '钝', '针', '醚', '钠', '钧', '钦', '钟', '钮', '钢', '钩', '钨', '钡', '钞', '钵', '钱', '钳', '铂', '钙', '铃', '钻', '铆', '钥', '铅', '铡', '铁', '铣', '铭', '铝', '铜', '铱', '铬', '铰', '铸', '链', '银', '销', '铺', '锁', '锈', '锅', '锄', '锐', '锌', '锋', '锑', '铲', '锗', '铀', '锡', '锤', '锣', '锦', '锥', '锨', '锚', '锭', '锰', '锯', '键', '锻', '锹', '镁', '错', '镍', '镐', '镑', '镀', '镇', '镰', '镜', '镶', '闰', '长', '门', '闭', '镊', '闪', '闯', '闲', '闸', '闷', '闺', '阀', '阅', '间', '阉', '问', '闻', '阎', '阐', '阁', '镣', '阔', '防', '阶', '阴', '队', '阜', '阳', '阵', '阻', '闽', '阿', '阂', '附', '阑', '陋', '陕', '陡', '陈', '限', '降', '院', '险', '阮', '除', '陪', '隅', '陷', '陶', '隆', '陀', '隋', '隘', '陛', '障', '随', '隙', '陨', '雁', '雅', '雄', '隶', '雀', '难', '雏', '雕', '集', '隔', '雌', '隧', '雷', '霄', '雨', '雪', '雹', '震', '霓', '霉', '雾', '需', '霞', '霸', '雍', '露', '靖', '青', '靡', '静', '革', '霍', '非', '靠', '靶', '霜', '面', '鞍', '鞘', '霖', '鞠', '鞭', '靛', '韩', '霹', '韵', '韧', '顷', '音', '页', '韦', '顶', '顺', '顽', '颁', '顾', '项', '须', '韭', '韶', '颐', '颊', '顿', '频', '颈', '领', '颖', '颜', '颗', '颠', '颓', '题', '额', '颤', '颧', '飘', '餐', '饥', '飞', '风', '食', '饲', '颇', '饰', '饱', '饭', '饵', '饶', '颅', '饮', '馆', '馋', '饿', '馏', '馅', '饯', '馁', '香', '驮', '驯', '马', '驱', '首', '驳', '驰', '驴', '驶', '驹', '馈', '驼', '骇', '驻', '验', '驭', '骄', '骏', '骚', '骤', '骑', '骗', '骡', '髓', '鬃', '骨', '骂', '骆', '高', '魂', '魄', '魔', '鲁', '鲍', '鬼', '鲤', '骋', '鳖', '鳞', '鱼', '鳃', '鸣', '鸦', '鸥', '魁', '鸡', '鸭', '鸿', '鸟', '鸽', '鹃', '鹅', '麓', '鹿', '鹰', '麦', '麻', '鹤', '鸳', '黄', '黎', '黑', '黔', '鸵', '默', '鼎', '鼓', '齿', '齐', '鹏', '鼻', '龙', '龚', '龋', '龟', '龄', '鼠', '鹊', '黍', '鲸', '鸯', '鲜', '魏', '驾', '饼', '馒', '预', '颂', '鞋', '靴', '靳', '零', '雇', '隐', '陵', '陇', '陆', '际', '骸', '闹', '镭', '钾', '钎', '饺', '重', '釉', '酸', '酥', '酣', '郧', '邵', '迪', '陌', '这', '迁', '较', '载', '身', '跑', '跋', '赔', '赁', '赏', '贾', '购', '酝', '货', '贡', '贞', '贝', '象', '谗', '谢', '谅', '调', '课', '赡', '评', '证', '许', '讥', '誉', '触', '裹', '貉', '袁', '蠕', '表', '衍', '谍', '衅', '蠢', '蝎', '诧', '蜂', '蜀', '蛀', '虑', '誊', '藕', '薯', '蕊', '蓖', '蕉', '蔬', '蔗', '蒲', '葵', '落', '萧', '蜗', '菇', '莆', '蘑', '药', '荫', '荡', '荆', '苯', '苦', '若', '苞', '艳', '舜', '膘', '臆', '膀', '菲', '菏', '胁', '胀', '莎', '脱', '肯', '肇', '肌', '耽', '耳', '羔', '翟', '耗', '罐', '缉', '缝', '继', '绎', '组', '练', '级', '糯', '约', '肄', '糙', '粪', '簇', '笋', '童', '稽', '程', '秸', '秆', '绦', '祈', '祸', '碾', '碰', '碟', '粳', '硼', '砖', '知', '矗', '瞎', '益', '盈', '癸', '盆', '瘸', '瘴', '磐', '磋', '痈', '症', '疾', '甸', '甄', '瓶', '琶', '瑞', '琼', '皑', '痪', '现', '痉', '率', '猛', '狠', '狂', '牟', '牙', '瑚', '爹', '琐', '琉', '熟', '琅', '煮', '煎', '玖', '焦', '猩', '焚', '狡', '灼', '炯', '烟', '激', '漠', '溯', '源', '熙', '渤', '渠', '渝', '渗', '渔', '清', '淮', '涪', '潦', '潞', '淡', '滁', '测', '浆', '泡', '溉', '法', '沸', '沃', '汹', '沥', '淬', '汇', '汀', '毗', '涟', '殴', '氏', '毕', '毒', '歧', '泞', '武', '沽', '步', '款', '欺', '樟', '欲', '欠', '榴', '氓', '椽', '棍', '栏', '柯', '果', '望', '朔', '最', '昼', '料', '昆', '斑', '敢', '敌', '携', '橇', '捌', '掇', '樱', '捉', '挣', '挠', '挖', '拔', '折', '手', '扇', '户', '戴', '我', '戍', '懒', '慨', '愿', '愧', '愤', '捍', '感', '愁', '惜', '惋', '悬', '悠', '悄', '恃', '总', '怨', '快', '忌', '忆', '徒', '弥', '弟', '并', '巴', '悸', '峪', '悯', '尘', '尖', '尔', '寄', '宠', '实', '宛', '彪', '季', '嫉', '婪', '嫂', '娥', '嫁', '巩', '媳', '巍', '婆', '峨', '姿', '如', '夹', '夸', '失', '够', '壶', '墓', '墅', '塌', '埠', '埋', '坊', '坷', '在', '圣', '圃', '图', '嗜', '喜', '哲', '哮', '哆', '哗', '哑', '哈', '咯', '呀', '吾', '叙', '县', '即', '危', '切', '啡', '内', '兢', '兽', '入', '党', '匡', '凯']
    for c in list(dictionary.char_dict):
        if c not in char_item_list:
            print(c)
            # 刊 这个字没检索到。。可能是爬虫卡了


def handle_scrapy_text(df, c):
    data = df.get_value(index=c, col='char2text_list')
    # print(str(data))
    list_data = eval(data)
    # print(len(list_data))
    list_data_segment = []
    for i in list_data:
        result = ''
        i_list = re.split(r'[，。：；？！]', i)
        for x in i_list:
            if c in x:
                result = x
                break
        if result == '':
            print(i)
        list_data_segment.append(result)
    # print(list_data_segment)
    # print(len(list_data_segment[0]))
    list_data_extract = []
    for i in list_data_segment:
        # pattern = re.compile(r"[\u4e00-\u9fa5]+")
        # result = pattern.findall(i)
        # sub(pattern,repl,string,count=0,flag=0)
        # re.sub(r'\w+','10',"xy 15 rt 3e,gep",2,flags=re.I )
        # pattern = r'[""“”“@（）（）()、×′．_－〔〕…\d□《》]A-Za-z３]'
        pattern = r'[^\u4e00-\u9fa5]'
        repl = ''
        # print(i)
        i = re.sub(pattern, repl, i)
        # print(i)
        if i == '':
            print(i)
        list_data_extract.append(i)
    # print(list_data_extract)
    # print(len(list_data_extract[0]))
    # print(len(list_data_extract))
    list_data_limit = []
    for i in list_data_extract:
        if 2 <= len(i) <= 12:
            list_data_limit.append(i)
    # print(list_data_limit)
    # print(len(list_data_limit))
    list_data_limit = list(set(list_data_limit))
    for i in list_data_limit:
        print(i)
    # print(len(list_data_limit))

    return list_data_limit


def text_gen(init_text_check_path,corpus_path):
    text_check(init_text_check_path)
    # csv_file_path = '../../res/zhonghuayuwen_3755.csv'
    csv_file_path = init_text_check_path
    df = pd.read_csv(csv_file_path)
    print(df.head())
    cols = ['char_item', 'char2text_list']
    df = df.set_index(keys='char_item')
    df = df.drop(columns='char2text_list_len')
    print(df.head())
    text_dict = {}
    c = '丁'
    for c in list(dictionary.char_dict):
        text_dict[c] = handle_scrapy_text(df, c)
    # with open('./text_dict.txt', 'a', encoding='UTF-8') as f:
    with open(corpus_path, 'a', encoding='UTF-8') as f:
        f.write(str(text_dict))
    # print(text_dict)


def run(args):
    dg = DataGenerator(hwdb_trn_dir=args.hwdb_trn_dir,hwdb_tst_dir=args.hwdb_tst_dir,hcl_dir=args.hcl_dir,
                       background_dir=args.background_dir,default_bg_path=args.default_bg_path,corpus_path=args.corpus_path,off_corpus=args.off_corpus,hcl_ratio=args.hcl_ratio,
                       img_height=args.image_height,img_width=args.image_width,const_char_num=args.const_char_num,
                       max_char_num=args.max_char_num,line_mix=args.line_mix,test_mode=args.test_mode,true_write_type=args.true_write_type)

    if args.is_first:
        ## 第一次使用需要构建HWDB和语料库
        print("is_firs:",args.is_first)
        # dg.gen_single_character_from_HWDB(args.init_hwdb_trn_dir,args.init_hwdb_tst_dir)
        if not self.off_corpus:         # 不使用语料库时候就不构建语料库
            corpus_dir = dg.corpus_path.rsplit("/", 1)[0]
            if not os.path.exists(corpus_dir):
                os.makedirs(corpus_dir)
            text_gen(args.init_text_check_path,dg.corpus_path)
            print("第一次使用，构建语料库完成！")

    ## 开始生成数据集
    data_dir = args.data_dir
    which = ""
    if dg.test_mode:
        print("Start test mode.")
        which = "train"
        gen_image_dir = os.path.join(data_dir, "test_mode", "ttest_mode_image")
        gen_info_path = os.path.join(data_dir, "test_mode", "test_mode_info", "info_local_train.txt")
        gen_frequency_list_path = os.path.join(data_dir, "test_mode", "local_train_character_word_frequency.txt")
        if args.data_augment == "":
            """
            line单行数据增强: l-画线; p-仿射变换; b-模糊(高斯、均匀、中值中随机); n-噪声(高斯噪声、弹性变换中随机); e-(浮雕、对比度中随机); c-像素(Dropout、加减、乘除中随机).
            char单字符数据增强: m-单字符随机上下;  r-单字符随机大小; a-单字符随机倾斜
            """
            data_augment = "lpbnecmra"
        else:
            data_augment = args.data_augment
        args.train_data_num = 1
        args.write_types = 1
        data_augment_percent = 1
        for i in range(len(data_augment)):
            aug = data_augment[i]
            dg.run(which=which, gen_data_num=args.train_data_num, gen_image_dir=gen_image_dir, gen_info_path=gen_info_path,
                   gen_frequency_list_path=gen_frequency_list_path, write_types=args.write_types,chars_gap_width=args.chars_gap_width,
                   data_augment=aug,data_augment_percent=data_augment_percent)
    else:
        if args.train_data_num >0:
            print("Start generate train set.")
            which = "train"
            gen_image_dir = os.path.join(data_dir,"train","train_image")
            gen_info_path = os.path.join(data_dir,"train","train_info","info_local_train.txt")
            gen_frequency_list_path = os.path.join(data_dir,"train","local_train_character_word_frequency.txt")
            data_augment = args.data_augment
            data_augment_percent = args.data_augment_percent
            dg.run(which=which,gen_data_num=args.train_data_num, gen_image_dir=gen_image_dir,gen_info_path=gen_info_path,
                                gen_frequency_list_path=gen_frequency_list_path,write_types=args.write_types,chars_gap_width=args.chars_gap_width,
                                data_augment=data_augment,data_augment_percent=data_augment_percent)
        if args.test_data_num >0:
            print("Start generate test set.")
            which = "test"
            gen_image_dir = os.path.join(data_dir,"test","test_image")
            gen_info_path = os.path.join(data_dir,"test","test_info","info_local_test.txt")
            gen_frequency_list_path = os.path.join(data_dir,"test","local_test_character_word_frequency.txt")
            data_augment = ""
            data_augment_percent = 0
            dg.run(which=which,gen_data_num=args.test_data_num, gen_image_dir=gen_image_dir,gen_info_path=gen_info_path,
                                gen_frequency_list_path=gen_frequency_list_path,write_types=args.write_types,chars_gap_width=args.chars_gap_width,
                                data_augment=data_augment,data_augment_percent=data_augment_percent)
        if args.valid_data_num >0:
            print("Start generate valid set.")
            which = "valid"
            gen_image_dir = os.path.join(data_dir,"valid","valid_image")
            gen_info_path = os.path.join(data_dir,"valid","valid_info","info_local_valid.txt")
            gen_frequency_list_path = os.path.join(data_dir,"valid","local_valid_character_word_frequency.txt")
            data_augment = ""
            data_augment_percent = 0
            dg.run(which=which,gen_data_num=args.valid_data_num, gen_image_dir=gen_image_dir,gen_info_path=gen_info_path,
                                gen_frequency_list_path=gen_frequency_list_path,write_types=args.write_types,chars_gap_width=args.chars_gap_width,
                                data_augment=data_augment,data_augment_percent=data_augment_percent)



