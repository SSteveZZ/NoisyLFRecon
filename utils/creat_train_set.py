# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  LF_reconstruction
   File Name    :  creat_49*txt
   Author       :  sjw
   Time         :  19-5-19 14:17
   Description  :  
-------------------------------------------------
   Change Activity: 
                   19-5-19 14:17
-------------------------------------------------
"""
# import os
# scence_name = 'books'
# scence_list = sorted(os.listdir('/home/sjw/Desktop/extra_hard_disk/dataset/training/decode_stanford2018/{}'.format(scence_name)))
# img_name_list = [60,61,62,63,64,65,66,
#                  74,75,76,77,78,79,80,
#                  88,89,90,91,92,93,94,
#                  102,103,104,105,106,107,108,
#                  116,117,118,119,120,121,122,
#                  130,131,132,133,134,135,136,
#                  144,145,146,147,148,149,150]
# w = open('/home/sjw/Desktop/extra_hard_disk/dataset/stanford2018_{}_test.txt'.format(scence_name), 'w')
# for scence in scence_list:
#     for img_name in img_name_list:
#         w.write('decode_stanford2018/{}/{}/Img_{:0>7d}.png '.format(scence_name, scence, img_name))
#     w.write('\n')
# w.close()
#
#
# f = open('/home/sjw/Desktop/extra_hard_disk/dataset/stanford2018_{}_test.txt'.format(scence_name))
# lines = f.readlines()
# f.close()
# input_index = [0, 3, 6, 21, 24, 27, 42, 45, 48]
#
# w = open('/home/sjw/Desktop/extra_hard_disk/dataset/stanford2018_{}_test_49.txt'.format(scence_name), 'w')
# for line in lines:
#     line = line.split(' ')
#     line[-1] = line[-1].rstrip('\n')
#     for x in range(7):
#         for y in range(7):
#             index = 7 * y + x
#             # if index not in [0, 3, 6, 21, 24, 27, 42, 45, 48]:
#             write_line = [line[i] for i in input_index]
#             write_line.append(line[index])
#             write_line.append(x)
#             write_line.append(y)
#             for item in range(11):
#                 w.write(str(write_line[item]) + ' ')
#             w.write(str(write_line[-1]) + ' ')
#             w.write('\n')
# w.close()


import os
# scence_name = 'books'
scence_list = sorted(os.listdir('/home/sjw/Desktop/test_data/extra/pic/'))
# img_name_list = [60,61,62,63,64,65,66,
#                  74,75,76,77,78,79,80,
#                  88,89,90,91,92,93,94,
#                  102,103,104,105,106,107,108,
#                  116,117,118,119,120,121,122,
#                  130,131,132,133,134,135,136,
#                  144,145,146,147,148,149,150]
img_name_list = list(range(49))
w = open('/home/sjw/Desktop/extra_test.txt', 'w')
for scence in scence_list:
    for img_name in img_name_list:
        w.write('extra/pic/{}/input_Cam{:0>3d}.png '.format(scence, img_name))
    w.write('\n')
w.close()


f = open('/home/sjw/Desktop/extra_test.txt')
lines = f.readlines()
f.close()
input_index = [0, 3, 6, 21, 24, 27, 42, 45, 48]

w = open('/home/sjw/Desktop/extra_test_49.txt', 'w')
for line in lines:
    line = line.split(' ')
    line[-1] = line[-1].rstrip('\n')
    for x in range(7):
        for y in range(7):
            index = 7 * y + x
            # if index not in [0, 3, 6, 21, 24, 27, 42, 45, 48]:
            write_line = [line[i] for i in input_index]
            write_line.append(line[index])
            write_line.append(x)
            write_line.append(y)
            for item in range(11):
                w.write(str(write_line[item]) + ' ')
            w.write(str(write_line[-1]) + ' ')
            w.write('\n')
w.close()

