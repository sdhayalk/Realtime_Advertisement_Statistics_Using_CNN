import tensorflow as tf
import numpy as np
import csv
import os
import io

from PIL import Image
from models.research.object_detection.utils import dataset_util

class Util:
	def __init__(self):
		pass

	def read_label(self, label_path):
		line = list(open(label_path))[0].split(',')
		print(line)

		image_file_name = line[0]
		class_label = line[1]
		class_index = int(line[2])
		x_min = float(line[3])
		y_min = float(line[4])
		x_max = float(line[5])
		y_max = float(line[6])

		return image_file_name, class_label, class_index, x_min, y_min, x_max, y_max


class Preprocessing(Util):
	def __init__(self):
		pass

	def create_tf_example(self, image_path, image_file_name, class_label, class_index, x_min, y_min, x_max, y_max):
		with tf.gfile.GFile(image_path, 'rb') as fid:
			encoded_image_data = fid.read()

		encoded_image_data_io = io.BytesIO(encoded_image_data)
		image = Image.open(encoded_image_data_io)
		width, height = image.size
		print(width, height)

		filename = image_file_name
		image_format = b'jpg'

		xmins = [x_min]
		xmaxs = [x_max]
		ymins = [y_min]
		ymaxs = [y_max]
		classes_text = [class_label.encode('utf8')]
		classes = [class_index + 1]

		tf_example = tf.train.Example(features=tf.train.Features(feature={
		  'image/height': dataset_util.int64_feature(height),
		  'image/width': dataset_util.int64_feature(width),
		  'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
		  'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
		  'image/encoded': dataset_util.bytes_feature(encoded_image_data),
		  'image/format': dataset_util.bytes_feature(image_format),
		  'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		  'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		  'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		  'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		  'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		  'image/object/class/label': dataset_util.int64_list_feature(classes),
		}))
		return tf_example

	def convert_to_tfrecords(self, images_folder_path, labels_folder_path, new_tfrecord_train_folder, new_tfrecord_test_folder, splitter, background_image_dim_1=640, background_image_dim_2=480):
		images_train_list = os.listdir(images_folder_path)[0:splitter]
		labels_train_list = os.listdir(labels_folder_path)[0:splitter]
		images_test_list = os.listdir(images_folder_path)[splitter:1000]
		labels_test_list = os.listdir(labels_folder_path)[splitter:1000]

		''' generating tf records for training '''
		writer = tf.python_io.TFRecordWriter(new_tfrecord_train_folder)

		for i in range(0, len(images_train_list)):
			image_path = images_folder_path + os.sep + images_train_list[i]
			label_path = labels_folder_path + os.sep + labels_train_list[i]

			image_file_name, class_label, class_index, x_min, y_min, x_max, y_max = self.read_label(label_path)

			tf_example = self.create_tf_example(image_path, image_file_name, class_label, class_index, x_min, y_min, x_max, y_max)
			writer.write(tf_example.SerializeToString())

		''' generating tf records for testing '''
		writer = tf.python_io.TFRecordWriter(new_tfrecord_test_folder)

		for i in range(0, len(images_test_list)):
			image_path = images_folder_path + os.sep + images_test_list[i]
			label_path = labels_folder_path + os.sep + labels_test_list[i]

			image_file_name, class_label, class_index, x_min, y_min, x_max, y_max = self.read_label(label_path)

			tf_example = self.create_tf_example(image_path, image_file_name, class_label, class_index, x_min, y_min, x_max, y_max)
			writer.write(tf_example.SerializeToString())


def main():
	IMAGES_FOLDER_PATH = 'G:/DL/data_logo/coco_data/overlayed_images/Images'
	LABELS_FOLDER_PATH = 'G:/DL/data_logo/coco_data/overlayed_images/Labels'
	NEW_TFRECORD_TRAIN_FOLDER = 'G:/DL/data_logo/coco_data/overlayed_images/TFTrainRecord.record'
	NEW_TFRECORD_TEST_FOLDER = 'G:/DL/data_logo/coco_data/overlayed_images/TFTestRecord.record'
	SPLITTER = 950
	BACKGROUND_IMAGE_DIM_1 = 640
	BACKGROUND_IMAGE_DIM_2 = 480

	preprocssing = Preprocessing()
	preprocssing.convert_to_tfrecords(IMAGES_FOLDER_PATH, LABELS_FOLDER_PATH, NEW_TFRECORD_TRAIN_FOLDER, NEW_TFRECORD_TEST_FOLDER, SPLITTER, background_image_dim_1=BACKGROUND_IMAGE_DIM_1, background_image_dim_2=BACKGROUND_IMAGE_DIM_2)

if __name__ == '__main__':
	main()


# python models/research/object_detection/train.py --logtostderr --train_dir=G:/DL/Realtime_Advertisement_Statistics_Using_CNN --pipeline_config_path=G:/DL/Realtime_Advertisement_Statistics_Using_CNN/faster_rcnn_inception_v2_coco.config