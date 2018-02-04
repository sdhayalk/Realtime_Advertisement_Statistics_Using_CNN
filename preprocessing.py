import tensorflow as tf
import numpy as np
import csv
import os
import io

from PIL import Image
from object_detection.utils import dataset_util

class Util:
	def __init__(self):
		pass

	def read_label(self, label_path):
		with open(label_path) as f:
			csv_reader = csv.reader(f, delimiter=",")
			line = csv_reader[0]

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

		filename = str(image_file_name) + '.jpg'
		image_format = b'jpg'

		xmins = [x_min]
		xmaxs = [x_max]
		ymins = [y_min]
		ymaxs = [y_max]
		classes_text = [class_label.encode('utf8')]
		classes = [class_index]

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

	def convert_to_tfrecords(self, images_folder_path, labels_folder_path, new_tfrecord_folder, background_image_dim_1=640, background_image_dim_2=480):
		images_list = os.listdir(images_folder_path)
		labels_list = os.listdir(labels_folder_path)

		writer = tf.python_io.TFRecordWriter(images_folder_path + os.sep + new_tfrecord_folder)

		for i in range(0, len(images_list)):
			image_path = images_folder_path + os.sep + images_list[i]
			label_path = labels_folder_path + os.sep + labels_list[i]

			image_file_name, class_label, class_index, x_min, y_min, x_max, y_max = self.read_label(label_path)

			tf_example = self.create_tf_example(image_path, image_file_name, class_label, class_index, x_min, y_min, x_max, y_max)
			writer.write(tf_example.SerializeToString())


def main():
	IMAGES_FOLDER_PATH = 
	LABELS_FOLDER_PATH = 
	NEW_TFRECORD_FOLDER = 
	BACKGROUND_IMAGE_DIM_1 = 640
	BACKGROUND_IMAGE_DIM_2 = 480

	preprocssing = Preprocessing()
	preprocssing.convert_to_tfrecords(IMAGES_FOLDER_PATH, LABELS_FOLDER_PATH, NEW_TFRECORD_FOLDER, background_image_dim_1=BACKGROUND_IMAGE_DIM_1, background_image_dim_2=BACKGROUND_IMAGE_DIM_2)

if __name__ == '__main__':
	main()