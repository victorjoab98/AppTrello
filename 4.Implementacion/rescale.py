#Usage:
#python scales_image.py --src_image_dir </path/to/src/> --dest_image_dir </path/to/dst/> --cores n --factor f

import tensorflow as tf
import argparse
import os.path
import sys
import PIL
import threading


from tensorflow.python.platform import gfile
from PIL import Image



FLAGS = None


#retrieve command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
	'--src_image_dir',
	type=str,
	default='./image_dir'
)

parser.add_argument(
	'--dest_image_dir',
	type=str,
	default='./reduce_image_dir'
)

parser.add_argument(
	'--cores',
	type=int,
	default=1
)

parser.add_argument(
	'--factor',
	type=float,
	default=0.5
)


FLAGS, unparsed = parser.parse_known_args()


#prepare directories
if not gfile.Exists(FLAGS.src_image_dir):
	print("Image directory '" + FLAGS.src_image_dir + "' not found.", file=sys.stderr)
	quit()
else:
	print("Reading directory '"+ FLAGS.src_image_dir+"'")

if not gfile.Exists(FLAGS.dest_image_dir):
	print("Creating directory '"+ FLAGS.dest_image_dir+"'")
	try:
		tf.io.gfile.mkdir(FLAGS.dest_image_dir)
	except:
		print("Could not create directory '"+FLAGS.dest_image_dir+"'", file=sys.stderr)
		quit()
else:
	print("Directory '"+FLAGS.dest_image_dir+"'already exists, using current directory")



#scaleimages
def tranform_data(colorImage, prefix_name, file_extention, semaphore):
	img_size = tuple(int(i * FLAGS.factor) for i in colorImage.size)
	new_image = colorImage.resize(img_size,Image.ANTIALIAS)
	new_image.save(  prefix_name+file_extention, quality=60 )

	colorImage.close()
	semaphore.release() #increase counter
	return



#Read sub-directories
sub_dirs = [x[0] for x in gfile.Walk(FLAGS.src_image_dir)]
extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

# The root directory comes first, so skip it.
is_root_dir = True
for sub_dir in sub_dirs:
	if is_root_dir:
		is_root_dir = False
		continue

	file_list = []
	dir_name = os.path.basename(sub_dir)
	print("Looking for images in '" + dir_name + "'")

	for extension in extensions:
		file_glob = os.path.join(FLAGS.src_image_dir, dir_name, '*.' + extension)
		file_list.extend(gfile.Glob(file_glob))
	if not file_list:
		print('No files found')
		continue

	#create sub-directory in destination
	sub_folder = os.path.join(FLAGS.dest_image_dir, dir_name)
	if not gfile.Exists( sub_folder ):
		try:
			tf.io.gfile.mkdir(sub_folder)
		except:
			print("Could not create directory '"+sub_folder+"'", file=sys.stderr)
			continue


	semaphore = threading.BoundedSemaphore(FLAGS.cores)

	for file_name in file_list:
		base_name = os.path.basename(file_name)
		print("Scaling file '"+base_name+"'")
		colorImage = Image.open(os.path.abspath( os.path.join(FLAGS.src_image_dir,  file_name)))
		prefix_name = os.path.abspath(
			os.path.join(FLAGS.dest_image_dir, dir_name, os.path.splitext(base_name)[0])
		)
		file_extention = os.path.splitext(file_name)[1]

		semaphore.acquire() #decrease counter

		#scale image size
		t = threading.Thread(target=tranform_data,
			args=(colorImage, prefix_name, file_extention, semaphore))
		t.start()
