#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 2:04
# @Author  : Barry_J
# @Email   : s.barry1994@foxmail.com
# @File    : noise_3cnn.py
# @Software: PyCharm


"""Functions for downloading and reading noise data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream,MSB=True):
  if MSB:
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  else:
    dt = numpy.dtype(numpy.uint32).newbyteorder('<')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f,gzip_compress=True,MSB=True):
  """Extract the images into a 4D float32 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D float32 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 3331.

  """
  print('Extracting', f.name)
  # 在文件不压缩的情况下，即gzip_compress=false，使用open函数打开文件，返回file类型的对象bytestream
  # 直接使用gfile类型的对象f在调用read方法时出现了问题，原因还不清楚，
  with (gzip.GzipFile(fileobj=f) if gzip_compress else open(f.name,'rb')) as bytestream:
    magic = _read32(bytestream,MSB=MSB)
    if magic != 3331:
      raise ValueError('Invalid magic number %d in SwallowSound image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream,MSB=MSB)
    rows = _read32(bytestream,MSB=MSB)
    cols = _read32(bytestream,MSB=MSB)
    buf = bytestream.read(rows * cols * num_images * 4)#由于一个float是4个字节，所以此处要乘以4
    #此处需要自定义一个numpy.dtype，其格式是float32类型也就是4个字节的浮点数，
    # 其字节排列顺序是大端存储的方式（MSB）,用newbyteorder('>')进行设置
    # （具体哪种存储方式要看数据文件写入的时候是如何写的）
    # numpy.float32类型默认是native order,在windows下应该是小端存储
    if MSB:
      dt = numpy.dtype(numpy.float32).newbyteorder('>')
    else:
      dt = numpy.dtype(numpy.float32).newbyteorder('<')
    data = numpy.frombuffer(buf, dtype=dt)
    # data = numpy.frombuffer(buf, dtype=numpy.float32)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=75, gzip_compress=True,MSB=True):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with (gzip.GzipFile(fileobj=f) if gzip_compress else open(f.name,'rb')) as bytestream:
    magic = _read32(bytestream,MSB=MSB)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in SwallowSound label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream,MSB=MSB)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be `float32`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype is not dtypes.float32:
      raise TypeError('Invalid image dtype %r, expected float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
      self._original_shape = images.shape
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  # 用来存放原始数据的维度信息
  @property
  def original_shape(self):
    return self._original_shape

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 2500
      if self.one_hot:
        fake_label = [1] + [0] * 75
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
        fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   num_classes=75,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL,
                   train_imgaes= 'train-swallowsound-images-idx3-float.gz',
                   train_labels='train-swallowsound-labels-idx1-ubyte.gz',
                   test_imgaes='t10k-swallowsound-images-idx3-float.gz',
                   test_labels='t10k-swallowsound-labels-idx1-ubyte.gz',
                   gzip_compress=True,
                   MSB=True):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  TRAIN_IMAGES = train_imgaes
  TRAIN_LABELS = train_labels
  TEST_IMAGES = test_imgaes
  TEST_LABELS = test_labels

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   source_url + TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train_images = extract_images(f,gzip_compress=gzip_compress,MSB=MSB)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   source_url + TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot,num_classes = num_classes,gzip_compress=gzip_compress,MSB=MSB)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   source_url + TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test_images = extract_images(f,gzip_compress=gzip_compress,MSB=MSB)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   source_url + TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot,num_classes = num_classes,gzip_compress=gzip_compress,MSB=MSB)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]


  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)

