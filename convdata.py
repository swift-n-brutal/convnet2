# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from python_util.data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from threading import Thread
from math import sqrt
import sys
#from matplotlib import pylab as pl
from PIL import Image
from StringIO import StringIO
from time import time
import itertools as it

class OttoLoaderThread(Thread):
    MINI_BATCH_SIZE = 128
    def __init__(self, dp, batch_idx, list_out):
        Thread.__init__(self)
        self.dp = dp
        self.batch_idx = batch_idx
        self.list_out = list_out
    
    @staticmethod
    def make_batch(dp, batch_idx, test):
        if test:
            num_samples = dp.val_set.shape[0]
            data_mat = dp.data_feat[dp.val_set]
            label_mat = n.zeros(shape=(num_samples, dp.num_classes), dtype=n.float32)
            labels = dp.labels[dp.val_set]
            for idx in xrange(num_samples):
                label_mat[idx, labels[idx]] = 1
            
            return {"data": data_mat,
                    "label_mat": label_mat,
                    "labels": labels.reshape(-1,1)}
        else:
            num_samples = dp.train_set.shape[0]
            train_rs = n.random.RandomState()
            rand_order = train_rs.permutation(num_samples)
            data_mat = dp.data_feat[dp.train_set[rand_order]]
            label_mat = n.zeros(shape=(num_samples, dp.num_classes), dtype=n.float32)
            labels = dp.labels[dp.train_set[rand_order]]
            for idx in xrange(num_samples):
                label_mat[idx, labels[idx]] = 1
            
            return {"data": data_mat,
                    "label_mat": label_mat,
                    "labels": labels.reshape(-1,1)}
    
    def run(self):
        p = OttoLoaderThread.make_batch(self.dp, self.batch_idx, self.dp.test)
        self.list_out.append(p)

class OttoDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
#        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        if batch_range == None:
            batch_range = DataProvider.get_batch_nums(data_dir)        
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]
        self.data_dir = data_dir
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
#        self.batch_meta = self.get_batch_meta(data_dir)
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)
        
        self.mini = dp_params['minibatch_size']
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread = None
        self.convnet = dp_params['convnet']
        
        self.batches_generated, self.loaders_started = 0, 0
        
        self.num_features = dp_params['inner_size']
        self.num_classes = dp_params['out_size']
        self.data_feat = -1
        self.labels = -1
#        self.data_path = "train.csv"
        self.data_path = data_dir
        self.M = 32

    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.num_features
        elif idx == 2:
            return self.num_classes
        return 1
        
    def get_num_classes(self):
        return self.num_classes
        
    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = OttoLoaderThread(self, batch_idx, self.load_data)
        self.loader_thread.start()
    
    def load_data_labels(self, path, num_features, num_classes, skip_header=1,
                         norm="var"):
#        cols_feat = tuple(range(1, num_features + 1))
#        data = n.genfromtxt(path, dtype=n.float32, delimiter=',', \
#                usecols=cols_feat, skip_header=skip_header)
##        D = data.sum(axis=1)
##        data = (1 + float(self.M)/D).reshape(-1,1) *data
#        labels = n.zeros(shape=data.shape[0], dtype=n.float32)
#        fh = open(path, 'r')
#        for idx in xrange(skip_header):
#            fh.readline()
#        for idx in xrange(data.shape[0]):
#            line = fh.readline()
#            assert(line)
#            line = line.strip('\n')
#            labels[idx] = int(line[-1]) - 1
#        fh.close()
        fh = open(path, 'r')
        data_dict = cPickle.load(fh)
        fh.close()
        data = data_dict["feat"]
        labels = data_dict["label"]
        if norm == "orsm":
            D = data.sum(axis=1)
            D[D==0] = 1
            data = (1 + float(self.M)/D).reshape(-1,1) *data
        elif norm == "var":
            data_dir = os.path.dirname(path)
            mean_var_name = "train_test_mean_var.pkl"
            fh = open(os.path.join(data_dir, mean_var_name), "r")
            mean_var = cPickle.load(fh)
            fh.close()
            data = data - mean_var["mean"]
            data = data / n.sqrt(mean_var["var"])
            print data.mean(), data.var()
            
        assert(data.shape[1] == num_features)
        assert(all(labels <= num_classes))
        return data, labels
    
    def split_train_val_set(self, train_ratio=0.9):
        num_samples = self.data_feat.shape[0]
        val_start = int(num_samples * train_ratio) \
            / OttoLoaderThread.MINI_BATCH_SIZE \
            * OttoLoaderThread.MINI_BATCH_SIZE
        tmp_idx = range(num_samples)
        split_rs = n.random.RandomState(1)
        split_rs.shuffle(tmp_idx)
        train_idx = tmp_idx[:val_start]
        val_idx = tmp_idx[val_start:]
        self.train_set = n.array(sorted(train_idx), dtype=n.int32)
        self.val_set = n.array(sorted(val_idx), dtype=n.int32)
        print "Training set:", self.train_set.shape[0], "samples"
        print "Validation set:", self.val_set.shape[0], "samples"
    
    def get_data_from_loader(self):
        if type(self.data_feat) == int:
            self.data_feat, self.labels = self.load_data_labels(self.data_path, 
                                                           self.num_features, 
                                                           self.num_classes)
            print 'Loaded', self.data_feat.shape[0], 'samples'
            self.split_train_val_set()
            
        if self.loader_thread == None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            # He actually means .join(0)
            self.loader_thread.join()
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
#  we do this in the data maker
#        self.data[self.d_idx]["noisy"] -= self.data_mean_crop
#        self.data[self.d_idx]["clean"] -= self.data_mean_crop
        
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]["data"].T, 
                                 self.data[self.d_idx]["labels"].T,
                                 self.data[self.d_idx]["label_mat"].T]
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
#        mean = self.data_mean_crop.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.reshape((data.shape[0],1))
        mean = 0
        size = int(n.sqrt(data.shape[0]))
#        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 1, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2), dtype=n.single)
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], size, size).swapaxes(1,2), dtype=n.single)

"""
    Gaussian noisy data provider
"""
class NoisyMakerThread(Thread):
    NUM_PATCH = 8
    def __init__(self, dp, batch_idx, list_out):
        Thread.__init__(self)
        self.dp = dp
        self.batch_idx = batch_idx
        self.list_out = list_out
    
    @staticmethod
    def make_batch(dp, batch_idx, test):
        image_batch = dp.image_batch_list[batch_idx]
        clean_list = list()
        noisy_list = list()
        sigma_base = dp.sigma_base
        scalar_mean = dp.scalar_mean
        inner_size = dp.inner_size
        out_size = dp.out_size
        border = (inner_size - out_size) / 2
        
        if test:
            test_nr = nr.RandomState(1)
            sigma = Sigma(dp.sigma_range)
            sigma.seed(1)
            for img in image_batch:
                height = img[0]
                width = img[1]
                data = n.fromstring(img[2], dtype=n.float32, count=height*width)
                assert len(data) == height*width
                data.shape = height, width
                hs = test_nr.randint(0, height - inner_size, (NoisyMakerThread.NUM_PATCH))
                ws = test_nr.randint(0, width - inner_size, (NoisyMakerThread.NUM_PATCH))
                noisy_list += [data[hs[i]:hs[i]+inner_size, ws[i]:ws[i]+inner_size] \
                    for i in xrange(NoisyMakerThread.NUM_PATCH)]
            
            num_patches = len(noisy_list)
            for i in xrange(num_patches):
                clean_list.append(noisy_list[i][border:-border, border:-border])
                
            clean_mat = n.empty((num_patches, out_size * out_size), dtype=n.float32)
            noisy_mat = n.empty((num_patches, inner_size * inner_size), dtype=n.float32)
            for i in xrange(num_patches):
                noise_level = sigma.getValue()
                clean_mat[i,:] = clean_list[i].reshape(out_size*out_size)
                noisy_mat[i,:] = noisy_list[i].reshape(inner_size*inner_size) \
                    + (noise_level * test_nr.randn(inner_size*inner_size)).astype(n.float32)
                mu = scalar_mean if scalar_mean >= 0 else noisy_mat[i,:].mean()
                clean_mat[i,:] = (clean_mat[i,:] - mu) * (sigma_base / noise_level)
                noisy_mat[i,:] = (noisy_mat[i,:] - mu) * (sigma_base / noise_level)
            return {"clean": clean_mat,
                    "noisy": noisy_mat}
        else:
            sigma = dp.sigma
            for img in image_batch:
                height = img[0]
                width = img[1]
                data = n.fromstring(img[2], dtype=n.float32, count=height*width)
                assert len(data) == height*width
                data.shape = height, width
                hs = nr.randint(0, height - inner_size, (NoisyMakerThread.NUM_PATCH))
                ws = nr.randint(0, width - inner_size, (NoisyMakerThread.NUM_PATCH))
                noisy_list += [data[hs[i]:hs[i]+inner_size, ws[i]:ws[i]+inner_size] for i in xrange(NoisyMakerThread.NUM_PATCH)]
            
            num_patches = len(noisy_list)
            for i in xrange(num_patches):
                clean_list.append(noisy_list[i][border:-border, border:-border])
            rand_order = range(num_patches)
            r.shuffle(rand_order) # train only
                
            clean_mat = n.empty((num_patches, out_size * out_size), dtype=n.float32)
            noisy_mat = n.empty((num_patches, inner_size * inner_size), dtype=n.float32)
            for i in xrange(num_patches):
                noise_level = sigma.getValue()
                clean_mat[i,:] = clean_list[rand_order[i]].reshape(out_size*out_size)
                noisy_mat[i,:] = noisy_list[rand_order[i]].reshape(inner_size*inner_size) \
                    + (noise_level * nr.randn(inner_size*inner_size)).astype(n.float32)
                mu = scalar_mean if scalar_mean >= 0 else noisy_mat[i,:].mean()
                clean_mat[i,:] = (clean_mat[i,:] - mu) * (sigma_base / noise_level)
                noisy_mat[i,:] = (noisy_mat[i,:] - mu) * (sigma_base / noise_level)
            return {"clean": clean_mat,
                    "noisy": noisy_mat}
            
    def run(self):
        p = NoisyMakerThread.make_batch(self.dp, self.batch_idx, self.dp.test)
        self.list_out.append(p)


class Sigma:
    def __init__(self, sigma_range, max_value=255.0):
        self.sig_min = min(sigma_range)
        self.sig_max = max(sigma_range)
        self.randstate = None
        if self.sig_min != self.sig_max:
            self.randstate = n.random.RandomState()
        self.sig_min = self.sig_min / max_value
        self.sig_max = self.sig_max / max_value
            
    def seed(self, sd=None):
        if self.randstate:
            self.randstate.seed(sd)
    
    def getValue(self):
        if self.randstate:
            return self.randstate.uniform(self.sig_min, self.sig_max)
        return self.sig_min

class NoisyDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.num_colors = 1
#        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.out_size = dp_params['out_size']
        assert (self.inner_size - self.out_size)%2 == 0
        self.border_size = (self.inner_size - self.out_size) / 2
        self.multiview = dp_params['multiview_test']
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.batch_size = self.batch_meta['batch_size']
        self.scalar_mean = dp_params['scalar_mean']
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread, self.noise_thread = None, None
        self.convnet = dp_params['convnet']
        
        self.sigma_base = dp_params['sigma_base'] / 255.0
        self.sigma_range = dp_params['sigma']
        self.sigma = Sigma(dp_params['sigma'])
            
#        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
#        self.data_mean_crop = self.data_mean.reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,self.num_colors*self.inner_size**2))
        self.data_mean_crop = 0

        if self.scalar_mean >= 0:
            self.data_mean_crop = self.scalar_mean
            
        self.image_batch_list = None

    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.out_size**2
        elif idx == 1:
            return self.inner_size**2
        return 1

    def start_maker(self, batch_idx):
        self.load_data = []
        self.noise_thread = NoisyMakerThread(self, batch_idx, self.load_data)
        self.noise_thread.start()
    
    def get_data_from_maker(self):
        if self.image_batch_list == None:
            self.image_batch_list = list()
            num_imgs = 0
            for batch_num in self.batch_range:
                self.image_batch_list += [self.get_batch(batch_num)["data"]]
                num_imgs += len(self.image_batch_list[-1])
            print 'Loaded', num_imgs, 'images'
        
        if self.noise_thread == None:
            self.start_maker(self.batch_idx)
            self.noise_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_maker(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            # He actually means .join(0)
            self.noise_thread.join()
            if not self.noise_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_maker(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_maker()

        # Subtract mean
#  we do this in the data maker
#        self.data[self.d_idx]["noisy"] -= self.data_mean_crop
#        self.data[self.d_idx]["clean"] -= self.data_mean_crop
        
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]["clean"].T, self.data[self.d_idx]["noisy"].T]
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
#        mean = self.data_mean_crop.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.reshape((data.shape[0],1))
        mean = self.data_mean_crop
        size = int(n.sqrt(data.shape[0]))
#        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 1, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2), dtype=n.single)
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], size, size).swapaxes(1,2), dtype=n.single)
       
"""
    Imagenet data provider
"""
class JPEGBatchLoaderThread(Thread):
    def __init__(self, dp, batch_num, label_offset, list_out):
        Thread.__init__(self)
        self.list_out = list_out
        self.label_offset = label_offset
        self.dp = dp
        self.batch_num = batch_num
        
    @staticmethod
    def load_jpeg_batch(rawdics, dp, label_offset):
        if type(rawdics) != list:
            rawdics = [rawdics]
        nc_total = sum(len(r['data']) for r in rawdics)

        jpeg_strs = list(it.chain.from_iterable(rd['data'] for rd in rawdics))
        labels = list(it.chain.from_iterable(rd['labels'] for rd in rawdics))
        
        img_mat = n.empty((nc_total * dp.data_mult, dp.inner_pixels * dp.num_colors), dtype=n.float32)
        lab_mat = n.zeros((nc_total, dp.get_num_classes()), dtype=n.float32)
        dp.convnet.libmodel.decodeJpeg(jpeg_strs, img_mat, dp.img_size, dp.inner_size, dp.test, dp.multiview)
        lab_vec = n.tile(n.asarray([(l[nr.randint(len(l))] if len(l) > 0 else -1) + label_offset for l in labels], dtype=n.single).reshape((nc_total, 1)), (dp.data_mult,1))
        for c in xrange(nc_total):
            lab_mat[c, [z + label_offset for z in labels[c]]] = 1
        lab_mat = n.tile(lab_mat, (dp.data_mult, 1))
        

        return {'data': img_mat[:nc_total * dp.data_mult,:],
                'labvec': lab_vec[:nc_total * dp.data_mult,:],
                'labmat': lab_mat[:nc_total * dp.data_mult,:]}
    
    def run(self):
        rawdics = self.dp.get_batch(self.batch_num)
        p = JPEGBatchLoaderThread.load_jpeg_batch(rawdics,
                                                  self.dp,
                                                  self.label_offset)
        self.list_out.append(p)
        
class ColorNoiseMakerThread(Thread):
    def __init__(self, pca_stdevs, pca_vecs, num_noise, list_out):
        Thread.__init__(self)
        self.pca_stdevs, self.pca_vecs = pca_stdevs, pca_vecs
        self.num_noise = num_noise
        self.list_out = list_out
        
    def run(self):
        noise = n.dot(nr.randn(self.num_noise, 3).astype(n.single) * self.pca_stdevs.T, self.pca_vecs.T)
        self.list_out.append(noise)

class ImageDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean'].astype(n.single)
        self.color_eig = self.batch_meta['color_pca'][1].astype(n.single)
        self.color_stdevs = n.c_[self.batch_meta['color_pca'][0].astype(n.single)]
        self.color_noise_coeff = dp_params['color_noise']
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.mini = dp_params['minibatch_size']
        self.inner_size = dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.img_size
        self.inner_pixels = self.inner_size **2
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.batch_size = self.batch_meta['batch_size']
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.scalar_mean = dp_params['scalar_mean'] 
        # Maintain pointers to previously-returned data matrices so they don't get garbage collected.
        self.data = [None, None] # These are pointers to previously-returned data matrices

        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']
            
        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
        self.data_mean_crop = self.data_mean.reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,3*self.inner_size**2))

        if self.scalar_mean >= 0:
            self.data_mean_crop = self.scalar_mean
            
    def showimg(self, img):
        from matplotlib import pylab as pl
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        pl.imshow(img, interpolation='nearest')
        pl.show()
            
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        if idx == 2:
            return self.get_num_classes()
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = JPEGBatchLoaderThread(self,
                                                   self.batch_range[batch_idx],
                                                   self.label_offset,
                                                   self.load_data)
        self.loader_thread.start()
        
    def start_color_noise_maker(self):
        color_noise_list = []
        self.color_noise_thread = ColorNoiseMakerThread(self.color_stdevs, self.color_eig, self.num_noise, color_noise_list)
        self.color_noise_thread.start()
        return color_noise_list

    def set_labels(self, datadic):
        pass
    
    def get_data_from_loader(self):
        if self.loader_thread is None:
            self.start_loader(self.batch_idx)
            self.loader_thread.join()
            self.data[self.d_idx] = self.load_data[0]

            self.start_loader(self.get_next_batch_idx())
        else:
            # Set the argument to join to 0 to re-enable batch reuse
            # He actually means .join(0)
            self.loader_thread.join()
            if not self.loader_thread.is_alive():
                self.data[self.d_idx] = self.load_data[0]
                self.start_loader(self.get_next_batch_idx())
            #else:
            #    print "Re-using batch"
        self.advance_batch()
    
    def add_color_noise(self):
        # At this point the data already has 0 mean.
        # So I'm going to add noise to it, but I'm also going to scale down
        # the original data. This is so that the overall scale of the training
        # data doesn't become too different from the test data.

        s = self.data[self.d_idx]['data'].shape
        cropped_size = self.get_data_dims(0) / 3
        ncases = s[0]

        if self.color_noise_thread is None:
            self.color_noise_list = self.start_color_noise_maker()
            self.color_noise_thread.join()
            self.color_noise = self.color_noise_list[0]
            self.color_noise_list = self.start_color_noise_maker()
        else:
            self.color_noise_thread.join(0)
            if not self.color_noise_thread.is_alive():
                self.color_noise = self.color_noise_list[0]
                self.color_noise_list = self.start_color_noise_maker()

        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases*3, cropped_size))
        self.color_noise = self.color_noise[:ncases,:].reshape((3*ncases, 1))
        self.data[self.d_idx]['data'] += self.color_noise * self.color_noise_coeff
        self.data[self.d_idx]['data'] = self.data[self.d_idx]['data'].reshape((ncases, 3* cropped_size))
        self.data[self.d_idx]['data'] *= 1.0 / (1.0 + self.color_noise_coeff) # <--- NOTE: This is the slow line, 0.25sec. Down from 0.75sec when I used division.
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        epoch, batchnum = self.curr_epoch, self.curr_batchnum

        self.get_data_from_loader()

        # Subtract mean
        self.data[self.d_idx]['data'] -= self.data_mean_crop
        
        if self.color_noise_coeff > 0 and not self.test:
            self.add_color_noise()
        self.batches_generated += 1
        
        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['labvec'].T, self.data[self.d_idx]['labmat'].T]
        
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean_crop.reshape((data.shape[0],1)) if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.reshape((data.shape[0],1))
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
       
class CIFARDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.img_size = 32 
        self.num_colors = 3
        self.inner_size =  dp_params['inner_size'] if dp_params['inner_size'] > 0 else self.batch_meta['img_size']
        self.border_size = (self.img_size - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 9
        self.scalar_mean = dp_params['scalar_mean'] 
        self.data_mult = self.num_views if self.multiview else 1
        self.data_dic = []
        for i in batch_range:
            self.data_dic += [unpickle(self.get_data_file_name(i))]
            self.data_dic[-1]["labels"] = n.require(self.data_dic[-1]['labels'], dtype=n.single)
            self.data_dic[-1]["labels"] = n.require(n.tile(self.data_dic[-1]["labels"].reshape((1, n.prod(self.data_dic[-1]["labels"].shape))), (1, self.data_mult)), requirements='C')
            self.data_dic[-1]['data'] = n.require(self.data_dic[-1]['data'] - self.scalar_mean, dtype=n.single, requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(self.data_dic[bidx]['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, self.data_dic[bidx]['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0), (0, self.border_size), (0, self.border_size*2),
                                  (self.border_size, 0), (self.border_size, self.border_size), (self.border_size, self.border_size*2),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views):
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

class DummyConvNetLogRegDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)

        self.img_size = int(sqrt(data_dim/3))
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        dic = {'data': dic[0], 'labels': dic[1]}
        print dic['data'].shape, dic['labels'].shape
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
