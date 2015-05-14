# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:31:06 2015

@author: v-wushi
"""


import os
import sys
from tarfile import TarFile, TarInfo
from matplotlib import pylab as pl
import numpy as n
import getopt as opt
from python_util.util import *
from math import sqrt, ceil, floor
from python_util.gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from python_util.options import *
from PIL import Image
from time import sleep

class ShowCurveError(Exception):
    pass

class ShowCurve(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)

    def init_data_providers(self):
        self.need_gpu = False
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
            
    def init_model_state(self):
#        if self.op.get_value('show_mse'):
#            self.show_mse = self.op.get_value('show_mse')
        pass
            
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
    
    def extract_error(self, skip=1):
        train_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.train_outputs]
        test_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.test_outputs]
        if self.smooth_test_errors:
            test_errors = [sum(test_errors[max(0,i-len(self.test_batch_range)):i])/(i-max(0,i-len(self.test_batch_range))) for i in xrange(1,len(test_errors)+1)]
        output_size = self.layers[self.output_layer]['outputs']
        if self.show_mse:
            train_errors /= output_size
            test_errors /= output_size
        else:
            train_errors = 10 * (n.log10(output_size) - n.log10(train_errors))
            test_errors = 10 * (n.log10(output_size) - n.log10(test_errors))
        test_errors = n.row_stack(test_errors)
        test_errors = n.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]
        
        x = range(0, len(train_errors))
        
        return x[::skip], test_errors[::skip], train_errors[::skip]
    
    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
#        print self.test_outputs
#        train_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.train_outputs]
#        test_errors = [eval(self.layers[self.show_cost]['outputFilter'])(o[0][self.show_cost], o[1])[self.cost_idx] for o in self.test_outputs]
#        if self.smooth_test_errors:
#            test_errors = [sum(test_errors[max(0,i-len(self.test_batch_range)):i])/(i-max(0,i-len(self.test_batch_range))) for i in xrange(1,len(test_errors)+1)]
#        output_size = self.layers[self.output_layer]['outputs']
#        if self.show_mse:
#            train_errors /= output_size
#            test_errors /= output_size
#        else:
#            train_errors = 10 * (n.log10(output_size) - n.log10(train_errors))
#            test_errors = 10 * (n.log10(output_size) - n.log10(test_errors))
        numbatches = len(self.train_batch_range)
#        test_errors = n.row_stack(test_errors)
#        test_errors = n.tile(test_errors, (1, self.testing_freq))
#        test_errors = list(test_errors.flatten())
#        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
#        test_errors = test_errors[:len(train_errors)]
        x, test_errors, train_errors = self.extract_error()

        numepochs = len(train_errors) / float(numbatches)
        pl.figure(1, figsize=(20,12))
#        x = range(0, len(train_errors))
        skip = self.step
        pl.plot(x[::skip], train_errors[::skip], 'k-', label='Training set')
        pl.plot(x[::skip], test_errors[::skip], 'r-', label='Test set')
#        pl.legend()
        epoch_label_gran = int(ceil(numepochs / 20.)) 
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) if numepochs >= 10 else epoch_label_gran 
        ticklocs = range(numbatches*epoch_label_gran, len(train_errors) - len(train_errors) % numbatches + 1, numbatches*epoch_label_gran)
        ticklabels = range(epoch_label_gran, int(ceil(numepochs)) + 1, epoch_label_gran)
                
        if self.show_mse:
            pl.ylim(ymax=0.002)
            pl.ylabel("MSE")
            pl.legend(loc='upper right')
        else:
            pl.ylim(27.0, 29.0)
            pl.ylabel("PSNR (dB)")
            pl.legend(loc='lower right')
        pl.xticks(ticklocs, ticklabels)
        pl.grid(b=True, axis='y', linewidth=1)
#        pl.xlabel('Epoch')
        pl.xlabel('Number of Patches (x %d)' % (self.patch_perbatch * numbatches))
#        pl.ylabel(self.show_cost)
        pl.title('%s[%d]' % (self.show_cost, self.cost_idx))
#        print "plotted cost"
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans, FILTERS_PER_ROW=16):
        MAX_ROWS = 24
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_pixels = filters.shape[1]
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        FILTERS_PER_ROW = 16
        filter_start = 0 # First filter to show
        if self.show_filters not in self.layers:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[self.show_filters]
        filters = layer['weights'][self.input_idx]
#        filters = filters - filters.min()
#        filters = filters / filters.max()
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
            filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], channels, layer['filterPixels'][self.input_idx], num_filters))
                filters = filters[:, :, :, self.local_plane] # first map for now (modules, channels, pixels)
                filters = filters.swapaxes(0,2).swapaxes(0,1)
                num_filters = layer['modules']
#                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
#                num_filters *= layer['modules']
                FILTERS_PER_ROW = layer['modulesX']
            else:
                filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        if self.norm_filters:
#            print filters.shape
            filters = filters - n.tile(filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).mean(axis=0).reshape(1, 1, filters.shape[2]), (filters.shape[0], filters.shape[1], 1))
            filters_var = filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).var(axis=0)
            filters_order = sorted(list(enumerate(filters_var)), key=lambda x:x[1], reverse=True)
#            filters = filters / n.sqrt(n.tile(filters_var.reshape(1, 1, filters.shape[2]), (filters.shape[0], filters.shape[1], 1)))
            filters = filters - n.tile(filters.min(axis=0).min(axis=0), (1, filters.shape[1], 1))
            filters = filters / n.tile(filters.max(axis=0).max(axis=0), (1, filters.shape[1], 1))
        #else:
        filters = filters - filters.min()
        filters = filters / filters.max()
        
        if self.norm_filters:
            filters_tmp = n.ndarray(filters.shape, dtype=n.float32)
            for i in range(filters.shape[2]):
                filters_tmp[:,:,i] = filters[:,:,filters_order[i][0]]
            filters = filters_tmp
        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans, FILTERS_PER_ROW=FILTERS_PER_ROW)

    def start(self):
        self.op.print_values()
#        print self.show_cost
        if self.show_cost:
            self.plot_cost()
        elif self.show_filters:
            self.plot_filters()
        if pl:
#            pl.show()
            if self.result_path:
                pl.savefig(self.result_path)
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('load_file', 'sigma'):
                op.delete_option(option)
        
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show filters of specified layer", default="")
        op.add_option("norm-filters", "norm_filters", BooleanOptionParser, "Individually normalize filters shown with --show-filters", default=0)
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("smooth-test-errors", "smooth_test_errors", BooleanOptionParser, "Use running average for test error plot?", default=0)
        op.add_option("output-layer", "output_layer", StringOptionParser, "Output layer that the cost is computed from", default="")
        op.add_option("show-mse", "show_mse", BooleanOptionParser, "Show mse error (or PSNR error)", default=False)
        op.add_option("step", "step", IntegerOptionParser, "Step of x axis", default=1)
        op.add_option("patch-perbatch", "patch_perbatch", IntegerOptionParser, "Num of patches per batch", default=4096)
        
        op.add_option("result-path", "result_path", StringOptionParser, "Path to store the PSNR curve", default="")

        op.options['load_file'].default = None
        return op
    
if __name__ == "__main__":
    #nr.seed(6)
    try:
        op = ShowCurve.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
#        if 'sigma' not in op.options:
#            op.add_option("")
        model = ShowCurve(op, load_dic)
        model.start()
    except (UnpickleError, ShowCurveError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
