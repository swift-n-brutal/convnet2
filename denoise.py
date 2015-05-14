# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:09:45 2015

@author: v-wushi
"""

import os, sys
import numpy as n
from time import time, asctime, localtime, strftime
from python_util.util import *
from python_util.data import *
from python_util.options import *
from python_util.gpumodel import *
import layer as lay
from math import ceil, floor, sqrt, isinf, isnan
import shutil
import platform
import struct
import getopt as opt
from os import linesep as NL
from threading import Thread
import tempfile as tf
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
from PIL import Image, ImageDraw, ImageFont
import cPickle

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ModelStateException(Exception):
    pass

class DenoiseNetError(Exception):
    pass

class DenoiseNet:
    def __init__(self, model_name, op, filename_options=[]):
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.filename_options = filename_options
        self.device_ids = self.op.get_value('gpu')
        self.fill_excused_options()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)
        
        self.pixels = self.patch_size * self.patch_size
        
        self.init_data_providers()
        
        self.model_state = {}
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
        
        self.import_model()
        self.init_model_lib()
        
    def fill_excused_options(self):
        pass
    
    def import_model(self):
        lib_name = "cudaconvnet._ConvNet"
        print "========================="
        print "Importing %s C++ module" % lib_name
        self.libmodel = __import__(lib_name,fromlist=['_ConvNet'])
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers,
                                self.device_ids,
                                self.minibatch_size,
                                False)#self.conserve_mem)
        
    def init_model_state(self):
        ms = self.model_state
        layers = {}
        ms['layers'] = lay.LayerParser.parse_layers(os.path.join(self.layer_path, self.layer_def),
                                                    os.path.join(self.layer_path, self.layer_params), self, layers=layers)
        
        self.do_decouple_conv()
        self.do_unshare_weights()

        self.op.set_value('conv_to_local', [], parse=False)
        self.op.set_value('unshare_weights', [], parse=False)
        
#        self.set_driver()
    
    def do_decouple_conv(self):
        # Convert convolutional layers to local
        if len(self.op.get_value('conv_to_local')) > 0:
            for lname in self.op.get_value('conv_to_local'):
                if self.model_state['layers'][lname]['type'] == 'conv':
                    lay.LocalLayerParser.conv_to_local(self.model_state['layers'], lname)
    
    def do_unshare_weights(self):
        # Decouple weight matrices
        if len(self.op.get_value('unshare_weights')) > 0:
            for name_str in self.op.get_value('unshare_weights'):
                if name_str:
                    name = lay.WeightLayerParser.get_layer_name(name_str)
                    if name is not None:
                        name, idx = name[0], name[1]
                        if name not in self.model_state['layers']:
                            raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                        layer = self.model_state['layers'][name]
                        lay.WeightLayerParser.unshare_weights(layer, self.model_state['layers'], matrix_idx=idx)
                    else:
                        raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
    
    def init_data_providers(self):
        class MyDataProvider:
            def __init__(self, data_dim):
                self.data_dim = data_dim
                self.curr_epoch = 1
                self.curr_batchnum = 1
                self.batch_idx = 0
            
            def get_data_dims(self, idx=0):
                if idx == 0:
                    return self.data_dim
                else:
                    return 1
            
        self.train_data_provider = self.test_data_provider = MyDataProvider(self.pixels)
    
    def get_batch_data(self, image, height, width, patch_size, output_size, offset=0, keep_border=True):
        border = (patch_size - output_size) / 2
        if keep_border and border > 0:
            y_idx = range(offset, height, output_size)
            if (height - offset) % output_size != 0:
                y_idx[-1] = height - output_size
            x_idx = range(offset, width, output_size)
            if (width - offset) % output_size != 0:
                x_idx[-1] = width - output_size
            # extend image by filpping on the four edges
            image_ex = n.zeros((height + border * 2, width + border * 2), dtype=n.float32)
            image_ex[border:-border, border:-border] = image[:,:]
            for y in range(border):
                image_ex[border - y,:] = image_ex[border + y,:]
                image_ex[-border - 1 + y,:] = image_ex[-border - 1 - y,:]
            for x in range(border):
                image_ex[:,border - x] = image_ex[:,border + x]
                image_ex[:,-border - 1 + x] = image_ex[:,-border - 1 - x]
                
            batch_data = n.zeros((len(y_idx)*len(x_idx), patch_size*patch_size), dtype=n.float32)
            case_id = 0
            for y in y_idx:
                for x in x_idx:
                    batch_data[case_id,:] = image_ex[y:y+patch_size,x:x+patch_size].reshape(patch_size*patch_size)
                    case_id += 1
            return batch_data, y_idx, x_idx
        else:
            output_height = height - border * 2
            y_idx = range(offset, output_height, output_size)
            if (output_height - offset) % output_size != 0:
                y_idx[-1] = output_height - output_size
            output_width = width - border * 2
            x_idx = range(offset, output_width, output_size)
            if (output_width - offset) % output_size != 0:
                x_idx[-1] = output_width - output_size
            batch_data = n.zeros((len(y_idx)*len(x_idx), patch_size*patch_size), dtype=n.float32)
            case_id = 0
            for y in y_idx:
                for x in x_idx:
                    batch_data[case_id,:] = image[y:y+patch_size,x:x+patch_size].reshape(patch_size*patch_size)
                    case_id += 1
            return batch_data, y_idx, x_idx
    
    def get_plottable_data(self, image, sizeY, sizeX):
        img_uint8 = image*255
        img_uint8[img_uint8<0] = 0
        img_uint8[img_uint8>255] = 255
        img_uint8 = n.require(img_uint8, dtype=n.uint8)
                
        return n.tile(img_uint8, (3,1)).T.reshape(sizeY, sizeX, 3)
    
    def denoise_in_patches(self, noisy_img, height, width,
                           channels, patch_size, output_size, num_outputs, offset,
                           re_image, re_weight, patch_weight=1.0,
                           sub_mean=0, sigma_base=25.0, sigma_est=25.0):
        BATCH_MAX_SIZE = 512
        
        # cut the image into patches
        batch_data, y_idx, x_idx = \
            self.get_batch_data(noisy_img.reshape(height,width), height, width, patch_size, output_size, offset=offset)
        num_cases = batch_data.shape[0]
        batch_data_mean = sub_mean
        if sub_mean < 0:
            batch_data_mean = batch_data.mean(axis=1).reshape(num_cases, 1)
        batch_data -= batch_data_mean
        batch_data *= (sigma_base / sigma_est)
        batch_output = n.zeros((num_cases, num_outputs), dtype=n.float32)
#            print batch_data.shape, batch_output.shape
        
        for start_id in range(0, num_cases, BATCH_MAX_SIZE):
            end_id = min(num_cases, start_id + BATCH_MAX_SIZE)
            self.libmodel.startFeatureWriter([batch_data.T[:,start_id:end_id]], [batch_output[start_id:end_id,:]], [self.output_layer])
            cost_outputs = self.finish_batch()
            self.sync_with_host()
        
        # aggregate through all channels
        if channels > 1:
            patch_pixels_i = output_size * output_size
            tmp_output = n.zeros((num_cases, patch_pixels_i), dtype=n.float32)
            for c in range(channels):
                tmp_output += batch_output[:,c*patch_pixels_i:(c+1)*patch_pixels_i]
            batch_output = tmp_output / channels
            
        batch_output *= (sigma_est / sigma_base)
        batch_output += batch_data_mean
        case_id = 0
        for y in y_idx:
            for x in x_idx:
                re_image[y:y+output_size, x:x+output_size] += \
                    (batch_output[case_id,:].reshape(output_size, output_size) * patch_weight)
                # simple average
                re_weight[y:y+output_size, x:x+output_size] += patch_weight
                case_id += 1
                
    def get_gaussian_window(self, patch_size, alpha=2.0):
        window_weights = n.zeros((patch_size, patch_size))
        mid = int(floor(patch_size / 2))
        sig = floor(patch_size/2) / alpha
        for i in xrange(patch_size):
            for j in xrange(patch_size):
                d = sqrt((i - mid)**2 + (j - mid)**2)
                window_weights[i,j] = n.exp((-d*d) / (2*sig*sig))
                
        window_weights /= n.max(window_weights)
        return window_weights
        
    def read_image_bin(self, path):
        raw_data = open(path, 'rb')
        hw = raw_data.read(8)
        height, width = struct.unpack("2i", hw)
        pixels_img = height * width
        clean_img = raw_data.read(pixels_img*4)
        clean_img = n.fromstring(clean_img, dtype=n.float32, count=pixels_img)
        noisy_img = raw_data.read(pixels_img*4)
        noisy_img = n.fromstring(noisy_img, dtype=n.float32, count=pixels_img)
        assert(pixels_img == len(clean_img))
        assert(pixels_img == len(noisy_img))
        raw_data.close()
        
        return height, width, clean_img, noisy_img
        
    def load_simga_est(self, path):
        file_est = open(path, 'r')
        ret_dict = dict()
        for line in file_est:
            line = line.strip('\n')
            name_sig = line.split(' ')
            if len(name_sig) == 2:
                ret_dict[name_sig[0]] = float(name_sig[1])
        
        return ret_dict
    
    def denoise(self, keep_border=True, show_comp=True, pickle_data=False):
        print "======================="
        print "Denoising whole images"
        print "======================="
        self.op.print_values()
        print "Running on CUDA device(s) %s" % ",".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "======================="
        makedir(self.result_folder)
        patch_size = self.patch_size
        NUM_OUTPUTS = self.model_state['layers'][self.output_layer]['outputs']
        sub_mean = self.scalar_mean
        channels = 1
#        if 'channels' in self.model_state['layers'][self.output_layer]:
#            channels = self.model_state['layers'][self.output_layer]['channels']
#            if type(channels) is list:
#                channels = channels[0]
#        channels = 16
        OUTPUT_SIZE = int(sqrt(NUM_OUTPUTS / channels))
        print "channels =", channels, "output_size =", OUTPUT_SIZE
        
        BORDER = (patch_size - OUTPUT_SIZE) / 2
#        patch_weight = self.get_patch_weight(OUTPUT_SIZE)
        patch_weight = 1.0
        sigma_model = self.sigma_model
        sigma_est = self.sigma_data
        if self.sigma_est:
            sigma_est = self.load_simga_est(self.sigma_est)
            
        image_names = open(self.image_names, 'r')
        for name in image_names:
            name = name.strip('\n')
            [height, width, clean_img, noisy_img] = self.read_image_bin(os.path.join(self.image_folder, name))
            print name, height, width,
            
            start_time = time()           
            output_height = height
            output_width = width
            if not keep_border:
                output_height -= 2*BORDER
                output_width -= 2*BORDER
            output_pixels = output_height * output_width
            restored_image = n.zeros(output_pixels, dtype=n.float32)
            restored_weight = n.zeros(output_pixels, dtype=n.float32)
            re_image = restored_image.reshape(output_height, output_width)
            re_weight = restored_weight.reshape(output_height, output_width)
            
            sigma_data = sigma_est[name] if type(sigma_est) == dict else sigma_est
            self.denoise_in_patches(noisy_img, height, width,
                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 0, 
                                    re_image, re_weight, patch_weight,
                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 5, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 10, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 15, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 20, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 25, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 30, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
#            self.denoise_in_patches(noisy_img, height, width,
#                                    channels, patch_size, OUTPUT_SIZE, NUM_OUTPUTS, 35, 
#                                    re_image, re_weight, patch_weight,
#                                    sub_mean, sigma_model, sigma_data)
            re_image /= re_weight
            self.print_time(time() - start_time)
            print "sigma[model/data] = ", sigma_model, sigma_data
            
            # compute PSNR loss
            if keep_border:
                mse = clean_img - restored_image
                mse = (mse*mse).mean()
                psnr = -10 * n.log10(mse)
                print "PSNR:", psnr, "dB"
            else:
                target_clean = clean_img[border:-border, border:-border]
                mse = target_clean - restored_image
                mse = (mse*mse).mean()
                psnr = -10 * n.log10(mse)
                print "PSNR:", psnr, "dB"
            n_mse = clean_img - noisy_img
            n_mse = (n_mse*n_mse).mean()
            n_psnr = -10 * n.log10(n_mse)
            
            # show image
            loss_height = 20
            pad = 1         
            font = ImageFont.truetype("arial.ttf", size=20)
            if show_comp:
                canvas = Image.new("RGB", \
                    (width*3 + pad*2, height + loss_height), \
                    "white")
                draw = ImageDraw.Draw(canvas)
                clean_obj = Image.fromarray(self.get_plottable_data( \
                    clean_img, height, width))
                restored_obj = Image.fromarray(self.get_plottable_data( \
                    restored_image, output_height, output_width))
                noisy_obj = Image.fromarray(self.get_plottable_data( \
                    noisy_img, height, width))
                canvas.paste(clean_obj, (0,0))
                canvas.paste(restored_obj, (width + pad + (0 if keep_border else BORDER), 0))
                canvas.paste(noisy_obj, ((width + pad)*2, 0))
                draw.text((0, height), 'Clean', font=font, \
                    fill="#000000")
                draw.text((width+pad, height), '%f' % psnr, font=font, \
                    fill="#000000")
                draw.text(((width+pad)*2, height), '%f' % n_psnr, font=font, \
                    fill="#000000")
            else:
                canvas = Image.new("RGB", \
                    (width, height + loss_height), \
                    "white")
                draw = ImageDraw.Draw(canvas)
                restored_obj = Image.fromarray(self.get_plottable_data( \
                    restored_image, output_height, output_width))
                canvas.paste(restored_obj, (0 if keep_border else BORDER, 0))
                draw.text((0, height), '%f' % psnr, font=font, \
                    fill="#000000")
            
            canvas.save(os.path.join(self.result_folder, '%s_%f.png' % (name[:-4], psnr)), 'PNG')
            if pickle_data:
                f_raw = file(os.path.join(self.result_folder, '%s.pkl' % name), "w")
                cPickle.dump((output_height, output_width, restored_image), f_raw)
                f_raw.close()
                
#            break
        
        image_names.close()
        
    def start(self):
        self.denoise()
        sys.exit(0)
        
    def finish_batch(self):
        return self.libmodel.finishBatch()
    
    def sync_with_host(self):
        self.libmodel.syncWithHost()
    
    def print_time(self, time_py):
        print "(%.3f sec)" % time_py
    
    def print_costs(self, cost_outputs):
        costs, num_cases = cost_outputs[0], cost_outputs[1]
        children = set()
        for errname in costs:
            # if not a chlld of any other cost layer
            if sum(errname in self.layers[z]['children'] for z in costs) == 0:
#                print self.layers[errname]['children']
                for child in set(self.layers[errname]['children']) & set(costs.keys()):
                    costs[errname] = [v + u for v, u in zip(costs[errname], costs[child])]
                    children.add(child)
            
                filtered_costs = eval(self.layers[errname]['outputFilter'])(costs[errname], num_cases)
                print "%s: " % errname,
                if 'outputFilterFormatter' not in self.layers[errname]:
                    print ", ".join("%.6f" % v for v in filtered_costs),
                else:
                    print eval(self.layers[errname]['outputFilterFormatter'])(self,filtered_costs),
                if isnan(filtered_costs[0]) or isinf(filtered_costs[0]):
                    print "<- error nan or inf!"
                    sys.exit(1)
        for c in children:
            del costs[c]
    
    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override")
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-path", "layer_path", StringOptionParser, "Layer file path prefix", default="")
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=False)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
        op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
        op.add_option("scalar-mean", "scalar_mean", FloatOptionParser, "Substract this scalar from image (-1 = patch mean)", default=-1)
        
        op.add_option("image-folder", "image_folder", StringOptionParser, "Folder of images to restore", default="")
        op.add_option("image-names", "image_names", StringOptionParser, "File of image names", default="")
        op.add_option("result-folder", "result_folder", StringOptionParser, "Folder of restored images", default="")
        op.add_option("patch-size", "patch_size", IntegerOptionParser, "Size of the input patch", default=100)
        op.add_option("output-layer", "output_layer", StringOptionParser, "Name of the output layer", default="")
        op.add_option("sigma-est", "sigma_est", StringOptionParser, "File of estimated noise level", default="")
        op.add_option("sigma-data", "sigma_data", FloatOptionParser, "Noise level of data", default=25.0)
        op.add_option("sigma-model", "sigma_model", FloatOptionParser, "Noise level of model", default=25.0)
        return op
        
    @staticmethod
    def parse_options(op):
        try:
            op.parse()
            return op
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        sys.exit()
            

if __name__ == "__main__":
    try:
        op = DenoiseNet.get_options_parser()
        op = DenoiseNet.parse_options(op)
        model = DenoiseNet("ConvNet", op)
        model.start()
    except (UnpickleError, DenoiseNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e