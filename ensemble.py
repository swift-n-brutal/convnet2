# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:51:57 2015

@author: v-wushi
"""

import os, sys
import cPickle
import matplotlib.pylab as pl
import numpy as n
import struct
from PIL import Image, ImageDraw, ImageFont

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_best_ratio(model1, model2, gnd_truth, result_folder):
    num_imgs = len(gnd_truth)
    num_xticks = 101
    x = n.linspace(0,1,num_xticks)
    y = n.zeros(x.shape)
    for i in xrange(num_xticks):
        eta = x[i]
        mse = n.zeros(num_imgs)
        for j in xrange(num_imgs):
            diff = model1[j] * eta + model2[j] * (1 - eta) - gnd_truth[j]
            mse[j] = (diff*diff).mean()
        psnr = -10*n.log10(mse)
        y[i] = psnr.mean()
    idx = n.argmax(y)
    pl.figure(1, figsize=(20, 12))
    pl.plot(x,y,label="ReLU * x + PReLU * (1-x)")
    pl.axvline(x=x[idx], ls='--', c="k", label="Best y=%f, x=%f" % (y[idx], x[idx]))
    pl.ylim(ymin=n.min(y))
    pl.xlabel("x")
    pl.ylabel("Average PSNR result of the ensembled model on the standard 11 images (dB)")
    pl.legend(loc="lower right")
    makedir(result_folder)
    pl.savefig(os.path.join(result_folder, "_curve_ensembled.png"))
    return x[idx]

def get_plottable_data(image, sizeY, sizeX):
        img_uint8 = image*255
        img_uint8[img_uint8<0] = 0
        img_uint8[img_uint8>255] = 255
        img_uint8 = n.require(img_uint8, dtype=n.uint8)
                
        return n.tile(img_uint8, (3,1)).T.reshape(sizeY, sizeX, 3)

def ensemble_models(model1_folder, model2_folder, image_folder, filenames, result_folder, eta=0.5, show_comp=True):
    makedir(result_folder)
    for name in open(filenames, 'r'):
        name = name.strip('\n')
        print name
        raw_data = open(os.path.join(image_folder, name), 'rb')
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
        
        mse_n = clean_img - noisy_img
        mse_n = (mse_n*mse_n).mean()
        psnr_n = -10 * n.log10(mse_n)
        print "Noisy image PSNR:", psnr_n, "dB"
        
        f1 = file(os.path.join(model1_folder, "%s.pkl" % name), "r")
        model1_img = cPickle.load(f1)
        f1.close()
        
        f2 = file(os.path.join(model2_folder, "%s.pkl" % name), "r")
        model2_img = cPickle.load(f2)
        f2.close()
        
        restored_img = model1_img[2] * eta + model2_img[2] * (1 - eta)
        
        mse = clean_img - restored_img
        mse = (mse*mse).mean()
        psnr = -10 * n.log10(mse)
        print "Restored image PSNR:", psnr, "dB"
        
        # show images
        loss_height = 20
        pad = 1
        font = ImageFont.truetype("arial.ttf", size=20)
        if show_comp:
            canvas = Image.new("RGB", \
                (width*3 + pad*2, height + loss_height), \
                "white")
            draw = ImageDraw.Draw(canvas)
            clean_obj = Image.fromarray(get_plottable_data( \
                clean_img, height, width))
            restored_obj = Image.fromarray(get_plottable_data( \
                restored_img, height, width))
            noisy_obj = Image.fromarray(get_plottable_data( \
                noisy_img, height, width))
            canvas.paste(clean_obj, (0,0))
            canvas.paste(restored_obj, (width + pad, 0))
            canvas.paste(noisy_obj, ((width + pad)*2, 0))
            draw.text((0, height), 'Clean', font=font, \
                fill="#000000")
            draw.text((width+pad, height), '%f' % psnr, font=font, \
                fill="#000000")
            draw.text(((width+pad)*2, height), '%f' % psnr_n, font=font, \
                fill="#000000")
        else:
            canvas = Image.new("RGB", \
                (width, height + loss_height), \
                "white")
            draw = ImageDraw.Draw(canvas)
            restored_obj = Image.fromarray(get_plottable_data( \
                restored_img, height, width))
            canvas.paste(restored_obj, (0, 0))
            draw.text((0, height), '%f' % psnr, font=font, \
                fill="#000000")
        canvas.save(os.path.join(result_folder, '%s_%f.png' % (name[:-4], psnr)), 'PNG')
    
def main():
    if len(sys.argv) == 6:
        model1 = list()
        model2 = list()
        gnd_truth = list()
        for name in open(sys.argv[4]):
            name = name.strip('\n')
            raw_data = open(os.path.join(sys.argv[3], name), 'rb')
            hw = raw_data.read(8)
            height, width = struct.unpack("2i", hw)
            pixels_img = height * width
            clean_img = raw_data.read(pixels_img*4)
            clean_img = n.fromstring(clean_img, dtype=n.float32, count=pixels_img)
            raw_data.close()
            gnd_truth.append(clean_img)
            
            f1 = file(os.path.join(sys.argv[1], "%s.pkl" % name), "r")
            model1_img = cPickle.load(f1)
            f1.close()
            model1.append(model1_img[2])
            
            f2 = file(os.path.join(sys.argv[2], "%s.pkl" % name), "r")
            model2_img = cPickle.load(f2)
            f2.close()
            model2.append(model2_img[2])
        eta = find_best_ratio(model1, model2, gnd_truth, sys.argv[5])
        print "Best ratio =", eta
        ensemble_models(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], eta)
    elif len(sys.argv) == 7:
        ensemble_models(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]))
    else:
        print "Usage: ensemble.py model1_folder model2_folder images_folder filenames result_folder [eta]"

def main2():
    f1 = file("d:/cuda-convnet2/tests/test1/results/std11/sig25/raw.pkl")
    f2 = file("d:/cuda-convnet2/tests/test9/results/std11/sig25/raw.pkl")
    model1 = cPickle.load(f1)
    model2 = cPickle.load(f2)
    f1.close()
    f2.close()
    
    psnr = list()
    num_imgs = len(model1)
    eta = 0.44
    for i in xrange(num_imgs):
        diff = model1[i][0] * eta + model2[i][0] * (1 - eta) - model1[i][1]
        mse = (diff*diff).mean()
        psnr.append(-10*n.log10(mse))
        
    print psnr
    print n.mean(psnr)
    


if __name__ == "__main__":
    main()
#    main2()