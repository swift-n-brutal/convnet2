/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/pyext.h"

using namespace std;

static PyMethodDef _MakeDataPyExtMethods[] = {{ "resizeJPEG", resizeJPEG, METH_VARARGS },
											  { "cropGrayPNG", cropGrayPNG, METH_VARARGS },
											  { "cvtGray32F", cvtGray32F, METH_VARARGS },
                                              { NULL, NULL }
};

//void init_MakeDataPyExt() {
PyMODINIT_FUNC init_MakeDataPyExt() {
    (void) Py_InitModule("_MakeDataPyExt", _MakeDataPyExtMethods);
}

PyObject* resizeJPEG(PyObject *self, PyObject *args) {

    PyListObject* pyListSrc;
    int tgtImgSize, numThreads;
    int cropToSquare;

    if (!PyArg_ParseTuple(args, "O!iii",
                          &PyList_Type, &pyListSrc,
                          &tgtImgSize,
                          &numThreads,
                          &cropToSquare)) {
        return NULL;
    }

    //DecoderThread* threads[numThreads];
    std::vector<DecoderThreadExt*> threads(numThreads);
    int num_imgs = PyList_GET_SIZE(pyListSrc);
    int num_imgs_per_thread = DIVUP(num_imgs, numThreads);
    for (int t = 0; t < numThreads; ++t) {
        int start_img = t * num_imgs_per_thread;
        int end_img = min(num_imgs, (t+1) * num_imgs_per_thread);

        threads[t] = new DecoderThreadExt((PyObject*)pyListSrc, start_img, end_img, tgtImgSize, cropToSquare);
        threads[t]->start();
    }

    PyObject* pyListTgt = PyList_New(0);
    for (int t = 0; t < numThreads; ++t) {
        threads[t]->join();
        PyList_Append(pyListTgt, threads[t]->getTargetList());
        delete threads[t]; // the thread's list too
    }

    return pyListTgt;
}

PyObject* cropGrayPNG(PyObject *self, PyObject *args) {

  PyListObject* pyListSrc;
  int tgtImgSize, numThreads;
  int cropToSquare;

  if (!PyArg_ParseTuple(args, "O!iii",
	  &PyList_Type, &pyListSrc, &tgtImgSize, &numThreads, &cropToSquare)) {
	return NULL;
  }

  std::vector<PNGDecoderThreadExt*> threads(numThreads);
  int num_imgs = PyList_GET_SIZE(pyListSrc);
  int num_imgs_per_thread = DIVUP(num_imgs, numThreads);
  for (int t = 0; t < numThreads; ++t) {
	int start_img = t * num_imgs_per_thread;
	int end_img = min(num_imgs, (t + 1)*num_imgs_per_thread);

	threads[t] = new PNGDecoderThreadExt((PyObject*)pyListSrc,
	    start_img, end_img, tgtImgSize, cropToSquare);
	threads[t]->start();
  }

  PyObject* pyListTgt = PyList_New(0);
  for (int t = 0; t < numThreads; ++t) {
	threads[t]->join();
	PyList_Append(pyListTgt, threads[t]->getTargetList());
	delete threads[t];
  }

  return pyListTgt;
}

PyObject* cvtGray32F(PyObject *self, PyObject *args) {
  PyListObject* pyListSrc;
  int tgtImgSize, numThreads;
  
  if (!PyArg_ParseTuple(args, "O!ii",
	  &PyList_Type, &pyListSrc, &tgtImgSize, &numThreads)) {
	return NULL;
  }

  std::vector<JPEGDecoderThreadExt*> threads(numThreads);
  int num_imgs = PyList_GET_SIZE(pyListSrc);
  int num_imgs_per_thread = DIVUP(num_imgs, numThreads);
  for (int t = 0; t < numThreads; ++t) {
	int start_img = t * num_imgs_per_thread;
	int end_img = min(num_imgs, (t + 1)*num_imgs_per_thread);
	threads[t] = new JPEGDecoderThreadExt((PyObject*)pyListSrc,
	    start_img, end_img, tgtImgSize);
	threads[t]->start();
  }

  PyObject* pyListTgt = PyList_New(0);
  for (int t = 0; t < numThreads; ++t) {
	threads[t]->join();
	PyList_Append(pyListTgt, threads[t]->getTargetList());
	delete threads[t];
  }

  return pyListTgt;
}

DecoderThreadExt::DecoderThreadExt(PyObject* py_list_src, int start_img, int end_img, int target_size, bool crop_to_square)
: Thread(true), _py_list_src(py_list_src), _start_img(start_img), _end_img(end_img), _target_size(target_size), _crop_to_square(crop_to_square) {

    _encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    _encode_params.push_back(JPEG_QUALITY);
    _py_list_tgt = PyList_New(0);
}

DecoderThreadExt::~DecoderThreadExt(){
    Py_DECREF(_py_list_tgt);
}

void* DecoderThreadExt::run() {
    for (int i = _start_img; i < _end_img; ++i) {
        makeJPEG(i);
    }
    return NULL;
}

PyObject* DecoderThreadExt::getTargetList() {
    return _py_list_tgt;
}


inline int round_d(double s)
{
	return int(s+0.5);
}

void DecoderThreadExt::makeJPEG(int idx) {
    /*
     * Decompress JPEG
     */
    PyObject* pySrc = PyList_GET_ITEM(_py_list_src, idx);
    uchar* src = (unsigned char*)PyString_AsString(pySrc);
    size_t src_len = PyString_GET_SIZE(pySrc);
    vector<uchar> src_vec(src, src + src_len);

    cv::Mat decoded_mat = cv::imdecode(cv::Mat(src_vec), CV_LOAD_IMAGE_COLOR);
    assert(decoded_mat.channels() == 3);

    /*
     * Resize
     */
    double min_dim = std::min(decoded_mat.size().height, decoded_mat.size().width);
    double scale_factor = _target_size / min_dim;
	
    //int new_height = round(scale_factor * decoded_mat.size().height);
    //int new_width = round(scale_factor * decoded_mat.size().width);
    int new_height = round_d(scale_factor * decoded_mat.size().height);
    int new_width = round_d(scale_factor * decoded_mat.size().width);
    assert((new_height == _target_size && new_width >= _target_size)
           || (new_width == _target_size && new_height >= _target_size));
    int interpolation = scale_factor == 1 ? cv::INTER_LINEAR
                      : scale_factor > 1 ? cv::INTER_CUBIC : cv::INTER_AREA;

    cv::resize(decoded_mat, _resized_mat_buffer, cv::Size(new_width, new_height), 0, 0, interpolation);

    /*
     * Conditionally crop and compress JPEG
     */
    if (_crop_to_square) {
        int crop_start_x = (new_width - _target_size) / 2;
        int crop_start_y = (new_height - _target_size) / 2;
        cv::Rect cropRect(crop_start_x, crop_start_y, _target_size, _target_size);
        cv::Mat cropped_mat_buffer = _resized_mat_buffer(cropRect);
        cv::imencode(".jpg", cropped_mat_buffer, _output_jpeg_buffer, _encode_params);
    } else {
        cv::imencode(".jpg", _resized_mat_buffer, _output_jpeg_buffer, _encode_params);
    }

    char* output_jpeg_buffer_ptr = reinterpret_cast<char*>(&_output_jpeg_buffer[0]);
    PyObject* pyStr = PyString_FromStringAndSize(output_jpeg_buffer_ptr, _output_jpeg_buffer.size());
    PyList_Append(_py_list_tgt, pyStr);
    Py_DECREF(pyStr);
}

/*
	PNGDecoderThreadExt
*/
PNGDecoderThreadExt::PNGDecoderThreadExt(PyObject* py_list_src, int start_img, int end_img, int target_size, bool crop_to_square)
: Thread(true), _py_list_src(py_list_src), _start_img(start_img), _end_img(end_img), _target_size(target_size), _crop_to_square(crop_to_square) {

  _encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  _encode_params.push_back(JPEG_QUALITY);
  _py_list_tgt = PyList_New(0);
}

PNGDecoderThreadExt::~PNGDecoderThreadExt(){
  Py_DECREF(_py_list_tgt);
}

void* PNGDecoderThreadExt::run() {
  for (int i = _start_img; i < _end_img; ++i) {
	makePNG(i);
  }
  return NULL;
}

PyObject* PNGDecoderThreadExt::getTargetList() {
  return _py_list_tgt;
}

void PNGDecoderThreadExt::makePNG(int idx) {
  /*
  * Decompress PNG
  */

}

/*
	JPEGDecoderThreadExt
*/
JPEGDecoderThreadExt::JPEGDecoderThreadExt(PyObject* py_list_src, int start_img, int end_img, int target_size)
: Thread(true), _py_list_src(py_list_src), _start_img(start_img), _end_img(end_img), _target_size(target_size) {

  _py_list_tgt = PyList_New(0);
}

JPEGDecoderThreadExt::~JPEGDecoderThreadExt() {
  Py_DECREF(_py_list_tgt);
}

void* JPEGDecoderThreadExt::run() {
  for (int i = _start_img; i < _end_img; ++i) {
	makeData(i);
  }
  return NULL;
}

PyObject* JPEGDecoderThreadExt::getTargetList() {
  return _py_list_tgt;
}

void JPEGDecoderThreadExt::makeData(int idx) {
  /*
  to gray and 32F
  */
  PyObject* pySrc = PyList_GET_ITEM(_py_list_src, idx);
  uchar* src = (unsigned char*)PyString_AsString(pySrc);
  size_t src_len = PyString_GET_SIZE(pySrc);
  vector<uchar> src_vec(src, src + src_len);

  cv::Mat decoded_mat = cv::imdecode(cv::Mat(src_vec), CV_LOAD_IMAGE_COLOR);
  assert(decoded_mat.channels() == 3);
  int height = decoded_mat.rows;
  int width = decoded_mat.cols;
  int size = height * width;

  // to 32F
  cv::Mat img_32f;
  decoded_mat.convertTo(img_32f, CV_32F, 1.0 / 255);
  // to grayscale
  cv::Mat img_gray_32f;
  cv::cvtColor(img_32f, img_gray_32f, CV_BGR2GRAY);
  if (_target_size > 0) {} // TODO crop image to _target_size

  //PyObject* pyInt1 = PyInt_FromSsize_t(height);
  //PyObject* pyInt2 = PyInt_FromSsize_t(width);

  PyObject* pyStr;
  if (!img_gray_32f.isContinuous()) {
	img_gray_32f = img_gray_32f.clone();
  }
  //pyStr = PyString_FromStringAndSize(reinterpret_cast<char*>(img_gray_32f.data), size * sizeof(float));
  //PyObject* pyTup = PyTuple_Pack(3, pyInt1, pyInt2, pyStr);
  //PyObject* pyTup = PyTuple_New(3);
  //PyTuple_SetItem(pyTup, 0, pyInt1);
  //PyTuple_SetItem(pyTup, 1, pyInt2);
  //PyTuple_SetItem(pyTup, 2, pyStr);
  //PyList_Append(_py_list_tgt, pyTup);

  PyObject* pyTup = Py_BuildValue("(iis#)",
	  height, width,
	  reinterpret_cast<char*>(img_gray_32f.data), size * sizeof(float));
  PyList_Append(_py_list_tgt, pyTup);

  //Py_DECREF(pyInt1);
  //Py_DECREF(pyInt2);
  //Py_DECREF(pyStr);
  Py_DECREF(pyTup);
}