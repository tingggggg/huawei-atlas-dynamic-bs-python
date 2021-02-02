"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-17 14:04:45
"""
import argparse
import numpy as np
import struct
import acl
import os
from PIL import Image
from timeit import default_timer as timer

import utils
from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_ERROR_NONE, IMG_EXT

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }

dynamicFlag = "ascend_mbatch_shape_data"

def check_ret(message, ret):
    """Check function's return err code
    Args:
        message(str): function name
        ret(int): error code
    """
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Net(object):
    """To Load Model From .om model(Huawei atlas model format)
    
    Attributes:
        device_id(int): Define which npu card
        model_path(str): Model path
        model_id(pointer): Got model id after load model
        context(pointer): Include 2 stream after init_resource()

        input_data(list): Temporarily store buffer and buffer size
        output_data(list): Temporarily store buffer and buffer size
        model_desc(pointer): aclmdlDesc address after init_resource()
        load_input_dataset(pointer): aclmdlDataset address after _gen_dataset()
        load_output_dataset(pointer): aclmdlDataset address after _gen_dataset()

        img_buffer_size(int): According to arg input_size to calculate C * W * H // 2. 
        Divided by 2 because atlas model do image preprocess with convert image to (YUV420SP_U8) format that size just only half.
        batch_size(int): Batch size, e.g. 1 or 16
        number_iterations(int): Total number of times for each execute  
    """
    def __init__(self, device_id, model_path, input_size=(224, 224, 3)):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.img_buffer_size = (input_size[0] * input_size[1] * input_size[2]) // 2
        self.batch_size = 1
        self.number_iterations = 3

        self.init_resource()

    def __del__(self):
        """ For release all buffer and address be applied that from acl function
        """
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)

    def init_resource(self):
        """Load model and init it
        """
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()

    def _get_model_info(self,):
        """Generate buffer of input and output 
        """
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        """Generate buffer then according to des("in" or "out") add to self.input_data or self.output_data
        Args:
            size(int): An number of input or output
            des(str): For choose to add in which list(input or output)
        """
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        """Copy data from host to device or from device to host  that according "policy"
        Copy data from host to device:
            dynamic batch size model need to apply 2 input buffer 

            for i in range(int input_num: 2):
                input_name = acl.mdl.get_input_name_by_index(model_desc, i)
                print(input_name)         
            out->
                X
                ascend_mbatch_shape_data

            just copy pointer of img to (input_name=="X") buffer
            Only 1 input buffer (input_name: "X") if your model is static batch size


            image need to be copied to device one by one, 
            according to (idx * self.img_buffer_size) to move your buffer pointer to stack your images data

            for idx, image in enumerate(images):
                ptr = acl.util.numpy_to_ptr(image)
                ret = acl.rt.memcpy(item["buffer"] + (idx * self.img_buffer_size),
                                    item["size"],
                                    ptr,
                                    self.img_buffer_size,
                                    policy)
            
            ret = acl.rt.memcpy(dst, dest_max, src, count, kind)
            dst(int)： des pointer address
            dest_max(int)： maximum size of des (Byte)
            src(int)： host pointer address
            count(int)： lenght of host data (Byte)

        Copy data from device to host:
            only 1 output if model is classification
            need to malloc host memory first for store output result


        Args:
            dataset(list(array)): image list
            policy(int): from host or from device
        
        Returns:
            None
        """
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                input_name = acl.mdl.get_input_name_by_index(self.model_desc, i)
                if input_name != dynamicFlag:
                    for idx, img in enumerate(dataset):
                        ptr = acl.util.numpy_to_ptr(img)
                        ret = acl.rt.memcpy(item["buffer"] + (idx * self.img_buffer_size),
                                            item["size"],
                                            ptr,
                                            self.img_buffer_size,
                                            policy)
                        check_ret("acl.rt.memcpy", ret)
            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        """According buffer to create dataset(acl.mdl.create_dataset()) object for execute 

        Args:
            type_str(str): For choose to create  which dataset(input or output)
        """
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        # copy images to device
        self.data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")

    def _data_from_device_to_host(self):
        """Copy model output from device to host
        You have not do call this function if you don't need print out predict results
        """
        res = []
        # copy device to host
        self.data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        result = self.get_result(res)
        self._print_result(result)

    def run(self, images, batch_size=1, number_iter=3):
        """ Copy host data to devcie then do inference, finally copy device data to host
        Args:
            images(numpy.array): A flatten image array e.g. images.shape(150528, )
            number_iter(int): Total number of times for each execute
        Returns:
            res_time(float): from forward() (ms)
        """
        self.batch_size = batch_size
        self._data_from_host_to_device(images)
        res_time = self.forward(number_iter)
        # self._data_from_device_to_host()
        return res_time

    def forward(self, number_iter=3):
        """Set batch szie first by acl.mdl.set_dynamic_batch_size()
        then do infer by acl.mdl.execute()

        Args:
            number_iter(int): Total number of times for each execute
        
        Returns: 
            res_time(float): Average times of execute (ms)
        """
        i, ret = acl.mdl.get_input_index_by_name(self.model_desc, dynamicFlag)
        check_ret("acl.mdl.get_input_index_by_name", ret)
        ret = acl.mdl.set_dynamic_batch_size(self.model_id, self.load_input_dataset, i, self.batch_size)
        check_ret("acl.mdl.set_dynamic_batch_size", ret)

        total_list = []
        while(number_iter):
            t0 = timer()
            ret = acl.mdl.execute(self.model_id,
                                self.load_input_dataset,
                                self.load_output_dataset)
            check_ret("acl.mdl.execute", ret)
            cost = timer() - t0
            total_list.append(cost)
            number_iter -= 1
        self._destroy_databuffer()
        res_time = (sum(total_list) * 1000) / len(total_list)
        return res_time

    def _print_result(self, result):
        MODEL_MAXIMUM_BS = 128
        """Print out predict result:
        e.g. 
        ======== top5 inference results: =============
        Idx -> 0
        [640]: 0.656250
        [678]: 0.046082
        [825]: 0.035309
        [199]: 0.022812
        [493]: 0.022812

        Args:
            result(numpy array): array format fo results
        
        Returns:
            None
        """
        tuple_st = struct.unpack("{:}f".format(str(result[0].shape[0] // 4)), bytearray(result[0]))
        # print(tuple_st)
        vals = np.array(tuple_st).flatten()
        vals = vals.reshape(MODEL_MAXIMUM_BS, -1)
        for b in range(self.batch_size):
            top_k = vals[b].argsort()[-1:-6:-1]
            print("======== top5 inference results: =============")
            print(f"Idx -> {b}")
            for j in top_k:
                print("[%d]: %f" % (j, vals[b][j]))

    def _destroy_databuffer(self):
        """Release data buffer by mormal way
        """
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def get_result(self, output_data):
        """Convert the already copied to host buffer result to numpy format
        Args:
            output_data(host buffer): results(in host buffer)

        Returns:
            numpy array results
        """
        dataset = []
        for temp in output_data:
            size = temp["size"]
            ptr = temp["buffer"]
            data = acl.util.ptr_to_numpy(ptr, (size,), 1)
            dataset.append(data)
        return dataset


# def transfer_pic(input_path):
#     """Read real image from path but have not be used in atomic benchmark framwork
#     """
#     input_path = os.path.abspath(input_path)
#     image_file = Image.open(input_path)
#     image_file = image_file.resize((256, 256))
#     img = np.array(image_file)
#     height = img.shape[0]
#     width = img.shape[1]
#     h_off = (height - 224) // 2
#     w_off = (width - 224) // 2
#     crop_img = img[h_off:height - h_off, w_off:width - w_off, :]
#     # rgb to bgr
#     img = crop_img[:, :, ::-1]
#     shape = img.shape
#     img = img.astype("float16")
#     img[:, :, 0] -= 104
#     img[:, :, 1] -= 117
#     img[:, :, 2] -= 123
#     img = img.reshape([1] + list(shape))
#     img = img.transpose([0, 3, 1, 2])
#     result = np.frombuffer(img.tobytes(), np.float16)
#     return result

def random_pic(size=(3,224,224)):
    """Generate a random array for do inference
    Args:
        size(tuple): Define the random array size
    Returns:
        A array be flatten and convert by numpy.frombuffer for copy to npu device buffer
        e.g. img.shape = (150528, )
    """
    img = np.random.random(size).astype("float16")
    shape = img.shape
    img = img.reshape([1] + list(shape))
    img = np.frombuffer(img.tobytes(), np.float16)
    return img

def build_argparser():
    parser = argparse.ArgumentParser(description='pyACL infer tool.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_path', "-m", type=str, default="./model/resnet50.om")
    parser.add_argument('--images_path', type=str, default="./data")
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=3, type=int)
    parser.add_argument("-nb", "--min_batch", help="Number of min batch size", default=1, type=int)
    parser.add_argument("-mb", "--max_batch", help="Number of max batch size", default=128, type=int)
    parser.add_argument("-s", "--size", help="Array of image size(format: WxH)", default="224x224", type=str)
    parser.add_argument("-p", "--precision", help="", default="FP32", type=str)
    return parser

def do(model_file, number_iter, batchs, shapes, device=0, **kwargs):
    """Do inference with shape and batch
    Args:
        model_file(str): model path
        number_iter(int): Total number of times for each execute
        batchs(list): list out each batch size which need to do inference e.g. [1, 2, 4, 8, ...]
        shapes(list(tuple)): list out each CxHxW which need to do inference e.g. [(3, 224, 224), ...]
        device(int): npu card id
    Returns:
        A list include average inference times of each batch & shape 
        e.g. [((1, 3, 224, 224), 4.437762840340535), ((2, 3, 224, 224), 3.2970468358447156), ...]
    """
    precision = None if 'precision' not in kwargs else kwargs['precision']
    times = []
    shapes = ([shapes] if isinstance(shapes, int) else shapes)
    net = Net(device, model_file)
    
    for shape in shapes:
          infer_img = random_pic() # this img be flatten, like (3,224,224,) - > (150528, )
          for batch in batchs:
              infer_imgs = [infer_img] * batch
              res_time = net.run(infer_imgs, len(infer_imgs), number_iter=number_iter)
              print("Average running time of one iteration: {} ms".format(res_time))
              print("Average running time of one input: {} ms".format(res_time / batch))
              if not isinstance(shape, int):
                times.append(((batch, shape[0], shape[1], shape[2]), res_time / batch))
              else:
                times.append(((batch, shape), res_time / batch))
    return times


if __name__ == '__main__':
    args = build_argparser().parse_args()
    
    batchs = utils.calc_batchs(args.min_batch, args.max_batch)
    shapes = (utils.split_shapes(args.size) if "x" in args.size else int(args.size))

    do(args.model_path, args.number_iter, batchs, shapes, device=args.device)

    # below from origin sample code
    # net = Net(args.device, args.model_path)
    # images_list = [os.path.join(args.images_path, img)
    #                for img in os.listdir(args.images_path)
    #                if os.path.splitext(img)[1] in IMG_EXT]

    # data_list = []
    # for image in images_list:
    #     dst_im = transfer_pic(image)
    #     data_list.append(dst_im)

    # net.run(data_list, len(data_list))

    # print("*****run finish******")
