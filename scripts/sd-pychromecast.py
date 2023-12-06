from math import ceil, sqrt
import pychromecast
import time
from datetime import datetime
from modules import script_callbacks, shared, scripts, ui_components
from modules.script_callbacks import ImageSaveParams, AfterCFGCallbackParams, ImageGridLoopParams
import logging
from modules.processing import StableDiffusionProcessingTxt2Img
import pathlib
import tempfile
import socket
from modules.shared import opts, cmd_opts, state
from modules.options import Options
from fastapi import FastAPI
from gradio import Blocks
from typing import Optional, Tuple, Union, List
from logging.handlers import RotatingFileHandler
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue, Empty, Full
import modules.scripts as scripts
from enum import Enum
from dataclasses import dataclass
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
# class PyChromeCastScript(scripts.Scripts):
#     pass

class ImageType(Enum):
    FILE= "File"
    TENSOR= "torch Tensor"
    PIL= "Pillow Image"
    
class CastType(Enum):
    CHROMECAST= "ChromeCast"
    AIRPLAY= "AirPlay"

@dataclass
class CastConfiguration:
    cast_type: CastType
    device_name: str
    base_callback_url: str
    temp_dir: pathlib.Path

@dataclass
class ImageInfo:
    image_type: ImageType
    obj: any
    creation_date: datetime
    message: Optional[str]


class CastingImageInfo():
    image_info: ImageInfo
    image_path: pathlib.Path
    mime_type: str
    config: CastConfiguration
    url: str
    _ready: bool
    _torch_to_pil: transforms.ToPILImage= transforms.ToPILImage()
    _path: pathlib.Path

    def _get_grid_size(number_of_items: int) -> Tuple[int,int]:
        columns = int(sqrt(number_of_items))
        lines = int(ceil(number_of_items / columns))
        return (columns, lines)
        
    def __init__(self, image_info: ImageInfo, config: CastConfiguration):
        self.image_info= image_info
        self.config= config
        self._ready= False
        if self.image_info.image_type == ImageType.TENSOR:
            obj: Tensor= self.image_info.obj
            dimensions= len(obj.size())
            logger.info("Dimensions %d", dimensions)
            if dimensions == 4:
                logger.warning("%d dimensions, len %s", dimensions, len(obj))
                obj= obj[0]
            im: Image.Image= CastingImageInfo._torch_to_pil(obj)
            self._file= tempfile.NamedTemporaryFile(suffix='.png', prefix='torch_to_pil', dir=self.config.temp_dir, delete=False)
            self._path= pathlib.Path(self._file.name)
            im.save(self._path)
            self.url= "{}{}".format(self.config.base_callback_url, self._file.name)
            self.mime_type= "image/png"
            self._ready= True
        elif self.image_info.image_type == ImageType.FILE:
            im: Image.Image= Image.Image()
            im.load(self.image_info.obj)
            self._file= tempfile.NamedTemporaryFile(suffix='.png', prefix='torch_to_pil', dir=self.config.temp_dir, delete=False)
            self._path= pathlib.Path(self._file.name)
            im.save(self._path)
            self.url= "{}{}".format(self.config.base_callback_url, self._file.name)
            self.mime_type= "image/png"
            self._ready= True
            
            
    def _tensor_to_pil(self, obj: Tensor) -> Image.Image:
        dimensions= len(obj.size())
        batch_size= 1
        if dimensions == 4:
            batch_size= len(obj)
        logger.info("Convert Tensor to PIL. Tensor has %d dimensions. Batch size %d", dimensions, batch_size)
        if dimensions < 4:
            im: Image.Image= CastingImageInfo._torch_to_pil(obj)
            return im
        rows, cols= CastingImageInfo._get_grid_size(batch_size)       
        images: List[Image.Image]= []
        max_w= 0
        max_h= 0
        for i in range(batch_size):
            im: Image.Image= CastingImageInfo._torch_to_pil(obj[i])
            i+= i
            images.append(im)
            w,h= im.size
            max_w= max(max_w, w)
            max_h= max(max_h, h)
        grid_w= max_w * cols
        grid_h= max_h * rows
        grid = Image.new('RGB', size=(grid_w, grid_h))
        im: Image.Image
        for i, im in enumerate(images):
            grid.paste(im, box=(i%cols*max_w, i//cols*max_h))
        return grid
    
    def is_ready(self) -> bool:
        return self._ready and self._path.is_file()
    
    

class CastThread(Thread):
    _chromecast: pychromecast.Chromecast= None
    
    def __init__(self, cast_device: str, stop_event: Event, image_queue: Queue, config: CastConfiguration):
        Thread.__init__(self)
        self.cast_device= cast_device
        self.stop_event= stop_event
        self.queue= image_queue
        self.config= config
    
    def run(self):
        logger.info("Start Cast Thread")
        if self.config.cast_type != CastType.CHROMECAST:
            logger.critical("Cast type %s not supported.", self.config.cast_type.value)
            return
        while not self.stop_event.is_set():
            try:
                # Using timeout of 5 seconds to allow teardown within 5 seconds if
                # we request a stop but the queue is empty and no other objects come in
                img_info= self.queue.get(timeout=5)
                logger.info("Processing image")
                
                if self._chromecast is None:
                    if self.config.cast_type == CastType.CHROMECAST:
                        chromecasts, browser= pychromecast.get_listed_chromecasts(friendly_names=[self.config.device_name]) 
                        obj: pychromecast.Chromecast
                        for obj in chromecasts:
                            if obj.name == self.config.device_name:
                                logger.debug("Found device")
                                self._chromecast= obj
                                break
                        else:
                            logger.warning("ChromeCast %s is unreachable, dropping image")
                            continue
                self._chromecast.wait(timeout=5)
                casting_image= CastingImageInfo(image_info=img_info, config=self.config)
                if casting_image.is_ready():
                    logger.info("Casting %s", casting_image.url)
                    self._chromecast.play_media(url=casting_image.url, content_type=casting_image.mime_type)
                    
                
                
# my_cast: pychromecast.Chromecast= chromecasts[0] if len(chromecasts) > 0 else None                    
            except Empty as e:
                pass
            except Exception as e:
                logger.exception("Error processing image")
        logger.warning("Cast thread ended, stop event received")

    
image_queue: Optional[Queue[ImageInfo]]= None
cast_thread: Optional[CastThread]= None
cast_thread_stop_event: Event= Event()

base_dir: pathlib.Path= pathlib.Path(scripts.basedir())
log_dir = base_dir.joinpath("log")
if not log_dir.exists():
    print ("Creating log directory %s" % log_dir.resolve())
    log_dir.mkdir(parent=True)

log_file= log_dir.joinpath("sd-pychromecast.log")
#print("Log file for sd-pychromecast: %s" % log_file.resolve())
fh = RotatingFileHandler(log_file, mode='a', maxBytes=512*1024, backupCount=5)
#fh.doRollover()

fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger= logging.getLogger(__name__)
if cmd_opts.loglevel is not None:
    logger.setLevel(cmd_opts.loglevel)
else:
    logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

logger.info("INIT")


# services, browser = pychromecast.discovery.discover_chromecasts()
# pychromecast.discovery.stop_discovery(browser)

# device: pychromecast.models.CastInfo
# device_name= "Chrome Cast EG"
# for device_uuid, device in browser.devices.items():
#     print ("Searching chromecast, found : {}".format(device.friendly_name))

# #pychromecast.get_listed_chromecasts()
# chromecasts, browser = pychromecast.get_listed_chromecasts(friendly_names=[device_name]) 
# my_cast: pychromecast.Chromecast= chromecasts[0] if len(chromecasts) > 0 else None

# # if my_cast is not None:
# #     my_cast.wait()
# #     #print("{!s} Status       {!s}".format(datetime.now(), cast.status))
# #     #print("{!s} Status event {!r}".format(datetime.now(), cast.status_event))
# #     for img in imgs:
# #         print("Push {}".format(img))
# #         my_cast.play_media(url=img, content_type="image/png")
# #         for i in range(7):
# #             #print("{!s} Status       {!s}".format(datetime.now(), cast.status))
# #             #print("{!s} Status event {!s}".format(datetime.now(), cast.status_event))
# #             time.sleep(1)
# #     my_cast.disconnect()        
# my_cast.wait()
# url= "{}{}".format(base_url, params.filename)
# my_cast.play_media(url=url, content_type="image/png")

for k, v in opts.data.items():
    logger.debug("SHARED OPTS DATA: %30s = %s", k, v)

for k, v in cmd_opts.__dict__.items():
    logger.debug("CMD_OPTS: %30s = %s", k, v)

temp_dir= base_dir.joinpath("tmp")
if not temp_dir.is_dir():
    logger.warning("mkdir %s", temp_dir.resolve())
    temp_dir.mkdir(parents=True)
logger.info("Add temp dir to gradio allowed path")
cmd_opts.gradio_allowed_path.append(temp_dir.resolve())
 
default_listen_ip= None
try:     
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    default_listen_ip= s.getsockname()[0]
    s.close()
except Exception as e:
    logger.exception("Error on finding local IP")

logger.debug("Default listen IP : %s", default_listen_ip)

## TEST VALUES TO HELP THE PROCESS
device_name= "Chrome Cast EG"
listen_port= cmd_opts.port
if listen_port is None:
    listen_port= 7860
base_url= "http://{}:{}/file=".format(default_listen_ip, listen_port)
##



def on_save_action(params: ImageSaveParams):
    logger.info("PUSH IMG")
    logger.info("Filename: %s", params.filename)
    logger.info("Pnginfo : %r", params.pnginfo)
    logger.info("p (%s) : %r", type(params.p), params.p)
    info= ImageInfo(ImageType.FILE, obj= params.filename, creation_date=datetime.now(), message= None)

    if image_queue.full():
        try:
            image_queue.get_nowait()
        except Empty:
            pass
    try:
        image_queue.put_nowait(info)
    except Full:
        logger.warning("Error queueing image, queue full")
        
def cfg_action(params: AfterCFGCallbackParams):
    logger.info("CFG_ACTION")
    loop= "{}/{}".format(params.sampling_step, params.total_sampling_steps)
    text= "cfg: {}".format(loop)
    if params.sampling_step == params.total_sampling_steps:
        text= "cfg: Completed {}".format(params.total_sampling_steps)

    info= ImageInfo(ImageType.TENSOR, obj= params.x, creation_date=datetime.now(), message=text)
    if image_queue.full():
        try:
            image_queue.get_nowait()
        except Empty:
            pass
    try:
        image_queue.put_nowait(info)
    except Full:
        logger.warning("Error queueing image, queue full")

def teardown():
    global cast_thread, cast_thread_stop_event
    logger.warning("TEARDOWN")
    if cast_thread is not None:
        logger.info("Request stop of cast thread")
        cast_thread_stop_event.set()
        logger.info("Wait for cast thread to teard down, should be max 5 seconds")
        cast_thread.join()
        cast_thread= None
        

def init(demo: Optional[Blocks], app: FastAPI):
    global config, image_queue, cast_thread, cast_thread_stop_event
    logger.warning("INIT")
    config= CastConfiguration(cast_type=CastType.CHROMECAST, device_name=device_name, base_callback_url=base_url, temp_dir=temp_dir)
    if image_queue is None:
        image_queue= Queue(maxsize=10)
    if cast_thread is None:
        #Just to be sure
        cast_thread_stop_event.clear()
        cast_thread= CastThread(cast_device=device_name, stop_event=cast_thread_stop_event, image_queue=image_queue, config=config)
        cast_thread.start()
    
#image_saved_callback
#script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_image_saved(on_save_action)
script_callbacks.on_cfg_after_cfg(cfg_action)
script_callbacks.on_script_unloaded(teardown)
script_callbacks.on_app_started(init)
