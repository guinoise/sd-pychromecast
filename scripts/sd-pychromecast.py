from math import ceil, sqrt
import gradio as gr
import pychromecast
import time
from datetime import datetime
from modules import script_callbacks, shared, scripts, ui_components
from modules.script_callbacks import ImageSaveParams, AfterCFGCallbackParams, ImageGridLoopParams
import logging
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessing, Processed
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
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL.Image import Image as ImagePIL
from PIL.ImageDraw import ImageDraw as ImageDrawPIL
from torch import Tensor
# class PyChromeCastScript(scripts.Scripts):
#     pass

class ImageType(Enum):
    FILE= "File"
    TENSOR= "torch Tensor"
    PIL= "Pillow Image"
    STABLE_DIFFUSION_PROCESSED= "Stable diffusion processed"
    STABLE_DIFFUSION_PROCESSING= "Stable diffusion processing"
    
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

    def add_text_to_im(self, im: ImagePIL, text: Optional[str], date: Optional[datetime]= None) -> ImagePIL:
        if text is None and date is None:
            return
        elif text is None:
            text=date.strftime('%Y-%m-%d %H:%M:%S')
        elif date is not None:
            text="{} : {}".format(date.strftime('%Y-%m-%d %H:%M:%S'), text)
        w,h= im.size
        zone= int(h*0.05)
        logger.debug("Add text. w %4d h %4d zone %4d : %s", w, h, zone, text)
        im2= ImageOps.expand(im, border=(0,zone,0,0))
        draw: ImageDrawPIL= ImageDraw.Draw(im2)
        #font= ImageFont.truetype('FreeMono.ttf', 24)
        draw.text((5,5), text, fill=(0,0,0))
        return im2
    
    def __init__(self, image_info: ImageInfo, config: CastConfiguration):
        self.image_info= image_info
        self.config= config
        self._ready= False
        try:
            im: ImagePIL= None
            if self.image_info.image_type == ImageType.TENSOR:
                prefix='torch_to_pil_'
                obj: Tensor= self.image_info.obj
                im= self._tensor_to_pil(obj)
            elif self.image_info.image_type == ImageType.FILE:
                prefix='from_file_'
                im: ImagePIL= Image.open(self.image_info.obj)
            elif self.image_info.image_type == ImageType.PIL:
                prefix='from_pil_'
                im: ImagePIL= self.image_info.obj
            elif self.image_info.image_type == ImageType.STABLE_DIFFUSION_PROCESSED:
                prefix='sd_processed_'
                obj: Processed= self.image_info.obj
                im: ImagePIL= self._join_images(obj.images)
            elif self.image_info.image_type == ImageType.STABLE_DIFFUSION_PROCESSING:
                prefix='sd_processing_'
                obj: StableDiffusionProcessing= self.image_info.obj
                im: ImagePIL= self._join_images()

            im= self.add_text_to_im(im, image_info.message, image_info.creation_date)
            self._file= tempfile.NamedTemporaryFile(suffix='.png', prefix=prefix, dir=self.config.temp_dir, delete=False)
            self._path= pathlib.Path(self._file.name)
            im.save(self._path)
            self.url= "{}{}".format(self.config.base_callback_url, self._file.name.replace('\\', '/'))
            self.mime_type= "image/png"
            self._ready= True

        except Exception as e:
            logger.exception("Error processing image : %s", str(e))
            
    def _join_images(self, images: List[ImagePIL]):
        count= len(images)

        if count == 1:
            return images[0]
        
        rows, cols= CastingImageInfo._get_grid_size(len(images))       
        max_w= 0
        max_h= 0
        for i in range(count):
            w,h= images[i].size
            max_w= max(max_w, w)
            max_h= max(max_h, h)
            
        grid_w= max_w * cols
        grid_h= max_h * rows
        grid = Image.new('RGB', size=(grid_w, grid_h))

        im: ImagePIL
        for i, im in enumerate(images):
            grid.paste(im, box=(i%cols*max_w, i//cols*max_h))
        return grid
        
    def _tensor_to_pil(self, obj: Tensor) -> ImagePIL:
        dimensions= len(obj.size())
        batch_size= 1
        if dimensions == 4:
            batch_size= len(obj)
        logger.info("Convert Tensor to PIL. Tensor has %d dimensions. Batch size %d", dimensions, batch_size)
        if dimensions < 4:
            im: ImagePIL= CastingImageInfo._torch_to_pil(obj)
            return im
        
        rows, cols= CastingImageInfo._get_grid_size(batch_size)       
        images: List[ImagePIL]= []
        for i in range(batch_size):
            im: ImagePIL= CastingImageInfo._torch_to_pil(obj[i])
            images.append(im)
            
        return self._join_images(images=images)
    
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
        last_sent= None
        min_wait= 10
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
                    if last_sent is not None:
                        while (datetime.now() - last_sent).total_seconds() < min_wait:
                            logging.debug("Wait, not %d seconds since last play", min_wait)
                            time.sleep(1)
                    logger.info("Casting %s", casting_image.url)
                    self._chromecast.play_media(url=casting_image.url, content_type=casting_image.mime_type)
                    last_sent= datetime.now()
                
                
# my_cast: pychromecast.Chromecast= chromecasts[0] if len(chromecasts) > 0 else None                    
            except Empty as e:
                pass
            except Exception as e:
                logger.exception("Error processing image")
        logger.warning("Cast thread ended, stop event received")

class PyChromeCastScript(scripts.Script):
    # Extension title in menu UI
    def title(self):
            return "Casting"
    def title(self):
            return "Casting"
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        check_box_cast = gr.inputs.Checkbox(label="Cast", default=True)        
        return [check_box_cast] 
               
    def postprocess_image(self, p: StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """
        logger.warning("postprocess_image")
        logger.info("p: (%s) %r", type(p), p)
        logger.info("pp: (%s) %r", type(pp), pp)
        logger.info("args: %r", args)
        if issubclass(type(pp), Processed):
            logger.info("postprocess_image: Processed")
            info= ImageInfo(ImageType.STABLE_DIFFUSION_PROCESSED, obj= pp, creation_date=datetime.now(), message= None)
            self._enqueue(info)
        if issubclass(type(pp), scripts.PostprocessImageArgs):
            logger.info("postprocess_image: PostprocessImageArgs")
            logger.info("pp.image : %s", type(pp.image))
            info= ImageInfo(ImageType.PIL, obj= pp.image, creation_date=datetime.now(), message= "PostprocessImageArgs {}/{}".format(p.iteration, p.batch_size))
            self._enqueue(info)
        elif issubclass(type(p), StableDiffusionProcessing):
            logger.info("postprocess_image: StableDiffusionProcessing")
            info= ImageInfo(ImageType.STABLE_DIFFUSION_PROCESSING, obj= pp, creation_date=datetime.now(), message= None)            
            self._enqueue(info)
        else:
            logger.warning("PP: Unsuported type %s", type(pp))
            

    def _enqueue(self, img_info: ImageInfo):
        if image_queue is None:
            logger.error("Queue object not found, discarding object")
            return
        if image_queue.full():
            logger.warning("Queue was full, dropping one object")
            try:
                image_queue.get_nowait()
            except Empty:
                pass
        try:
            image_queue.put_nowait(img_info)
        except Full:
            logger.warning("Error queueing image, queue full")            
        
    
image_queue: Optional[Queue[ImageInfo]]= None
cast_thread: Optional[CastThread]= None
cast_thread_stop_event: Event= Event()

base_dir: pathlib.Path= pathlib.Path(scripts.basedir())
log_dir = base_dir.joinpath("log")
if not log_dir.exists():
    print ("Creating log directory %s" % log_dir.resolve())
    log_dir.mkdir(parents=True)

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
    pass
    # logger.info("PUSH IMG")
    # logger.info("Filename: %s", params.filename)
    # logger.info("Pnginfo : %r", params.pnginfo)
    # logger.info("p (%s) : %r", type(params.p), params.p)
    # info= ImageInfo(ImageType.FILE, obj= params.filename, creation_date=datetime.now(), message= None)

    # if image_queue.full():
    #     try:
    #         image_queue.get_nowait()
    #     except Empty:
    #         pass
    # try:
    #     image_queue.put_nowait(info)
    # except Full:
    #     logger.warning("Error queueing image, queue full")
        
def cfg_action(params: AfterCFGCallbackParams):
    pass
    #logger.info("CFG_ACTION")
    # loop= "{}/{}".format(params.sampling_step, params.total_sampling_steps)
    # text= "cfg: {}".format(loop)
    # if params.sampling_step == params.total_sampling_steps:
    #     text= "cfg: Completed {}".format(params.total_sampling_steps)

    # info= ImageInfo(ImageType.TENSOR, obj= params.x, creation_date=datetime.now(), message=text)
    # if image_queue.full():
    #     try:
    #         image_queue.get_nowait()
    #     except Empty:
    #         pass
    # try:
    #     image_queue.put_nowait(info)
    # except Full:
    #     logger.warning("Error queueing image, queue full")

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
