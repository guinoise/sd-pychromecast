import inspect
from matplotlib import font_manager
from math import ceil, sqrt
import gradio as gr
import pychromecast
import time
from datetime import datetime
from modules import script_callbacks, shared, scripts, ui_components
from modules.shared_state import State
from modules.script_callbacks import ImageSaveParams, AfterCFGCallbackParams, ImageGridLoopParams
import logging
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessing, Processed
import pathlib
import tempfile
import socket
from modules.options import Options, OptionInfo
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
from textwrap import TextWrapper
import itertools
from PIL.ImageFont import FreeTypeFont
import gradio.blocks
import gradio.layouts
import gradio.components.dropdown
import gradio.components.checkbox
import gradio.components.button 
## Directory variables and early variables
# Some are required for logger, setting here
# Set other variables in block Variables
base_dir: pathlib.Path= pathlib.Path(scripts.basedir())
data_path: pathlib.Path= pathlib.Path(shared.data_path)
log_dir: pathlib.Path= base_dir.joinpath("log")
temp_dir: pathlib.Path= data_path.joinpath("tmp")

## INIT Logger
if not log_dir.exists():
    print ("Creating log directory %s" % log_dir.resolve())
    log_dir.mkdir(parents=True)

log_file= log_dir.joinpath("sd-pychromecast.log")
fh = RotatingFileHandler(log_file, mode='a', maxBytes=100*1024*1024, backupCount=5)

fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger= logging.getLogger(__name__)
logger.addHandler(fh)

if shared.cmd_opts.loglevel:
    if shared.cmd_opts.loglevel.upper() == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif shared.cmd_opts.loglevel.upper() == "INFO":
        logger.setLevel(logging.INFO)
    elif shared.cmd_opts.loglevel.upper() in ["WARN", "WARNING"]:
        logger.setLevel(logging.WARNING)
    elif shared.cmd_opts.loglevel.upper() == "ERROR":
        logger.setLevel(logging.ERROR)
    elif shared.cmd_opts.loglevel.upper() == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)    
        logger.critical("ERROR SETTING Debug level, value %r could not be parsed, defaulting to INFO", shared.cmd_opts.loglevel)
else:
    logger.setLevel(logging.DEBUG)

logger.setLevel(logging.DEBUG)
# End logger

## Datatypes
# Define datatypes (dataclass) and Enum

class TvAspects(Enum):
    ASPECT_4_3= (4,3)
    ASPECT_16_9= (16,9)

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
    message: Union[str, List[str], None]

# End Datatypes

def aspect_resize(w: int, h: int, aspect: TvAspects) -> Tuple[int,int]:
    target_ratio= aspect.value[0] / aspect.value[1]
    current_ratio= w / h
    if current_ratio < target_ratio:
        new_w = w * (target_ratio/current_ratio)
        new_h = h
    else:
        new_w = w 
        new_h = h * (target_ratio/current_ratio)
    logger.debug("Aspect resize : ratio %d/%d %7.5f w: %5d %5d h: %5d %5d final ratio %7.5f", aspect.value[0], aspect.value[1],target_ratio, w, h, new_w, new_h, new_w/new_h)
    return (new_w, new_h)

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
        self.aspect= TvAspects.ASPECT_16_9
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
            if image_info.message is None:
                text= []
            elif issubclass(type(image_info.message), str):
                text= [image_info.message]
            else:
                text= image_info.message
            
            text_padding= "_" * 22
            if im.info is not None:
                v: str
                for k,v in im.info.items():
                    lines= v.splitlines()
                    text.append(f"{k:20s} : {lines[0]}")
                    for i in range(1,len(lines)):
                        text.append(f"{text_padding} {lines[i]}")

            im= self.add_text_to_im(im, image_info.message, image_info.creation_date)
            self._file= tempfile.NamedTemporaryFile(suffix='.png', prefix=prefix, dir=self.config.temp_dir, delete=False)
            self._path= pathlib.Path(self._file.name)
            im.save(self._path)
            self.url= "{}{}".format(self.config.base_callback_url, self._file.name.replace('\\', '/'))
            self.mime_type= "image/png"
            self._ready= True

        except Exception as e:
            logger.exception("Error processing image : %s", str(e))

    def add_text_to_im(self, im: ImagePIL, text: Union[str, List[str], None], date: datetime) -> ImagePIL:
        td= datetime.now().replace(microsecond=0) - date.replace(microsecond=0)
        date_text= "{} ({})".format(date.strftime('%Y-%m-%d %H:%M:%S'), td)
        if text is None:
            text= date_text
        if issubclass(type(text), str):
            text= [date_text, text]
        else:
            text.insert(0, date_text)

        w,h= im.size
        logger.debug("SIZE %20s w %5d h %5d", "orig", w, h)

        wrapper: TextWrapper= TextWrapper()
        font_size= int(h*0.03)
        logger.info("Selected font: %s", selected_font)
        try:
            font: FreeTypeFont= ImageFont.truetype(selected_font, size=font_size)
            font_w= font.getlength("_")
        except Exception as e:
            logger.exception("Error loading TTF font, fallback to default : %s", str(e))
            font: ImageFont= ImageFont.load_default()
            # Approximation
            font_w= int(font_size * 0.4)
        logger.debug("SIZE %20s w %5d h %5d", "font", font_w, font_size)
        r_w, r_h= aspect_resize(w + font_size * (len(text) + 2 ), h, self.aspect)
        logger.debug("SIZE %20s w %5d h %5d", "aspect1", r_w, r_h)

        max_char= int(r_w / font_w)
        logger.debug("Font size : %5d max_char %d", font_size, max_char)

        wrapper.width= int(r_w / font_w)
        lines = [wrapper.wrap(i) for i in text]
        lines = list(itertools.chain.from_iterable(lines))

        r_h= int(h + ((len(lines)+1) * (font_size + 3)))
        r_w= int(self.aspect.value[0]/self.aspect.value[1]*r_h)
        lr_borders= int((r_w - w) / 2)
        b_border= r_h - h
        # Redo text wrapping after image resizing
        max_char= int(r_w / font_w)
        logger.debug("Font size : %5d max_char %d", font_size, max_char)

        wrapper.width= int(r_w / font_w)
        lines = [wrapper.wrap(i) for i in text]
        lines = list(itertools.chain.from_iterable(lines))

        logger.debug("SIZE %20s w %5d h %5d", "aspect2", r_w, r_h)
        logger.debug("Add text. w %4d h %4d font size : %d nb lines %d", w, h, font_size, len(lines))
        # left top right bottom
        im2= ImageOps.expand(im, border=(lr_borders, 0, lr_borders, b_border))
        draw: ImageDrawPIL= ImageDraw.Draw(im2)

        y= h + 2
        draw.rectangle([
            (2, y), 
            (r_w - 2, y + 2 + (len(lines) * (font_size+3)))
            ], outline=(255,255,255), width=1)
        for text in lines:
            draw.text((3,y), text, fill=(255,255,255), font=font)
            y+= font_size + 3
        return im2
            
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
    
    def __init__(self, stop_event: Event, image_queue: Queue, config: CastConfiguration):
        Thread.__init__(self)
        logger.info("Init Cast Thread Config: %r", config)
        self.stop_event= stop_event
        self.queue= image_queue
        self.config= config
    
    def run(self):
        logger.info("Start Cast Thread")
        last_sent= None
        min_wait= 2
        last_id_live_preview: int= -1
        current_job: str= ""
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
                                gr.Info(f"Connecting to {self.config.device_name}")
                                #Info App CC1AD845
                                break
                        else:
                            logger.warning("ChromeCast %s is unreachable, dropping image")
                            gr.Warning(f"Unable to connect to {self.config.device_name}")
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
            except Empty as e:
                # Empty queue, check if live preview availabe
                #shared.state.id_live_preview != req.id_live_preview
                #shared.state.set_current_image()
                #logger.debug("Empty queue. State (%s) : (%r)", type(shared.state), shared.state)
                if (getattr(shared.opts, "cast_live_preview", False) == True):
                    state: State= shared.state
                    if (state.job != "" 
                        and (current_job != state.job or state.id_live_preview != last_id_live_preview)
                        and state.current_image is not None):
                        logger.debug("Casting a preview")
                        start_time= datetime.fromtimestamp(state.time_start)
                        messages=[f"Preview {state.id_live_preview}", f"Job : {state.job}"]
                        logger.debug(messages)
                        info= ImageInfo(ImageType.PIL, obj= state.current_image, creation_date=start_time, message= messages)
                        current_job= state.job
                        last_id_live_preview= state.id_live_preview
                        self.queue.put_nowait(info)
            except Exception as e:
                gr.Warning('Error casting image')
                logger.exception("Error processing image")
        logger.warning("Cast thread ended, stop event received")
        if self._chromecast is not None:
            gr.Warning(f"Disconnect from {self.config.device_name}")
            self._chromecast.disconnect()
        

class CastingScript(scripts.Script):

    def __init__(self):
        super().__init__()
        logger.debug("CastingScript%25s", inspect.stack()[0][3])
        self._send_to_cast_thread= False
        script_callbacks.on_app_started(lambda block, _: self.on_app_started(block))

    # def start_casting(self):
    #     pass

    # def stop_casting(self):
    #     pass

    # def set_casting(self, cast: bool):
    #     if cast:
    #         self.start_casting()
    #     else:
    #         self.stop_casting()

    def on_app_started(self, block: gradio.blocks.Blocks):
        logger.debug("CastingScript%25s", inspect.stack()[0][3])

    def title(self):
        logger.debug("CastingScript%25s", inspect.stack()[0][3])
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        return "Casting"

    # def ui(self, is_img2img):
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "is_img2img", type(is_img2img), is_img2img)
    #     """this function should create gradio UI elements. See https://gradio.app/docs/#components
    #     The return value should be an array of all components that are used in processing.
    #     Values of those returned components will be passed to run() and process() functions.
    #     """
    #     return []


    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        logger.debug("CastingScript%25s", inspect.stack()[0][3])
        logger.debug("Args %20s (%s) %r", "is_img2img", type(is_img2img), is_img2img)
        return scripts.AlwaysVisible
        #return False
    
    # def run(self, p, *args):
    #     """
    #     This function is called if the script has been selected in the script dropdown.
    #     It must do all processing and return the Processed object with results, same as
    #     one returned by processing.process_images.

    #     Usually the processing is done by calling the processing.process_images function.

    #     args contains all values returned by components from ui()
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)

    #     pass

    # def setup(self, p, *args):
    #     """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
    #     args contains all values returned by components from ui().
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)
    #     pass


    # def before_process(self, p, *args):
    #     """
    #     This function is called very early during processing begins for AlwaysVisible scripts.
    #     You can modify the processing object (p) here, inject hooks, etc.
    #     args contains all values returned by components from ui()
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)

    #     pass

    # def process(self, p, *args):
    #     """
    #     This function is called before processing begins for AlwaysVisible scripts.
    #     You can modify the processing object (p) here, inject hooks, etc.
    #     args contains all values returned by components from ui()
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)

    #     pass

    # def before_process_batch(self, p, *args, **kwargs):
    #     """
    #     Called before extra networks are parsed from the prompt, so you can add
    #     new extra network keywords to the prompt with this callback.

    #     **kwargs will have those items:
    #       - batch_number - index of current batch, from 0 to number of batches-1
    #       - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
    #       - seeds - list of seeds for current batch
    #       - subseeds - list of subseeds for current batch
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)

    #     pass

    # def after_extra_networks_activate(self, p, *args, **kwargs):
    #     """
    #     Called after extra networks activation, before conds calculation
    #     allow modification of the network after extra networks activation been applied
    #     won't be call if p.disable_extra_networks

    #     **kwargs will have those items:
    #       - batch_number - index of current batch, from 0 to number of batches-1
    #       - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
    #       - seeds - list of seeds for current batch
    #       - subseeds - list of subseeds for current batch
    #       - extra_network_data - list of ExtraNetworkParams for current stage
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)

    #     pass

    # def process_batch(self, p, *args, **kwargs):
    #     """
    #     Same as process(), but called for every batch.

    #     **kwargs will have those items:
    #       - batch_number - index of current batch, from 0 to number of batches-1
    #       - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
    #       - seeds - list of seeds for current batch
    #       - subseeds - list of subseeds for current batch
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)

    #     pass

    # def postprocess_batch(self, p, *args, **kwargs):
    #     """
    #     Same as process_batch(), but called for every batch after it has been generated.

    #     **kwargs will have same items as process_batch, and also:
    #       - batch_number - index of current batch, from 0 to number of batches-1
    #       - images - torch tensor with all generated images, with values ranging from 0 to 1;
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)

    #     pass

    # def postprocess_batch_list(self, p, pp: scripts.PostprocessBatchListArgs, *args, **kwargs):
    #     """
    #     Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
    #     This is useful when you want to update the entire batch instead of individual images.

    #     You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
    #     If the number of images is different from the batch size when returning,
    #     then the script has the responsibility to also update the following attributes in the processing object (p):
    #       - p.prompts
    #       - p.negative_prompts
    #       - p.seeds
    #       - p.subseeds

    #     **kwargs will have same items as process_batch, and also:
    #       - batch_number - index of current batch, from 0 to number of batches-1
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s (%s) %r", "pp", type(pp), pp)
    #     logger.debug("Args %20s %r", "args", args)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)

    #     pass

    def postprocess_image(self, p: StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """
        logger.debug("CastingScript%25s", inspect.stack()[0][3])
        if not casting:
            logger.error("NOT CASTING")
            return
        logger.debug("Args %20s (%s) %r", "p", type(p), p)
        logger.debug("Args %20s (%s) %r", "pp", type(pp), pp)
        logger.debug("Args %20s %r", "args", args)
        if issubclass(type(pp), Processed):
            logger.info("postprocess_image: Processed")
            info= ImageInfo(ImageType.STABLE_DIFFUSION_PROCESSED, obj= pp, creation_date=datetime.now(), message= None)
            self._enqueue(info)
        if issubclass(type(pp), scripts.PostprocessImageArgs):
            logger.info("postprocess_image: PostprocessImageArgs")
            logger.info("p :(%s) %s", type(p), p)
            messages=[]
            messages.append(f"PostprocessImageArgs {p.n_iter}/{p.batch_size}")
            messages.append(f"seed {p.seed}")
            #messages.append(f"eta {p.eta}")
            info= ImageInfo(ImageType.PIL, obj= pp.image, creation_date=datetime.now(), message= messages)
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

    # def postprocess(self, p, processed, *args):
    #     """
    #     This function is called after processing ends for AlwaysVisible scripts.
    #     args contains all values returned by components from ui()
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s (%s) %r", "processed", type(processed), processed)
    #     logger.debug("Args %20s %r", "args", args)
    #     pass

    # def before_component(self, component, **kwargs):
    #     """
    #     Called before a component is created.
    #     Use elem_id/label fields of kwargs to figure out which component it is.
    #     This can be useful to inject your own components somewhere in the middle of vanilla UI.
    #     You can return created components in the ui() function to add them to the list of arguments for your processing functions
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "component", type(component), component)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)

    #     pass

    # def after_component(self, component, **kwargs):
    #     """
    #     Called after a component is created. Same as above.
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "component", type(component), component)
    #     logger.debug("Args %20s %r", "kwargs", kwargs)
    #     if component.elem_id == "quicksettings":
    #         logger.debug("**HERE**")
    #     pass

    # def before_hr(self, p, *args):
    #     """
    #     This function is called before hires fix start.
    #     """
    #     logger.debug("CastingScript%25s", inspect.stack()[0][3])
    #     logger.debug("Args %20s (%s) %r", "p", type(p), p)
    #     logger.debug("Args %20s %r", "args", args)
    #     pass

## Variables
logger.info("Importing module, defining variable")
selected_font= None
casting: bool= False
initial_scan_done= False
cast_devices= []
image_queue: Optional[Queue[ImageInfo]]= None
cast_thread: Optional[CastThread]= None
cast_thread_stop_event: Event= Event()
base_url: str= None
# End Variables

def cast_devices_list():
    return ["---"] + cast_devices

def refresh_cast_devices():
    logger.info("Scan from casting devices, chromecast")
    chromecast_devices= []
    browser: pychromecast.CastBrowser
    services, browser = pychromecast.discovery.discover_chromecasts()
    #logger.debug("Services: ", services)
    browser.stop_discovery()
    device: pychromecast.models.CastInfo
    for device_uuid, device in browser.devices.items():
        logger.debug("Chrome discovery. UUID : %40s Name : %s", device_uuid, device.friendly_name)
        chromecast_devices.append(device.friendly_name)
    cast_devices.clear()
    cast_devices.extend(chromecast_devices)         

def cast_setting_change(*args, **kwargs):
    global image_queue, cast_thread, cast_thread_stop_event, cast_devices, casting
    logger.debug("CastingScript%25s", inspect.stack()[0][3])
    logger.debug("Args %20s %r", "args", args)
    logger.debug("Args %20s %r", "kwargs", kwargs)
    device_name= getattr(shared.opts, "cast_device", None)
    casting= (getattr(shared.opts, "casting", False) 
              and device_name is not None
              and device_name in cast_devices)
    logger.debug("Cast setting change. Casting option %5s Casting %5s device_name : %s devices: %s", getattr(shared.opts, "casting", None), casting, device_name, cast_devices)
    if casting and device_name and cast_thread is None:
        logger.warning("Starting casting thread")
        config= CastConfiguration(cast_type=CastType.CHROMECAST, device_name=device_name, base_callback_url=base_url, temp_dir=temp_dir)
        #Just to be sure
        cast_thread_stop_event.clear()
        if image_queue is None:
            image_queue= Queue(maxsize=10)        
        cast_thread= CastThread(stop_event=cast_thread_stop_event, image_queue=image_queue, config=config)
        cast_thread.start()
    elif cast_thread is not None:
        logger.warning("Shutdown casting thread")
        logger.info("Request stop of cast thread")
        cast_thread_stop_event.set()
        logger.info("Wait for cast thread to teard down, max 5 seconds")
        cast_thread.join(timeout=5)
        cast_thread= None
    if getattr(shared.opts, "casting", False) and cast_thread is None:
        shared.opts.set("casting", False, run_callbacks=False)       
        gr.Error(f"Device {device_name} is not discovered")






def on_ui_settings():
    logger.debug("PyChromeCast MAIN.%25s", inspect.stack()[0][3])
    port_info= """
Casting to chromecast require a webserver in http diffusing locally on the network.
This setting is to set the port used for this webserver.
Default is 7861, one port over the default stable diffusion webui port.
Limited to unprivileged port between 1024 and 32000
"""
    section = ("sd-cast", "Casting")
    # shared.opts.add_option(
    #     "cast_output",
    #     shared.OptionInfo(
    #         False,
    #         "Cast",
    #         gr.Checkbox,
    #         {"interactive": True},
    #         section=section,
            
    #     ),
    # )
    section = ("sd-cast", "Casting")
    shared.opts.add_option(
        "cast_devices_on_quicksettings",
        shared.OptionInfo(
            True,
            "List of cast devices in quick settings bar",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).needs_reload_ui(),
    )
    shared.opts.add_option(
        "cast_grid",
        shared.OptionInfo(
            False,
            "Cast the generated grid",
            gr.Checkbox,
            {"interactive": True, "visible": False},
            section=section,
        ),
    )
    shared.opts.add_option(
        "web_server_port",
        shared.OptionInfo(
            7861,
            "Port for the webserver for casting (chromecast required)",
            gr.Number,
            {"minimum": 1024, "maximum": 32000, "step": 1, "interactive": True, "info": port_info, "visible": False},
            section=section,
        ),
    )
    shared.opts.add_option(
        "cast_device",
        shared.OptionInfo(
            "---",
            "Cast device name",
            gr.Dropdown,
            lambda: {"choices": cast_devices_list()},
            refresh=refresh_cast_devices,
            section=section,
            onchange=cast_setting_change,
        ),
    )
    shared.opts.add_option(
        "casting",
        shared.OptionInfo(
            False,
            "Cast to device",
            gr.Checkbox,
            {"interactive": True},
            section=section,
            onchange=cast_setting_change,
        ),
    )
    shared.opts.add_option(
        "cast_live_preview",
        shared.OptionInfo(
            False,
            "Cast live preview while idle",
            gr.Checkbox,
            {"interactive": True, "visible": True},
            section=section,
        ),
    )    

def on_script_unloaded():
    global cast_thread, cast_thread_stop_event
    logger.info("on_script_unloaded (module)")
    if cast_thread is not None:
        logger.info("Request stop of cast thread")
        cast_thread_stop_event.set()
        logger.info("Wait for cast thread to teard down, should be max 5 seconds")
        cast_thread.join()
        cast_thread= None

def on_app_started(block: gr.Blocks, app):
    global cast_thread, cast_thread_stop_event, initial_scan_done, base_url, casting, selected_font
    logger.info("App started (module)")
    logger.info("Data path: %s", shared.data_path)
    casting= getattr(shared.opts, "casting", False)
    prefered_font=["FreeMono", "Ubuntu Mono"]
    all_fonts= font_manager.get_font_names()
    selected_font= None
    for f in prefered_font:
        if f in all_fonts:
            selected_font= f
            break
    else:
        for f in all_fonts:
            if "Mono" in f:
                selected_font= f
                break
        else:
            selected_font= all_fonts[0]
    selected_font= font_manager.findfont(selected_font)
    if not temp_dir.is_dir():
        logger.warning("Creating temp dir %s", temp_dir.resolve())
        temp_dir.mkdir(parents=True)
    gradio_allowed_path= getattr(shared.opts, "gradio_allowed_path", [])
    if temp_dir.resolve() not in gradio_allowed_path:   
        #TODO removed when local webserver is added     
        logger.info("Add temp dir to gradio allowed path")
        shared.cmd_opts.gradio_allowed_path.append(temp_dir.resolve()) 

    port= getattr(shared.opts, "port", 7860)
    default_listen_ip= None
    try:     
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        default_listen_ip= s.getsockname()[0]
        s.close()
    except Exception as e:
        logger.exception("Error on finding local IP")

    logger.debug("Default listen IP : %s", default_listen_ip)
    base_url= "http://{}:{}/file=".format(default_listen_ip, port)

    if casting and cast_thread is None:
        logger.debug("Casting was enabled but just restarting, disable it")
        shared.opts.set("casting", False, run_callbacks=False) 
    if not initial_scan_done:
        logger.info("Initial scanning of devices")               
        refresh_cast_devices()
        initial_scan_done= True


def on_before_ui():
    logger.info("Before UI (module)")
    quicksettings: List= getattr(shared.opts, "quicksettings_list", [])
    cast_devices_on_quicksettings= getattr(shared.opts, "cast_devices_on_quicksettings", None)
    if "cast_device" not in quicksettings and cast_devices_on_quicksettings:
        logger.info("Adding cast_device to quick settings")
        quicksettings.append("cast_device")
    elif "cast_device" in quicksettings and cast_devices_on_quicksettings == False:
        logger.info("Remove cast_device to quick settings")
        quicksettings.remove("cast_device")
    if "casting" not in quicksettings:
        quicksettings.append("casting")

# for k, v in shared.opts.data.items():
#     logger.debug("SHARED OPTS DATA: %30s = %s", k, v)

# for k, v in shared.cmd_opts.__dict__.items():
#     logger.debug("CMD_OPTS: %30s = %s", k, v)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_before_ui(on_before_ui)
script_callbacks.on_script_unloaded(on_script_unloaded)
