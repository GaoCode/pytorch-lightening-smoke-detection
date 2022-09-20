import argparse
import configure
from inference import BinaryFire,SmokeyNet
import logging
import os,sys
from waggle.plugin import Plugin
from waggle.data.vision import Camera

parser = argparse.ArgumentParser(description='Smoke Detector Plugin')

parser.add_argument('-st',
                        '--smoke_threshold',
                        metavar='smoke_threshold',
                        type=float,
                        default=0.9,
                        help='Threshold for model inference'
                    )

parser.add_argument('-c',
                        '--camera',
                        metavar='camera_endpoint',
                        type=str,
                        required=False,
                        help='Camera endpoint connected to the edge device.'
                    )

parser.add_argument('-hcid',
                        '--hpwren-camera-id',
                        metavar='hpwren_camera_id',
                        type=int,
                        default=0,
                        help='Camera ID for HPWREN. Optional if HPWREN camera API endpoint is being used.'
                    )

parser.add_argument('-hsid',
                        '--hpwren-site-id',
                        metavar='hpwren_site_id',
                        type=int,
                        default=0,
                        help='Site ID for HPWREN. Optional if HPWREN camera API endpoint is being used.'
                    )

args = parser.parse_args()

smoke_threshold=args.smoke_threshold
camera_endpoint=args.camera
hpwren_site_id = args.hpwren_site_id
hpwren_camera_id = args.hpwren_camera_id

TOPIC_SMOKE = os.getenv('TOPIC_SMOKE','env.smoke.')
MODEL_FILE = os.getenv('MODEL_FILE')
MODEL_ABS_PATH = os.path.abspath(MODEL_FILE)
MODEL_TYPE = os.getenv('MODEL_TYPE','smokeynet')
CAMERA_TYPE = os.getenv('CAMERA_TYPE','mp4')

FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="%Y/%m/%d %H:%M:%S",
)

if CAMERA_TYPE == 'mp4':
    camera_device = configure.Recorded_MP4()
elif CAMERA_TYPE == 'device':
    if camera_endpoint is None:
        print(f'No camera device specified. Exiting...')
        exit(1)
    camera_device = configure.Camera_Device(camera_endpoint)
elif CAMERA_TYPE == 'hpwren':
    camera_device = configure.Hpwren(hpwren_camera_id,hpwren_site_id)
else:
    logging.error(f'Error: not supported case for CAMERA_TYPE: {CAMERA_TYPE}.')
    sys.exit()

camera_meta = camera_device.get_metadata()
camera_src = camera_meta['camera_src']
server_name = camera_meta['server_name']
image_url = camera_meta['image_url']
description = camera_meta['description']

camera = Camera(camera_src)

logging.info(f'Starting smoke detection inferencing')
logging.info(f'Get image from {server_name}')
logging.info(f'Image url: {image_url}')
logging.info(f'Description: {description}')
logging.info(f'Using {MODEL_TYPE}')

sample = camera.snapshot()
imageArray = sample.data
timestamp = sample.timestamp

logging.info('Perform an inference based on trainned model')
if MODEL_TYPE == 'binary-classifier':
    binaryFire = BinaryFire(MODEL_ABS_PATH)
    binaryFire.setImageFromArray(imageArray)
    result  = binaryFire.inference()
    percent = result[1]
    if percent >= smoke_threshold:
        sample.save("sample.jpg")
        logging.info('Publish')
        with Plugin() as plugin:
            plugin.upload_file("sample.jpg", timestamp=timestamp)
            plugin.publish(TOPIC_SMOKE + 'certainty', percent, timestamp=timestamp,meta={"camera": f'{camera_src}'})
elif MODEL_TYPE == 'smokeynet':
    previousImg = imageArray
    sample_current = camera.snapshot()
    timestamp_current = sample_current.timestamp
    currentImg = sample_current.data
    smokeyNet = SmokeyNet(MODEL_ABS_PATH,smoke_threshold)
    image_preds, tile_preds, tile_probs = smokeyNet.inference(currentImg,previousImg)
    logging.info('Publish')
    sample.save("sample_previous.jpg")
    sample_current.save("sample_current.jpg")
    with Plugin() as plugin:
        plugin.upload_file("sample_previous.jpg", timestamp=timestamp)
        plugin.upload_file("sample_current.jpg", timestamp=timestamp_current)
        tile_probs_list = str(tile_probs.tolist())
        plugin.publish(TOPIC_SMOKE + 'tile_probs', tile_probs_list, timestamp=timestamp_current,meta={"camera": f'{camera_src}'})
