import asyncio
import sys
from io import BytesIO
from pathlib import Path

import gdown
import torch
import uvicorn
from fastai.basic_train import load_learner
from fastai.vision import Image, to_np, open_image
from object_detection_fastai.helper.object_detection_helper import process_output, nms, \
    rescale_boxes
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url: str = 'https://drive.google.com/uc?id=1-3uuNy4lXWL6H5LZaWv8yWEbDfI5TxKv'
export_file_name: str = 'export.pkl'
anchors_file_url: str = 'https://drive.google.com/uc?id=1gUxRm4JzR-xl6imkZCGwDn89AVuKPIkO'
anchors_file_name: str = 'anchors.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists():
        return
    gdown.download(url, str(dest))


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    await download_file(anchors_file_url, path / anchors_file_name)
    try:
        learn = load_learner(path, export_file_name)
        with (path / anchors_file_name).open('rb') as f:
            anchors = torch.load(f)
        return learn, anchors
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn, anchors = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


def prediction(img: Image, detect_thresh: float = 0.2, nms_thresh: float = 0.3):
    with torch.no_grad():
        # Hack to get the model's image size; there's probably a cleaner way to do this
        dummy_batch, _ = learn.data.one_item(img)
        input_size = tuple(dummy_batch[0].shape)
        img_resized = img.clone().resize(input_size)
        img_batch, _ = learn.data.one_item(img_resized)
        prediction_batch = learn.model(img_batch)
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]
        _, clas_pred, bbox_pred = img_batch[0], class_pred_batch[0], bbox_pred_batch[0]
        t_sz = torch.Tensor([*img.size])[None].cpu()
        bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
        if bbox_pred is None:
            return {'predictions': []}
        to_keep = nms(bbox_pred, scores, nms_thresh)
        bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[
            to_keep].cpu()
        bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
        # change from center to top left
        bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2
        # Each bbox is [pixels from top, pixels from left, height, width]
        classes = learn.data.train_ds.classes[1:]
        # show_results(img, bbox_pred, preds, scores, classes, [], [])
        pred_strings = [classes[i] for i in preds]
    return {
        'predictions': [
            {'bbox': bbox, 'class_id': pred, 'class': pred_string, 'score': score}
            for bbox, pred, pred_string, score in zip(bbox_pred.tolist(), preds.tolist(),
                                                      pred_strings, scores.tolist())
        ]
    }


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    pred = prediction(img)
    return JSONResponse({'result': pred})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
