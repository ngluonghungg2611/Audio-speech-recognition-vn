import os
import shutil
import datetime
# import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn
from pathlib import Path
from tempfile import NamedTemporaryFile
# from s2t_rec.s2t_ja import Vosk_S2T
from s2t_rec.s2t_vi import VN_S2T
import yaml

app = FastAPI()

# allow CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("s2t_rec/config.yaml") as f:
    cfg = yaml.safe_load(f)
# ja_s2t = Vosk_S2T()
vn_s2t = VN_S2T(cfg)

if not os.path.exists("./save_audio/vi"):
    os.makedirs("./save_audio/vi")



if not os.path.exists("./s2t_rec/audio_chunks"):
    os.makedirs("./s2t_rec/audio_chunks")


def save_upload_file_tmp(upload_file: UploadFile, lang):
    """
    save file user uploaded

    Args:
        upload_file (UploadFile): file user uploaded
        lang (str): language (ja or vi)
    Returns:
        Path: path save file
    """
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)

        save_path = "./save_audio/{}/audio_{}_{}_{}.{}".format(
            lang,
            datetime.datetime.now().hour,
            datetime.datetime.now().minute,
            datetime.datetime.now().second,
            str(tmp_path).split(".")[-1],
        )
        shutil.copy(tmp_path, save_path)

    finally:
        upload_file.file.close()

    return tmp_path


# @app.post("/speech-to-text-ja")
# async def pred_stt_ja(request: Request, file1: UploadFile = File(...)):

#     if request.method == "POST":
#         in_file = save_upload_file_tmp(file1, lang="ja")
#         text = ja_s2t.speech2text(in_file)
#         os.remove(in_file)
#         if "error" in text:
#             return text
#         return {"text": text}


@app.post("/speech-to-text-vi")
def pred_stt_vi(request: Request, file1: UploadFile = File(...)):

    if request.method == "POST":
        in_file = save_upload_file_tmp(file1, lang="vi")

        text = vn_s2t.speech2text(in_file)
        os.remove(in_file)
        if "error" in text:
            return text

        return {"text": text}

# @app.post("/speech-to-text-vi-")

if __name__ == "__main__":
    print("* Starting web service...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
