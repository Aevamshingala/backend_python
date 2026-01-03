from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import requests, os, uuid, subprocess, shutil, asyncio, random, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional
from urllib.parse import urlparse
from google import genai

app = FastAPI()

# ---------------------- API KEYS ----------------------
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# ---------------------- DIR SETUP ----------------------
os.makedirs("public", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("temp_media", exist_ok=True)

app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# ---------------------- CORS ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- MODELS ----------------------
class SentenceList(BaseModel):
    keywords: list[str]

class OrderedItem(BaseModel):
    type: Literal["image", "video"]
    url: str
    start: float = 0
    end: float = 4
    duration: float = 4


class VideoRequest(BaseModel):
    items: list[OrderedItem]
    start: Optional[float] = None      # for videos
    end: Optional[float] = None
    music_url: str | None = None
    subtitles: dict | None = None

class Script(BaseModel):
    description: str

# ---------------------- UTIL ----------------------
def find_ffmpeg():
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path
    raise RuntimeError("ffmpeg not found.")

FFMPEG_BIN = find_ffmpeg()

def run_cmd(cmd, check=True):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("FFmpeg failed.")
    return result

def download_file(url, path):
    try:
        r = requests.get(url, stream=True, timeout=25)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                f.write(chunk)
        return True
    except Exception as e:
        print("[Download failed]", e)
        return False

def safe_extension(url):
    return os.path.splitext(urlparse(url).path)[1] or ""

# ---------------------- FIXED FFMPEG HELPERS ----------------------
def reencode_video_to_mp4(input_path, output_path, fps=24):
    cmd = [
        FFMPEG_BIN, "-y",
        "-fflags", "+genpts",
        "-i", input_path,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "-r", str(fps),
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    run_cmd(cmd)
    return output_path

def trim_video_to_duration(input_path, output_path,start,end):
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(end),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "-r", "24",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    run_cmd(cmd)
    return output_path

def concat_mp4_list(file_list_path, output_path):
    """
    Concatenate all clips with proper timestamps and consistent frame rate.
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-filter_complex", "fps=24,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "-r", "24",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    run_cmd(cmd)
    return output_path

def mix_audio_into_video(video_path, audio_path, output_path):
    """
    Attach background music to a (possibly silent) video.
    Works even if:
    - the video has no audio stream
    - the video is shorter or longer than the music
    - the AAC encoder behaves differently across platforms
    """
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg not found.")

    # Determine total video length
    duration = get_video_duration(video_path)
    if not duration or duration <= 0:
        duration = 30.0  # fallback to 30s

    # Build ffmpeg command
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[1:a]aloop=loop=-1:size=2e+09,apad[aud];[aud]atrim=duration={duration}[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]

    print("[INFO] Mixing audio into video...")
    result = run_cmd(cmd)
    print("[INFO] Audio mix complete:", output_path)
    return output_path


# ---------------------- VIDEO PROCESS ----------------------
def get_video_duration(video_path):
    try:
        cmd = [FFMPEG_BIN, "-i", video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stderr.split("\n"):
            if "Duration:" in line:
                d = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = d.split(":")
                return float(h)*3600 + float(m)*60 + float(s)
    except:
        pass
    return 4.0

def process_video_sync(url, idx, start, end, subtitle_text=None):
    ext = safe_extension(url) or ".mp4"
    raw_clip = os.path.join("temp_media", f"clip_{idx}{ext}")

    if not download_file(url, raw_clip):
        return None

    dur = get_video_duration(raw_clip)
    trimmed = os.path.join("temp_media", f"trim_{idx}.mp4")

    if dur > end:
        trim_video_to_duration(raw_clip, trimmed, start, end)
    else:
        reencode_video_to_mp4(raw_clip, trimmed)

    norm = os.path.join("temp_media", f"norm_{idx}.mp4")
    reencode_video_to_mp4(trimmed, norm)

    # 🆕 Add subtitle overlay
    subbed = os.path.join("temp_media", f"sub_{idx}.mp4")
    add_subtitle_to_video(norm, subbed, subtitle_text)

    try:
        os.remove(raw_clip)
        os.remove(trimmed)
        os.remove(norm)
    except:
        pass
    return subbed

# ---------------------- IMAGE PROCESS ----------------------
def resize_with_padding(img, size=(1920,1080)):
    target_w, target_h = size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - new_w)//2
    pad_y = (target_h - new_h)//2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas

def cinematic_motion(img, steps):
    h, w = img.shape[:2]
    z1, z2 = random.uniform(1.0,1.1), random.uniform(1.1,1.25)
    frames=[]
    for i in range(steps):
        t=i/(steps-1)
        zoom=z1+(z2-z1)*t
        M=cv2.getRotationMatrix2D((w/2,h/2),0,zoom)
        f=cv2.warpAffine(img,M,(w,h))
        frames.append(f)
    return frames

def draw_subtitle(frame, text):
    if not text:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    # Measure text correctly
    (tw, th2), _ = cv2.getTextSize(text, font, scale, thickness)

    # Position centered above bottom
    x = (w - tw) // 2
    y = h - 80

    # Draw black translucent background
    cv2.rectangle(
        overlay,
        (x - 25, y - th2 - 25),
        (x + tw + 25, y + 25),
        (0, 0, 0),
        -1
    )

    # Blend background with frame
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Draw white text
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

def add_subtitle_to_video(video_path, output_path, text):
    if not text:
        shutil.copy(video_path, output_path)
        return output_path

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vf", f"drawtext=text='{text}':fontcolor=white:fontsize=48:borderw=2:x=(w-text_w)/2:y=h-100",
        "-c:a", "copy",
        output_path
    ]
    run_cmd(cmd)
    return output_path

def process_image_sync(img_path, idx, subtitle_text, duration=4, fps=24):
    img=cv2.imread(img_path)
    if img is None: return None
    img=resize_with_padding(img)
    frames = cinematic_motion(img, int(duration * fps))
    if subtitle_text: frames=[draw_subtitle(f,subtitle_text) for f in frames]
    avi=os.path.join("temp_media",f"temp_{idx}.avi")
    out=cv2.VideoWriter(avi,cv2.VideoWriter_fourcc(*"MJPG"),fps,(1920,1080))
    for f in frames: out.write(f)
    out.release()
    mp4=os.path.join("temp_media",f"img_{idx}.mp4")
    reencode_video_to_mp4(avi,mp4,fps)
    os.remove(avi)
    return mp4


async def process_image_async(img_path, idx, subtitle_text, duration, fps):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool,
            process_image_sync,
            img_path,
            idx,
            subtitle_text,
            duration,
            fps
        )

async def process_video_async(url, idx, start, end, subtitle_text=None):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, process_video_sync, url, idx, start, end, subtitle_text)


# ---------------------- MAIN VIDEO BUILDER ----------------------
async def create_cinematic_video(ordered_items, fps=24, music_url=None, subtitles=None):
    temp = []
    tasks=[]
    for idx,item in enumerate(ordered_items):
       if item.type == "image":
            ext = safe_extension(item.url) or ".jpg"
            img = os.path.join("temp_media", f"img_{idx}{ext}")

            if download_file(item.url, img):
                t = asyncio.create_task(
                    process_image_async(
                        img,
                        idx,
                        subtitles.get(str(idx)) if subtitles else None,
                        item.duration,
                        fps
                    )
                )
                tasks.append((t, "image", img))
       else:
            subtitle_text = subtitles.get(str(idx)) if subtitles else None
            t = asyncio.create_task(
                process_video_async(item.url, idx, item.start, item.end, subtitle_text)
            )
            tasks.append((t, "video", None))
    for task,typ,f in tasks:
        try:
            res=await task
            if res: temp.append(res)
            if typ=="image" and f: os.remove(f)
        except Exception as e:
            print("[ERR]",e)

    if not temp: raise Exception("No valid media.")

    listfile=os.path.join("temp_media",f"concat_{uuid.uuid4()}.txt")
    with open(listfile,"w") as f:
        for p in temp: f.write(f"file '{os.path.abspath(p)}'\n")

    merged=os.path.join("videos",f"merged_{uuid.uuid4()}.mp4")
    concat_mp4_list(listfile,merged)

    final=os.path.join("videos",f"final_{uuid.uuid4()}.mp4")
    print("\n music url",music_url)
    if music_url:
            aud = os.path.join("temp_media", f"a_{uuid.uuid4()}.mp3")
            print("\n aud", aud)

            # ✅ Handle both local and online URLs
            if os.path.exists(music_url):
                print("[INFO] Using local music file:", music_url)
                shutil.copy(music_url, aud)
            else:
                print("[INFO] Downloading music from URL:", music_url)
                if not download_file(music_url, aud):
                    print("[WARN] Failed to fetch audio — proceeding without it.")
                    shutil.copy(merged, final)
                    return final

            # ✅ Now safely mix audio into video
            mix_audio_into_video(merged, aud, final)

            # Clean up
            try:
                os.remove(aud)
            except:
                pass
    else:
            shutil.copy(merged, final)


    return final

# ---------------------- ROUTES ----------------------
import traceback

@app.post("/create_video")
async def create_video(req: VideoRequest):
    try:
        out = await create_cinematic_video(req.items, music_url=req.music_url, subtitles=req.subtitles)
        os.remove(os.path.join("temp_media"))
        return {"status": "success", "video_url": f"http://127.0.0.1:8000/{out.replace(os.sep, '/')}"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ---------------------- ROUTES ----------------------
@app.get("/proxy_video")
def proxy_video(url: str):
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        return StreamingResponse(
            r.iter_content(1024 * 64),
            media_type="video/mp4",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cross-Origin-Resource-Policy": "cross-origin"
            }
        )
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".mp3"
        save_path = os.path.join("temp_media", f"{uuid.uuid4()}{ext}")

        os.makedirs("temp_media", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(await file.read())

        print("[INFO] Audio uploaded:", save_path)

        duration = 0.0
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                save_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stdout.strip():
                duration = float(result.stdout.strip())
            else:
                print("[WARN] ffprobe returned no duration output.")
        except Exception as e:
            print("[WARN] Could not extract duration:", e)

        return {
            "status": "success",
            "file_path": save_path,
            "duration": duration,
        }

    except Exception as e:
        print("[ERROR] upload_audio failed:", e)
        # 🔒 Always return JSON
        return {
            "status": "error",
            "message": str(e),
        }
@app.post("/fetch_image")
async def fetch_image_video(body: SentenceList):
    headers = {"Authorization": PEXELS_API_KEY}
    results = []

    for sentence in body.keywords:
        if not sentence.strip():
            continue
        imgs, vids = [], []
        try:
            pexels_img = f"https://api.pexels.com/v1/search?query={sentence.strip()}&per_page=10"
            resp = requests.get(pexels_img, headers=headers, timeout=10)
            if resp.status_code == 200:
                for p in resp.json().get("photos", []):
                    src=p.get("src",{})
                    best=src.get("large2x") or src.get("large") or src.get("original")
                    if best: imgs.append(best)
            pixabay_url="https://pixabay.com/api/"
            params={'key':PIXABAY_API_KEY,'q':sentence.strip(),'image_type':'photo','per_page':10,'safesearch':'true'}
            resp=requests.get(pixabay_url,params=params,timeout=10)
            if resp.status_code==200:
                for hit in resp.json().get("hits",[]):
                    url=hit.get("largeImageURL") or hit.get("webformatURL")
                    if url: imgs.append(url)
            pexels_vid=f"https://api.pexels.com/videos/search?query={sentence.strip()}&per_page=10"
            resp=requests.get(pexels_vid,headers=headers,timeout=10)
            if resp.status_code==200:
                for v in resp.json().get("videos",[]):
                    best=None;h=0
                    for vf in v.get("video_files",[]):
                        link=vf.get("link");hh=vf.get("height",0)
                        if link and link.endswith(".mp4") and hh>h:
                            best=link;h=hh
                    pics=v.get("video_pictures",[])
                    thumb=pics[len(pics)//2].get("picture") if pics else None
                    if best and thumb: vids.append({"mp4":best,"thumbnail":thumb})
            video_params={'key':PIXABAY_API_KEY,'q':sentence.strip(),'per_page':10,'safesearch':'true'}
            resp=requests.get("https://pixabay.com/api/videos/",params=video_params,timeout=10)
            if resp.status_code==200:
                for v in resp.json().get("hits",[]):
                    vd=v.get("videos",{})
                    info=vd.get("medium") or vd.get("small") or vd.get("large")
                    if info:
                        vids.append({"mp4":info.get("url"),"thumbnail":v.get("thumbnail")})
        except Exception as e:
            print("Fetch error",e)
        results.append({"sentence":sentence,"images":imgs,"videos":vids})
    return {"results":results}

@app.post("/keyword")
async def Find_keyword(script: Script):
    prompt=f"""
Read the following paragraph or story carefully.

Understand the context and sequence of events — what is happening in each part of the story.

For every major scene, event, or action, extract the main idea or keyword phrase (2-6 words max) that best summarizes what is happening. 
Focus on context — who is doing what, what happens next, and what the result or discovery is.

The extracted phrases should:
- Follow the story order.
- Reflect the actual situation or action (not just random nouns).
- Capture emotional or logical transitions (e.g., realization, problem, reaction).
- Be concise but descriptive enough for video scene tagging or script segmentation.

Give the maximum relevant keyword phrases covering all main parts of the story.

Format output as:
- Only the keyword phrases.
- Each phrase separated by a dot (.)
- No numbering, no bullet points, no extra explanations, and no titles.

Example:

Input:
A human took an Airtel SIM. After that his whole life became hell. When you go to buy a SIM, telecom companies sometimes hand over a recycled number. That means the number is not fresh. Calls started coming from the bank asking for 30,000. Recovery agents threatened visits. The number belonged to someone who hadn't paid a loan. On government websites, it shows “number already registered.” Later, police complaints in Ghaziabad's Vijayanagar station were found linked to that number. On Facebook and social media, accounts already existed. Today, since Aadhar, bank, income tax, and UPI are all tied to numbers, this has become a big issue. The person believes the government should make rules to delete old links before reassigning numbers.

Output:
new sim card. bank calls. recovery agent. recycled number. government website issue. aadhar or bank problem. social media account already exists. police complaint. harassment and confusion. government regulation needed.

Now apply the same logic to this paragraph:
{script.description}
"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    )
    return {"response": response.text.strip()}