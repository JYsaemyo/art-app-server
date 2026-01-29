from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import os
import boto3
import uuid
import json
import requests
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional
import asyncio

# 1. 환경 변수 로드
load_dotenv()

# --- Gemini API 설정 ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ 경고: .env 파일에 GEMINI_API_KEY가 없습니다.")
else:
    genai.configure(api_key=api_key)

app = FastAPI()

# 2. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. AWS S3 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION = os.getenv("AWS_REGION")

# --- Pydantic Models (위치: 사용 전 정의 필수) ---
class MusicUrlUpdate(BaseModel):
    music_url: str

# --- Helper Functions ---

def upload_file_to_s3(file: UploadFile):
    try:
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_client.upload_fileobj(
            file.file, BUCKET_NAME, unique_filename,
            ExtraArgs={'ContentType': file.content_type}
        )
        return f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{unique_filename}"
    except Exception as e:
        print(f"❌ S3 업로드 에러: {e}")
        return None

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception: return None

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# --- AI Core Functions ---

# 1. 그림 분석 (이미지 우선)
def run_gemini_vision(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    style_text = style if style else "특별히 지정되지 않음"
    
    if genre in ["그림", "조각", "Painting", "Sculpture", "유화", "수채화", "동양화", "드로잉"]:
        prompt_context = f"""
        이 작품의 장르는 '{genre}'이며, 텍스트상 화풍은 '{style_text}'입니다.
        **[필수] 텍스트보다 이미지를 우선하세요.**
        이미지에서 보이는 붓터치, 질감, 색채, 조형적 특징을 관찰하고, 이것이 화풍 '{style_text}'와 일치하는지 시각적 근거를 들어 설명하세요.
        """
    else:
        prompt_context = f"""
        이 작품의 장르는 '{genre}'입니다.
        스타일 정보보다는 **이미지 자체의 시각적 연출(구도, 빛, 분위기)**과 제목 '{title}'의 상징적 연결성을 분석하세요.
        """

    prompt = f"""
    당신은 예술 전문 큐레이터입니다. 
    **[가장 중요한 지시] 제공된 이미지(사진)를 면밀히 시각적으로 분석하세요.**

    [작품 정보] 제목:{title}, 작가:{artist}, 장르:{genre}, 스타일:{style_text}
    [지침] {prompt_context}
    [출력 포맷(JSON)] - 모든 설명은 완성된 문장으로 서술하세요.
    {{
        "artist_intro": "작가 설명 (2문장 내외)",
        "title_meaning": "제목과 이미지 연관성 (2문장 내외)",
        "art_review": "이미지의 시각적 특징을 바탕으로 한 종합 감상평 (3문장 내외)"
    }}
    """
    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Vision 에러: {e}"); return None

# 2. 음악 프롬프트 생성
def run_gemini_music(description, title, artist):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f"""
    음악 프롬프트 엔지니어로서, 다음 정보를 바탕으로 JSON을 출력하세요.
    [정보] 제목:{title}, 작가:{artist}, 내용:{description}
    [출력] {{"mood": "...", "instruments": "...", "tempo": "...", "music_prompt": "영어 프롬프트", "explanation": "이유"}}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        res = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        return res if not isinstance(res, list) else res[0]
    except Exception: return None

# --- 🛡️ [통합 로직] AI 처리 및 데이터 보호 함수 ---
def process_ai_logic(post_id: int, image_url: str, title: str, artist: str, genre: str, style1: str, description: str, tags: str, force_update: bool = False):
    """
    모든 AI 생성 로직을 담당하는 중앙 함수입니다.
    force_update=False이면 기존 데이터가 있을 경우 절대 덮어쓰지 않습니다.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 현재 상태 조회
        cursor.execute("SELECT ai_summary, music_prompt FROM posts WHERE id = %s", (post_id,))
        current_data = cursor.fetchone()
        
        if not current_data: return

        # 1. 그림 분석 (ai_summary)
        if not current_data['ai_summary'] or force_update:
            print(f"🖌️ [Processing] ID {post_id} 그림 분석 시작...")
            vision_res = run_gemini_vision(image_url, title, artist, genre, style1)
            
            if vision_res:
                summary = vision_res.get('art_review', '')
                
                # [DB 보호]
                if force_update:
                    sql = "UPDATE posts SET ai_summary = %s WHERE id = %s"
                else:
                    sql = "UPDATE posts SET ai_summary = %s WHERE id = %s AND (ai_summary IS NULL OR ai_summary = '')"
                
                cursor.execute(sql, (summary, post_id))
                conn.commit()
                current_data['ai_summary'] = summary
        else:
            print(f"🛡️ [Protected] ID {post_id} 그림 분석 데이터 보존됨.")

        # 2. 음악 프롬프트 생성 (music_prompt)
        # [확인] 여기에 음악 프롬프트 생성 로직이 포함되어 있습니다.
        if not current_data['music_prompt'] or force_update:
            # 재료 준비
            source_text = description or current_data['ai_summary'] or tags or "Art"
            
            if source_text:
                print(f"🎵 [Processing] ID {post_id} 음악 프롬프트 생성 시작...")
                music_res = run_gemini_music(f"{source_text} / 태그: {tags}", title, artist)
                
                if music_res:
                    prompt = music_res.get('music_prompt')
                    
                    # [DB 보호]
                    if force_update:
                        sql = "UPDATE posts SET music_prompt = %s WHERE id = %s"
                    else:
                        sql = "UPDATE posts SET music_prompt = %s WHERE id = %s AND (music_prompt IS NULL OR music_prompt = '')"
                    
                    cursor.execute(sql, (prompt, post_id))
                    conn.commit()
        else:
             print(f"🛡️ [Protected] ID {post_id} 음악 프롬프트 데이터 보존됨.")

    except Exception as e:
        print(f"❌ [Error] ID {post_id} 처리 중 오류: {e}")
    finally:
        cursor.close(); conn.close()

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Art App Backend is Live!"}

@app.post("/posts/")
async def create_post(
    background_tasks: BackgroundTasks, 
    user_id: int = Form(...), title: str = Form(...), artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None), tags: Optional[str] = Form(None),
    genre: Optional[str] = Form("인상주의"), style1: Optional[str] = Form("유화"),
    style2: Optional[str] = Form(None), style3: Optional[str] = Form(None),
    style4: Optional[str] = Form(None), style5: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    # 1. S3 업로드
    image_url = upload_file_to_s3(image)
    if not image_url: raise HTTPException(500, "S3 실패")

    # 2. DB 선 저장 (빠른 응답)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts (user_id, title, artist_name, image_url, description, tags, genre, style1, style2, style3, style4, style5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (user_id, title, artist_name, image_url, description, tags, genre, style1, style2, style3, style4, style5)
        cursor.execute(sql, val)
        conn.commit()
        new_post_id = cursor.lastrowid
        
        # 3. ✨ 백그라운드 AI 작업 등록
        background_tasks.add_task(
            process_ai_logic, 
            new_post_id, image_url, title, artist_name, genre, style1, description, tags,
            True 
        )
        
        return {"message": "업로드 완료. AI 분석이 백그라운드에서 진행됩니다.", "id": new_post_id}
        
    finally: cursor.close(); conn.close()

@app.get("/posts/")
def get_posts():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT p.*, u.nickname FROM posts p JOIN users u ON p.user_id = u.id ORDER BY p.id DESC")
        return {"posts": cursor.fetchall()}
    finally:
        cursor.close(); conn.close()

# 수동 분석 요청
@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int, force_update: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post: raise HTTPException(404, "게시글 없음")

        # 기존 함수 재사용
        process_ai_logic(
            post['id'], post['image_url'], post['title'], post['artist_name'], 
            post['genre'], post['style1'], post['description'], post['tags'],
            force_update
        )
        
        cursor.execute("SELECT ai_summary FROM posts WHERE id = %s", (post_id,))
        updated_post = cursor.fetchone()
        return {"message": "요청 완료", "ai_summary": updated_post['ai_summary']}
    finally: cursor.close(); conn.close()

@app.post("/posts/{post_id}/music")
def generate_music_prompt(post_id: int, force_update: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post: raise HTTPException(404, "게시글 없음")

        process_ai_logic(
            post['id'], post['image_url'], post['title'], post['artist_name'], 
            post['genre'], post['style1'], post['description'], post['tags'],
            force_update
        )

        cursor.execute("SELECT music_prompt FROM posts WHERE id = %s", (post_id,))
        updated_post = cursor.fetchone()
        return {"message": "요청 완료", "music_prompt": updated_post['music_prompt']}
    finally: cursor.close(); conn.close()

@app.post("/posts/{post_id}/register_music_url")
def register_music_url(post_id: int, body: MusicUrlUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE posts SET music_url = %s WHERE id = %s", (body.music_url, post_id))
        conn.commit()
        return {"message": "등록 완료", "music_url": body.music_url}
    finally:
        cursor.close(); conn.close()

# 수동 동기화 요청
@app.post("/posts/sync-ai")
def sync_missing_ai_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
        empty_posts = cursor.fetchall()

        if not empty_posts: return {"message": "모든 데이터가 최신입니다."}

        for post in empty_posts:
            process_ai_logic(
                post['id'], post['image_url'], post['title'], post['artist_name'], 
                post['genre'], post['style1'], post['description'], post['tags'],
                False 
            )

        return {"message": f"총 {len(empty_posts)}건 보정 요청 완료"}
    finally:
        cursor.close(); conn.close()

# --- ⏰ [Scheduler] 1분 주기 자동 보정 스케줄러 ---
async def periodic_sync_task():
    """1분마다 무한 반복하며 비어있는 AI 데이터를 채웁니다."""
    print("⏰ [Scheduler] 1분 주기 자동 보정 스케줄러가 시작되었습니다.")
    
    while True:
        try:
            await asyncio.sleep(60)
            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
            empty_posts = cursor.fetchall()
            
            if empty_posts:
                print(f"🔍 [Scheduler] {len(empty_posts)}개의 누락 데이터 발견. 보정 시작...")
                for post in empty_posts:
                    process_ai_logic(
                        post['id'], post['image_url'], post['title'], post['artist_name'], 
                        post['genre'], post['style1'], post['description'], post['tags'],
                        False # 안전 모드
                    )
            cursor.close(); conn.close()
        except Exception as e:
            print(f"⚠️ [Scheduler] 에러 발생 (1분 후 재시도): {e}")

# 서버 시작 시 스케줄러 실행
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(periodic_sync_task())
