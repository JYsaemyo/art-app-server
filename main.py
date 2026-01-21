from fastapi import FastAPI, HTTPException, Form, UploadFile, File
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

# --- S3 업로드 헬퍼 함수 ---
def upload_file_to_s3(file: UploadFile):
    try:
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            unique_filename,
            ExtraArgs={'ContentType': file.content_type}
        )
        return f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{unique_filename}"
    except Exception as e:
        print(f"❌ S3 업로드 에러: {e}")
        return None

# --- AI 헬퍼 함수들 ---

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        return None

# 🖼️ 그림 분석 시 style1 정보를 적극 활용하도록 프롬프트 구성
def run_gemini_vision(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None
    
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    prompt = f"""
    당신은 미술 평론가입니다. 제공된 이미지와 아래의 스타일 정보를 바탕으로 작품을 정밀 분석하세요.
    특별히 지정된 화풍({style})의 특징이 그림에서 어떻게 나타나는지 주목하여 JSON으로 출력하세요.

    [정보] 제목: {title}, 작가: {artist}, 장르: {genre}, 화풍(Style): {style}
    
    [출력 포맷 (JSON)]
    {{
        "artist_intro": "작가 설명 (2문장)",
        "title_meaning": "제목 의미 (2문장)",
        "art_review": "화풍의 특징이 반영된 종합 감상평 (3문장)"
    }}
    """
    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Vision 에러: {e}")
        return None

def run_gemini_music(description, title, artist):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f"""
    전문 음악 프롬프트 엔지니어로서, 아래 작품 정보를 바탕으로 음악 생성 프롬프트를 작성하세요.
    [작품 정보] 제목: {title}, 작가: {artist}, 내용: {description}

    [출력 포맷 (JSON)]
    {{
        "mood": "...", "instruments": "...", "tempo": "...",
        "music_prompt": "생성용 영어 프롬프트",
        "explanation": "추천 이유 (한글)"
    }}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        result = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        return result if not isinstance(result, list) else result[0]
    except Exception as e:
        print(f"Gemini Music 에러: {e}")
        return None

def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"DB 접속 실패: {err}")

# --- API 엔드포인트 ---

@app.get("/")
def read_root():
    return {"message": "Art App Backend is Live!"}

# (3) 게시글 업로드 (style1, genre 저장 추가 ✨)
@app.post("/posts/")
async def create_post(
    user_id: int = Form(...), 
    title: str = Form(...), 
    artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None), 
    tags: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    style1: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    image_url = upload_file_to_s3(image)
    if not image_url: raise HTTPException(status_code=500, detail="S3 업로드 실패")

    # 업로드 시 음악 프롬프트 자동 생성 시도
    generated_prompt = None
    if description or tags:
        input_ctx = f"설명: {description or ''} / 태그: {tags or ''}"
        res = run_gemini_music(input_ctx, title, artist_name)
        if res: generated_prompt = res.get('music_prompt')

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts (user_id, title, artist_name, image_url, description, tags, music_prompt, genre, style1)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (user_id, title, artist_name, image_url, description, tags, generated_prompt, genre, style1))
        conn.commit()
        return {"id": cursor.lastrowid, "image_url": image_url, "music_prompt": generated_prompt}
    finally:
        cursor.close(); conn.close()

# (4) 피드 조회
@app.get("/posts/")
def get_posts():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT p.*, u.nickname FROM posts p JOIN users u ON p.user_id = u.id ORDER BY p.id DESC")
        return {"posts": cursor.fetchall()}
    finally:
        cursor.close(); conn.close()

# (5) 그림 분석 (style1 적용 ✨)
@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post: raise HTTPException(status_code=404, detail="게시글 없음")
        
        # DB의 style1 컬럼 값을 우선적으로 분석에 사용
        target_style = post.get('style1')
        target_genre = post.get('genre')
        
        ai_result = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], target_genre, target_style)
        if not ai_result: raise HTTPException(status_code=500, detail="AI 분석 실패")

        summary_text = ai_result.get('art_review', '')
        cursor.execute("UPDATE posts SET ai_summary = %s WHERE id = %s", (summary_text, post_id))
        conn.commit()
        
        return {"message": "분석 완료", "ai_summary": summary_text, "result": ai_result}
    finally:
        cursor.close(); conn.close()

# (6) 음악 프롬프트 수동 생성
@app.post("/posts/{post_id}/music")
def generate_music_prompt(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        desc = post.get('description') or post.get('ai_summary')
        if not desc: raise HTTPException(status_code=400, detail="정보 부족")

        music_res = run_gemini_music(f"{desc} / {post.get('tags','')}", post['title'], post['artist_name'])
        if music_res:
            prompt_text = music_res.get('music_prompt', '')
            cursor.execute("UPDATE posts SET music_prompt = %s WHERE id = %s", (prompt_text, post_id))
            conn.commit()
            return {"music_prompt": prompt_text, "explanation": music_res.get('explanation')}
    finally:
        cursor.close(); conn.close()

# (7) 음악 URL 등록
class MusicUrlUpdate(BaseModel):
    music_url: str

@app.post("/posts/{post_id}/register_music_url")
def register_music_url(post_id: int, body: MusicUrlUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE posts SET music_url = %s WHERE id = %s", (body.music_url, post_id))
        conn.commit()
        return {"message": "등록 완료"}
    finally:
        cursor.close(); conn.close()

# (8) 기존 DB 데이터 보정 (비어있는 값 자동 생성)
@app.post("/posts/sync-ai")
def sync_missing_ai_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
        empty_posts = cursor.fetchall()
        if not empty_posts: return {"message": "모든 데이터가 최신입니다."}

        sync_count = 0
        for post in empty_posts:
            post_id = post['id']
            updates = {}

            # [A] ai_summary 생성 (style1 컬럼 활용 ✨)
            if not post.get('ai_summary'):
                style = post.get('style1')
                genre = post.get('genre')
                res = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], genre, style)
                if res:
                    updates['ai_summary'] = res.get('art_review', '')
                    post['ai_summary'] = res.get('art_review') # 다음 단계를 위해 갱신

            # [B] music_prompt 생성
            if not post.get('music_prompt'):
                source = post.get('description') or post.get('ai_summary')
                if source:
                    res = run_gemini_music(f"{source} / {post.get('tags','')}", post['title'], post['artist_name'])
                    if res: updates['music_prompt'] = res.get('music_prompt')

            if updates:
                cols = ", ".join([f"{k} = %s" for k in updates.keys()])
                cursor.execute(f"UPDATE posts SET {cols} WHERE id = %s", list(updates.values()) + [post_id])
                conn.commit()
                sync_count += 1
        return {"message": f"{sync_count}건 보정 완료"}
    finally:
        cursor.close(); conn.close()

import asyncio

# --- [추가] 서버 시작 시 실행될 백그라운드 동기화 함수 ---
async def startup_sync():
    """서버 시작 5초 후부터 비어있는 AI 데이터를 자동으로 채웁니다."""
    await asyncio.sleep(5) # 서버가 완전히 준비될 때까지 잠시 대기
    print("🚀 [System] 서버 시작: 누락된 AI 데이터 자동 보정을 시작합니다...")
    
    try:
        # 기존에 만든 sync_missing_ai_data 로직을 그대로 호출하거나 
        # 아래처럼 직접 로직을 수행합니다.
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 보정이 필요한 데이터 조회
        cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
        empty_posts = cursor.fetchall()

        if not empty_posts:
            print("✅ [System] 보정할 데이터가 없습니다. 모든 데이터가 최신입니다.")
            return

        for post in empty_posts:
            post_id = post['id']
            updates = {}

            # 1. ai_summary 보정 (style1 컬럼 활용 ✨)
            if not post.get('ai_summary'):
                style = post.get('style1')
                genre = post.get('genre')
                res = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], genre, style)
                if res:
                    updates['ai_summary'] = res.get('art_review', '')
                    post['ai_summary'] = updates['ai_summary']

            # 2. music_prompt 보정
            if not post.get('music_prompt'):
                source = post.get('description') or post.get('ai_summary')
                if source:
                    res = run_gemini_music(f"{source} / {post.get('tags','')}", post['title'], post['artist_name'])
                    if res:
                        updates['music_prompt'] = res.get('music_prompt')

            # DB 반영
            if updates:
                cols = ", ".join([f"{k} = %s" for k in updates.keys()])
                cursor.execute(f"UPDATE posts SET {cols} WHERE id = %s", list(updates.values()) + [post_id])
                conn.commit()
                print(f"✨ [System] ID {post_id}번 데이터 보정 완료")

        print(f"✅ [System] 총 {len(empty_posts)}건의 데이터 보정 프로세스가 종료되었습니다.")

    except Exception as e:
        print(f"❌ [System] 자동 보정 중 에러 발생: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# --- FastAPI 스타트업 이벤트 등록 ---
@app.on_event("startup")
async def on_startup():
    # 서버 실행 시 백그라운드에서 동기화 함수 실행 (비동기 처리)
    asyncio.create_task(startup_sync())
