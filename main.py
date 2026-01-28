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

# 🖼️ [핵심] 이미지 시각 분석을 최우선으로 하도록 프롬프트 강화
def run_gemini_vision(image_url, title, artist, genre, style):
    """
    style: style1 단일 문자열 값
    이미지 자체의 시각적 특징을 텍스트 정보(장르, 스타일)와 결합하여 분석합니다.
    """
    img = load_image_from_url(image_url)
    if not img: return None
    
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    # 1. 스타일 텍스트 처리
    style_text = style if style else "특별히 지정되지 않음"

    # 2. 장르에 따른 분석 초점 설정 (이미지 관찰 중심)
    if genre in ["그림", "조각", "Painting", "Sculpture", "유화", "수채화", "동양화", "드로잉"]:
        # [Case A] 그림/조각: style1 정보를 집중적으로 확인
        prompt_context = f"""
        이 작품의 장르는 '{genre}'이며, 핵심 화풍(Style)은 '{style_text}'입니다.
        
        **[중요] 반드시 제공된 이미지(사진)를 시각적으로 분석하세요.**
        이미지 속의 붓터치, 질감, 색채, 조형적 특징이 텍스트로 제시된 화풍 '{style_text}'와 어떻게 일치하는지 시각적 근거를 들어 설명하세요.
        만약 텍스트 정보와 이미지가 다르다면, 이미지에서 보이는 실제 특징을 우선하여 묘사하세요.
        """
    else:
        # [Case B] 그 외 (사진, 미디어아트 등): 이미지의 연출과 제목의 관계를 찾아라
        prompt_context = f"""
        이 작품의 장르는 '{genre}'입니다. 
        스타일 정보보다는 **이미지 자체의 시각적 연출**과 작품의 제목 '{title}'이 주는 상징성에 집중하여 분석하세요.
        이미지에서 느껴지는 분위기가 주제를 어떻게 전달하는지 설명하세요.
        """

    # 3. 최종 프롬프트 조합 (이미지 분석 강조)
    prompt = f"""
    당신은 예리한 관찰력을 가진 예술 전문 큐레이터입니다. 
    
    **[가장 중요한 지시]** 텍스트 정보에만 의존하지 말고, **반드시 함께 제공된 이미지(사진)를 면밀히 시각적으로 분석**해야 합니다. 당신의 분석 결과는 실제 눈으로 본 이미지의 특징에 기반해야 합니다.

    [작품 텍스트 정보]
    - 제목: {title}
    - 작가: {artist}
    - 장르: {genre}
    - 스타일: {style_text}
    
    [분석 지침]
    {prompt_context}

    [출력 포맷 (JSON)]
    - 모든 설명은 완성된 문장으로 서술하세요.
    {{
        "artist_intro": "작가 설명 (2문장 내외)",
        "title_meaning": "제목이 이미지와 어떤 관련이 있는지 설명 (2문장 내외)",
        "art_review": "이미지의 시각적 특징을 바탕으로 한 종합 감상평 (3문장 내외)"
    }}
    """
    
    try:
        # 이미지 객체(img)와 강화된 텍스트 프롬프트(prompt)를 함께 전송
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Vision 에러: {e}")
        return None

# 🎵 음악 프롬프트 생성 함수 (필수)
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

# (3) 게시글 업로드: 사진 등록 시 모든 AI 데이터 즉시 생성 ✨
@app.post("/posts/")
async def create_post(
    user_id: int = Form(...), 
    title: str = Form(...), 
    artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None), 
    tags: Optional[str] = Form(None),
    genre: Optional[str] = Form("인상주의"), # 기본값
    style1: Optional[str] = Form("유화"),    # AI 분석에 사용될 주요 스타일
    style2: Optional[str] = Form(None), 
    style3: Optional[str] = Form(None), 
    style4: Optional[str] = Form(None), 
    style5: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    # 1. 이미지 S3 업로드
    image_url = upload_file_to_s3(image)
    if not image_url:
        raise HTTPException(status_code=500, detail="S3 업로드 실패")

    # 2. ✨ [즉시 실행 1] 그림 분석 (style1 사용)
    # 사진이 들어오자마자 분석을 돌려서 ai_summary를 확보합니다.
    ai_summary = None
    try:
        print(f"🖼️ [{title}] 그림 분석 중... (장르: {genre}, 스타일: {style1})")
        # style1 하나만 넘깁니다.
        vision_res = run_gemini_vision(image_url, title, artist_name, genre, style1)
        if vision_res:
            ai_summary = vision_res.get('art_review')
    except Exception as e:
        print(f"❌ 그림 분석 실패: {e}")

    # 3. ✨ [즉시 실행 2] 음악 프롬프트 생성 (music_prompt 생성)
    # 사용자의 설명이 없더라도 위에서 만든 ai_summary를 재료로 사용합니다.
    generated_prompt = None
    try:
        source_text = description or ai_summary or tags or "아름다운 예술 작품"
        print(f"🎵 [{title}] 음악 프롬프트 생성 중...")
        music_res = run_gemini_music(f"{source_text} / 태그: {tags or ''}", title, artist_name)
        if music_res:
            generated_prompt = music_res.get('music_prompt')
    except Exception as e:
        print(f"❌ 음악 프롬프트 생성 실패: {e}")

    # 4. DB 저장: 모든 style 컬럼 저장
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts 
            (user_id, title, artist_name, image_url, description, tags, ai_summary, music_prompt, genre, style1, style2, style3, style4, style5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (user_id, title, artist_name, image_url, description, tags, ai_summary, generated_prompt, genre, style1, style2, style3, style4, style5)
        cursor.execute(sql, val)
        conn.commit()
        
        return {
            "message": "등록 및 AI 분석 완료",
            "id": cursor.lastrowid,
            "ai_summary": ai_summary,
            "music_prompt": generated_prompt
        }
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"DB 저장 실패: {err}")
    finally:
        cursor.close(); conn.close()

# (4) 피드 조회: DB 내용만 빠르게 응답
@app.get("/posts/")
def get_posts():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT p.*, u.nickname FROM posts p JOIN users u ON p.user_id = u.id ORDER BY p.id DESC")
        return {"posts": cursor.fetchall()}
    finally:
        cursor.close(); conn.close()

# (5) 그림 분석 버튼 클릭 (style1 사용)
@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post: raise HTTPException(status_code=404, detail="게시글 없음")
        
        # 1. Gemini를 호출하여 분석 수행 (style1 사용)
        target_style = post.get('style1')
        target_genre = post.get('genre')
        
        ai_result = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], target_genre, target_style)
        
        if not ai_result: raise HTTPException(status_code=500, detail="AI 분석 실패")

        # 2. 결과 저장
        summary_text = ai_result.get('art_review', '')
        cursor.execute("UPDATE posts SET ai_summary = %s WHERE id = %s", (summary_text, post_id))
        conn.commit()
        
        return {
            "message": "분석 완료 및 저장 성공",
            "ai_summary": summary_text,
            "result": ai_result
        }
    finally:
        cursor.close(); conn.close()

# (6) 음악 프롬프트 수동 요청
@app.post("/posts/{post_id}/music")
def generate_music_prompt(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        
        # 재료 확인
        desc = post.get('description') or post.get('ai_summary')
        if not desc:
            raise HTTPException(status_code=400, detail="분석 결과나 설명이 없어 프롬프트를 만들 수 없습니다.")

        tags = post.get('tags') or ""
        music_res = run_gemini_music(f"{desc} / {tags}", post['title'], post['artist_name'])
        
        if music_res:
            prompt_text = music_res.get('music_prompt', '')
            cursor.execute("UPDATE posts SET music_prompt = %s WHERE id = %s", (prompt_text, post_id))
            conn.commit()
            return {"music_prompt": prompt_text, "explanation": music_res.get('explanation')}
    finally:
        cursor.close(); conn.close()

# (7) 음악 URL 등록 API
class MusicUrlUpdate(BaseModel):
    music_url: str

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

# (8) 수동 보정 API
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

            # [A] ai_summary 채우기 (style1 사용)
            if not post.get('ai_summary'):
                style = post.get('style1')
                genre = post.get('genre')
                res = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], genre, style)
                if res:
                    updates['ai_summary'] = res.get('art_review', '')
                    post['ai_summary'] = updates['ai_summary'] # 임시 갱신

            # [B] music_prompt 채우기
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
                sync_count += 1

        return {"message": f"총 {sync_count}건 보정 완료"}
    finally:
        cursor.close(); conn.close()

# --- [서버 시작 시] 백그라운드 자동 동기화 ---
async def startup_sync():
    """서버 시작 5초 후부터 비어있는 AI 데이터를 자동으로 채웁니다."""
    await asyncio.sleep(5)
    print("🚀 [System] 서버 시작: 누락된 AI 데이터 자동 보정 시작...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
        empty_posts = cursor.fetchall()

        if not empty_posts:
            print("✅ [System] 보정할 데이터가 없습니다.")
            return

        for post in empty_posts:
            post_id = post['id']
            updates = {}

            # 1. ai_summary 보정
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

            if updates:
                cols = ", ".join([f"{k} = %s" for k in updates.keys()])
                cursor.execute(f"UPDATE posts SET {cols} WHERE id = %s", list(updates.values()) + [post_id])
                conn.commit()
                print(f"✨ [System] ID {post_id}번 데이터 보정 완료")

        print(f"✅ [System] 총 {len(empty_posts)}건의 데이터 보정 프로세스 종료.")
    except Exception as e:
        print(f"❌ [System] 자동 보정 중 에러: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(startup_sync())
