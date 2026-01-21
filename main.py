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

# 1. 이미지 다운로드
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        return None

# 2. Gemini 그림 분석
def run_gemini_vision(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None
    
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    prompt = f"""
    당신은 미술 평론가입니다. 작품 정보를 바탕으로 분석하여 JSON으로 출력하세요.
    [정보] 제목: {title}, 작가: {artist}, 장르: {genre}, 화풍: {style}
    [출력 포맷 (JSON)]
    {{
        "artist_intro": "작가 설명 (2문장)",
        "title_meaning": "제목 의미 (2문장)",
        "art_review": "종합 감상평 (3문장)"
    }}
    """
    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Vision 에러: {e}")
        return None

# 3. Gemini 음악 프롬프트 생성 (제목/작가/태그 반영 ✨)
def run_gemini_music(description, title, artist):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    # 프롬프트에 제목과 작가 정보를 명시적으로 추가했습니다.
    prompt = f"""
    전문 음악 프롬프트 엔지니어로서, 아래 [작품 정보]를 바탕으로 AI 음악 생성 프롬프트를 작성하세요.
    제목과 작가가 주는 뉘앙스, 그리고 설명/태그의 분위기를 음악 스타일에 적극 반영하세요.

    [작품 정보]
    1. 제목: {title}
    2. 작가: {artist}
    3. 설명 및 태그: 
    {description}

    [출력 포맷 (JSON)]
    {{
        "mood": "...", "instruments": "...", "tempo": "...",
        "music_prompt": "실제 생성용 프롬프트 (영어)",
        "explanation": "추천 이유 (한글)"
    }}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)
        if isinstance(result, list): result = result[0]
        return result
    except Exception as e:
        print(f"Gemini Music 에러: {e}")
        return None


# --- DB 연결 함수 ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        return connection
    except mysql.connector.Error as err:
        print(f"DB 접속 에러: {err}")
        raise HTTPException(status_code=500, detail="Database connection failed")


# --- API 엔드포인트 ---

@app.get("/")
def read_root():
    return {"message": "Art App Backend is Live!"}

class UserCreate(BaseModel):
    username: str
    password: str
    nickname: str

class UserLogin(BaseModel):
    username: str
    password: str

# (1) 회원가입
@app.post("/users/signup")
def signup(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO users (username, password, nickname) VALUES (%s, %s, %s)"
        cursor.execute(sql, (user.username, user.password, user.nickname))
        conn.commit()
        return {"message": "가입 성공", "id": cursor.lastrowid, "nickname": user.nickname}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"가입 실패: {err}")
    finally:
        cursor.close()
        conn.close()

# (2) 로그인
@app.post("/users/login")
def login(user: UserLogin):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        sql = "SELECT id, nickname FROM users WHERE username = %s AND password = %s"
        cursor.execute(sql, (user.username, user.password))
        result = cursor.fetchone()
        
        if result:
            return {"message": "로그인 성공", "user_id": result['id'], "nickname": result['nickname']}
        else:
            raise HTTPException(status_code=401, detail="로그인 실패")
    finally:
        cursor.close()
        conn.close()

# (3) 게시글 업로드: 사진 등록 시 음악 프롬프트 자동 생성 및 저장 ✨
@app.post("/posts/")
async def create_post(
    user_id: int = Form(...),
    title: str = Form(...),
    artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None), 
    ai_summary: Optional[str] = Form(None),
    music_url: Optional[str] = Form(None),
    rating: int = Form(5),
    image: UploadFile = File(...)
):
    # 1. 이미지 S3 업로드
    image_url = upload_file_to_s3(image)
    if not image_url:
        raise HTTPException(status_code=500, detail="S3 업로드 실패")

    # 2. ✨ [자동 생성] 사진 등록과 동시에 음악 프롬프트 제작
    generated_prompt = None
    
    # 프롬프트 생성을 위한 소스 데이터 준비 (설명이나 태그 중 하나라도 있으면 실행)
    if description or tags:
        try:
            # 제목, 작가, 감상평, 태그를 모두 조합하여 AI에게 전달
            input_context = f"감상평: {description or ''} / 태그: {tags or ''}"
            
            # Gemini Music 함수 호출
            music_result = run_gemini_music(input_context, title, artist_name)
            
            if music_result:
                generated_prompt = music_result.get('music_prompt')
                print(f"✅ 자동 생성된 프롬프트: {generated_prompt}")
        except Exception as e:
            print(f"❌ 음악 프롬프트 자동 생성 중 오류 발생: {e}")
            # 생성에 실패하더라도 업로드는 계속 진행되도록 예외 처리

    # 3. DB 저장 (생성된 프롬프트를 music_prompt 컬럼에 함께 넣음)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts 
            (user_id, title, artist_name, image_url, description, tags, ai_summary, music_url, rating, music_prompt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (
            user_id, title, artist_name, image_url, description, tags, 
            ai_summary, music_url, rating, generated_prompt
        )
        cursor.execute(sql, val)
        conn.commit()
        
        return {
            "message": "사진 등록 및 음악 프롬프트 생성 완료",
            "id": cursor.lastrowid,
            "image_url": image_url,
            "music_prompt": generated_prompt
        }
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"DB 저장 실패: {err}")
    finally:
        cursor.close()
        conn.close()

# (4) 피드 조회
@app.get("/posts/")
def get_posts():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        sql = """
            SELECT p.*, u.nickname 
            FROM posts p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.id DESC
        """
        cursor.execute(sql)
        posts = cursor.fetchall()
        return {"posts": posts}
    finally:
        cursor.close()
        conn.close()

# (5) 그림 분석 요청 (Style1 컬럼 반영 ✨)
@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int, genre: str = Form("인상주의"), style: str = Form("유화")):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 1. 게시글 정보 가져오기
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        
        # 2. ✨ [핵심 수정] DB에 저장된 'style1'과 'genre' 값이 있으면 그걸 우선 사용
        # (DB 값이 없으면, API 요청 시 받은 기본값 genre, style을 사용)
        db_style = post.get('style1')
        db_genre = post.get('genre')

        target_style = db_style if db_style else style
        target_genre = db_genre if db_genre else genre

        print(f"🤖 AI 분석 시작: {post['title']} | 화풍: {target_style}, 장르: {target_genre}")

        # 3. Gemini Vision 실행 (수정된 style 정보 전달)
        ai_result = run_gemini_vision(
            post['image_url'], 
            post['title'], 
            post['artist_name'], 
            target_genre, 
            target_style
        )
        
        if not ai_result:
            raise HTTPException(status_code=500, detail="AI 분석 실패")

        # 4. 결과 저장
        summary_text = ai_result.get('art_review', '')
        
        update_sql = "UPDATE posts SET ai_summary = %s WHERE id = %s"
        cursor.execute(update_sql, (summary_text, post_id))
        conn.commit()
        
        return {"message": "분석 완료", "result": ai_result}
        
    finally:
        conn.close()

# (6) 음악 프롬프트 요청 (Tags + Title + Artist 반영 ✨)
@app.post("/posts/{post_id}/music")
def generate_music_prompt(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # DB에서 게시글 정보 가져오기
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="게시글 없음")

        # 1. 감상평 가져오기
        description = post['description']
        if not description or len(description) < 5:
            description = post['ai_summary'] 
        
        if not description:
            raise HTTPException(status_code=400, detail="감상평이나 AI 분석 결과가 필요합니다.")

        # 2. 정보 조합 (설명 + 태그)
        tags = post.get('tags')
        ai_input_text = f"작품 감상: {description}"
        if tags:
            ai_input_text += f"\n관련 키워드/태그(Tags): {tags}"
        
        # 3. 추가 정보 (제목, 작가) 가져오기 ✨
        title = post.get('title', '제목 미상')
        artist = post.get('artist_name', '작가 미상')

        print(f"🎵 AI 입력 프롬프트: 제목[{title}], 작가[{artist}], 내용[{ai_input_text}]")

        # 4. Gemini Music 실행 (제목과 작가도 함께 전달) ✨
        music_result = run_gemini_music(ai_input_text, title, artist)
        
        if not music_result:
            raise HTTPException(status_code=500, detail="음악 프롬프트 생성 실패")

        # 5. DB 저장
        prompt_text = music_result.get('music_prompt', '')
        update_sql = "UPDATE posts SET music_prompt = %s WHERE id = %s"
        cursor.execute(update_sql, (prompt_text, post_id))
        conn.commit()
        
        return {"message": "생성 완료", "result": music_result}

    finally:
        conn.close()

# (7) 음악 URL 등록 API (테스트용/로컬 AI 연동용)
class MusicUrlUpdate(BaseModel):
    music_url: str

@app.post("/posts/{post_id}/register_music_url")
def register_music_url(post_id: int, body: MusicUrlUpdate):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="게시글 없음")

        sql = "UPDATE posts SET music_url = %s WHERE id = %s"
        cursor.execute(sql, (body.music_url, post_id))
        conn.commit()

        return {"message": "등록 완료", "music_url": body.music_url}
    finally:
        conn.close()
