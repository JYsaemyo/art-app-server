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
import google.generativeai as genai  # ✨ Gemini 라이브러리 추가
from dotenv import load_dotenv
from typing import Optional

# 1. 환경 변수 로드
load_dotenv()

# --- ✨ [추가됨] Gemini API 설정 ---
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

# --- ✨ [추가됨] AI 헬퍼 함수들 (이미지 다운로드 & Gemini 호출) ---

# 1. URL에서 이미지 다운로드 (S3 이미지를 Gemini에게 넘겨주기 위해 필요)
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        return None

# 2. Gemini 그림 분석 실행 함수
def run_gemini_vision(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None
    
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    prompt = f"""
    당신은 사려 깊고 관찰력이 뛰어난 미술 평론가입니다. 
    제공된 이미지와 정보를 바탕으로 작품을 분석하여 JSON 형식으로 출력하세요.

    [지침]
    1. 단정적인 표현 대신 추측성 어조 사용 ("~인 것 같습니다").
    2. 정중하고 감성적인 문체 사용.
    3. 한국어로 출력.

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

# 3. Gemini 음악 프롬프트 생성 함수
def run_gemini_music(description):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    prompt = f"""
    전문 음악 프롬프트 엔지니어로서, 아래 감상평을 바탕으로 AI 음악 생성 프롬프트를 작성하세요.

    [감상평] {description}

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


# --- 데이터베이스 연결 함수 ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            # ssl_ca="/etc/ssl/certs/ca-certificates.crt" if os.name != 'nt' else None
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
            raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    finally:
        cursor.close()
        conn.close()

# (3) 게시글 업로드
@app.post("/posts/")
def create_post(
    user_id: int = Form(...),
    title: str = Form(...),
    artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None),
    ai_summary: Optional[str] = Form(None),
    music_url: Optional[str] = Form(None),
    rating: int = Form(5),
    image: UploadFile = File(...)
):
    image_url = upload_file_to_s3(image)
    if not image_url:
        raise HTTPException(status_code=500, detail="S3 업로드 실패")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts 
            (user_id, title, artist_name, image_url, description, ai_summary, music_url, rating)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (user_id, title, artist_name, image_url, description, ai_summary, music_url, rating)
        cursor.execute(sql, val)
        conn.commit()
        return {"message": "업로드 성공", "id": cursor.lastrowid, "image_url": image_url}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"업로드 실패: {err}")
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

# --- ✨ [추가된 API 1] 그림 분석 요청 ---
# 프론트엔드에서: POST /posts/{id}/analyze (body: genre, style)
@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int, genre: str = Form("인상주의"), style: str = Form("유화")):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 1. 게시글 정보 가져오기 (이미지 URL 확인)
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        
        # 2. Gemini Vision 실행
        print(f"🤖 AI 분석 시작: {post['title']}")
        ai_result = run_gemini_vision(post['image_url'], post['title'], post['artist_name'], genre, style)
        
        if not ai_result:
            raise HTTPException(status_code=500, detail="AI 분석 실패")

        # 3. DB에 저장 (ai_summary 컬럼 업데이트)
        # JSON 결과 중 'art_review'(감상평)를 뽑아서 저장합니다.
        summary_text = ai_result.get('art_review', '')
        
        update_sql = "UPDATE posts SET ai_summary = %s WHERE id = %s"
        cursor.execute(update_sql, (summary_text, post_id))
        conn.commit()
        
        return {"message": "분석 완료", "result": ai_result}
        
    finally:
        conn.close()

# --- ✨ [추가된 API 2] 음악 프롬프트 요청 ---
# 프론트엔드에서: POST /posts/{id}/music
@app.post("/posts/{post_id}/music")
def generate_music_prompt(post_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        post = cursor.fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

        # 1. 감상평 가져오기 (사용자가 쓴 게 없으면 방금 만든 AI 요약본 사용)
        description = post['description']
        if not description or len(description) < 5:
            description = post['ai_summary'] 
        
        if not description:
            raise HTTPException(status_code=400, detail="감상평(description)이나 AI 분석 결과(ai_summary)가 없습니다.")

        # 2. Gemini Music 실행
        print(f"🎵 음악 프롬프트 생성 시작: {post['title']}")
        music_result = run_gemini_music(description)
        
        if not music_result:
            raise HTTPException(status_code=500, detail="음악 프롬프트 생성 실패")

        # 3. DB에 저장 (music_prompt 컬럼 업데이트)
        prompt_text = music_result.get('music_prompt', '')
        
        update_sql = "UPDATE posts SET music_prompt = %s WHERE id = %s"
        cursor.execute(update_sql, (prompt_text, post_id))
        conn.commit()
        
        return {"message": "생성 완료", "result": music_result}

    finally:
        conn.close()
