from fastapi import FastAPI, HTTPException, Form, UploadFile, File # File, UploadFile, Form 추가됨
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import os
import boto3 # AWS S3용 라이브러리 추가
import uuid  # 파일명 중복 방지용
from dotenv import load_dotenv
from typing import Optional

# 1. 환경 변수 로드
load_dotenv()

app = FastAPI()

# 2. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [추가됨] AWS S3 설정 초기화 ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION = os.getenv("AWS_REGION")

# --- [추가됨] S3 업로드 헬퍼 함수 ---
def upload_file_to_s3(file: UploadFile):
    try:
        # 파일명 중복 방지를 위해 UUID 사용 (예: a1b2.jpg)
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # S3 업로드 (ExtraArgs는 브라우저에서 이미지가 바로 보이게 함)
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            unique_filename,
            ExtraArgs={'ContentType': file.content_type}
        )
        # 업로드된 URL 생성해서 반환
        return f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{unique_filename}"
    except Exception as e:
        print(f"❌ S3 업로드 에러: {e}")
        return None


# 3. 데이터베이스 연결 함수 (기존 동일)
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            ssl_ca="/etc/ssl/certs/ca-certificates.crt" if os.name != 'nt' else None
        )
        return connection
    except mysql.connector.Error as err:
        print(f"DB 접속 에러: {err}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# 4. 데이터 모델 정의 (기존 동일 - 참고용)
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    nickname: str

# PostCreate 모델은 파일 업로드 시 Form 데이터로 대체되므로 여기선 쓰이지 않지만 남겨둡니다.

# --- [API 엔드포인트] ---

@app.get("/")
def read_root():
    return {"message": "Art App Backend is Live!"}

# (1) 회원가입 (기존 동일)
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

# (2) 로그인 (기존 동일)
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

# (3) [수정됨] 게시글 업로드 (S3 연동)
# 기존 PostCreate 모델 대신 Form(...)과 UploadFile을 사용해야 합니다.
@app.post("/posts/")
def create_post(
    user_id: int = Form(...),
    title: str = Form(...),
    artist_name: Optional[str] = Form("작가 미상"),
    description: Optional[str] = Form(None),
    ai_summary: Optional[str] = Form(None),
    music_url: Optional[str] = Form(None),
    rating: int = Form(5),
    image: UploadFile = File(...)  # 여기가 핵심! 파일을 직접 받음
):
    # 1. 먼저 이미지를 S3에 업로드하고 URL을 받아옵니다.
    image_url = upload_file_to_s3(image)
    
    if not image_url:
        raise HTTPException(status_code=500, detail="이미지 S3 업로드 실패")

    # 2. 받아온 URL을 DB에 저장합니다.
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts 
            (user_id, title, artist_name, image_url, description, ai_summary, music_url, rating)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        # post.image_url 대신 방금 만든 image_url 변수를 넣습니다.
        val = (user_id, title, artist_name, image_url, description, ai_summary, music_url, rating)
        
        cursor.execute(sql, val)
        conn.commit()
        return {"message": "업로드 성공", "id": cursor.lastrowid, "image_url": image_url}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"업로드 실패: {err}")
    finally:
        cursor.close()
        conn.close()

# (4) 피드 조회 (기존 동일)
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
