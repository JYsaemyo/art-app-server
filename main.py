from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import os
from dotenv import load_dotenv
from typing import Optional

# 1. 환경 변수 로드
load_dotenv()

app = FastAPI()

# 2. CORS 설정 (중요: 이게 없으면 프론트엔드에서 접속 거부당함)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 곳에서의 접속 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 데이터베이스 연결 함수
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            # 클라우드 DB 접속 시 SSL 설정 (필수)
            ssl_ca="/etc/ssl/certs/ca-certificates.crt" if os.name != 'nt' else None
        )
        return connection
    except mysql.connector.Error as err:
        print(f"DB 접속 에러: {err}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# 4. 데이터 모델 정의 (Pydantic)
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    nickname: str

class PostCreate(BaseModel):
    user_id: int
    title: str
    artist_name: Optional[str] = "작가 미상"
    image_url: str
    description: Optional[str] = None  # 사용자 감상평
    ai_summary: Optional[str] = None   # AI 요약
    music_url: Optional[str] = None    # 생성된 음악
    rating: int = 5

# --- [API 엔드포인트] ---

@app.get("/")
def read_root():
    return {"message": "Art App Backend is Live!"}

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

# (3) 게시글 업로드 (음악, 감상평 포함)
@app.post("/posts/")
def create_post(post: PostCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = """
            INSERT INTO posts 
            (user_id, title, artist_name, image_url, description, ai_summary, music_url, rating)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (post.user_id, post.title, post.artist_name, post.image_url, 
               post.description, post.ai_summary, post.music_url, post.rating)
        
        cursor.execute(sql, val)
        conn.commit()
        return {"message": "업로드 성공", "id": cursor.lastrowid}
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
