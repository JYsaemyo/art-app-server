from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import os
import uvicorn
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
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from datetime import date as date_type

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
    # 브라우저(웹)에서는 allow_credentials=True + allow_origins="*" 조합이 차단되어
    # CORS가 "Network Error"로 보일 수 있습니다. (쿠키 기반 인증도 현재 사용하지 않음)
    allow_credentials=False,
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

# --- Pydantic Models ---
class MusicUrlUpdate(BaseModel):
    music_url: str

# --- [추가] Admin Auth Pydantic Models ---
class AdminRegisterIn(BaseModel):
    email: str
    name: str
    password: str

class AdminLoginIn(BaseModel):
    email: str
    password: str

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
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME")

    db_ssl = (os.getenv("DB_SSL", "") or "").strip().lower() in ("1", "true", "yes", "y")

    kwargs = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }

    # TiDB Cloud 등에서 TLS가 강제인 경우가 많아 옵션을 반영합니다.
    # CA 경로를 따로 주지 않는 환경도 있어, 우선 verify는 끈 형태로 연결합니다.
    if db_ssl:
        kwargs["ssl_disabled"] = False
        kwargs["ssl_verify_cert"] = False

    return mysql.connector.connect(**kwargs)

# --- [추가] Admin Auth Helpers ---

_ADMIN_PBKDF2_ITERATIONS = int(os.getenv("ADMIN_PBKDF2_ITERATIONS", "200000"))
_ADMIN_SESSION_TTL_HOURS = int(os.getenv("ADMIN_SESSION_TTL_HOURS", str(24 * 7)))

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def hash_password(password: str) -> str:
    if not password or len(password) < 4:
        raise HTTPException(400, "비밀번호가 너무 짧습니다.")

    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _ADMIN_PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("utf-8")
    return f"pbkdf2_sha256${_ADMIN_PBKDF2_ITERATIONS}${salt_b64}${dk_b64}"

def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iters_str, salt_b64, dk_b64 = (password_hash or "").split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iters_str)
        salt = base64.urlsafe_b64decode(salt_b64.encode("utf-8"))
        expected = base64.urlsafe_b64decode(dk_b64.encode("utf-8"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False

def ensure_admin_auth_tables():
    """
    기존 기능에 영향 없이, admin 전용 인증 테이블만 준비합니다.
    (CREATE TABLE IF NOT EXISTS 만 사용)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                password_hash VARCHAR(512) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                token VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                revoked TINYINT(1) NOT NULL DEFAULT 0,
                INDEX idx_admin_sessions_user_id (user_id),
                INDEX idx_admin_sessions_expires_at (expires_at),
                CONSTRAINT fk_admin_sessions_user_id
                    FOREIGN KEY (user_id) REFERENCES admin_users(id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        conn.commit()
        print("✅ [Auth] admin_users/admin_sessions 테이블 준비 완료")
    finally:
        cursor.close()
        conn.close()

def ensure_admin_demo_tables():
    """
    통계/알림용 테이블을 안전하게 준비합니다.
    (CREATE TABLE IF NOT EXISTS 만 사용)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exhibition_daily_usage (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exhibition_id INT NOT NULL,
                date DATE NOT NULL,
                count INT NOT NULL DEFAULT 0,
                UNIQUE KEY uniq_exhibition_date (exhibition_id, date),
                INDEX idx_exhibition_date (exhibition_id, date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_purchase_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exhibition_id INT NOT NULL,
                art_title VARCHAR(255) NOT NULL,
                buyer_name VARCHAR(255) NOT NULL,
                price INT NOT NULL DEFAULT 0,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_alerts_exhibition (exhibition_id),
                INDEX idx_alerts_status_created (status, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        conn.commit()
        print("✅ [Demo] exhibition_daily_usage/admin_purchase_alerts 테이블 준비 완료")
    finally:
        cursor.close()
        conn.close()

def _normalize_purchase_status(value: str) -> str:
    """
    DB status 값을 프론트(StatusBadge)에서 쓰는 소문자 형태로 통일합니다.
    """
    s = (value or "").strip().lower()
    if s in ("approved", "accept", "accepted", "ok", "y", "yes"):
        return "approved"
    if s in ("rejected", "reject", "denied", "no", "n"):
        return "rejected"
    if s in ("pending", "wait", "waiting"):
        return "pending"
    # 기존 백엔드에서 사용하던 형태(APPROVED/REJECTED)도 처리
    if s == "approved":
        return "approved"
    if s == "rejected":
        return "rejected"
    return "pending"

def _get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(401, "Authorization 헤더가 필요합니다.")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(401, "Authorization 형식이 올바르지 않습니다. (Bearer <token>)")
    return parts[1].strip()

def create_admin_session(conn, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=_ADMIN_SESSION_TTL_HOURS)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO admin_sessions (user_id, token, expires_at, revoked) VALUES (%s, %s, %s, %s)",
            (user_id, token, expires_at, 0),
        )
        conn.commit()
        return token
    finally:
        cursor.close()

def require_admin_user(authorization: Optional[str] = Header(None)):
    token = _get_bearer_token(authorization)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT u.id, u.email, u.name
            FROM admin_sessions s
            JOIN admin_users u ON u.id = s.user_id
            WHERE s.token = %s
              AND s.revoked = 0
              AND s.expires_at > UTC_TIMESTAMP()
            LIMIT 1
            """,
            (token,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(401, "세션이 만료되었거나 유효하지 않습니다.")
        return row
    finally:
        cursor.close()
        conn.close()

# --- AI Core Functions ---

# 1. 그림 분석 (Style 집중 & 이미지 증거 찾기)
def run_gemini_vision(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    style_text = style if style else "특별히 지정되지 않은 화풍"
    
    if genre in ["그림", "조각", "Painting", "Sculpture", "유화", "수채화", "동양화", "드로잉", "일러스트", "판화"]:
        prompt_context = f"""
        이 작품의 장르는 '{genre}'이며, **가장 핵심적인 화풍(Style)은 '{style_text}'**입니다.
        
        **[분석 미션]**
        당신은 '{style_text}' 전문 비평가입니다. 텍스트 정보에 의존하지 말고, **이미지(Picture)**에서 '{style_text}' 양식의 시각적 증거를 찾아내세요.
        1. **화풍의 정의**: 이미지 속 붓터치, 질감, 색채 사용이 '{style_text}'의 전형적인 특징과 어떻게 일치하는지 묘사하세요.
        2. **기법 분석**: 작가가 이 스타일을 표현하기 위해 사용한 재료적/기법적 시도를 분석하세요.
        3. **비평**: 이 화풍이 작품의 주제를 전달하는 데 어떤 효과를 주는지 평가하세요.
        """
    else:
        prompt_context = f"""
        이 작품은 '{genre}' 장르입니다. (스타일 참고: {style_text})
        스타일보다는 **이미지 자체의 시각적 연출(구도, 빛, 분위기)**과 제목 '{title}'의 상징적 연결성을 분석하세요.
        """

    prompt = f"""
    당신은 통찰력 있는 예술 큐레이터입니다.
    제공된 **이미지(사진)**를 면밀히 분석하되, **주어진 스타일 정보('{style_text}')를 분석의 기준으로 삼으세요.**

    [작품 정보] 제목:{title}, 작가:{artist}, 장르:{genre}, 스타일:{style_text}
    [지침] {prompt_context}

    [출력 포맷 (JSON)]
    * 답변은 한국어 경어체(~합니다)로 작성하세요.
    {{
        "artist_intro": "작가와 해당 스타일의 관계를 설명하는 소개 (2문장)",
        "title_meaning": "제목이 스타일 및 이미지와 어떻게 연결되는지 해석 (2문장)",
        "art_review": "화풍('{style_text}')의 특징이 이미지에서 어떻게 드러나는지 구체적으로 서술한 비평 (3~4문장)"
    }}
    """
    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Vision 에러: {e}"); return None

# 2. 음악 프롬프트 생성 (이미지 + 태그 + 설명 반영)
def run_gemini_music(image_url, description, title, artist, tags):
    img = load_image_from_url(image_url)
    if not img: return None
    
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    prompt = f"""
    당신은 영화 음악 감독(Film Scorer)입니다. 
    **제공된 이미지(Picture)**를 보고, 그 시각적 분위기를 소리로 번역(Sonification)하세요.
    동시에 제공된 설명과 **태그(Tags)** 정보도 참고하여 음악 생성 AI용 프롬프트를 작성하세요.
    
    [입력 정보]
    - 시각 자료: (첨부된 이미지)
    - 작품 제목/작가: {title} / {artist}
    - 작품 설명: {description}
    - **사용자 태그(Tags): {tags}**
    
    [지침]
    1. **시각-청각 변환**: 이미지의 색감이 차가우면 Cool pad/Reverb를, 거칠면 Distortion/Staccato를 매칭하세요.
    2. **태그 반영**: 태그({tags})가 있다면 그 키워드를 music_prompt에 적극 반영하세요.
    3. **music_prompt**: Suno/MusicGen이 이해하기 쉬운 **영어 키워드(Tag)** 위주로 작성하세요.

    [출력 포맷 (JSON)]
    {{
        "mood": "분위기 (한글)",
        "instruments": "주요 악기 (한글)",
        "tempo": "템포 (예: Adagio, 80 BPM)",
        "music_prompt": "음악 생성용 영어 프롬프트 (High quality, Cinematic, ...)",
        "explanation": "이미지와 태그를 보고 이 음악을 추천한 이유 (한글 1문장)"
    }}
    """
    
    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        res = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        return res if not isinstance(res, list) else res[0]
    except Exception as e:
        print(f"Gemini Music 에러: {e}"); return None

# --- 🛡️ [통합 로직] AI 처리 및 데이터 보호 함수 ---
def process_ai_logic(post_id: int, image_url: str, title: str, artist: str, genre: str, style1: str, description: str, tags: str, force_update: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT ai_summary, music_prompt FROM posts WHERE id = %s", (post_id,))
        current_data = cursor.fetchone()
        
        if not current_data: return

        # 1. 그림 분석
        if not current_data['ai_summary'] or force_update:
            print(f"🖌️ [Processing] ID {post_id} 그림 분석 시작...")
            vision_res = run_gemini_vision(image_url, title, artist, genre, style1)
            
            if vision_res:
                summary = vision_res.get('art_review', '')
                if force_update:
                    sql = "UPDATE posts SET ai_summary = %s WHERE id = %s"
                else:
                    sql = "UPDATE posts SET ai_summary = %s WHERE id = %s AND (ai_summary IS NULL OR ai_summary = '')"
                cursor.execute(sql, (summary, post_id))
                conn.commit()
                current_data['ai_summary'] = summary
        else:
            print(f"🛡️ [Protected] ID {post_id} 그림 분석 데이터 보존됨.")

        # 2. 음악 프롬프트 생성
        if not current_data['music_prompt'] or force_update:
            desc_text = description or current_data['ai_summary'] or "예술 작품"
            tag_text = tags or ""
            
            print(f"🎵 [Processing] ID {post_id} 음악 프롬프트 생성 시작...")
            music_res = run_gemini_music(image_url, desc_text, title, artist, tag_text)
            
            if music_res:
                prompt = music_res.get('music_prompt')
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

# --- [추가] Admin Auth Endpoints ---

@app.post("/auth/register")
def auth_register(body: AdminRegisterIn):
    email = _normalize_email(body.email)
    name = (body.name or "").strip()
    password = body.password or ""

    if not email or "@" not in email:
        raise HTTPException(400, "이메일 형식이 올바르지 않습니다.")
    if not name:
        raise HTTPException(400, "이름이 필요합니다.")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor2 = conn.cursor()
    try:
        cursor.execute("SELECT id FROM admin_users WHERE email = %s LIMIT 1", (email,))
        if cursor.fetchone():
            raise HTTPException(409, "이미 가입된 이메일입니다.")

        pw_hash = hash_password(password)
        cursor2.execute(
            "INSERT INTO admin_users (email, name, password_hash) VALUES (%s, %s, %s)",
            (email, name, pw_hash),
        )
        conn.commit()
        user_id = cursor2.lastrowid

        token = create_admin_session(conn, user_id)
        return {"token": token, "user": {"id": user_id, "email": email, "name": name}}
    except mysql.connector.IntegrityError:
        # 레이스 컨디션 등으로 UNIQUE 충돌 시
        raise HTTPException(409, "이미 가입된 이메일입니다.")
    finally:
        cursor.close()
        cursor2.close()
        conn.close()

@app.post("/auth/login")
def auth_login(body: AdminLoginIn):
    email = _normalize_email(body.email)
    password = body.password or ""

    if not email or "@" not in email:
        raise HTTPException(400, "이메일 형식이 올바르지 않습니다.")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, email, name, password_hash FROM admin_users WHERE email = %s LIMIT 1",
            (email,),
        )
        user = cursor.fetchone()
        if not user or not verify_password(password, user.get("password_hash")):
            raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")

        token = create_admin_session(conn, int(user["id"]))
        return {"token": token, "user": {"id": int(user["id"]), "email": user["email"], "name": user["name"]}}
    finally:
        cursor.close()
        conn.close()

@app.get("/auth/me")
def auth_me(user=Depends(require_admin_user)):
    return {"user": {"id": int(user["id"]), "email": user["email"], "name": user["name"]}}

@app.post("/auth/logout")
def auth_logout(authorization: Optional[str] = Header(None)):
    token = _get_bearer_token(authorization)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE admin_sessions SET revoked = 1 WHERE token = %s", (token,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(401, "세션이 유효하지 않습니다.")
        return {"message": "ok"}
    finally:
        cursor.close()
        conn.close()

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
    image_url = upload_file_to_s3(image)
    if not image_url: raise HTTPException(500, "S3 실패")

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
        
        # [빠른 응답] 즉시 트리거 (실패해도 스케줄러가 30초 안에 처리함)
        background_tasks.add_task(
            process_ai_logic, 
            new_post_id, image_url, title, artist_name, genre, style1, description, tags,
            True 
        )
        
        return {"message": "업로드 완료. AI 분석 시작됨.", "id": new_post_id}
        
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

@app.post("/posts/{post_id}/analyze")
def analyze_art(post_id: int, force_update: bool = False):
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
        
        cursor.execute("SELECT ai_summary FROM posts WHERE id = %s", (post_id,))
        updated_post = cursor.fetchone()
        return {"message": "완료", "ai_summary": updated_post['ai_summary']}
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
        return {"message": "완료", "music_prompt": updated_post['music_prompt']}
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

@app.post("/posts/sync-ai")
def sync_missing_ai_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
        empty_posts = cursor.fetchall()
        if not empty_posts: return {"message": "최신 상태입니다."}

        for post in empty_posts:
            process_ai_logic(
                post['id'], post['image_url'], post['title'], post['artist_name'], 
                post['genre'], post['style1'], post['description'], post['tags'],
                False 
            )
        return {"message": f"{len(empty_posts)}건 요청 완료"}
    finally:
        cursor.close(); conn.close()

# --- ⏰ [수정됨] 30초 주기 무조건 스위핑 ---
async def periodic_sync_task():
    print("⏰ [Scheduler] 30초 주기 자동 보정 스케줄러가 시작되었습니다.")
    while True:
        try:
            # 60초 -> 30초로 단축 (더 자주 체크)
            await asyncio.sleep(30)
            
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            # 비어있는 것만 찾아서 채움 (누락 방지)
            cursor.execute("SELECT * FROM posts WHERE ai_summary IS NULL OR music_prompt IS NULL")
            empty_posts = cursor.fetchall()
            
            if empty_posts:
                print(f"🔍 [Scheduler] {len(empty_posts)}건 발견. 보정 시작...")
                for post in empty_posts:
                    process_ai_logic(
                        post['id'], post['image_url'], post['title'], post['artist_name'], 
                        post['genre'], post['style1'], post['description'], post['tags'],
                        False # 안전 모드 (이미 있으면 패스)
                    )
            cursor.close(); conn.close()
        except Exception as e:
            print(f"⚠️ [Scheduler] 에러 발생 (재시도): {e}")

@app.on_event("startup")
async def on_startup():
    try:
        ensure_admin_auth_tables()
    except Exception as e:
        # auth 테이블 준비 실패가 기존 기능까지 죽이지 않도록 보호
        print(f"⚠️ [Auth] 테이블 준비 실패: {e}")
    try:
        ensure_admin_demo_tables()
    except Exception as e:
        print(f"⚠️ [Demo] 테이블 준비 실패: {e}")
    asyncio.create_task(periodic_sync_task())


if __name__ == "__main__":
    # ✅ `python main.py`로도 바로 실행 가능하게 엔트리포인트를 추가합니다.
    # - host=0.0.0.0 : 실기기/에뮬레이터에서 PC로 접근 가능
    # - port : .env의 PORT를 사용하되, 없으면 8000
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

# --- [추가] Admin 전용 Pydantic Models ---
class ExhibitionCreate(BaseModel):
    title: str
    date: str
    location: str
    description: Optional[str] = None

class ArtworkCreate(BaseModel):
    exhibition_id: int
    title: str
    artist_name: str
    price: int
    image_url: str
    nfc_uuid: str

class PurchaseStatusUpdate(BaseModel):
    status: str  # 'APPROVED' or 'REJECTED'

# --- [추가] Admin 전용 Update Models ---
class ExhibitionUpdate(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None

# --- 🚀 [Admin] 1. 전시회 관리 함수 섹션 ---

# 모든 전시회 목록 조회 (사용자 태깅 수 계산 포함)
@app.get("/admin/exhibitions/")
def get_admin_exhibitions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 전시회 제목과 posts 테이블의 제목을 매칭하여 '전체 태그' 수를 실시간 집계합니다.
        sql = """
            SELECT
                e.*,
                COALESCE(
                    (SELECT SUM(u.count) FROM exhibition_daily_usage u WHERE u.exhibition_id = e.id),
                    COUNT(p.id)
                ) as total_tags
            FROM exhibitions e 
            LEFT JOIN posts p ON p.title = e.title 
            GROUP BY e.id ORDER BY e.id DESC
        """
        cursor.execute(sql)
        return cursor.fetchall()
    finally: cursor.close(); conn.close()

# 새 전시회 생성 (중앙 + 버튼 연동)
@app.post("/admin/exhibitions/")
def create_exhibition(ex: ExhibitionCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO exhibitions (title, date, location, description) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (ex.title, ex.date, ex.location, ex.description))
        conn.commit()
        return {"id": cursor.lastrowid, "message": "전시회 정보가 등록되었습니다."}
    finally: cursor.close(); conn.close()

# 전시회 정보 수정
@app.put("/admin/exhibitions/{ex_id}")
def update_exhibition(ex_id: int, body: ExhibitionUpdate):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM exhibitions WHERE id = %s", (ex_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(404, "전시회를 찾을 수 없습니다.")

        new_title = body.title if body.title is not None else existing.get("title")
        new_date = body.date if body.date is not None else existing.get("date")
        new_location = body.location if body.location is not None else existing.get("location")
        new_description = body.description if body.description is not None else existing.get("description")

        # 업데이트할 값이 하나도 없으면 그대로 반환
        if (
            body.title is None
            and body.date is None
            and body.location is None
            and body.description is None
        ):
            return {"message": "변경 사항이 없습니다.", "id": ex_id}

        cursor2 = conn.cursor()
        sql = """
            UPDATE exhibitions
            SET title = %s, date = %s, location = %s, description = %s
            WHERE id = %s
        """
        cursor2.execute(sql, (new_title, new_date, new_location, new_description, ex_id))
        conn.commit()
        cursor2.close()

        return {"message": "전시회 정보가 수정되었습니다.", "id": ex_id}
    finally:
        cursor.close(); conn.close()

# 전시회 상세 조회 (필요 시 프론트에서 사용)
@app.get("/admin/exhibitions/{ex_id}")
def get_exhibition_detail(ex_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM exhibitions WHERE id = %s", (ex_id,))
        ex = cursor.fetchone()
        if not ex:
            raise HTTPException(404, "전시회를 찾을 수 없습니다.")
        return ex
    finally:
        cursor.close(); conn.close()

# 특정 전시회 상세 통계 (Google Analytics 스타일)
@app.get("/admin/exhibitions/{ex_id}/stats")
def get_exhibition_analytics(ex_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT title FROM exhibitions WHERE id = %s", (ex_id,))
        ex = cursor.fetchone()
        if not ex: raise HTTPException(404, "전시회를 찾을 수 없습니다.")

        # 1) ✅ DB에 직접 저장된 "전시회 일별 이용추이"가 있으면 그걸 우선 사용
        try:
            start_date = (datetime.utcnow().date() - timedelta(days=6)).strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT date, count
                FROM exhibition_daily_usage
                WHERE exhibition_id = %s
                  AND date >= %s
                ORDER BY date ASC
                LIMIT 7
                """,
                (ex_id, start_date),
            )
            rows = cursor.fetchall() or []
            if rows:
                out = []
                for r in rows:
                    d = r.get("date")
                    if isinstance(d, (datetime, date_type)):
                        d_str = d.strftime("%Y-%m-%d")
                    else:
                        d_str = str(d)
                    out.append({"date": d_str, "count": int(r.get("count") or 0)})
                return {"title": ex["title"], "daily_stats": out, "source": "exhibition_daily_usage"}
        except Exception:
            pass

        # 2) fallback: posts 기반 (기존 로직)
        sql = """
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM posts WHERE title = %s 
            GROUP BY DATE(created_at) ORDER BY date ASC LIMIT 7
        """
        cursor.execute(sql, (ex['title'],))
        return {"title": ex['title'], "daily_stats": cursor.fetchall(), "source": "posts"}
    finally: cursor.close(); conn.close()

@app.get("/admin/purchase-alerts")
def get_admin_purchase_alerts(status: Optional[str] = None, limit: int = 50):
    """
    My Page에서 쓰는 "구매 희망 알림"용 API.
    DB에 넣어둔 목업/현실 데이터(admin_purchase_alerts)를 전시회 제목과 함께 반환합니다.
    """
    limit = max(1, min(int(limit or 50), 200))
    st = (status or "").strip().lower() if status else None

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if st:
            cursor.execute(
                """
                SELECT a.id, e.title AS exhibition, a.art_title, a.buyer_name, a.price, a.status, a.created_at
                FROM admin_purchase_alerts a
                JOIN exhibitions e ON e.id = a.exhibition_id
                WHERE LOWER(a.status) = %s
                ORDER BY a.created_at DESC
                LIMIT %s
                """,
                (st, limit),
            )
        else:
            cursor.execute(
                """
                SELECT a.id, e.title AS exhibition, a.art_title, a.buyer_name, a.price, a.status, a.created_at
                FROM admin_purchase_alerts a
                JOIN exhibitions e ON e.id = a.exhibition_id
                ORDER BY a.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )

        rows = cursor.fetchall() or []
        out = []
        for r in rows:
            created = r.get("created_at")
            if isinstance(created, datetime):
                created_str = created.strftime("%Y.%m.%d %H:%M")
            else:
                created_str = str(created)

            out.append(
                {
                    "id": str(r.get("id")),
                    "exhibition": r.get("exhibition") or "",
                    "art_title": r.get("art_title") or "작품",
                    "buyer_name": r.get("buyer_name") or "",
                    "price": f"₩ {int(r.get('price') or 0):,}",
                    "status": _normalize_purchase_status(r.get("status") or "pending"),
                    "created_at": created_str,
                }
            )
        return {"alerts": out}
    finally:
        cursor.close()
        conn.close()


# --- 🚀 [Admin] 2. 공식 작품 등록 섹션 (NFC 매칭용) ---

# --- [Admin] 공식 작품 등록 (순수 작가 설명 저장) ---
@app.post("/admin/artworks/")
async def register_artwork(
    ex_id: int = Form(...), 
    title: str = Form(...), 
    artist: str = Form(...), 
    description: str = Form(""), 
    price: int = Form(0), 
    image: UploadFile = File(...)
):
    print(f"📥 요청 도착: {title}, {artist}") # 로그 확인용
    
    # 1. S3 업로드 시도
    image_url = upload_file_to_s3(image)
    if not image_url:
        print("❌ S3 업로드 실패")
        raise HTTPException(500, "S3 업로드 실패")
    
    print(f"✅ S3 업로드 성공: {image_url}")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        nfc_uuid = f"nfc_{uuid.uuid4().hex[:8]}"
        sql = "INSERT INTO artworks (exhibition_id, title, artist_name, description, price, image_url, nfc_uuid) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (ex_id, title, artist, description, price, image_url, nfc_uuid))
        conn.commit()
        print("✅ DB 저장 성공!")
        return {"message": "저장 성공", "artwork_id": cursor.lastrowid}
    except Exception as e:
        print(f"❌ DB 에러 발생: {e}") # 여기서 에러 내용이 Render 로그에 찍힙니다.
        raise HTTPException(500, f"DB 에러: {str(e)}")
    finally:
        cursor.close(); conn.close()

def _db_column_exists(conn, table_name: str, column_name: str) -> bool:
    """
    MySQL 테이블의 컬럼 존재 여부를 확인합니다.
    (스키마 변경 없이, 런타임 fallback에 사용)
    """
    db_name = os.getenv("DB_NAME")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
            """,
            (db_name, table_name, column_name),
        )
        row = cursor.fetchone()
        return bool(row and row.get("cnt", 0) > 0)
    finally:
        cursor.close()

# --- [Admin] 공식 작품 수정 (PUT) ---
@app.put("/admin/artworks/{artwork_id}")
async def update_artwork(
    artwork_id: int,
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    price: Optional[int] = Form(None),
    genre: Optional[str] = Form(None),  # 프론트에서 보내는 필드(테이블에 없을 수 있음)
    image: Optional[UploadFile] = File(None),
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM artworks WHERE id = %s", (artwork_id,))
        existing = cursor.fetchone()
        if not existing:
            raise HTTPException(404, "작품을 찾을 수 없습니다.")

        updates = []
        params = []

        if title is not None:
            updates.append("title = %s")
            params.append(title)

        if artist is not None:
            updates.append("artist_name = %s")
            params.append(artist)

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if price is not None:
            updates.append("price = %s")
            params.append(price)

        # genre 컬럼이 실제로 존재하면 업데이트(없으면 무시)
        if genre is not None and _db_column_exists(conn, "artworks", "genre"):
            updates.append("genre = %s")
            params.append(genre)

        if image is not None:
            new_image_url = upload_file_to_s3(image)
            if not new_image_url:
                raise HTTPException(500, "S3 업로드 실패")
            updates.append("image_url = %s")
            params.append(new_image_url)

        if not updates:
            return {"message": "변경 사항이 없습니다.", "artwork_id": artwork_id}

        sql = f"UPDATE artworks SET {', '.join(updates)} WHERE id = %s"
        params.append(artwork_id)

        cursor2 = conn.cursor()
        cursor2.execute(sql, tuple(params))
        conn.commit()
        cursor2.close()

        return {"message": "작품 정보가 수정되었습니다.", "artwork_id": artwork_id}
    finally:
        cursor.close(); conn.close()


# --- 🚀 [Admin] 3. 판매 및 구매 요청 섹션 ---

# 전시회별로 그룹화된 구매 요청 목록 조회
@app.get("/admin/sales/requests")
def get_purchase_requests():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 어느 전시회의 어떤 작품인지 JOIN을 통해 상세히 가져옵니다.
        sql = """
            SELECT e.title as exhibition_name, pr.id as request_id, a.title as art_title, 
                   pr.buyer_name, pr.price as requested_price, pr.status
            FROM purchase_requests pr
            JOIN artworks a ON pr.artwork_id = a.id
            JOIN exhibitions e ON a.exhibition_id = e.id
            ORDER BY e.title, pr.created_at DESC
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        # 프론트엔드 UI(SectionList) 구성을 위해 전시회별로 그룹화
        grouped_data = {}
        for row in rows:
            name = row['exhibition_name']
            if name not in grouped_data: grouped_data[name] = []
            # 프론트 호환 alias 추가 (기존 키 유지 + 새 키 추가)
            normalized = dict(row)
            normalized["id"] = row.get("request_id")
            normalized["price"] = row.get("requested_price")
            grouped_data[name].append(normalized)
        
        # 하위호환: 기존 `data` 유지 + 프론트가 쓰는 `requests`도 함께 제공
        return [{"exhibition": k, "data": v, "requests": v} for k, v in grouped_data.items()]
    finally: cursor.close(); conn.close()

# 구매 요청 승인/거절 처리
@app.post("/admin/sales/requests/{req_id}/status")
def update_purchase_status(req_id: int, body: PurchaseStatusUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "UPDATE purchase_requests SET status = %s WHERE id = %s"
        cursor.execute(sql, (body.status, req_id))
        conn.commit()
        return {"message": f"요청 상태가 {body.status}(으)로 변경되었습니다."}
    finally: cursor.close(); conn.close()
        
# 특정 전시회에 등록된 모든 작품 목록 가져오기
@app.get("/admin/exhibitions/{ex_id}/artworks")
def get_exhibition_artworks(ex_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        sql = "SELECT * FROM artworks WHERE exhibition_id = %s ORDER BY created_at DESC"
        cursor.execute(sql, (ex_id,))
        return cursor.fetchall()
    finally: cursor.close(); conn.close()

# --- [Admin] 전시회별 Top3 통계 ---
@app.get("/admin/exhibitions/{ex_id}/top3")
def get_exhibition_top3(ex_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 1) posts.artwork_id 컬럼이 있으면 "태깅(방문)" 기준 집계 (우선)
        if _db_column_exists(conn, "posts", "artwork_id"):
            sql = """
                SELECT a.id AS artwork_id, a.title, a.artist_name, COUNT(p.id) AS count
                FROM posts p
                JOIN artworks a ON p.artwork_id = a.id
                WHERE a.exhibition_id = %s
                GROUP BY a.id
                ORDER BY count DESC
                LIMIT 3
            """
            cursor.execute(sql, (ex_id,))
            return {"metric": "posts", "top3": cursor.fetchall()}

        # 2) fallback: purchase_requests 기반 집계(대체 지표)
        sql = """
            SELECT a.id AS artwork_id, a.title, a.artist_name, COUNT(pr.id) AS count
            FROM artworks a
            LEFT JOIN purchase_requests pr ON pr.artwork_id = a.id
            WHERE a.exhibition_id = %s
            GROUP BY a.id
            ORDER BY count DESC
            LIMIT 3
        """
        cursor.execute(sql, (ex_id,))
        return {"metric": "purchase_requests", "top3": cursor.fetchall()}
    finally:
        cursor.close(); conn.close()
