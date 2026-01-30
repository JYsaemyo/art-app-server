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

# --- Pydantic Models ---
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
    asyncio.create_task(periodic_sync_task())

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

# --- 🚀 [Admin] 1. 전시회 관리 함수 섹션 ---

# 모든 전시회 목록 조회 (사용자 태깅 수 계산 포함)
@app.get("/admin/exhibitions/")
def get_admin_exhibitions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 전시회 제목과 posts 테이블의 제목을 매칭하여 '전체 태그' 수를 실시간 집계합니다.
        sql = """
            SELECT e.*, COUNT(p.id) as total_tags 
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

# 특정 전시회 상세 통계 (Google Analytics 스타일)
@app.get("/admin/exhibitions/{ex_id}/stats")
def get_exhibition_analytics(ex_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT title FROM exhibitions WHERE id = %s", (ex_id,))
        ex = cursor.fetchone()
        if not ex: raise HTTPException(404, "전시회를 찾을 수 없습니다.")
        
        # 최근 7일간의 날짜별 태깅(방문) 추이를 가져옵니다.
        sql = """
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM posts WHERE title = %s 
            GROUP BY DATE(created_at) ORDER BY date ASC LIMIT 7
        """
        cursor.execute(sql, (ex['title'],))
        return {"title": ex['title'], "daily_stats": cursor.fetchall()}
    finally: cursor.close(); conn.close()


# --- 🚀 [Admin] 2. 공식 작품 등록 섹션 (NFC 매칭용) ---

# 3. 작품 등록 (AI 제거, 장르/설명 직접 입력)
@app.post("/admin/artworks/")
async def register_artwork(
    ex_id: int = Form(...), 
    title: str = Form(...), 
    artist: str = Form(...), 
    genre: str = Form("회화"), # 기본값 설정
    description: str = Form(""), 
    price: int = Form(0), 
    image: UploadFile = File(...)
):
    print(f"📥 작품 등록 요청: {title} ({genre})")

    # S3 업로드
    image_url = upload_file_to_s3(image)
    if not image_url:
        raise HTTPException(500, "S3 업로드 실패")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # DB 저장 (AI 관련 필드 제거됨)
        nfc_uuid = f"nfc_{uuid.uuid4().hex[:8]}"
        sql = """
            INSERT INTO artworks (exhibition_id, title, artist_name, genre, description, price, image_url, nfc_uuid) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (ex_id, title, artist, genre, description, price, image_url, nfc_uuid))
        conn.commit()
        print("✅ DB 저장 성공!")
        return {"message": "저장 성공", "artwork_id": cursor.lastrowid}
    except Exception as e:
        print(f"❌ DB 에러: {e}")
        raise HTTPException(500, f"DB 에러: {str(e)}")
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
            grouped_data[name].append(row)
        
        return [{"exhibition": k, "data": v} for k, v in grouped_data.items()]
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
        
# 5. 작품 수정 (이미지 변경 없으면 기존 유지)
@app.put("/admin/artworks/{art_id}")
async def update_artwork(
    art_id: int,
    title: str = Form(...),
    artist: str = Form(...),
    genre: str = Form(...),
    description: str = Form(""),
    # 이미지는 없을 수도 있음 (None 허용)
    image: UploadFile = File(None) 
):
    print(f"🔄 작품 수정 요청 ID: {art_id}, 제목: {title}")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. 기존 이미지 URL 가져오기
        cursor.execute("SELECT image_url FROM artworks WHERE id = %s", (art_id,))
        existing_art = cursor.fetchone()
        
        if not existing_art:
            raise HTTPException(404, "작품을 찾을 수 없습니다.")

        final_image_url = existing_art['image_url']

        # 2. 새 이미지가 왔다면 S3 업로드 후 URL 교체
        if image:
            print("📸 새 이미지 업로드 중...")
            new_url = upload_file_to_s3(image)
            if new_url:
                final_image_url = new_url

        # 3. DB 업데이트 (artist -> artist_name 매핑 주의)
        sql = """
            UPDATE artworks 
            SET title = %s, artist_name = %s, genre = %s, description = %s, image_url = %s
            WHERE id = %s
        """
        cursor.execute(sql, (title, artist, genre, description, final_image_url, art_id))
        conn.commit()
        
        print("✅ 수정 완료")
        return {"message": "수정되었습니다.", "image_url": final_image_url}

    except Exception as e:
        print(f"❌ 수정 에러: {e}")
        raise HTTPException(500, f"에러: {str(e)}")
    finally:
        cursor.close(); conn.close()
