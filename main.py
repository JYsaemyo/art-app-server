import streamlit as st
import mysql.connector
import os
import json
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 환경 변수 및 Gemini 설정
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ .env 파일에 GEMINI_API_KEY가 없습니다!")
    st.stop()

genai.configure(api_key=api_key)

# --- [DB 연결 함수] ---
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        # ssl_ca="/etc/ssl/certs/ca-certificates.crt" # 필요한 경우 주석 해제
    )

# --- [DB 저장 함수 1] 음악 프롬프트만 저장 ---
def update_music_data(post_id, prompt):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "UPDATE posts SET music_prompt = %s WHERE id = %s"
        cursor.execute(sql, (prompt, post_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")
        return False

# --- [DB 저장 함수 2] 그림 분석 결과(ai_summary) 저장 ---
def update_art_summary(post_id, summary_text):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "UPDATE posts SET ai_summary = %s WHERE id = %s"
        cursor.execute(sql, (summary_text, post_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")
        return False

# --- [이미지 로드 함수] ---
def load_image_from_url(url):
    try:
        if "localhost" in url:
            filename = url.split("/")[-1]
            local_path = os.path.join("server", "uploads", filename)
            
            if os.path.exists(local_path):
                return Image.open(local_path)
            else:
                local_path_v2 = os.path.join("uploads", filename)
                if os.path.exists(local_path_v2):
                    return Image.open(local_path_v2)
                return None
        else:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"이미지 로드 실패: {url}")
        return None

# --- [AI 분석 함수 1] 그림 분석 ---
def analyze_art_ai(image_url, title, artist, genre, style):
    img = load_image_from_url(image_url)
    if not img: return None

    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    # 복원된 상세 프롬프트
    prompt = f"""
    당신은 사려 깊고 관찰력이 뛰어난 미술 평론가입니다. 
    제공된 이미지와 정보를 바탕으로 작품을 분석하여 JSON 형식으로 출력하세요.

    [핵심 지침: 말투와 어조]
    1. **단정적인 표현을 절대 피하세요.**
    2. **관찰자의 입장에서 추측하고 해석하는 어조를 사용하세요.** (예: "~인 것 같습니다", "~으로 보입니다")
    3. 정중하고 감성적인 문체를 유지하세요.
    4. 한국어로 출력하세요.
    
    [작품 정보]
    - 제목: {title}
    - 작가: {artist}
    - 장르: {genre}, 화풍: {style}
    
    [출력 요구사항 (JSON)]
    반드시 아래 3가지 키(key)를 가진 JSON 형식으로만 답변하세요.
    1. "artist_intro": 작가 설명 (2문장 내외)
    2. "title_meaning": 제목 의미 (2문장 내외)
    3. "art_review": 종합 감상평 (3문장 내외)
    """

    try:
        response = model.generate_content([prompt, img], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI 분석 실패: {e}")
        return None

# [수정됨] 음악 프롬프트 생성 함수 (제목, 작가, 태그 반영)
def create_music_prompt_ai(description, title, artist, tags):
    model = genai.GenerativeModel('models/gemini-2.0-flash')

    # 백엔드와 동일한 고품질 프롬프트 적용
    prompt = f"""
    전문 음악 프롬프트 엔지니어로서, 아래 [작품 정보]를 바탕으로 AI 음악 생성 프롬프트를 작성하세요.
    제목과 작가가 주는 뉘앙스, 그리고 설명/태그의 분위기를 음악 스타일에 적극 반영하세요.

    [작품 정보]
    1. 제목: {title}
    2. 작가: {artist}
    3. 설명 및 태그: 
    {description}
    관련 태그: {tags}

    [출력 요구사항 (JSON)]
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
        st.error(f"음악 프롬프트 생성 실패: {e}")
        return None

# --- [메인 화면 UI] ---
st.set_page_config(page_title="🎨 Art AI Manager", layout="wide")
st.title("🎨 Art App: AI 관리자")

# 1. DB 목록 불러오기
try:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM posts ORDER BY id DESC")
    posts = cursor.fetchall()
    conn.close()
except Exception as e:
    st.error("DB 연결 실패")
    posts = []

if posts:
    post_options = {p['id']: f"[{p['id']}] {p['title']} - {p['artist_name']}" for p in posts}
    selected_post_id = st.selectbox("작업할 작품 선택", options=list(post_options.keys()), format_func=lambda x: post_options[x])
    post = next((p for p in posts if p['id'] == selected_post_id), None)

    if post:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if post['image_url'].startswith("http"):
                st.image(post['image_url'], caption=post['title'], use_container_width=True)
            else:
                st.warning("이미지 URL 오류")
            st.info(f"**작가:** {post['artist_name']}")

        with col2:
            st.subheader("💎 Gemini 작업실")
            
            tab1, tab2 = st.tabs(["🖼️ 그림 분석", "🎵 음악 프롬프트"])

            # --- [탭 1] 그림 분석 (여기가 수정되었습니다!) ---
            with tab1:
                st.markdown("### 1. 작품 3단 분석")
                genre = st.text_input("장르", value="인상주의")
                style = st.text_input("화풍", value="유화")
                
                analyze_btn = st.button("🖼️ 분석 시작")
                
                # 1. 분석 실행 및 세션 저장
                if analyze_btn:
                    with st.spinner("Gemini가 그림을 분석 중입니다..."):
                        result = analyze_art_ai(post['image_url'], post['title'], post['artist_name'], genre, style)
                        if result:
                            st.session_state['art_result'] = result
                            st.session_state['art_target_id'] = post['id']
                            st.rerun() # 새로고침
                
                # 2. 결과 표시 및 저장 버튼
                if 'art_result' in st.session_state and st.session_state.get('art_target_id') == post['id']:
                    res = st.session_state['art_result']
                    
                    st.success("분석 완료!")
                    st.write(f"**🧑‍🎨 작가 소개:** {res.get('artist_intro')}")
                    st.write(f"**🏷️ 제목 의미:** {res.get('title_meaning')}")
                    st.write(f"**📝 감상평:** {res.get('art_review')}")
                    
                    st.divider()
                    
                    # 저장할 내용 미리보기 (art_review -> ai_summary)
                    summary_to_save = res.get('art_review', '')
                    st.info(f"💾 **DB에 저장될 내용 (AI 요약):**\n{summary_to_save}")

                    # [저장 버튼 추가됨]
                    if st.button("💾 분석 결과(감상평) DB에 저장하기"):
                        if summary_to_save:
                            if update_art_summary(post['id'], summary_to_save):
                                st.toast("✅ AI 요약(ai_summary) 저장 성공!")
                        else:
                            st.warning("저장할 내용이 없습니다.")

            # --- [탭 2] 음악 프롬프트 ---
            with tab2:
                st.markdown("### 2. 음악 프롬프트 생성")
                
                # DB에 있는 내용 가져오기
                default_desc = post['description'] if post['description'] else ""
                tags_info = post.get('tags', '') # 태그 가져오기

                # 화면 표시
                st.info(f"**정보:** 제목[{post['title']}] / 작가[{post['artist_name']}] / 태그[{tags_info}]")
                desc_text = st.text_area("감상평 입력", value=default_desc, height=100)
                
                generate_btn = st.button("🎵 프롬프트 만들기")

                if generate_btn:
                    if not desc_text:
                        st.warning("감상평을 입력해주세요.")
                    else:
                        with st.spinner("작곡가는 생각 중..."):
                            # [수정됨] 함수에 제목, 작가, 태그 정보를 함께 전달
                            music_res = create_music_prompt_ai(
                                desc_text, 
                                post['title'], 
                                post['artist_name'], 
                                tags_info
                            )
                            
                            if music_res:
                                st.session_state['music_result'] = music_res
                                st.session_state['target_post_id'] = post['id'] 
                                st.rerun()

else:
    st.info("게시글이 없습니다.")
