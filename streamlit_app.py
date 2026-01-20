import streamlit as st
from groq import Groq
import base64
import fitz
from PIL import Image
import io
import pandas as pd
from openpyxl.styles import Border, Side

KEY_1 = 'gsk_vD7EHyaBqg3bPLeQfVaXWGdyb3'

KEY_2 = 'FYRFTnIHfwoSnh5U7wdLWm6QyN'

GROQ_API_KEY = KEY_1 + KEY_2

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="AI Vision Sheet Scanner",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        border-right: 2px solid #00d9ff;
    }

    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none !important;
    }

    .header-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding-top: 80px;      
        padding-bottom: 40px;   
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3em;         
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00d9ff 0%, #7b2ff7 50%, #f107a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
        margin: 0;
        padding: 0;
        letter-spacing: 3px;    
        line-height: 1.2;
    }

    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2em;       
        font-weight: 400;
        text-align: center;
        color: #00d9ff;
        margin-top: 15px;       
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.9;
    }

    .developer-credit {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9em;       
        text-align: center;
        color: #888;
        font-style: italic;
        margin-top: 15px;       
    }

    .stChatMessage {
        background: rgba(26, 26, 46, 0.6) !important;
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }

    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid #00d9ff;
        color: white;
        border-radius: 8px;
    }
    
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-wrapper">
    <div class="main-title">AI VISION SHEET SCANNER</div>
    <div class="subtitle">Intelligent Document Processing AI</div>
    <div class="developer-credit">Developed by Andy Ting Zhi Wei</div>
</div>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_images' not in st.session_state:
    st.session_state.file_images = None
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=False,
        help="Maximum file size: 200MB"
    )

    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)
        if file_size > 200:
            st.error("File size exceeds 200MB!")
            uploaded_file = None
        else:
            if uploaded_file != st.session_state.uploaded_file:
                st.session_state.uploaded_file = uploaded_file

                with st.spinner("Loading document..."):
                    try:
                        if uploaded_file.type == "application/pdf":
                            pdf_bytes = uploaded_file.read()
                            pdf_document = fitz.open(
                                stream=pdf_bytes, filetype="pdf")
                            images = []
                            for page_num in range(pdf_document.page_count):
                                page = pdf_document[page_num]
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                img = Image.frombytes(
                                    "RGB", [pix.width, pix.height], pix.samples)
                                images.append(img)
                            pdf_document.close()
                            st.session_state.file_images = images
                        else:
                            image = Image.open(uploaded_file)
                            st.session_state.file_images = [image]

                        st.success(
                            f"✅ {uploaded_file.name} ({file_size:.2f} MB)")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            else:
                st.success(f"✅ {uploaded_file.name} ({file_size:.2f} MB)")

    st.markdown("---")

    model = st.selectbox(
        "Vision Model",
        options=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct"
        ],
        index=0,
        help="Llama 4 Scout is best for vision and multimodal tasks. Llama 4 Maverick is best for advanced reasoning and complex tasks."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=512,
        max_value=8192,
        value=2048,
        step=256
    )


def encode_image(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')


def get_vision_response(prompt, images, model_name, temp, tokens):
    enhanced_prompt = f"""
    You are an intelligent document assistant (Neural Ledger).
    
    User Request: "{prompt}"
    
    ### STRICT INSTRUCTION: MODE SELECTION ###
    You must determine if the user wants to CHAT (summarize/explain) or EXTRACT DATA (convert to table/excel).
    
    --- OPTION A: CHAT / SUMMARY / ANALYSIS ---
    Condition: User asks "Summarize", "Explain", "What is this", "Analyze", or asks a general question.
    Output Format:
    1. Start response EXACTLY with: ###CHAT_MODE###
    2. Then provide your natural language answer.
    3. Do NOT provide any CSV data.
    
    --- OPTION B: DATA EXTRACTION ---
    Condition: User asks "Extract", "Table", "Excel", "CSV", or "Get data".
    Output Format:
    1. Start response EXACTLY with: ###DATA_MODE###
    2. Followed immediately by the raw CSV data.
    3. NO introduction text (e.g. "Here is the data").
    4. NO summary sentences.
    5. Use comma (,) delimiter.
    6. If multiple tables, separate with: ###TABLE_SPLIT###
    """

    messages = [{"role": "user", "content": []}]

    for img in images:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
        })

    messages[0]["content"].append({"type": "text", "text": enhanced_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temp,
        max_tokens=tokens
    )

    return response.choices[0].message.content


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me to extract data or analyze your document..."):
    if not st.session_state.file_images:
        st.error("⚠️ Please upload a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("🧠 Processing..."):
            try:
                raw_response = get_vision_response(
                    prompt,
                    st.session_state.file_images,
                    model,
                    temperature,
                    max_tokens
                )

                clean_response = raw_response.strip()

                if "###DATA_MODE###" in clean_response:
                    data_content = clean_response.replace(
                        "###DATA_MODE###", "").strip()

                    for marker in ["```csv", "```"]:
                        if data_content.startswith(marker):
                            data_content = data_content[len(marker):]
                        if data_content.endswith(marker):
                            data_content = data_content[:-len(marker)]
                    data_content = data_content.strip()

                    table_strings = data_content.split('###TABLE_SPLIT###')
                    excel_buffer = io.BytesIO()
                    tables_found = False

                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        workbook = writer.book
                        worksheet = workbook.create_sheet('Data')
                        writer.sheets['Data'] = worksheet

                        thin_border = Border(left=Side(style='thin'),
                                             right=Side(style='thin'),
                                             top=Side(style='thin'),
                                             bottom=Side(style='thin'))

                        current_row = 1

                        for table_str in table_strings:
                            table_str = table_str.strip()
                            if not table_str:
                                continue

                            csv_lines = [line for line in table_str.split(
                                '\n') if ',' in line]
                            if not csv_lines:
                                continue
                            clean_table_str = '\n'.join(csv_lines)

                            try:
                                df = pd.read_csv(
                                    io.StringIO(clean_table_str),
                                    header=0,
                                    engine='python',
                                    on_bad_lines='skip'
                                )

                                if not df.empty and len(df.columns) > 1:
                                    tables_found = True
                                    df.to_excel(
                                        writer, sheet_name='Data', startrow=current_row-1, index=False)

                                    end_row = current_row + df.shape[0]
                                    end_col = df.shape[1]
                                    for row in worksheet.iter_rows(min_row=current_row, max_row=end_row, min_col=1, max_col=end_col):
                                        for cell in row:
                                            cell.border = thin_border

                                    current_row += df.shape[0] + 3
                            except:
                                continue

                    with st.chat_message("assistant"):
                        if tables_found:
                            st.markdown("✅ **Data Extracted Successfully**")
                            # Preview
                            try:
                                first_table_str = [
                                    s for s in table_strings if ',' in s][0]
                                first_valid_lines = '\n'.join(
                                    [line for line in first_table_str.split('\n') if ',' in line])
                                df_preview = pd.read_csv(io.StringIO(
                                    first_valid_lines), header=0, engine='python', on_bad_lines='skip')
                                st.caption("Preview:")
                                st.dataframe(df_preview, width='stretch')
                            except:
                                pass

                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_buffer.getvalue(),
                                file_name="extracted_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            st.session_state.messages.append(
                                {"role": "assistant", "content": "✅ **Data Extracted** "})
                        else:
                            st.error(
                                "AI entered Data Mode but produced invalid CSV data. Please try again.")

                else:
                    chat_content = clean_response.replace(
                        "###CHAT_MODE###", "").strip()

                    with st.chat_message("assistant"):
                        st.markdown(chat_content)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": chat_content})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg})
