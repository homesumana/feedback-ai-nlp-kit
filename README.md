
# Feedback AI/NLP Kit (Google Forms → Insights)

ชุดนี้ช่วยสรุปความคิดเห็นแบบข้อความยาว (multi-line) หลายพันเรคคอร์ด ให้เห็นทั้งภาพรวมและเชิงลึกต่อทีมงาน โดยใช้ Python + NLP + (ออปชั่น) LLM

## โครงสร้างโฟลเดอร์
```
feedback-ai-nlp-kit/
├─ README.md
├─ requirements.txt
├─ scripts/
│  └─ pipeline.py          # รันวิเคราะห์แบบ batch
├─ app/
│  └─ streamlit_app.py     # Dashboard แบบ Interactive
└─ outputs/                # ไฟล์ผลลัพธ์จะมาอยู่ที่นี่เมื่อรัน pipeline
```

## เตรียมข้อมูล (จาก Google Forms)
1) เปิด Google Form → Responses → ไอคอน Sheets → เปิดใน Google Sheets  
2) ใน Google Sheets: File → Download → **CSV**  
3) ตรวจสอบหัวคอลัมน์ (อย่างน้อยควรมี)
   - `timestamp` (วันที่/เวลา)
   - `team` (ทีมงานที่เกี่ยวข้อง เช่น วิชาการ, เทคนิค, ประสานงาน) — ถ้าไม่มี ให้เติมเองภายหลัง
   - `feedback` (ข้อความยาวจากผู้เรียน)

> คุณสามารถแก้ชื่อคอลัมน์ได้ในไฟล์ `scripts/pipeline.py` (ตัวแปร `COLS`).

## วิธีติดตั้ง
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> ถ้าบางแพ็กเกจ (เช่น `torch`) ใช้เวลา/ทรัพยากร ให้พิจารณารันใน Google Colab

## รันวิเคราะห์แบบ Batch
```bash
python scripts/pipeline.py --csv_path path/to/your_feedback.csv --out_dir outputs
```
ออปชั่น:
- `--model paraphrase-multilingual-MiniLM-L12-v2`  (ค่าเริ่มต้น)  
- `--min_topic_size 30`  (ปรับขนาดหัวข้อขั้นต่ำ)  
- `--use_openai`  (สร้างสรุปด้วย GPT ถ้ามี `OPENAI_API_KEY` ใน env)

ผลลัพธ์หลักในโฟลเดอร์ `outputs/`:
- `feedback_enriched.csv` (เพิ่ม sentiment, topic, prob, keywords)
- `topics_summary.csv` (รหัสหัวข้อ → คำอธิบาย, ตัวอย่างคอมเมนต์)
- `team_findings.csv` (สรุปเชิงทีม: ปริมาณ/สัดส่วน sentiment, top keywords, top issues)
- โฟลเดอร์ `figures/` (กราฟภาพรวม และต่อทีม เช่น บาร์ชาร์ต/เวิร์ดคลาวด์)

## เปิด Dashboard (Streamlit)
หลังจากมีไฟล์ `feedback_enriched.csv` แล้ว:
```bash
streamlit run app/streamlit_app.py --server.headless true
```
> ถ้าต้องการสรุปข้อความ/ข้อเสนอแนะด้วย GPT ใส่ `OPENAI_API_KEY` ใน environment ก่อน:
```bash
export OPENAI_API_KEY="sk-xxxx"
```

## แนวทางใช้งาน
- มุมมองภาพรวม: สัดส่วน sentiment, หัวข้อยอดฮิต, คีย์เวิร์ด, ตัวอย่างคอมเมนต์เด่น
- มุมมองเชิงทีม: กรองตาม `team`, ดูหัวข้อ/ประเด็นที่เจอบ่อย, ดูคอมเมนต์ตัวอย่าง, คำแนะนำลำดับการปรับปรุง
- Actionable: ใช้ `topics_summary.csv` และหน้า "Recommendations" ใน dashboard เพื่อดึงข้อเสนอแนะต่อทีม/ต่อหัวข้อ

## หมายเหตุเรื่องภาษาไทย
- ใช้ `pythainlp` สำหรับตัดคำ + stopwords ภาษาไทย
- ใช้ sentence embeddings แบบ multilingual → จับกลุ่มหัวข้อหลายภาษาได้
- Sentiment: มีตัวเลือก rule-based (เร็ว), หรือโมเดลเสริม (ปรับเพิ่มได้ในภายหลัง)

---

Made with ❤️
