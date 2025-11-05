# gpt_service.py

from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import json # (json 임포트)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

def analyze_memory(text: str) -> dict:
    """GPT-4o를 이용하여 기억 텍스트를 분석하고 JSON 구조로 반환"""
    
    # 🔹 1. (수정) GPT에게 '영어 키'와 '한글 값'을 요청
    system_prompt = """
    당신은 감성 분석 전문가입니다.
    사용자가 제공한 기억 텍스트를 분석하여 아래 JSON 구조로 출력하세요.
    'Key'는 반드시 영어를 사용하고, 'Value'는 반드시 유창한 한국어로 작성합니다.
    출력 형식은 JSON으로만 제한하며, 절대 다른 텍스트를 포함하지 마세요.
    
     "조건"이라 이름 붙인 규칙은 절대, 지켜져야 합니다.
    (아래에 "조건" 글자 뒤에 숫자를 붙여 조건들을 각각 구별할 예정입니다.)
    조건1 : 19금 요소가 있거나, 선정적인 결과물은 나와선 안됩니다.(이미지 생성과정 중 선정적이거나, 19금 적인 프롬포트 사용을 금지합니다.)
    조건2 : 'Key'값의 반환값'Value'에 내용은, 사용자가 제공한 기억 텍스트를 가능한 면밀하게 분석하여 도출해야합니다.
    조건3 : 'Key'값의 반환값'Value'에 내용은 최대한 간결하면서도, 명확하게 의미가 전달되도록 표현되어야합니다.
    조건4 : 'Key'값의 반환값'Value'에 내용은, 나오는 모든 내용이 한국어로 작성되어야 합니다 
    조건5:  이미지 생성에 사용되는 프롬포트는 입력 데이터 값을 활용할시, 최대한 현실적인 그림체를 유지하도록 프롬포트 키워드를 도출해야 합니다.
    조건6:  비현실적 그림체를 유발하는 프롬포트 키워드는 이미지 생성시에 사용을 금합니다.
    조건7: ultra photorealistic, natural skin texture, realistic lighting and shadows, physically accurate proportions, natural colors, subtle film grain, 
    35mm photography style, high dynamic range, shallow depth of field 해당 프롬포트 단어들을 이미지 생성시 무조건 긍정 프롬포트에 포함합니다.
    조건8: no cartoon, no anime, no illustration, no artstyle filters, no exaggerated features,no AI artifacts, no plastic skin, no smooth wax-like texture,no distortion, 
    no extra limbs, no odd proportions, no unrealistic lighting, no text, no watermark, no logo 해당 프롬포트 단어들을 이미지 생성시 무조건 부정 프롬포트에 포함합니다.
    조건9: 위 조건이라 묶은 규칙들을 항상 적용합니다.

    
    출력 JSON 구조:
    {
      "emotion": "",      # 기억에서 느껴지는 감정 (한글 값) 
      "imagery": "",      # 기억을 시각적으로 표현할 수 있는 장면 (한글 값)
      "time_period": "",  # 기억이 연상되는 시대적 배경 (한글 값)
      "symbolism": ""     # 기억이 상징하는 의미나 주제 (한글 값)
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7
        )
        result_text = response.choices[0].message.content
        try:
            # 🔹 2. JSON 문자열 -> 파이썬 딕셔너리 (영어 키 + 한글 값)
            result_json = json.loads(result_text)

            # 🔹 3. (추가) index.js를 위한 'analysis' 키를 수동으로 생성
            # 4개의 한글 값을 하나의 요약 문자열로 합칩니다.
            analysis_summary = (
                f"감정: {result_json.get('emotion', '-')}\n"
                f"이미지: {result_json.get('imagery', '-')}\n"
                f"시대: {result_json.get('time_period', '-')}\n"
                f"상징: {result_json.get('symbolism', '-')}"
            )
            
            # 🔹 4. (추가) 'analysis' 키를 딕셔너리에 추가
            result_json["analysis"] = analysis_summary

        except json.JSONDecodeError:
            # GPT가 JSON 형식을 반환하지 않았을 경우
            result_json = {"analysis": result_text, "analysis_text": result_text}

        # 🔹 5. (최종 반환) 영어 키 4개 + analysis 키 1개가 포함된 딕셔너리
        return result_json
        
    except Exception as e:
        logging.error(f"GPT 분석 실패: {e}")
        return {"analysis_error": str(e), "analysis": "분석 중 오류가 발생했습니다."}