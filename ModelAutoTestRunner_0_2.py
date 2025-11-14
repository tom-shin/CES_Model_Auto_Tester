import os
import subprocess
import re
import json
from datetime import datetime
import pandas as pd
import sys
import io
import time
import adbutils
import openpyxl
from openpyxl.styles import Alignment
import pexpect

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"



def format_elapsed_time(start_time, end_time=None):
    """
    시작 시간과 종료 시간을 받아 경과 시간을
    day, hour, minute, second 형식으로 문자열 반환
    """
    if end_time is None:
        end_time = time.time()
    
    elapsed = end_time - start_time

    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds:.2f}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"

def check_adb_devices():
    devices = adbutils.adb.device_list()

    if not devices:
        print("❌ No ADB devices connected.")
        return False
    else:
        return True

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def kill_running_processes():
    kill_cmds = [
        ["adb", "shell", "pkill -f MambaTest"],
        ["adb", "shell", "pkill -f llama"],
        ["adb", "shell", "killall -9 MambaTest"],
        ["adb", "shell", "killall -9 llama"]
    ]
    for cmd in kill_cmds:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass



def run_conversation_processor(execute_info, model):
    """
    Executes MambaTest or LLaMA interactively on Android device via adb.
    Waits until the model is ready ([Input]: for Mamba, system message for LLaMA).
    Does not send any input; just checks readiness.
    Sets global recrusive_process for later use.
    """
    global recrusive_process

    execute_path = execute_info[model]["execute_path"]
    model_file   = execute_info[model]["model"]
    execute_cmd  = execute_info[model]["execute_cmd"]

    print(f"[Starting {model_file} Recrusive session on device]")

    if "Mamba" in model_file:
        adb_cmd = (
            f'adb shell -t -t "export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; '
            f'{execute_path}{execute_cmd} -output-buffer-size 1"'
        )
    else:
        adb_cmd = (
            f'adb shell -t -t "export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; '
            f'{execute_path}{execute_cmd} '
            f'-m {execute_path}{model_file} '
            '--simple-io --no-display-prompt -ngl 999 --temp 0.4"'
        )

    # pexpect 실행
    recrusive_process = pexpect.spawn(adb_cmd, encoding='utf-8', timeout=60)
    print("[Waiting for model warmup...]")

    try:
        if "Mamba" in model_file:
            recrusive_process.expect(r'\[Input\]:')
            print("[Interactive session ready - MAMBA ✅]")
        else:
            recrusive_process.expect(r'- Not using system message.')
            print("[Interactive session ready - LLAMA ✅]")
    except pexpect.TIMEOUT:
        print("[ERROR] Timeout waiting for interactive prompt")
        recrusive_process = None
        return None

    return recrusive_process





def run_conversation(llm_processor, prompt, execute_info, model):


    if llm_processor is None:
        print("[ERROR] Recrusive process is not running!")
        return None

    # print(f"\n📢 Input:\n{prompt}")

    # pexpect 사용: sendline
    llm_processor.sendline(prompt)

    output_lines = []

    while True:
        idx = llm_processor.expect([r'\[Input\]:', r'- Host .*llama_memory_breakdown_print', pexpect.EOF, pexpect.TIMEOUT], timeout=60)
        if idx == 0 or idx == 1:
            # 프롬프트 전 또는 Mamba/LLAMA 종료 신호
            output_lines.append(llm_processor.before)
            break
        elif idx == 2:
            print("[ERROR] Process ended unexpectedly")
            output_lines.append(llm_processor.before)
            break
        elif idx == 3:
            print("[WARN] Timeout waiting for output")
            output_lines.append(llm_processor.before)
            break

    output = ''.join(output_lines)
    inference_result = parse_output_conversation(prompt, output, execute_info, model)
    return inference_result


def remove_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_output_conversation(prompt, output: str, execute_info, model):
    lines = output.splitlines()

    inference_lines = []
    info_lines = []

    for line in lines:
        # [INFO] 정보 수집
        if "[INFO_TSK]" in line:
            info_lines.append(remove_ansi(line))
        # 모델 응답 텍스트
        elif line.strip() not in ["", ">"] and not line.lstrip().startswith("llama_memory_breakdown_print"):
            inference_lines.append(remove_ansi(line))

    
    # # 기본값 설정
    # initial_latency = 0.0
    # total_inference_time = 0.0
    # generated_tokens = 0
    # tps = 0.0

    # try:
    #     for line in info_lines:
    #         # Initial Latency
    #         m = re.search(r"initial latency:\s*([\d.]+)\s*ms", line)
    #         if m:
    #             initial_latency = float(m.group(1))

    #         # Total inference time + generated tokens
    #         m = re.search(r"total time\s*=\s*([\d.]+)\s*/\s*(\d+)\s*tokens", line)
    #         if m:
    #             total_inference_time = float(m.group(1))
    #             generated_tokens = int(m.group(2))
    #             if total_inference_time > 0:
    #                 tps = generated_tokens / (total_inference_time / 1000)
    # except Exception as e:
    #     # 예외 발생 시 기본값 유지하고 로그 출력
    #     print(f"[WARNING] Failed to parse profile info: {e}")

    # # 계산된 항목을 리스트로 추가
    # info_lines.append("")  # 빈 줄 추가  
    # info_lines.append("")  # 또 다른 빈 줄 추가 

    # info_lines.append(f">> Initial Latency: {initial_latency} ms")
    # info_lines.append(f">> Tokens Per Second (TPS): {tps:.2f} tokens/sec")
    # info_lines.append(f">> Total Inference Time: {total_inference_time} ms")
    # info_lines.append(f">> Total Generated Tokens: {generated_tokens} tokens")


    # Mamaba Case
    # 첫 번째 등장 기준으로 분리

    if "Mamba" in execute_info[model]["model"]:
        inference_out = "\n".join(inference_lines).strip()
        
        parts = inference_out.split("** Profile Summary **", 1)  # 1번만 split
        before_profile = parts[0].split("[Mamba]")
        after_profile = "** Profile Summary **" + parts[1]  # 포함해서 이후

        result = {
            "Question": prompt,
            "Inference Result": "[Mamba]" + before_profile[1],
            "Information": after_profile #+ "\n" + "\n".join(info_lines)
        }
        
        print(result["Inference Result"])        


    return result



def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_output(output_text):
    """
    Mamba 출력 파싱:
    - inference_result: 실제 답변 텍스트
    - info_lines: 통계/정보 텍스트 리스트 (빈 줄 제거)
    """
    try:
        lines = output_text.splitlines()
        info_lines = []

        # ----------------------------
        # 1) Profile Summary / Duration 라인 그대로 info_lines에 추가
        # ----------------------------
        profile_capture = False
        for line in lines:
            if "** Profile Summary **" in line:
                profile_capture = True
            if profile_capture:
                # [INFO_TSK] 라인은 제외
                if not line.startswith("[INFO_TSK]"):
                    stripped_line = line.rstrip()
                    if stripped_line:  # 빈 줄 제거
                        info_lines.append(stripped_line)

        # ----------------------------
        # 2) [INFO_TSK] 값 추출 후 >> 라인으로 추가
        # ----------------------------
        info_tsk_pattern = re.compile(r"\[INFO_TSK\]\s*(\d+),\s*(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)")

        for line in lines:
            match = info_tsk_pattern.search(line)
            if match:
                token_generation_length_inference = int(match.group(1))
                token_generation_length_prompt = int(match.group(2))
                input_token_processing_speed = float(match.group(3))
                token_generation_processing_speed = float(match.group(4))
                total_processing_latency = float(match.group(5))

                # [INFO_TSK] 내용은 출력하지 않고, >> 요약만 출력
                info_lines.append(f">> Token Generation Length Inference: {token_generation_length_inference}")
                info_lines.append(f">> Token Generation Length Prompt: {token_generation_length_prompt}")
                info_lines.append(f">> Input Token Processing Speed: {input_token_processing_speed:.2f} tps")
                info_lines.append(f">> Token Generation Processing Speed: {token_generation_processing_speed:.2f} tps")
                info_lines.append(f">> Total Processing Latency (runPipeline): {total_processing_latency:.2f} s")

                break  # [INFO_TSK]는 1개만 있음          
        

        # ----------------------------
        # 3) inference_result 추출
        # ----------------------------
        mamba_pattern = r"🐍 Mamba:\s*(.*?)(?=\*\* Profile Summary \*\*|$)"
        mamba_match = re.search(mamba_pattern, output_text, re.DOTALL)
        inference_result = mamba_match.group(1).strip() if mamba_match else ""
        inference_result = remove_ansi_codes(inference_result)

        return inference_result, info_lines

    except Exception as e:
        print(f"[WARN] parse_mamba_output failed: {e}")
        return "", ["Parsing failed"]

def run_single_shot(prompt, execute_info, model):
    print(f"{RED}[KILL] Existing LLM processes...{RESET}")
    kill_running_processes()
    time.sleep(2)

    # print(f"\n📢 Input:\n{GREEN}{prompt}{RESET}")

    execute_path = execute_info[model]["execute_path"].rstrip("/")
    execute_cmd = execute_info[model]["execute_cmd"]   

    cmd = [
    "adb", "shell",
    f"sh -c 'export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; "
    f"{execute_path}/{execute_cmd} -p \"{prompt}\" -output-buffer-size 1'"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore',
        bufsize=1
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    output = ''.join(output_lines)

    inference_result, profile_info = parse_output(output)

    return {
        "Question": prompt,
        "Inference Result": inference_result,
        "Information": profile_info
    }

def main(file_path, language, execute_info, model):

    device_exist = None
    all_results = []
    llm_processor = None

    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
   
    
    if execute_info[model]["type"] == "Recrusive":
        print(f"{RED}[KILL] [Recrusive] -Existing LLM processes...{RESET}")
        kill_running_processes()
        time.sleep(2)
        llm_processor = run_conversation_processor(execute_info, model)

    
    all_results = []
    idx = 1
    total_cnt = sum(len(v) for v in questions.values())

    for category, question_list in questions.items():    
            
        print(f"\n{RED}================================ Category: {category} ================================ {RESET}")
        for _, q in enumerate(question_list):
            prompt = q[language]
            print(f"Input:\n{prompt}")
            time.sleep(1.5)

            if execute_info[model]["type"] == "One-Shot":
                result = run_single_shot(prompt, execute_info, model)
            else:
                result = run_conversation(llm_processor, prompt, execute_info, model)

            all_results.append(result)
            print(f"{BLUE}=================== 현재 Category: {category},  ({idx}/{total_cnt}) th Test Done. {round(idx/total_cnt*100, 2)} %. ==================={RESET}\n")
            idx += 1

            device_exist = check_adb_devices()

            if not device_exist:
                break

        if not device_exist:
            break

    
    # Result 폴더 생성
    RESULT_DIR = "Result"
    os.makedirs(RESULT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "full" if device_exist else "partial"

    # 파일 경로 변경 (Result/ 폴더에 저장)
    json_filename = os.path.join(RESULT_DIR, f"{model}_Result_{language}_{suffix}_{timestamp}.json")

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"{YELLOW}✅ JSON saved to: {json_filename}{RESET}")


    excel_filename = os.path.join(RESULT_DIR, f"{model}_Result_{language}_{suffix}_{timestamp}.xlsx")
    excel_data = []

    for result in all_results:
        row = {
            "Question": result["Question"],
            "Inference Result": result["Inference Result"],
            "Information": "\n".join(result["Information"]) if isinstance(result["Information"], list) else str(result["Information"])
        }
        excel_data.append(row)

    df = pd.DataFrame(excel_data)
    df.to_excel(excel_filename, index=False)

    # 셀 정렬
    wb = openpyxl.load_workbook(excel_filename)
    ws = wb.active

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical='top', wrap_text=True)

    wb.save(excel_filename)
    print(f"{YELLOW}✅ Excel saved to: {excel_filename} (Left & Top aligned){RESET}")


    return all_results

def get_model_info():
    execute_info = {
        "NNC-Mamba": {
            "execute_cmd": "MambaTest",
            "execute_path": "/data/local/tmp/MAMBA/",
            "model": "Mamba",
            "type": "Recrusive"   #"One-Shot"
        },
        "llama-1B": {
            "execute_cmd": "llama-cli",
            "execute_path": "/data/local/tmp/GPU_LLAMA_USE_VULKAN/",
            "model": "llama-3.2-1b-instruct-q4_k_m.gguf",
            "type": "Recrusive"
        },
        "llama-3B": {
            "execute_cmd": "llama-cli",
            "execute_path": "/data/local/tmp/GPU_LLAMA_USE_VULKAN/",
            "model": "llama-3.2-3b-instruct-q4_k_m.gguf",
            "type": "Recrusive"
        }
    }

    return execute_info

if __name__ == "__main__":

    ##################### User Selection #####################

    test_language = "English"   # 또는 "Chinese"
    # test_language = "Chinese"

    model = "NNC-Mamba"  
    # model = "NNC-llama-8B"
    # model = "llama-1B"
    # model = "llama-3B"

    scenario_file = "Scenario/test_ces_llm_questions_all_categories_100.json" 
    # scenario_file = "Scenario/ces_llm_questions_all_categories_100.json"

    ##################### User Selection End #####################
    

    scenario_items = os.path.join(os.getcwd(), scenario_file)

    start_time = time.time()
    results = main(file_path=scenario_items, language=test_language, execute_info=get_model_info(), model=model)
    elapsed_text = format_elapsed_time(start_time)

    print(f"\nTotal Execution Time: {elapsed_text}")






