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
    # global recrusive_process

    execute_path = execute_info[model]["execute_path"]
    model_file   = execute_info[model]["model"]
    execute_cmd  = execute_info[model]["execute_cmd"]

    print(f"[Starting {model_file} Recrusive session on device]")

    if "Mamba" in model_file:
        adb_cmd = (
            f'adb shell -t -t "export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; '
            f'{execute_path}{execute_cmd} -output-buffer-size 1"'
        )
    elif "llama-8B" in model_file:
        adb_cmd = (
            f'adb shell -t -t "export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; '
            f'{execute_path}{execute_cmd}'
        )    
    else:
        adb_cmd = (
            f'adb shell -t -t "export LD_LIBRARY_PATH={execute_path}:$LD_LIBRARY_PATH; '
            f'{execute_path}{execute_cmd} '
            f'-m {execute_path}{model_file} '
            '--simple-io --no-display-prompt -ngl 999 --temp 0.4"'
        )

    # pexpect 실행
    recrusive_process = pexpect.spawn(adb_cmd, encoding='utf-8', timeout=180)
    recrusive_process.logfile_read = sys.stdout  # 실시간 출력
    print("[Waiting for model warmup...]")

    try:
        if "Mamba" in model_file:
            # recrusive_process.expect(r'\[Input\]:')
            recrusive_process.expect_exact("[Input]:")
            # print("[Interactive session ready - MAMBA ✅]")
        elif "llama-8B" in model_file:
            recrusive_process.expect_exact("Please enter your question:")
            # print("[Interactive session ready - llama-8b ✅]")
        else:
            recrusive_process.expect('- Not using system message. To change it, set a different value via -sys PROMPT')
            # print("[Interactive session ready - LLAMA ✅]")
    except pexpect.TIMEOUT:
        print("[ERROR] Timeout waiting for interactive prompt")
        recrusive_process = None

    return recrusive_process




def response_mamba(llm_processor):
    output_lines = []

    while True:
        try:
            idx = llm_processor.expect_exact(["[Input]:", pexpect.EOF, pexpect.TIMEOUT], timeout=180)
        except pexpect.EOF:
            print("[INFO] Process finished")
            output_lines.append(llm_processor.before)
            break
        except pexpect.TIMEOUT:
            print("[WARN] Timeout waiting for output")
            output_lines.append(llm_processor.before)
            break

        # 항상 before에 있는 내용을 누적
        output_lines.append(llm_processor.before)

        if idx == 0:
            # "[Input]:" 프롬프트 도착 → 루프 종료
            break
        elif idx == 1:
            # EOF → 프로세스 종료
            print("[INFO] Process finished")
            break
        elif idx == 2:
            # TIMEOUT
            print("[WARN] Timeout waiting for output")
            break

    return output_lines


def response_llama_1b3b_gguf(llm_processor):
    all_lines = ""

    try:

        def to_text(x):
            if isinstance(x, bytes):
                return x.decode('utf-8', errors='ignore')
            return x if x is not None else ""

        while True:
            try:
                # 정규식 말고 문자열 포함 여부를 위해 우선 ANY 문자만 매칭
                idx = llm_processor.expect([r'.+', pexpect.EOF, pexpect.TIMEOUT],
                                           timeout=120)
            except pexpect.TIMEOUT:
                print("\n[WARN] Timeout…")
                break
            except pexpect.EOF:
                print("\n[INFO] Process finished")
                break

            # 직전 출력 + 매칭된 문자열(이번 라인 전체)
            before_txt = to_text(llm_processor.before)
            matched_txt = to_text(llm_processor.match.group())

            # 한 줄로 합쳐서 저장
            full_line = before_txt + matched_txt
            all_lines += full_line

            # 문자열 포함 여부만으로 체크
            if "[INFO_TSK]" in full_line:
                break

    except:
        pass

    return [all_lines]




def response_llama_8b(llm_processor):
    all_lines = []

    while True:
        try:
            idx = llm_processor.expect([r'Please enter your question:', pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        except pexpect.TIMEOUT:
            print("\n[WARN] Timeout…")
            break
        except pexpect.EOF:
            print("\n[INFO] Process finished")
            break

        # 스트림 출력 누적
        all_lines.append(llm_processor.before)

        if idx == 0:
            break

    return all_lines


def run_conversation(llm_processor, prompt, execute_info, model):

    if llm_processor is None:
        print("[ERROR] Recrusive process is not running!")
        return None

    llm_processor.sendline(prompt)

    output_lines = []

    if "llama-8B" in execute_info[model]["model"]:
        output_lines = response_llama_8b(llm_processor=llm_processor)

    elif "gguf" in execute_info[model]["model"]:
        output_lines = response_llama_1b3b_gguf(llm_processor=llm_processor)


    inference_result = parse_output_conversation(prompt, output_lines, execute_info, model)
    return inference_result


def remove_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_output_conversation(prompt, output_lines: list, execute_info, model):
    result = {
        "Question": "prompt",
        "Inference Result": "final_output",
        "Detailed Items": "info_text"
    }

    if "llama-8B" in execute_info[model]["model"]:
        try:
            list2string = "".join(output_lines)
            lines = list2string.splitlines()

            info_lines = lines[-1]
            inference_lines = "\n".join(lines[1:-2])
        except:
            inference_lines = "idx error"
            info_lines = "not exist"

        result = {
            "Question": prompt,
            "Inference Result": inference_lines,
            "Detailed Items": info_lines
        }
    elif "gguf" in execute_info[model]["model"]:
        try:
            sep = "".join(output_lines).split("[INFO_TSK]")

            full = sep[0].replace(prompt, "", 1)
            result = {
                "Question": prompt,
                "Inference Result": "".join(full),
                "Detailed Items": rf"[INFO_TSK]{sep[1]}"
            }
        except:
            pass

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

                info_lines.append("")
                info_lines.append("")
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
        "Detailed Items": profile_info
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
        llm_processor = run_conversation_processor(execute_info, model)   # PROCESS 시작


    all_results = []
    idx = 1
    total_cnt = sum(len(v) for v in questions.values())


    cnt_count = 10
    cnt_idx = 0

    for category, question_list in questions.items():    
            
        # print(f"\n{RED}================================ Category: {category} ================================ {RESET}")
        for _, q in enumerate(question_list):

            prompt = q[language]
            # print(f"{prompt}")
            time.sleep(1.5)

            if execute_info[model]["type"] == "One-Shot":
                result = run_single_shot(prompt, execute_info, model)
            else:
                result = run_conversation(llm_processor, prompt, execute_info, model)

            all_results.append(result)
            print(f"{BLUE}=================== 현재 Category: {category},  ({idx}/{total_cnt}) th Test Done. {round(idx/total_cnt*100, 2)} %. ==================={RESET}\n")
            idx += 1

            cnt_idx += 1
            if (cnt_idx % cnt_count) == 0:
                data_save(all_results, language)

    if llm_processor is not None:
        llm_processor.close()

    data_save(all_results, language)
    return all_results

def data_save(all_results, language):
    # Result 폴더 생성
    RESULT_DIR = "Result"
    os.makedirs(RESULT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "full" #if device_exist else "partial"

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
            "Detailed Items": "\n".join(result["Detailed Items"]) if isinstance(result["Detailed Items"],
                                                                                list) else str(result["Detailed Items"])
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


def get_model_info():
    execute_info = {
        "NNC-Mamba": {
            "execute_cmd": "MambaTest",
            "execute_path": "/data/local/tmp/MAMBA/",
            "model": "Mamba",
            "type": "One-Shot"   #"One-Shot"
        },
        "llama-1B": {
            "execute_cmd": "llama-cli",
            "execute_path": "/data/local/tmp/CPU_GPU_LLAMA/",
            "model": "llama-1b.gguf",
            "type": "Recrusive"
        },
        "llama-3B": {
            "execute_cmd": "llama-cli",
            "execute_path": "/data/local/tmp/CPU_GPU_LLAMA/",
            "model": "llama-3B.gguf",
            "type": "Recrusive"
        },
        "llama-8B": {
            "execute_cmd": "llm_executable",
            "execute_path": "/data/local/tmp/",
            "model": "llama-8B",
            "type": "Recrusive"
        }
    }

    return execute_info

if __name__ == "__main__":

    ##################### User Selection #####################

    # test_language = "English"   # 또는 "Chinese"
    test_language = "Chinese"

    # model = "NNC-Mamba"
    model = "llama-8B"
    # model = "llama-1B"
    # model = "llama-3B"

    # scenario_file = "Scenario/test_ces_llm_questions_all_categories_100.json"
    scenario_file = "Scenario/ces_llm_questions_all_categories_100.json"

    ##################### User Selection End #####################
    

    scenario_items = os.path.join(os.getcwd(), scenario_file)

    start_time = time.time()
    results = main(file_path=scenario_items, language=test_language, execute_info=get_model_info(), model=model)
    elapsed_text = format_elapsed_time(start_time)

    print(f"\nTotal Execution Time: {elapsed_text}")






