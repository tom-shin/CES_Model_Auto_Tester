import pexpect
import sys
import json
from datetime import datetime

adb_cmd = (
    'adb shell -t -t "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH; '
    '/data/local/tmp/llm_executable"'
)

child = pexpect.spawn(adb_cmd, encoding='utf-8', timeout=120)
child.logfile_read = sys.stdout

child.expect("Please enter your question:")

questions = ["explain quantum computing in some items."]
results = {}

for q in questions:
    print(f"\n[ASK] {q}")
    child.sendline(q)

    all_lines = []

    while True:
        try:
            idx = child.expect([
                r'Please enter your question:',
                pexpect.EOF,
                pexpect.TIMEOUT
            ], timeout=120)
        except pexpect.TIMEOUT:
            print("\n[WARN] Timeout…")
            break
        except pexpect.EOF:
            print("\n[INFO] Process finished")
            break

        # 스트림 출력 누적
        chunk = child.before
        if chunk:
            for line in chunk.splitlines():
                stripped = line.strip()
                if stripped != "":
                    all_lines.append(stripped)

        if idx == 0:
            break

    # 마지막 [generate tokens so far batch_id] 이후부터 [INFO_TSK] 직전까지 추출
    last_gen_idx = None
    info_idx = None
    for i, line in enumerate(all_lines):
        if line.startswith("[tsk_llama_8b_begin]:"):
            last_gen_idx = i
    for i, line in enumerate(all_lines):
        if line.startswith("[INFO_TSK]"):
            info_idx = i
            break

    if last_gen_idx is not None:
        start = last_gen_idx + 1
    else:
        start = 0

    end = info_idx if info_idx is not None else len(all_lines)
    final_output = "\n".join(all_lines[start:end]).strip()

    results[q] = final_output
    print(f"\n[SAVED FINAL OUTPUT] {final_output}")

child.close()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"llm_result_{timestamp}.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[DONE] Saved to {filename}")
