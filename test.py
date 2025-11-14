import pexpect
import sys

# adb 실행 명령
adb_cmd = (
    'adb shell -t -t "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH; '
    '/data/local/tmp/llm_executable"'
)

# pexpect로 프로세스 실행
child = pexpect.spawn(adb_cmd, encoding='utf-8', timeout=120)
child.logfile_read = sys.stdout  # 실시간 출력

# 첫 번째 질문 전 프롬프트 대기
child.expect("Please enter your question:")

# 질문 목록
questions = ["Hi?", "hello"]

for q in questions:
    print("\n" + "="*50)
    print(f">>> Sending question: {q}")
    print("="*50 + "\n")

    child.sendline(q)

    INFER = ""  # 최종 완성 답변만 저장
    accumulated_output = ""  # 출력 누적

    while True:
        try:
            idx = child.expect([
                r'Please enter your question:',  # 다음 질문 프롬프트
                pexpect.EOF,
                pexpect.TIMEOUT
            ], timeout=120)
        except pexpect.TIMEOUT:
            print("\n[WARN] Timeout waiting for output")
            break
        except pexpect.EOF:
            print("\n[INFO] Process finished")
            break

        # child.before에 나온 출력 누적
        if child.before:
            accumulated_output = child.before.strip()  # 항상 마지막 출력으로 덮어쓰기

        if idx == 0:
            # 다음 질문 프롬프트 도착 → 마지막 누적 출력이 최종 완성 답변
            INFER = accumulated_output
            break
        elif idx == 1:
            # EOF → 마지막 누적 출력 저장
            INFER = accumulated_output
            break

    print("\n" + ">"*50)
    print(f"[INFER] =\n{INFER}")
    print("+"*50 + "\n")

print("\nAll questions processed. Exiting...")
child.close()
